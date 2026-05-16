"""Benchmark TF-RED-PAN (MTAN R2U-Net 60s) on the RoSE test split.

Mirrors bench_pickers_rose.py exactly (same 2000-trace random sample,
same threshold sweep, same matching tolerance, same metrics) so the
resulting CSV merges cleanly with the SeisBench benchmark output.

The TF model has two heads:
  * picker:   (B, 6000, 3) → P / S / N per-sample probabilities
  * detector: (B, 6000, 2) → event / no-event probabilities

We run sliding-window inference over each ~143 s test trace via the
``redpan.core.REDPAN`` wrapper (spectrum-matched noise padding +
direct-array accumulation). The 1-D probability arrays returned are
treated identically to a SeisBench ``annotation`` Stream:

  * P / S picks via scipy.find_peaks(prob, height=thr, distance=100)
  * detections via obspy.signal.trigger.trigger_onset(M, thr, thr)

Same-phase precision/recall/F1, cross-phase MCC, residual stats, and
detection-box metrics use the helpers from bench_pickers_rose.py to
guarantee numeric parity.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Configure TF before any heavy import
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks

import seisbench.data as sbd

_THIS_DIR = Path(__file__).resolve().parent          # phase_picking/benchmark/
REPO_ROOT = _THIS_DIR.parent.parent          # repo root (phase_picking/benchmark -> ..)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))                    # sibling bench_* + redpan_inference

# Import metric helpers from the SeisBench bench script for parity.
from bench_pickers_rose import (  # noqa: E402
    BenchConfig,
    _aggregate_summary,
    evaluate_trace_detections,
    match_picks,
    match_picks_cross_phase,
    parse_true_picks,
    trace_to_stream,
)


logger = logging.getLogger("bench_redpan_rose")


@dataclass
class SimplePick:
    phase: str
    peak_time: UTCDateTime
    peak_value: float


@dataclass
class SimpleDetection:
    start_time: UTCDateTime
    end_time: UTCDateTime
    peak_value: float


def _ensure_cudnn_on_ld_library_path() -> None:
    """Re-exec ourselves with LD_LIBRARY_PATH including the pip-bundled
    cuDNN dir, if it's not already there. TF 2.16 was built against
    cuDNN 8.9; on boxes whose system cuDNN is older (e.g. 8.8.1 under
    /usr/local/cuda-11.8/) TF crashes every conv1d with "No DNN in
    stream executor". ld.so reads LD_LIBRARY_PATH at exec time — not
    from runtime os.environ changes — so we have to re-exec.

    The lookup path matches `_pipeline._bundled_cudnn_lib_dir` (derived
    from `sys.executable` so editable installs and venvs both resolve
    to the right interpreter's site-packages).

    No-op if the bundled libcudnn dir is absent, or already on the
    path. Only call this from a __main__ entry point; importing the
    module shouldn't replace the host process.
    """
    import sys
    py_lib = Path(sys.executable).resolve().parent.parent / "lib" \
        / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "nvidia" / "cudnn" / "lib"
    if not py_lib.is_dir() or not any(py_lib.glob("libcudnn*.so*")):
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    if str(py_lib) in current.split(":"):
        return
    os.environ["LD_LIBRARY_PATH"] = f"{py_lib}:{current}" if current else str(py_lib)
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)


def configure_tf_memory_growth(intra_threads: int | None = None):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    if gpus:
        logger.info("TF GPU enabled: %s", [g.name for g in gpus])
    # When running on GPU, intra/inter-op threading doesn't help; only set
    # it when forced to CPU (via CUDA_VISIBLE_DEVICES="" or absent GPU).
    if not gpus and intra_threads is not None and intra_threads > 0:
        try:
            tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
            tf.config.threading.set_inter_op_parallelism_threads(intra_threads)
        except Exception:
            pass


def load_redpan(model_path: str, pred_interval_sec: float = 10.0,
                batch_size: int = 32):
    """Load the TF model + wrap in the RED-PAN sliding-window inferer."""
    import tensorflow as tf
    from rose.redpan_inference.core import REDPAN
    logger.info("loading TF model from %s", model_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("loaded TF model (%d params); pred_npts=6000",
                model.count_params())
    redpan = REDPAN(
        model=model, pred_npts=6000, dt=0.01,
        pred_interval_sec=pred_interval_sec,
        batch_size=batch_size,
        use_compiled_infer=True,
    )
    return redpan


def picks_from_arrays(
    p_arr: np.ndarray, s_arr: np.ndarray,
    starttime: UTCDateTime, dt: float,
    p_thresh: float, s_thresh: float, distance: int,
) -> list[SimplePick]:
    out: list[SimplePick] = []
    for prob, phase, thresh in (
        (p_arr, "P", p_thresh), (s_arr, "S", s_thresh),
    ):
        peaks, props = find_peaks(prob, height=thresh, distance=distance)
        for idx, height in zip(peaks, props["peak_heights"]):
            out.append(SimplePick(
                phase=phase,
                peak_time=starttime + float(idx) * dt,
                peak_value=float(height),
            ))
    return out


def detections_from_array(
    m_arr: np.ndarray, starttime: UTCDateTime, dt: float,
    det_thresh: float,
) -> list[SimpleDetection]:
    triggers = trigger_onset(m_arr, det_thresh, det_thresh)
    out: list[SimpleDetection] = []
    for on_idx, off_idx in triggers:
        seg = m_arr[on_idx : off_idx + 1]
        peak_idx = on_idx + int(np.argmax(seg))
        out.append(SimpleDetection(
            start_time=starttime + float(on_idx) * dt,
            end_time=starttime + float(off_idx) * dt,
            peak_value=float(m_arr[peak_idx]),
        ))
    return out


def make_redpan_stream(wf_zne: np.ndarray, meta, sampling_rate: float,
                       bandpass: tuple[float, float] | None):
    """Reorder ZNE → ENZ for RED-PAN.

    RoSE WaveformDataset is configured with component_order='ZNE' so the
    waveform array is [Z, N, E]. RED-PAN's TF model was trained on
    [E, N, Z] (per redpan/legacy/data_aug.py:207). Feeding ZNE-ordered
    data yields essentially garbage (verified by viz_models_rose.py).
    """
    wf_enz = wf_zne[[2, 1, 0]]  # [Z, N, E] → [E, N, Z]
    return trace_to_stream(wf_enz, meta, sampling_rate,
                           components="ENZ", bandpass=bandpass)


def _empty_per_thr(thresholds: list[float]) -> dict[float, dict]:
    return {
        t: {
            "stats": {
                "P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
                "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
            },
            "mcc_true": [],
            "mcc_pred": [],
            "det_records": [],
        }
        for t in thresholds
    }


def _partial_path(out_dir: Path, model_name: str) -> Path:
    """`<out_dir>/<model>.partial.json` — incremental dump location."""
    return out_dir / f"{model_name}.partial.json"


def _dump_partial(path: Path, per_thr: dict[float, dict],
                  completed: set[int], n_evaluated: int, n_failed: int,
                  elapsed_s: float) -> None:
    """Atomically dump in-progress state so the run can resume after a kill.

    JSON keys are stringified floats (JSON doesn't preserve float keys);
    every other value is naturally JSON-serializable. Atomic via
    write-then-rename so a kill mid-write can't leave a corrupt file.
    """
    payload = {
        "per_thr": {str(t): d for t, d in per_thr.items()},
        "completed_indices": sorted(completed),
        "n_evaluated": n_evaluated,
        "n_failed": n_failed,
        "elapsed_s": elapsed_s,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, default=str))
    tmp.replace(path)


def _load_partial(path: Path, thresholds: list[float]
                  ) -> tuple[dict[float, dict], set[int], int, int, float] | None:
    """Inverse of `_dump_partial`. Returns None if the file's threshold set
    doesn't exactly match the current sweep (resume only works when the
    sweep grid is identical)."""
    if not path.is_file():
        return None
    payload = json.loads(path.read_text())
    saved = {float(s) for s in payload["per_thr"]}
    if saved != set(thresholds):
        logger.warning("partial file %s has threshold set %s != current %s; ignoring",
                       path, sorted(saved), sorted(thresholds))
        return None
    per_thr = {float(t): d for t, d in payload["per_thr"].items()}
    completed = set(int(i) for i in payload["completed_indices"])
    return (per_thr, completed,
            int(payload["n_evaluated"]), int(payload["n_failed"]),
            float(payload.get("elapsed_s", 0.0)))


def evaluate_redpan_sweep(
    redpan, test_dataset: sbd.WaveformDataset,
    indices: np.ndarray, cfg: BenchConfig, thresholds: list[float],
    *, partial_path: Path | None = None, save_every: int = 500,
) -> dict:
    """Sweep RED-PAN over `indices`, optionally checkpointing to
    ``partial_path`` every ``save_every`` evaluated traces so the run can
    resume after a kill. Pass ``partial_path=None`` to disable checkpointing.
    """
    base_thresh = min(thresholds)
    per_thr = _empty_per_thr(thresholds)
    n_evaluated, n_failed = 0, 0
    completed: set[int] = set()
    elapsed_carry = 0.0
    # Track failure modes so a silent 100% n_failed loop is no longer invisible.
    failure_counts: Counter[str] = Counter()
    LOG_FIRST_N = 5  # surface the first few full tracebacks at WARNING

    # Resume if a usable partial dump exists.
    if partial_path is not None:
        loaded = _load_partial(partial_path, thresholds)
        if loaded is not None:
            per_thr, completed, n_evaluated, n_failed, elapsed_carry = loaded
            logger.info(
                "resuming from %s: %d already done (%d evaluated, %d failed, "
                "%.0fs prior elapsed)",
                partial_path, len(completed), n_evaluated, n_failed, elapsed_carry,
            )

    md = test_dataset.metadata.reset_index(drop=True)
    t0 = time.time()
    dt = 1.0 / cfg.sampling_rate

    for k, idx in enumerate(indices):
        if int(idx) in completed:
            continue
        try:
            wf, _ = test_dataset.get_sample(int(idx))
        except Exception as exc:
            key = f"get_sample/{type(exc).__name__}"
            if failure_counts[key] < LOG_FIRST_N:
                logger.warning("get_sample failed on idx %d: %s: %s", idx, type(exc).__name__, exc)
            failure_counts[key] += 1
            n_failed += 1
            continue
        meta = md.iloc[int(idx)]
        true_picks = parse_true_picks(meta)
        if not true_picks:
            continue

        # RED-PAN expects ENZ channel ordering (not ZNE).
        stream = make_redpan_stream(
            wf, meta, cfg.sampling_rate, bandpass=cfg.bandpass,
        )
        try:
            p_arr, s_arr, m_arr = redpan.predict(stream, postprocess=False)
        except Exception as exc:
            key = f"predict/{type(exc).__name__}"
            if failure_counts[key] < LOG_FIRST_N:
                logger.warning(
                    "predict failed on idx %d: %s: %s", idx, type(exc).__name__, exc,
                    exc_info=failure_counts[key] == 0,  # full traceback on the very first
                )
            failure_counts[key] += 1
            n_failed += 1
            continue

        starttime = stream[0].stats.starttime

        all_picks = picks_from_arrays(
            p_arr, s_arr, starttime, dt,
            p_thresh=base_thresh, s_thresh=base_thresh,
            distance=int(cfg.sampling_rate),  # 1 s spacing
        )

        for thr in thresholds:
            picks_t = [p for p in all_picks if p.peak_value >= thr]
            detections_t = detections_from_array(m_arr, starttime, dt, thr)

            stats = per_thr[thr]["stats"]
            matched, unmatched = match_picks(picks_t, true_picks, cfg.pick_tol_s)
            for phase in ("P", "S"):
                if phase in true_picks:
                    if phase in matched:
                        stats[phase]["tp"] += 1
                        stats[phase]["residuals"].append(matched[phase][1])
                    else:
                        stats[phase]["fn"] += 1
            for p in unmatched:
                ph = str(getattr(p, "phase", "")).upper()
                if ph in stats:
                    stats[ph]["fp"] += 1
            for true_phase, pred_phase in match_picks_cross_phase(
                picks_t, true_picks, cfg.pick_tol_s,
            ):
                per_thr[thr]["mcc_true"].append(0 if true_phase == "P" else 1)
                per_thr[thr]["mcc_pred"].append(0 if pred_phase == "P" else 1)

            det_summary = evaluate_trace_detections(detections_t, true_picks)
            if det_summary is not None:
                per_thr[thr]["det_records"].append(det_summary)

        n_evaluated += 1
        completed.add(int(idx))
        if (k + 1) % 100 == 0:
            elapsed = elapsed_carry + (time.time() - t0)
            done = n_evaluated + n_failed
            eta = (elapsed / done * (len(indices) - done)) if done else 0.0
            logger.info("  %d/%d traces (%.1fs elapsed, ETA %.1fs)",
                        done, len(indices), elapsed, eta)
        if (partial_path is not None and save_every
                and n_evaluated > 0 and n_evaluated % save_every == 0):
            _dump_partial(
                partial_path, per_thr, completed, n_evaluated, n_failed,
                elapsed_carry + (time.time() - t0),
            )

    elapsed = elapsed_carry + (time.time() - t0)
    # Clean shutdown: the partial file is no longer needed once we have the
    # final aggregated result. Leave it on exception (caller writes final).
    if partial_path is not None and partial_path.is_file():
        try:
            partial_path.unlink()
        except OSError:
            pass
    if n_failed:
        top = failure_counts.most_common(5)
        breakdown = ", ".join(f"{k}={v}" for k, v in top)
        log_fn = logger.error if n_failed == len(indices) else logger.warning
        log_fn("evaluation summary: %d/%d traces failed (top causes: %s)",
               n_failed, len(indices), breakdown)
    return {
        thr: _aggregate_summary(
            d["stats"], d["mcc_true"], d["mcc_pred"],
            n_evaluated, n_failed, elapsed,
            det_records=d["det_records"],
        )
        for thr, d in per_thr.items()
    }


def main() -> None:
    # Ensure the right cuDNN is visible to ld.so BEFORE anything imports TF.
    # This re-execs if LD_LIBRARY_PATH needs amending. No-op when the
    # wrapping subprocess (run_inference.py via threadcap_env) already set it.
    _ensure_cudnn_on_ld_library_path()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-path", required=True,
                    help="Path to TF Keras model (.hdf5)")
    ap.add_argument("--rose-dir",
                    default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pick-tol-s", type=float, default=0.5,
                    help="Uniform tolerance (s). Ignored if "
                         "--pick-tol-p / --pick-tol-s-phase set.")
    ap.add_argument("--pick-tol-p", type=float, default=None,
                    help="P-pick tolerance (s) — pick-benchmark uses 0.5.")
    ap.add_argument("--pick-tol-s-phase", type=float, default=None,
                    help="S-pick tolerance (s) — pick-benchmark uses 1.0.")
    ap.add_argument("--sampling-rate", type=int, default=100)
    ap.add_argument("--component-order", default="ZNE")
    ap.add_argument("--bandpass-low", type=float, default=1.0)
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    ap.add_argument("--pred-interval-sec", type=float, default=10.0,
                    help="Sliding-window step for the inference loop.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--sweep-thresholds",
                    default="0.05,0.1,0.2,0.3,0.5,0.7")
    ap.add_argument("--tf-threads", type=int, default=0,
                    help="Cap TF intra/inter op threads (0 = no cap). "
                         "Set to 2 when running alongside other CPU jobs.")
    ap.add_argument("--model-name", default="RED-PAN-60s",
                    help="Display name in the output CSV.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from <out_dir>/<model>.partial.json if it "
                         "exists. Partial state is dumped every "
                         "--save-every traces during evaluation.")
    ap.add_argument("--save-every", type=int, default=500,
                    help="Dump partial state every N successful evaluations "
                         "(0 = never). Only meaningful with --resume.")
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / "bench.log")],
        force=True,
    )

    bandpass = ((args.bandpass_low, args.bandpass_high)
                if args.bandpass_low > 0 and args.bandpass_high > 0
                else None)
    if args.pick_tol_p is not None and args.pick_tol_s_phase is not None:
        pick_tol = {"P": float(args.pick_tol_p),
                    "S": float(args.pick_tol_s_phase)}
    else:
        pick_tol = float(args.pick_tol_s)
    cfg = BenchConfig(
        rose_dir=args.rose_dir, out_dir=str(out_dir),
        num_test=args.num_test, seed=args.seed,
        p_threshold=0.0, s_threshold=0.0, detection_threshold=0.0,
        pick_tol_s=pick_tol,
        sampling_rate=args.sampling_rate,
        component_order=args.component_order,
        bandpass=bandpass,
    )
    logger.info("config = %s", json.dumps(cfg.__dict__, indent=2))

    configure_tf_memory_growth(intra_threads=args.tf_threads or None)

    data = sbd.WaveformDataset(
        path=cfg.rose_dir, sampling_rate=cfg.sampling_rate,
        component_order=cfg.component_order, cache=None,
    )
    test = data.test()
    logger.info("RoSE test split: %d traces", len(test))

    rng = np.random.default_rng(cfg.seed)
    if 0 < cfg.num_test < len(test):
        indices = np.sort(rng.choice(len(test), size=cfg.num_test, replace=False))
    else:
        indices = np.arange(len(test))
    logger.info("evaluating %d traces", len(indices))

    thresholds = sorted(
        {float(x.strip()) for x in args.sweep_thresholds.split(",")
         if x.strip()}
    )
    logger.info("threshold sweep: %s", thresholds)

    redpan = load_redpan(
        args.model_path,
        pred_interval_sec=args.pred_interval_sec,
        batch_size=args.batch_size,
    )

    partial_path = _partial_path(out_dir, args.model_name) if args.resume else None
    sweep = evaluate_redpan_sweep(
        redpan, test, indices, cfg, thresholds,
        partial_path=partial_path, save_every=int(args.save_every),
    )

    out_json = out_dir / f"{args.model_name}.json"
    with out_json.open("w") as fh:
        json.dump({str(t): s for t, s in sweep.items()}, fh, indent=2,
                  default=str)
    logger.info("wrote %s", out_json)

    rows = []
    for thr in thresholds:
        s = sweep[thr]
        det = s.get("detection", {}) or {}
        for phase in ("P", "S"):
            d = s.get(phase, {})
            rows.append({
                "model": args.model_name,
                "phase": phase,
                "threshold": thr,
                "tp": d.get("tp"), "fp": d.get("fp"), "fn": d.get("fn"),
                "precision": d.get("precision"),
                "recall": d.get("recall"),
                "f1": d.get("f1"),
                "mae_s": d.get("mae_s"),
                "rmse_s": d.get("rmse_s"),
                "median_s": d.get("median_s"),
                "iqr_s": d.get("iqr_s"),
                "n_residuals": d.get("n_residuals"),
                "mcc_phase_id": s.get("mcc_phase_id"),
                "mcc_phase_accuracy": s.get("mcc_phase_accuracy"),
                "mcc_n_pairs": s.get("mcc_n_pairs"),
                "mcc_n_phase_confusions": s.get("mcc_n_phase_confusions"),
                "det_recall": det.get("det_recall"),
                "det_iou_mean": det.get("det_iou_mean"),
                "det_per_trace_mean": det.get("det_per_trace_mean"),
                "det_start_mae_s": det.get("det_start_mae_s"),
                "det_end_mae_s": det.get("det_end_mae_s"),
                "n_evaluated": s.get("n_evaluated"),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "sweep_comparison.csv", index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 240)
    pd.set_option("display.float_format",
                  lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
