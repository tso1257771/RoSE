"""Joint detector+picker benchmark on RoSE for EQT-RoSE and RED-PAN-60s.

Both models have a detection head and a phase-pick head. Production
deployment uses the detector to gate the picker — a pick is only
"emitted" if it falls inside a triggered detection window. This script
computes the picking metrics (precision/recall/F1, MAE/RMSE) under that
joint operation, and side-by-side against the un-gated picker-only
baseline.

Per (model, trace, threshold):
  1. Run model.classify (or REDPAN.predict) once at low thresholds.
  2. Build:
       PICKS_RAW       — picks above the per-phase threshold
       DETECTIONS_GATE — detection windows above the detection threshold
       PICKS_GATED     — PICKS_RAW filtered to those inside any DETECTION
  3. Compute same-phase TP/FP/FN with bench_pickers_rose.match_picks
     once for each pick-set; report both.

For RoSE local events the gating matters most for *false positives*:
spurious picker peaks on background noise should fall outside detection
boxes. Recall should drop only marginally if the detector is well-
calibrated; precision should rise.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import torch
from obspy import UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks

import seisbench.data as sbd
import seisbench.models as sbm

_THIS_DIR = Path(__file__).resolve().parent          # benchmark/
REPO_ROOT = _THIS_DIR.parent                          # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))                    # sibling bench_* + redpan_inference
from rose.checkpoint_io import safe_torch_load  # noqa: E402

from bench_pickers_rose import (  # noqa: E402
    BenchConfig, _aggregate_summary, evaluate_trace_detections,
    match_picks, match_picks_cross_phase, parse_true_picks, trace_to_stream,
)
from bench_redpan_rose import make_redpan_stream  # noqa: E402


logger = logging.getLogger("bench_joint_rose")


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


# ---------- helpers ----------------------------------------------------------

def gate_picks_by_detections(picks, detections):
    """Return only picks whose peak_time falls inside any detection box."""
    if not detections:
        return []
    intervals = [(d.start_time, d.end_time) for d in detections]
    out = []
    for p in picks:
        for s, e in intervals:
            if s <= p.peak_time <= e:
                out.append(p)
                break
    return out


def picks_from_redpan_arrays(p_arr, s_arr, starttime, dt, p_thr, s_thr,
                             distance):
    out = []
    for prob, phase, thr in ((p_arr, "P", p_thr), (s_arr, "S", s_thr)):
        peaks, props = find_peaks(prob, height=thr, distance=distance)
        for idx, h in zip(peaks, props["peak_heights"]):
            out.append(SimplePick(phase=phase,
                                  peak_time=starttime + float(idx) * dt,
                                  peak_value=float(h)))
    return out


def detections_from_redpan_array(m_arr, starttime, dt, det_thr):
    triggers = trigger_onset(m_arr, det_thr, det_thr)
    out = []
    for on, off in triggers:
        seg = m_arr[on : off + 1]
        peak_idx = on + int(np.argmax(seg))
        out.append(SimpleDetection(
            start_time=starttime + float(on) * dt,
            end_time=starttime + float(off) * dt,
            peak_value=float(m_arr[peak_idx]),
        ))
    return out


# ---------- evaluation -------------------------------------------------------

def evaluate_one_model(predict_fn, test_dataset, indices, cfg,
                      p_thr, s_thr, det_thr) -> dict:
    """Run picker+detector on each trace; produce gated and un-gated stats.

    predict_fn(stream) → (picks, detections, true_picks, starttime).
    Returns dict with 'raw' and 'gated' summaries.
    """
    accumulators = {
        mode: {
            "stats": {
                "P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
                "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
            },
            "mcc_true": [],
            "mcc_pred": [],
            "det_records": [],
        }
        for mode in ("raw", "gated")
    }
    n_evaluated, n_failed = 0, 0
    md = test_dataset.metadata.reset_index(drop=True)
    t0 = time.time()

    for k, idx in enumerate(indices):
        try:
            wf, _ = test_dataset.get_sample(int(idx))
        except Exception:
            n_failed += 1
            continue
        meta = md.iloc[int(idx)]
        true_picks = parse_true_picks(meta)
        if not true_picks:
            continue

        try:
            picks_all, detections_all, _ = predict_fn(wf, meta)
        except Exception as exc:
            logger.debug("predict failed on idx %d: %s", idx, exc)
            n_failed += 1
            continue

        # Apply per-phase pick threshold
        picks_above = [
            p for p in picks_all
            if (p.phase == "P" and p.peak_value >= p_thr)
            or (p.phase == "S" and p.peak_value >= s_thr)
        ]
        # Apply detection threshold
        detections_above = [d for d in detections_all
                            if d.peak_value >= det_thr]
        picks_gated = gate_picks_by_detections(picks_above, detections_above)

        for mode, picks in (("raw", picks_above), ("gated", picks_gated)):
            stats = accumulators[mode]["stats"]
            matched, unmatched = match_picks(picks, true_picks, cfg.pick_tol_s)
            for phase in ("P", "S"):
                if phase in true_picks:
                    if phase in matched:
                        stats[phase]["tp"] += 1
                        stats[phase]["residuals"].append(matched[phase][1])
                    else:
                        stats[phase]["fn"] += 1
            for p in unmatched:
                if p.phase in stats:
                    stats[p.phase]["fp"] += 1
            for tp_, pp_ in match_picks_cross_phase(picks, true_picks,
                                                    cfg.pick_tol_s):
                accumulators[mode]["mcc_true"].append(0 if tp_ == "P" else 1)
                accumulators[mode]["mcc_pred"].append(0 if pp_ == "P" else 1)

        det_summary = evaluate_trace_detections(detections_above, true_picks)
        if det_summary is not None:
            accumulators["raw"]["det_records"].append(det_summary)
            accumulators["gated"]["det_records"].append(det_summary)

        n_evaluated += 1
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(indices) - k - 1)
            logger.info("  %d/%d traces (%.1fs elapsed, ETA %.1fs)",
                        k + 1, len(indices), elapsed, eta)

    elapsed = time.time() - t0
    return {
        mode: _aggregate_summary(
            d["stats"], d["mcc_true"], d["mcc_pred"],
            n_evaluated, n_failed, elapsed,
            det_records=d["det_records"],
        )
        for mode, d in accumulators.items()
    }


# ---------- model wrappers ---------------------------------------------------

def make_eqt_predict_fn(model: sbm.EQTransformer, cfg: BenchConfig):
    def predict(wf, meta):
        stream = trace_to_stream(
            wf, meta, cfg.sampling_rate,
            components=cfg.component_order, bandpass=cfg.bandpass,
        )
        # classify at very low thresholds so we capture everything,
        # then filter post-hoc per pick / detection threshold.
        out = model.classify(stream, P_threshold=0.05, S_threshold=0.05,
                              detection_threshold=0.05)
        picks_eqt = []
        for p in (getattr(out, "picks", []) or []):
            picks_eqt.append(SimplePick(
                phase=str(getattr(p, "phase", "")).upper(),
                peak_time=p.peak_time,
                peak_value=float(getattr(p, "peak_value", 0.0)),
            ))
        detections_eqt = []
        for d in (getattr(out, "detections", []) or []):
            detections_eqt.append(SimpleDetection(
                start_time=d.start_time,
                end_time=d.end_time,
                peak_value=float(getattr(d, "peak_value", 0.0)),
            ))
        return picks_eqt, detections_eqt, stream[0].stats.starttime
    return predict


def make_redpan_predict_fn(redpan, cfg: BenchConfig):
    dt = 1.0 / cfg.sampling_rate
    distance = int(cfg.sampling_rate)

    def predict(wf, meta):
        stream = make_redpan_stream(
            wf, meta, cfg.sampling_rate, bandpass=cfg.bandpass,
        )
        p_arr, s_arr, m_arr = redpan.predict(stream, postprocess=False)
        starttime = stream[0].stats.starttime
        picks_rp = picks_from_redpan_arrays(
            p_arr, s_arr, starttime, dt,
            p_thr=0.05, s_thr=0.05, distance=distance,
        )
        detections_rp = detections_from_redpan_array(
            m_arr, starttime, dt, det_thr=0.05,
        )
        return picks_rp, detections_rp, starttime
    return predict


# ---------- main -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rose-dir", default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-test", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eqt-ckpt",
                    default=str(REPO_ROOT / "models" / "eqt_rose" / "eqt_rose.pt"))
    ap.add_argument("--redpan-tf",
                    default=str(REPO_ROOT / "models" / "redpan_tf60" / "train.hdf5"))
    ap.add_argument("--eqt-p", type=float, default=0.20)
    ap.add_argument("--eqt-s", type=float, default=0.20)
    ap.add_argument("--redpan-p", type=float, default=0.30)
    ap.add_argument("--redpan-s", type=float, default=0.30)
    ap.add_argument("--det-thr", type=float, default=0.30)
    ap.add_argument("--pick-tol-s", type=float, default=0.5)
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

    cfg = BenchConfig(
        rose_dir=args.rose_dir, out_dir=str(out_dir),
        num_test=args.num_test, seed=args.seed,
        p_threshold=0.0, s_threshold=0.0, detection_threshold=0.0,
        pick_tol_s=args.pick_tol_s, sampling_rate=100,
        component_order="ZNE", bandpass=(1.0, 45.0),
    )

    data = sbd.WaveformDataset(
        path=cfg.rose_dir, sampling_rate=cfg.sampling_rate,
        component_order=cfg.component_order, cache=None,
    )
    test = data.test()
    rng = np.random.default_rng(cfg.seed)
    indices = np.sort(rng.choice(len(test), size=cfg.num_test, replace=False))
    logger.info("evaluating %d traces", len(indices))

    # -------- EQT-RoSE --------
    logger.info("=== EQT-RoSE ===")
    state = safe_torch_load(args.eqt_ckpt, map_location="cpu")
    cfg_ckpt = state.get("config", {})
    eqt = sbm.EQTransformer(
        in_samples=int(cfg_ckpt.get("model_window", 6000)),
        sampling_rate=int(cfg_ckpt.get("sampling_rate", 100)),
        phases=["P", "S"], norm="peak",
    )
    eqt.load_state_dict(state["model"])
    eqt.norm = "peak"
    eqt.to("cpu").eval()
    logger.info("loaded EQT (epoch=%s, dev_loss=%.4f)",
                state.get("epoch"), state.get("dev_loss", float("nan")))
    eqt_pred = make_eqt_predict_fn(eqt, cfg)
    eqt_results = evaluate_one_model(
        eqt_pred, test, indices, cfg,
        p_thr=args.eqt_p, s_thr=args.eqt_s, det_thr=args.det_thr,
    )
    del eqt; torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # -------- RED-PAN-60s --------
    logger.info("=== RED-PAN-60s ===")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU TF
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    from rose.redpan_inference.core import REDPAN
    rp_model = tf.keras.models.load_model(args.redpan_tf, compile=False)
    redpan = REDPAN(model=rp_model, pred_npts=6000, dt=0.01,
                    pred_interval_sec=10.0, batch_size=8,
                    use_compiled_infer=False)
    rp_pred = make_redpan_predict_fn(redpan, cfg)
    rp_results = evaluate_one_model(
        rp_pred, test, indices, cfg,
        p_thr=args.redpan_p, s_thr=args.redpan_s, det_thr=args.det_thr,
    )

    rows = []
    for model_name, results in (("EQT-RoSE", eqt_results),
                                ("RED-PAN-60s", rp_results)):
        for mode, summary in results.items():
            det = summary.get("detection", {}) or {}
            for phase in ("P", "S"):
                d = summary.get(phase, {})
                rows.append({
                    "model": model_name, "mode": mode, "phase": phase,
                    "tp": d.get("tp"), "fp": d.get("fp"), "fn": d.get("fn"),
                    "precision": d.get("precision"),
                    "recall": d.get("recall"),
                    "f1": d.get("f1"),
                    "mae_s": d.get("mae_s"),
                    "rmse_s": d.get("rmse_s"),
                    "mcc_phase_id": summary.get("mcc_phase_id"),
                    "mcc_phase_accuracy": summary.get("mcc_phase_accuracy"),
                    "mcc_n_phase_confusions": summary.get(
                        "mcc_n_phase_confusions"),
                    "det_recall": det.get("det_recall"),
                    "det_iou_mean": det.get("det_iou_mean"),
                    "det_per_trace_mean": det.get("det_per_trace_mean"),
                    "n_evaluated": summary.get("n_evaluated"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "joint_comparison.csv", index=False)
    pd.set_option("display.float_format",
                  lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    logger.info("\n%s", df.to_string(index=False))

    with (out_dir / "summary.json").open("w") as fh:
        json.dump({"EQT-RoSE": eqt_results,
                   "RED-PAN-60s": rp_results}, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
