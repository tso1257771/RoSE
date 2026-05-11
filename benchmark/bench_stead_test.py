"""STEAD-test benchmark with proper T1 event-vs-noise metrics.

Loads a prepared STEAD test split (paired event + noise traces, npy files,
ENZ-ordered, 27 000 samples @ 100 Hz) — this is the missing piece for the
RoSE benchmark: STEAD's noise traces give us real T1 negatives, so we can
compute event-detection F1 / AUC alongside the per-phase precision/recall/F1,
MCC, and onset-residual metrics.

Inputs (point ``--stead-dir`` or ``$STEAD_DIR`` at this directory):
  $STEAD_DIR/
    metadata.csv         103,040 events (cols: trace_name, p_sample,
                          s_sample, p_sample_original, s_sample_original,
                          ps_residual_sec, ps_group)
    metadata_noise.csv    23,526 noise (col: trace_name)
    waveforms/*.npy       (27 000, 3) float32, ENZ
    noise_waveforms/*.npy (27 000, 3) float32, ENZ

The exact ``metadata.csv`` / ``metadata_noise.csv`` files used to produce the
published numbers are pinned under
``application/seisbench-rose-benchmark/data/stead_{test,noise}_index.csv``.

Models evaluated:
  * EQT-RoSE-v3              (PyTorch / SeisBench)
  * RED-PAN-60s              (TF Keras, ENZ-native)
  * EQT-instance pretrained  (SeisBench, optional pretrained reference)

Usage:
    export STEAD_DIR=/path/to/STEAD/benchmark_stead
    python benchmark/bench_stead_test.py \\
        --out-dir eval/bench_stead_test \\
        --num-events 2000 --num-noise 1000 \\
        --models eqt_rose_v3 redpan eqt_instance
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
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks
from sklearn.metrics import f1_score, roc_auc_score

import seisbench.models as sbm

_THIS_DIR = Path(__file__).resolve().parent          # benchmark/
REPO_ROOT = _THIS_DIR.parent                          # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))                    # sibling bench_* + redpan_inference
from rose.checkpoint_io import safe_torch_load  # noqa: E402

from bench_pickers_rose import (  # noqa: E402
    _aggregate_summary, evaluate_trace_detections,
    match_picks, match_picks_cross_phase,
)


logger = logging.getLogger("bench_stead_test")


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


# ---------- STEAD test loaders -----------------------------------------------

STEAD_DEFAULT = Path(os.environ["STEAD_DIR"]) if "STEAD_DIR" in os.environ else None
SAMPLING_RATE = 100.0
DT = 1.0 / SAMPLING_RATE
# Synthetic starttime for npy traces; only relative offsets matter.
SYNTH_T0 = UTCDateTime("2000-01-01")


def load_stead_event(idx: int, name: str, p_sample: int, s_sample: int,
                     wf_dir: Path):
    """Returns (waveform_enz, true_picks dict, starttime)."""
    arr = np.load(wf_dir / f"{name}.npy")  # (T, 3) ENZ
    starttime = SYNTH_T0 + idx * 1000.0  # spread starttimes apart
    p_time = starttime + p_sample * DT
    s_time = starttime + s_sample * DT
    true_picks = {"P": p_time, "S": s_time}
    return arr, true_picks, starttime


def load_stead_noise(idx: int, name: str, wf_dir: Path):
    arr = np.load(wf_dir / f"{name}.npy")
    starttime = SYNTH_T0 + (idx + 100000) * 1000.0
    return arr, starttime


def array_to_stream(arr_enz: np.ndarray, starttime: UTCDateTime,
                    components_for_model: str) -> Stream:
    """Build a 3-component ObsPy Stream from a STEAD ENZ array.

    components_for_model:
      "ENZ" — keep order (RED-PAN convention, native)
      "ZNE" — reorder to Z,N,E (SeisBench convention)
    """
    if components_for_model == "ENZ":
        data = arr_enz
        chan_order = ("E", "N", "Z")
    elif components_for_model == "ZNE":
        data = arr_enz[:, [2, 1, 0]]
        chan_order = ("Z", "N", "E")
    else:
        raise ValueError(components_for_model)
    traces = []
    for i, comp in enumerate(chan_order):
        tr = Trace(data=data[:, i].astype(np.float32))
        tr.stats.starttime = starttime
        tr.stats.sampling_rate = SAMPLING_RATE
        tr.stats.network = "XX"
        tr.stats.station = "STA"
        tr.stats.channel = f"HH{comp}"
        traces.append(tr)
    s = Stream(traces)
    return s


def apply_bandpass(stream: Stream, low: float, high: float) -> Stream:
    stream = stream.copy()
    stream.detrend("demean")
    stream.filter("bandpass", freqmin=low, freqmax=high,
                  corners=4, zerophase=True)
    return stream


# ---------- model wrappers ---------------------------------------------------

def predict_seisbench(model, stream: Stream):
    out = model.classify(stream, P_threshold=0.05, S_threshold=0.05,
                         detection_threshold=0.05)
    picks = []
    for p in (getattr(out, "picks", []) or []):
        picks.append(SimplePick(
            phase=str(getattr(p, "phase", "")).upper(),
            peak_time=p.peak_time,
            peak_value=float(getattr(p, "peak_value", 0.0)),
        ))
    detections = []
    for d in (getattr(out, "detections", []) or []):
        detections.append(SimpleDetection(
            start_time=d.start_time, end_time=d.end_time,
            peak_value=float(getattr(d, "peak_value", 0.0)),
        ))
    return picks, detections


def predict_redpan(redpan, stream: Stream):
    p_arr, s_arr, m_arr = redpan.predict(stream, postprocess=False)
    starttime = stream[0].stats.starttime
    picks = []
    for prob, phase in ((p_arr, "P"), (s_arr, "S")):
        peaks, props = find_peaks(prob, height=0.05,
                                  distance=int(SAMPLING_RATE))
        for idx, h in zip(peaks, props["peak_heights"]):
            picks.append(SimplePick(
                phase=phase,
                peak_time=starttime + float(idx) * DT,
                peak_value=float(h),
            ))
    detections = []
    triggers = trigger_onset(m_arr, 0.05, 0.05)
    for on, off in triggers:
        seg = m_arr[on : off + 1]
        peak_idx = on + int(np.argmax(seg))
        detections.append(SimpleDetection(
            start_time=starttime + float(on) * DT,
            end_time=starttime + float(off) * DT,
            peak_value=float(m_arr[peak_idx]),
        ))
    return picks, detections, m_arr


# ---------- evaluation -------------------------------------------------------

def evaluate_model(name, model_kind, model, ev_df, noise_df, ev_wf_dir,
                   nz_wf_dir, n_events, n_noise,
                   thresholds: list[float],
                   pick_tol_s=None, bandpass=(1.0, 45.0),
                   shard: int = 0, total_shards: int = 1):
    """If total_shards > 1, evaluate only this process's slice of the
    sampled (event + noise) indices: shard k of N takes indices where
    (i % N) == k. Use to parallelize across processes; merge per-shard
    JSONs after."""
    """Run model.classify ONCE per trace at the minimum threshold;
    compute metrics at every threshold in `thresholds` by post-hoc
    filtering. Returns {threshold (str): summary}.

    pick_tol_s: float (uniform) or dict {"P": 0.5, "S": 1.0} —
    pick-benchmark convention default.
    """
    if pick_tol_s is None:
        pick_tol_s = {"P": 0.5, "S": 1.0}
    if not thresholds:
        thresholds = [0.30]
    base_thr = float(min(thresholds))
    thresholds = sorted(set(float(t) for t in thresholds))
    rng = np.random.default_rng(42)
    ev_idx = rng.choice(len(ev_df), size=min(n_events, len(ev_df)),
                        replace=False)
    nz_idx = rng.choice(len(noise_df), size=min(n_noise, len(noise_df)),
                        replace=False)
    if total_shards > 1:
        ev_idx_full, nz_idx_full = ev_idx, nz_idx
        ev_idx = ev_idx[shard::total_shards]
        nz_idx = nz_idx[shard::total_shards]
        logger.info("[%s] shard %d/%d → %d events + %d noise (full set %d/%d)",
                    name, shard, total_shards, len(ev_idx), len(nz_idx),
                    len(ev_idx_full), len(nz_idx_full))
    else:
        logger.info("[%s] %d events + %d noise", name, len(ev_idx), len(nz_idx))

    # Per-threshold accumulators — same structure as bench_pickers_rose.py
    per_thr: dict[float, dict] = {
        t: {
            "stats": {
                "P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
                "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
            },
            "mcc_true": [], "mcc_pred": [], "det_records": [],
            "t1_y_true": [], "t1_y_pred": [],
        }
        for t in thresholds
    }
    event_max_det = []   # collected once; AUC is threshold-independent
    noise_max_det = []

    n_failed = 0
    t0 = time.time()
    components = "ENZ" if model_kind == "redpan" else "ZNE"

    # ---- events ----
    for k, i in enumerate(ev_idx):
        row = ev_df.iloc[int(i)]
        try:
            arr, true_picks, st0 = load_stead_event(
                int(i), row["trace_name"], int(row["p_sample"]),
                int(row["s_sample"]), ev_wf_dir,
            )
        except Exception:
            n_failed += 1
            continue
        stream = array_to_stream(arr, st0, components)
        if bandpass is not None:
            stream = apply_bandpass(stream, *bandpass)

        try:
            if model_kind == "redpan":
                picks, detections, m_arr = predict_redpan(model, stream)
                event_max_det.append(float(m_arr.max()))
            else:
                picks, detections = predict_seisbench(model, stream)
                event_max_det.append(
                    max([d.peak_value for d in detections] + [0.0])
                )
        except Exception as exc:
            logger.debug("predict failed (event %d): %s", i, exc)
            n_failed += 1
            continue

        # Loop over thresholds, filtering the same picks/detections list
        for thr in thresholds:
            picks_above = [
                p for p in picks
                if (p.phase == "P" and p.peak_value >= thr)
                or (p.phase == "S" and p.peak_value >= thr)
            ]
            acc = per_thr[thr]
            acc["t1_y_true"].append(1)
            acc["t1_y_pred"].append(1 if len(picks_above) > 0 else 0)

            matched, unmatched = match_picks(picks_above, true_picks,
                                             pick_tol_s)
            for ph in ("P", "S"):
                if ph in true_picks:
                    if ph in matched:
                        acc["stats"][ph]["tp"] += 1
                        acc["stats"][ph]["residuals"].append(
                            matched[ph][1])
                    else:
                        acc["stats"][ph]["fn"] += 1
            for p in unmatched:
                if p.phase in acc["stats"]:
                    acc["stats"][p.phase]["fp"] += 1
            for tp_, pp_ in match_picks_cross_phase(picks_above, true_picks,
                                                    pick_tol_s):
                acc["mcc_true"].append(0 if tp_ == "P" else 1)
                acc["mcc_pred"].append(0 if pp_ == "P" else 1)

            detections_above = [d for d in detections
                                if d.peak_value >= thr]
            det_summary = evaluate_trace_detections(detections_above,
                                                    true_picks)
            if det_summary is not None:
                acc["det_records"].append(det_summary)

        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(ev_idx) + len(nz_idx) - k - 1)
            logger.info("  events %d/%d (%.0fs elapsed, ETA %.0fs)",
                        k + 1, len(ev_idx), elapsed, eta)

    # ---- noise ----
    for k, i in enumerate(nz_idx):
        row = noise_df.iloc[int(i)]
        try:
            arr, st0 = load_stead_noise(int(i), row["trace_name"], nz_wf_dir)
        except Exception:
            n_failed += 1
            continue
        stream = array_to_stream(arr, st0, components)
        if bandpass is not None:
            stream = apply_bandpass(stream, *bandpass)
        try:
            if model_kind == "redpan":
                picks, _, m_arr = predict_redpan(model, stream)
                noise_max_det.append(float(m_arr.max()))
            else:
                picks, detections = predict_seisbench(model, stream)
                noise_max_det.append(
                    max([d.peak_value for d in detections] + [0.0])
                )
        except Exception as exc:
            logger.debug("predict failed (noise %d): %s", i, exc)
            n_failed += 1
            continue

        # T1 noise contribution per threshold
        for thr in thresholds:
            picks_above_nz = [
                p for p in picks
                if (p.phase == "P" and p.peak_value >= thr)
                or (p.phase == "S" and p.peak_value >= thr)
            ]
            per_thr[thr]["t1_y_true"].append(0)
            per_thr[thr]["t1_y_pred"].append(
                1 if len(picks_above_nz) > 0 else 0
            )

        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (len(ev_idx) + k + 1) * (len(nz_idx) - k - 1)
            logger.info("  noise %d/%d (%.0fs elapsed, ETA %.0fs)",
                        k + 1, len(nz_idx), elapsed, eta)

    # AUC of detection-head score is threshold-independent (computed once)
    ev_scores = np.asarray(event_max_det, dtype=np.float64)
    nz_scores = np.asarray(noise_max_det, dtype=np.float64)
    if (ev_scores.size and nz_scores.size
        and (ev_scores.max() > 0 or nz_scores.max() > 0)):
        ys_auc = np.concatenate([np.ones_like(ev_scores, dtype=int),
                                 np.zeros_like(nz_scores, dtype=int)])
        sc_auc = np.concatenate([ev_scores, nz_scores])
        try:
            auc_det = float(roc_auc_score(ys_auc, sc_auc))
        except ValueError:
            auc_det = float("nan")
    else:
        auc_det = float("nan")

    from sklearn.metrics import (precision_score, recall_score,
                                 matthews_corrcoef as _mcc)
    elapsed = time.time() - t0
    out_per_thr: dict[str, dict] = {}
    for thr in thresholds:
        acc = per_thr[thr]
        summary = _aggregate_summary(
            acc["stats"], acc["mcc_true"], acc["mcc_pred"],
            n_evaluated=len(ev_idx) + len(nz_idx),
            n_failed=n_failed, elapsed_s=elapsed,
            det_records=acc["det_records"],
        )
        if acc["t1_y_true"]:
            yt = np.asarray(acc["t1_y_true"])
            yp = np.asarray(acc["t1_y_pred"])
            try:
                mcc_v = float(_mcc(yt, yp))
            except ValueError:
                mcc_v = float("nan")
            summary["T1"] = {
                "threshold": thr,
                "n_event": int((yt == 1).sum()),
                "n_noise": int((yt == 0).sum()),
                "f1": float(f1_score(yt, yp, zero_division=0)),
                "precision": float(precision_score(yt, yp, zero_division=0)),
                "recall": float(recall_score(yt, yp, zero_division=0)),
                "mcc": mcc_v,
                "auc_det_head": auc_det,
                "definition": ("any pick emitted at threshold "
                               "(Münchmeyer 2022)"),
            }
        else:
            summary["T1"] = {"f1": float("nan"), "auc_det_head": auc_det}
        out_per_thr[str(thr)] = summary
    return out_per_thr


# ---------- main -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stead-dir", default=(str(STEAD_DEFAULT) if STEAD_DEFAULT else None))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-events", type=int, default=2000)
    ap.add_argument("--num-noise", type=int, default=1000)
    ap.add_argument("--pick-tol-p", type=float, default=0.5,
                    help="P-pick matching tolerance (s). pick-benchmark = 0.5.")
    ap.add_argument("--pick-tol-s", type=float, default=1.0,
                    help="S-pick matching tolerance (s). pick-benchmark = 1.0 "
                         "(asymmetric — S onsets are noisier than P).")
    ap.add_argument("--bandpass-low", type=float, default=1.0)
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    ap.add_argument("--models", nargs="+",
                    default=["eqt_rose_v3", "redpan", "eqt_instance"],
                    choices=["eqt_rose_v3", "redpan", "eqt_instance",
                             "phasenet_rose_v2", "phasenet_instance",
                             "eqt_ethz", "phasenet_ethz",
                             "eqt_stead", "phasenet_stead"])
    ap.add_argument("--eqt-rose-v3-ckpt",
                    default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "eqt_rose_v3" / "eqt_rose_v3.pt"))
    ap.add_argument("--phasenet-rose-v2-ckpt",
                    default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "phasenet_rose_v2" / "phasenet_rose_v2.pt"))
    ap.add_argument("--redpan-tf",
                    default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "redpan_tf60" / "train.hdf5"))
    ap.add_argument("--p-threshold", type=float, default=0.30,
                    help="(legacy) ignored if --sweep-thresholds is set.")
    ap.add_argument("--s-threshold", type=float, default=0.30)
    ap.add_argument("--detection-threshold", type=float, default=0.30)
    ap.add_argument("--sweep-thresholds",
                    default="0.05,0.1,0.2,0.3,0.5,0.7",
                    help="Comma-separated thresholds. Inference runs ONCE "
                         "per (model, trace) at the lowest threshold; "
                         "metrics are computed at each threshold by "
                         "post-hoc filtering. No re-inference required.")
    ap.add_argument("--shard", type=int, default=0,
                    help="0-indexed shard for parallelization across "
                         "processes. Each shard processes indices where "
                         "(i %% total_shards) == shard.")
    ap.add_argument("--total-shards", type=int, default=1)
    args = ap.parse_args()

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

    stead_dir = Path(args.stead_dir)
    ev_df = pd.read_csv(stead_dir / "metadata.csv", low_memory=False)
    noise_df = pd.read_csv(stead_dir / "metadata_noise.csv", low_memory=False)
    ev_wf_dir = stead_dir / "waveforms"
    nz_wf_dir = stead_dir / "noise_waveforms"
    logger.info("STEAD: %d events / %d noise available",
                len(ev_df), len(noise_df))

    all_results = {}

    for model_id in args.models:
        logger.info("=== %s ===", model_id)
        try:
            if model_id == "eqt_rose_v3":
                state = safe_torch_load(args.eqt_rose_v3_ckpt, map_location="cpu")
                cfg_ckpt = state.get("config", {})
                m = sbm.EQTransformer(
                    in_samples=int(cfg_ckpt.get("model_window", 6000)),
                    sampling_rate=int(cfg_ckpt.get("sampling_rate", 100)),
                    phases=["P", "S"], norm="peak",
                )
                m.load_state_dict(state["model"]); m.norm = "peak"
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "phasenet_rose_v2":
                state = safe_torch_load(args.phasenet_rose_v2_ckpt, map_location="cpu")
                m = sbm.PhaseNet(phases="PSN", norm="peak",
                                 default_args={"blinding": (200, 200)})
                m.load_state_dict(state["model"]); m.norm = "peak"
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "eqt_instance":
                m = sbm.EQTransformer.from_pretrained("instance")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "phasenet_instance":
                m = sbm.PhaseNet.from_pretrained("instance")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "eqt_ethz":
                m = sbm.EQTransformer.from_pretrained("ethz")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "phasenet_ethz":
                m = sbm.PhaseNet.from_pretrained("ethz")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "eqt_stead":
                m = sbm.EQTransformer.from_pretrained("stead")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "phasenet_stead":
                m = sbm.PhaseNet.from_pretrained("stead")
                m.to("cpu").eval()
                kind = "seisbench"
            elif model_id == "redpan":
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                import tensorflow as tf
                for g in tf.config.list_physical_devices("GPU"):
                    try:
                        tf.config.experimental.set_memory_growth(g, True)
                    except RuntimeError:
                        pass
                # Cap TF threads to avoid over-subscription when multiple
                # shards run in parallel on the same machine.
                # n cores per process = max(1, total_cores // total_shards)
                try:
                    n_threads = max(
                        1, os.cpu_count() // max(1, args.total_shards),
                    )
                    tf.config.threading.set_intra_op_parallelism_threads(
                        n_threads)
                    tf.config.threading.set_inter_op_parallelism_threads(
                        n_threads)
                    logger.info("TF threading capped to %d intra/inter "
                                "ops per process (cores=%d, shards=%d)",
                                n_threads, os.cpu_count(),
                                args.total_shards)
                except Exception:
                    pass
                from redpan_inference.core import REDPAN
                tf_model = tf.keras.models.load_model(args.redpan_tf,
                                                     compile=False)
                m = REDPAN(model=tf_model, pred_npts=6000, dt=0.01,
                           pred_interval_sec=10.0, batch_size=32,
                           use_compiled_infer=True)
                kind = "redpan"
            else:
                continue
        except Exception as exc:
            logger.error("load failed: %s", exc)
            all_results[model_id] = {"error": str(exc)}
            continue

        pick_tol = {"P": float(args.pick_tol_p), "S": float(args.pick_tol_s)}
        thresholds = sorted(
            {float(x.strip()) for x in args.sweep_thresholds.split(",")
             if x.strip()}
        )
        per_thr = evaluate_model(
            model_id, kind, m, ev_df, noise_df, ev_wf_dir, nz_wf_dir,
            args.num_events, args.num_noise,
            thresholds=thresholds,
            pick_tol_s=pick_tol, bandpass=bandpass,
            shard=args.shard, total_shards=args.total_shards,
        )
        all_results[model_id] = per_thr
        suffix = (f".shard{args.shard}of{args.total_shards}"
                  if args.total_shards > 1 else "")
        with (out_dir / f"{model_id}{suffix}.json").open("w") as fh:
            json.dump(per_thr, fh, indent=2, default=str)

        if kind != "redpan":
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compose CSV — iterate (model, threshold, phase)
    rows = []
    for name, per_thr in all_results.items():
        if "error" in per_thr:
            rows.append({"model": name, "error": per_thr["error"]})
            continue
        for thr_str, s in per_thr.items():
            det = s.get("detection", {}) or {}
            t1 = s.get("T1", {}) or {}
            for phase in ("P", "S"):
                d = s.get(phase, {})
                rows.append({
                    "model": name, "threshold": float(thr_str),
                    "phase": phase,
                    "tp": d.get("tp"), "fp": d.get("fp"), "fn": d.get("fn"),
                    "precision": d.get("precision"),
                    "recall": d.get("recall"),
                    "f1": d.get("f1"),
                    "mae_s": d.get("mae_s"), "rmse_s": d.get("rmse_s"),
                    "mcc_phase_id": s.get("mcc_phase_id"),
                    "mcc_phase_accuracy": s.get("mcc_phase_accuracy"),
                    "mcc_n_phase_confusions":
                        s.get("mcc_n_phase_confusions"),
                    "T1_f1": t1.get("f1"),
                    "T1_precision": t1.get("precision"),
                    "T1_recall": t1.get("recall"),
                    "T1_mcc": t1.get("mcc"),
                    "T1_auc_det_head": t1.get("auc_det_head"),
                    "T1_n_event": t1.get("n_event"),
                    "T1_n_noise": t1.get("n_noise"),
                    "det_recall": det.get("det_recall"),
                    "det_iou_mean": det.get("det_iou_mean"),
                    "det_per_trace_mean": det.get("det_per_trace_mean"),
                    "det_start_mae_s": det.get("det_start_mae_s"),
                    "det_end_mae_s": det.get("det_end_mae_s"),
                })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "stead_test_comparison.csv", index=False)
    pd.set_option("display.float_format",
                  lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
