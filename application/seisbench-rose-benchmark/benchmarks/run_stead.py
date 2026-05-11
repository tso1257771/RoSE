"""STEAD-test loop with paired event + noise traces.

Expects a directory layout (see data/README.md):
    <stead-dir>/metadata.csv          103,040 rows (event labels)
    <stead-dir>/metadata_noise.csv    23,526 rows (noise listing)
    <stead-dir>/waveforms/*.npy       (27000, 3) ENZ float32 per row
    <stead-dir>/noise_waveforms/*.npy (27000, 3) ENZ float32 per row

Yields the same per-threshold summary as the RoSE pipeline plus a T1
(trace-level event-vs-noise) entry.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from obspy import UTCDateTime
from sklearn.metrics import f1_score, matthews_corrcoef, roc_auc_score

THIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS))

from pickerbench.matching import (
    match_picks, match_picks_cross_phase, evaluate_trace_detections,
    aggregate_summary,
)
from pickerbench.trace_io import waveform_to_stream
from benchmarks.models import HAS_DETECTION_HEAD


def _load_npy(wf_dir: Path, trace_name: str) -> np.ndarray:
    """Load `<wf_dir>/<trace_name>.npy`. Rejects path-traversal-style
    trace_name values to keep the loader safe against untrusted CSVs."""
    if not isinstance(trace_name, str) or not trace_name:
        raise ValueError(f"invalid trace_name: {trace_name!r}")
    if "/" in trace_name or "\\" in trace_name or ".." in trace_name:
        raise ValueError(f"unsafe trace_name (path traversal): {trace_name!r}")
    path = wf_dir / f"{trace_name}.npy"
    arr = np.load(path)
    if arr.ndim == 2 and arr.shape[0] == 27000 and arr.shape[1] == 3:
        arr = arr.T  # (samples, ch) → (ch, samples)
    if arr.ndim != 2 or arr.shape[0] != 3:
        raise ValueError(f"unexpected shape {arr.shape} in {path}")
    return arr


def run_stead(model, model_id: str, stead_dir: Path, out_path: Path,
              num_events: int | None, num_noise: int | None,
              bandpass=(1.0, 45.0),
              thresholds: Iterable[float] = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7),
              pick_tol=None) -> dict:
    pick_tol = pick_tol or {"P": 0.5, "S": 1.0}
    thresholds = sorted(thresholds)
    logger = logging.getLogger("stead")
    has_det = HAS_DETECTION_HEAD.get(model_id, False)
    base_thr = min(thresholds)

    ev_df = pd.read_csv(stead_dir / "metadata.csv", low_memory=False)
    nz_df = pd.read_csv(stead_dir / "metadata_noise.csv", low_memory=False)
    ev_wf_dir = stead_dir / "waveforms"
    nz_wf_dir = stead_dir / "noise_waveforms"

    if num_events is None:
        num_events = len(ev_df)
    if num_noise is None:
        num_noise = len(nz_df)
    ev_df = ev_df.head(num_events).reset_index(drop=True)
    nz_df = nz_df.head(num_noise).reset_index(drop=True)
    logger.info("[%s] %d events + %d noise", model_id, len(ev_df), len(nz_df))

    classify_kw = {"P_threshold": base_thr, "S_threshold": base_thr,
                   "detection_threshold": base_thr}

    per_thr = {
        t: {
            "stats": {"P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
                      "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []}},
            "mcc_true": [], "mcc_pred": [], "det_records": [],
            "ev_with_pick": 0, "noise_with_pick": 0,
            "ev_max_prob": [], "noise_max_prob": [],
        }
        for t in thresholds
    }

    t0 = time.time()
    n_failed = 0

    # ---- events ------------------------------------------------------
    for k, row in ev_df.iterrows():
        try:
            wf = _load_npy(ev_wf_dir, str(row["trace_name"]))
        except Exception as exc:
            logger.warning("trace %s failed: %s: %s",
                           row.get("trace_name", "?"),
                           type(exc).__name__, exc)
            n_failed += 1; continue
        sr = 100
        starttime = UTCDateTime("2000-01-01T00:00:00")
        true_picks = {}
        for ph in ("P", "S"):
            samp = row.get(f"{ph.lower()}_sample")
            if pd.notna(samp):
                true_picks[ph] = starttime + float(samp) / sr
        if not true_picks:
            continue
        stream = waveform_to_stream(wf, sampling_rate=sr,
                                    components="ENZ", starttime=starttime,
                                    bandpass=bandpass)
        try:
            out = model.classify(stream, **classify_kw)
        except Exception as exc:
            logger.warning("trace %s failed: %s: %s",
                           row.get("trace_name", "?"),
                           type(exc).__name__, exc)
            n_failed += 1; continue
        picks_all = list(getattr(out, "picks", []) or [])
        dets_all = list(getattr(out, "detections", []) or [])
        max_prob = max((float(getattr(p, "peak_value", 0)) for p in picks_all),
                       default=0.0)

        for thr in thresholds:
            picks_t = [p for p in picks_all
                       if float(getattr(p, "peak_value", 0)) >= thr]
            dets_t = [d for d in dets_all
                      if float(getattr(d, "peak_value", 0)) >= thr]
            matched, unmatched = match_picks(picks_t, true_picks, pick_tol)
            stats = per_thr[thr]["stats"]
            for ph in ("P", "S"):
                if ph in true_picks:
                    if ph in matched:
                        stats[ph]["tp"] += 1
                        stats[ph]["residuals"].append(matched[ph][1])
                    else:
                        stats[ph]["fn"] += 1
            for p in unmatched:
                ph = str(getattr(p, "phase", "")).upper()
                if ph in stats:
                    stats[ph]["fp"] += 1
            for tp_ph, pp_ph in match_picks_cross_phase(picks_t, true_picks, pick_tol):
                per_thr[thr]["mcc_true"].append(0 if tp_ph == "P" else 1)
                per_thr[thr]["mcc_pred"].append(0 if pp_ph == "P" else 1)
            if has_det:
                rec = evaluate_trace_detections(dets_t, true_picks, factor=1.4)
                if rec is not None:
                    per_thr[thr]["det_records"].append(rec)
            if picks_t:
                per_thr[thr]["ev_with_pick"] += 1
            per_thr[thr]["ev_max_prob"].append(max_prob)
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(ev_df) - k - 1)
            logger.info("  events %d/%d (%.1fs elapsed, ETA %.1fs)",
                        k + 1, len(ev_df), elapsed, eta)

    # ---- noise -------------------------------------------------------
    for k, row in nz_df.iterrows():
        try:
            wf = _load_npy(nz_wf_dir, str(row["trace_name"]))
        except Exception as exc:
            logger.warning("trace %s failed: %s: %s",
                           row.get("trace_name", "?"),
                           type(exc).__name__, exc)
            n_failed += 1; continue
        starttime = UTCDateTime("2000-01-01T00:00:00")
        stream = waveform_to_stream(wf, sampling_rate=100,
                                    components="ENZ", starttime=starttime,
                                    bandpass=bandpass)
        try:
            out = model.classify(stream, **classify_kw)
        except Exception as exc:
            logger.warning("trace %s failed: %s: %s",
                           row.get("trace_name", "?"),
                           type(exc).__name__, exc)
            n_failed += 1; continue
        picks_all = list(getattr(out, "picks", []) or [])
        max_prob = max((float(getattr(p, "peak_value", 0)) for p in picks_all),
                       default=0.0)
        for thr in thresholds:
            picks_t = [p for p in picks_all
                       if float(getattr(p, "peak_value", 0)) >= thr]
            stats = per_thr[thr]["stats"]
            for p in picks_t:
                ph = str(getattr(p, "phase", "")).upper()
                if ph in stats:
                    stats[ph]["fp"] += 1
            if picks_t:
                per_thr[thr]["noise_with_pick"] += 1
            per_thr[thr]["noise_max_prob"].append(max_prob)
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            logger.info("  noise %d/%d (%.1fs elapsed)",
                        k + 1, len(nz_df), elapsed)

    # ---- aggregate ---------------------------------------------------
    summary = {}
    for thr, s in per_thr.items():
        agg = aggregate_summary(
            s["stats"], s["mcc_true"], s["mcc_pred"],
            n_evaluated=len(ev_df) + len(nz_df), n_failed=n_failed,
            elapsed_s=time.time() - t0, det_records=s["det_records"],
            keep_residuals=True,
        )
        # T1 = trace-level event-vs-noise classification
        n_ev = len(ev_df); n_nz = len(nz_df)
        tp = s["ev_with_pick"]; fn = n_ev - tp
        fp = s["noise_with_pick"]; tn = n_nz - fp
        prec = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rec = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2*prec*rec / (prec+rec) if (prec+rec) > 0 else float("nan")
        try:
            mcc_T1 = matthews_corrcoef([1]*tp + [1]*fn + [0]*fp + [0]*tn,
                                       [1]*tp + [0]*fn + [1]*fp + [0]*tn)
        except Exception:
            mcc_T1 = float("nan")
        # AUC only meaningful if we have continuous probs from a detection head;
        # we use the per-trace max pick prob as a proxy.
        try:
            auc = roc_auc_score(
                [1]*len(s["ev_max_prob"]) + [0]*len(s["noise_max_prob"]),
                s["ev_max_prob"] + s["noise_max_prob"]
            ) if has_det else float("nan")
        except Exception:
            auc = float("nan")
        agg["T1"] = {"threshold": thr, "n_event": n_ev, "n_noise": n_nz,
                     "f1": f1, "precision": prec, "recall": rec,
                     "mcc": mcc_T1, "auc_det_head": auc,
                     "definition": "any pick emitted at threshold"}
        summary[str(thr)] = agg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("wrote %s", out_path)
    return summary
