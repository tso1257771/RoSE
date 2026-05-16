"""Canonical SeisBench-API benchmark on the RoSE test split.

For each (model class, weight set):
    1. Iterate the RoSE test split.
    2. Build an ObsPy Stream from each trace's waveform + metadata.
    3. Call model.classify(stream, P_threshold=..., S_threshold=...,
       detection_threshold=...) — exactly what scdlpicker / production
       deployments call. SeisBench handles sliding-window annotation,
       blinding, stitching, and peak extraction internally.
    4. Greedy-match predicted picks to true picks (per phase) within a
       configurable tolerance.
    5. Aggregate per-phase precision / recall / F1 + onset residuals.

Reports per (model, weight):
    precision, recall, F1 for P and S
    MAE, RMSE, median, IQR of onset residuals (s) for matched picks
    MCC across matched picks (P-vs-S confusion)

Default model set: PhaseNet & EQTransformer pretrained on ETHZ / INSTANCE
/ STEAD, plus a user-provided EQT-RoSE checkpoint via --custom-eqt.

Notes on cost
-------------
model.classify is per-trace (sliding-window inference + Stream IO).
With 32,374 RoSE test traces and ~0.3 s/trace per model, evaluating all
seven models on the full test set is ~20 minutes per model = ~2.5 h.
Default --num-test 2000 keeps a single run under 5 min/model and is
ample for stable metrics.
"""
from __future__ import annotations

import argparse
import os
import sys
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

# Allow `from rose.checkpoint_io import safe_torch_load` when run as a script.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from rose.checkpoint_io import safe_torch_load  # noqa: E402

import numpy as np
import pandas as pd
import torch
from obspy import Stream, Trace, UTCDateTime
from sklearn.metrics import matthews_corrcoef

import seisbench.data as sbd
import seisbench.models as sbm


logger = logging.getLogger("bench_pickers_rose")


# Default model lineup.  Pretrained weights chosen to mirror common
# deployment baselines (small/regional, large/regional, large/global).
DEFAULT_PRETRAINED = [
    ("PhaseNet-ethz", "PhaseNet", "ethz"),
    ("PhaseNet-instance", "PhaseNet", "instance"),
    ("PhaseNet-stead", "PhaseNet", "stead"),
    ("EQT-ethz", "EQTransformer", "ethz"),
    ("EQT-instance", "EQTransformer", "instance"),
    ("EQT-stead", "EQTransformer", "stead"),
]


@dataclass
class BenchConfig:
    rose_dir: str
    out_dir: str
    num_test: int
    seed: int
    p_threshold: float
    s_threshold: float
    detection_threshold: float
    # float (uniform) or dict {"P": 0.5, "S": 1.0} (pick-benchmark convention).
    pick_tol_s: float | dict[str, float]
    sampling_rate: int
    component_order: str
    bandpass: tuple[float, float] | None


# --------------------------------------------------------------------- #
# Stream construction
# --------------------------------------------------------------------- #
def trace_to_stream(
    wf: np.ndarray,
    meta: pd.Series,
    sampling_rate: float,
    components: str = "ZNE",
    bandpass: tuple[float, float] | None = None,
) -> Stream:
    """Build a 3-component ObsPy Stream from a SeisBench-row waveform.

    bandpass: (low_hz, high_hz) to apply via stream.filter('bandpass', ...).
    Set to mirror training-time pre-augment bandpass; both EQT-RoSE and
    pretrained EQT (Mousavi 2020) were trained on 1-45 Hz filtered data.
    """
    starttime = UTCDateTime(str(meta["trace_start_time"]))
    network = str(meta.get("station_network_code", "XX")).strip() or "XX"
    station = str(meta.get("station_code", "STA")).strip() or "STA"
    raw_chan = str(meta.get("station_channel", "HH"))
    chan_prefix = (raw_chan[:2] if raw_chan and raw_chan != "nan" else "HH")

    traces = []
    for i, comp in enumerate(components):
        tr = Trace(data=wf[i].astype(np.float32))
        tr.stats.starttime = starttime
        tr.stats.sampling_rate = sampling_rate
        tr.stats.network = network
        tr.stats.station = station
        tr.stats.channel = f"{chan_prefix}{comp}"
        traces.append(tr)
    stream = Stream(traces)
    if bandpass is not None:
        stream.detrend("demean")
        stream.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1],
                      corners=4, zerophase=True)
    return stream


# --------------------------------------------------------------------- #
# Pick matching
# --------------------------------------------------------------------- #
def parse_true_picks(meta: pd.Series) -> dict[str, UTCDateTime]:
    """Pull ground-truth P / S pick times from the metadata row."""
    out: dict[str, UTCDateTime] = {}
    for key, phase in (
        ("trace_p_arrival_time", "P"),
        ("trace_s_arrival_time", "S"),
    ):
        val = meta.get(key)
        if not isinstance(val, str) or not val.strip() or val == "nan":
            continue
        try:
            out[phase] = UTCDateTime(val)
        except Exception:
            continue
    return out


def _phase_tol(tol_s, phase: str) -> float:
    """Resolve tolerance for one phase from a float (uniform) or
    a dict {"P": 0.5, "S": 1.0} (per-phase, pick-benchmark convention)."""
    if isinstance(tol_s, dict):
        return float(tol_s.get(phase, 0.5))
    return float(tol_s)


def match_picks(
    predicted, true_picks: dict[str, UTCDateTime],
    tol_s,
) -> tuple[dict, list]:
    """Greedy nearest match per phase (same-phase only).

    tol_s: either a float (same tolerance for P and S) or a dict
    ``{"P": 0.5, "S": 1.0}`` for the pick-benchmark convention with
    asymmetric tolerances (S onsets are noisier).

    Used for precision / recall / F1 / residual stats. A predicted P near
    a true S would not match here — that becomes an FP for P and an FN for
    S. Cross-phase confusion is captured separately by
    ``match_picks_cross_phase`` which feeds MCC.
    """
    matched: dict = {}
    used: set = set()

    for phase, true_time in true_picks.items():
        tol = _phase_tol(tol_s, phase)
        candidates = [
            (i, p) for i, p in enumerate(predicted)
            if str(getattr(p, "phase", "")).upper() == phase and i not in used
        ]
        if not candidates:
            continue
        diffs = [
            (i, p, abs(float(p.peak_time - true_time)))
            for i, p in candidates
        ]
        best_i, best_p, best_d = min(diffs, key=lambda x: x[2])
        if best_d <= tol:
            residual = float(best_p.peak_time - true_time)
            matched[phase] = (best_p.peak_time, residual,
                              float(getattr(best_p, "peak_value", float("nan"))))
            used.add(best_i)

    unmatched = [p for i, p in enumerate(predicted) if i not in used]
    return matched, unmatched


def evaluate_trace_detections(
    detections, true_picks: dict[str, UTCDateTime], factor: float = 1.4,
) -> dict | None:
    """Evaluate EQT-style detection boxes against the labelled event window.

    True event window per Mousavi 2020 / SeisBench DetectionLabeller:
        start = P_arrival
        end   = S_arrival + factor * (S_arrival - P_arrival)

    Returns None if the trace lacks both P and S (cannot define truth).
    Otherwise returns a dict per-trace summary. PhaseNet's empty
    detections list collapses to 'detected=False, n=0'.
    """
    if "P" not in true_picks or "S" not in true_picks:
        return None
    true_start = true_picks["P"]
    true_end = true_picks["S"] + factor * float(true_picks["S"] - true_picks["P"])
    true_dur = float(true_end - true_start)
    if true_dur <= 0:
        return None

    if not detections:
        return {"detected": False, "n_detections": 0,
                "best_iou": 0.0, "start_res_s": None, "end_res_s": None}

    best_iou, best_det = 0.0, None
    for det in detections:
        d_start = float(det.start_time - true_start)            # seconds, relative
        d_end = float(det.end_time - true_start)
        rel_true_end = true_dur
        intersect = max(0.0, min(d_end, rel_true_end) - max(d_start, 0.0))
        union = max(d_end, rel_true_end) - min(d_start, 0.0)
        iou = intersect / union if union > 0 else 0.0
        if iou > best_iou:
            best_iou = iou
            best_det = det

    if best_det is None:
        return {"detected": True, "n_detections": len(detections),
                "best_iou": 0.0, "start_res_s": None, "end_res_s": None}

    return {
        "detected": True,
        "n_detections": len(detections),
        "best_iou": float(best_iou),
        "start_res_s": float(best_det.start_time - true_start),
        "end_res_s": float(best_det.end_time - true_end),
    }


def match_picks_cross_phase(
    predicted, true_picks: dict[str, UTCDateTime], tol_s,
) -> list[tuple[str, str]]:
    """Cross-phase matching for MCC.

    For each true pick, find the nearest predicted pick of ANY phase
    within tolerance (per-phase if dict, uniform if float). Returns
    ``(true_phase, predicted_phase)`` pairs. A pair where predicted_phase
    != true_phase is a phase confusion — model detected a pick in the
    right place but mislabelled P vs S.

    Greedy: each predicted pick is consumed by at most one true pick;
    true picks are processed in order P, S.
    """
    pairs: list[tuple[str, str]] = []
    used: set = set()
    for true_phase in ("P", "S"):
        if true_phase not in true_picks:
            continue
        true_time = true_picks[true_phase]
        tol = _phase_tol(tol_s, true_phase)
        candidates = []
        for i, p in enumerate(predicted):
            if i in used:
                continue
            ph = str(getattr(p, "phase", "")).upper()
            if ph not in ("P", "S"):
                continue
            d = abs(float(p.peak_time - true_time))
            if d <= tol:
                candidates.append((i, p, d))
        if not candidates:
            continue
        best_i, best_p, _ = min(candidates, key=lambda x: x[2])
        pred_phase = str(best_p.phase).upper()
        pairs.append((true_phase, pred_phase))
        used.add(best_i)
    return pairs


# --------------------------------------------------------------------- #
# Per-model evaluation
# --------------------------------------------------------------------- #
def _aggregate_detection_stats(det_records: list[dict]) -> dict:
    if not det_records:
        return {
            "det_recall": float("nan"), "det_n_traces": 0,
            "det_iou_mean": float("nan"),
            "det_per_trace_mean": float("nan"),
            "det_start_mae_s": float("nan"),
            "det_end_mae_s": float("nan"),
        }
    n_total = len(det_records)
    detected = [r for r in det_records if r.get("detected")]
    n_detected = len(detected)
    iou = [r["best_iou"] for r in detected]
    n_per_trace = [r["n_detections"] for r in det_records]
    start_res = [r["start_res_s"] for r in detected if r.get("start_res_s") is not None]
    end_res = [r["end_res_s"] for r in detected if r.get("end_res_s") is not None]
    return {
        "det_recall": n_detected / n_total,
        "det_n_traces": n_total,
        "det_iou_mean": float(np.mean(iou)) if iou else float("nan"),
        "det_per_trace_mean": float(np.mean(n_per_trace)) if n_per_trace else float("nan"),
        "det_start_mae_s": float(np.mean(np.abs(start_res))) if start_res else float("nan"),
        "det_end_mae_s": float(np.mean(np.abs(end_res))) if end_res else float("nan"),
    }


def _aggregate_summary(stats: dict, mcc_true: list[int], mcc_pred: list[int],
                       n_evaluated: int, n_failed: int, elapsed_s: float,
                       det_records: list[dict] | None = None) -> dict:
    summary = {"n_evaluated": n_evaluated, "n_failed": n_failed,
               "elapsed_s": float(elapsed_s)}
    for phase, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))
        residuals = np.asarray(s["residuals"], dtype=np.float64)
        if residuals.size:
            res_stats = {
                "mae_s": float(np.mean(np.abs(residuals))),
                "rmse_s": float(np.sqrt(np.mean(residuals ** 2))),
                "median_s": float(np.median(residuals)),
                "iqr_s": float(np.percentile(residuals, 75)
                               - np.percentile(residuals, 25)),
            }
        else:
            res_stats = {"mae_s": float("nan"), "rmse_s": float("nan"),
                         "median_s": float("nan"), "iqr_s": float("nan")}
        summary[phase] = {"tp": tp, "fp": fp, "fn": fn,
                          "precision": precision, "recall": recall, "f1": f1,
                          "n_residuals": int(residuals.size), **res_stats,
                          "residuals_s": residuals.tolist()}

    if mcc_true:
        n_total = len(mcc_true)
        n_correct = sum(1 for t, p in zip(mcc_true, mcc_pred) if t == p)
        n_classes = len(set(mcc_true)) >= 2 and len(set(mcc_pred)) >= 2
        summary["mcc_phase_id"] = (
            float(matthews_corrcoef(mcc_true, mcc_pred)) if n_classes
            else float("nan")
        )
        summary["mcc_n_pairs"] = n_total
        summary["mcc_phase_accuracy"] = n_correct / n_total
        summary["mcc_n_phase_confusions"] = n_total - n_correct
    else:
        summary["mcc_phase_id"] = float("nan")
        summary["mcc_n_pairs"] = 0
        summary["mcc_phase_accuracy"] = float("nan")
        summary["mcc_n_phase_confusions"] = 0

    summary["detection"] = _aggregate_detection_stats(det_records or [])
    return summary


def evaluate_model(
    model, test_dataset: sbd.WaveformDataset,
    indices: np.ndarray, cfg: BenchConfig,
) -> dict:
    stats: dict = {
        "P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
        "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
    }
    classify_kwargs = {
        "P_threshold": cfg.p_threshold,
        "S_threshold": cfg.s_threshold,
        "detection_threshold": cfg.detection_threshold,
    }
    n_evaluated = 0
    n_failed = 0
    mcc_true: list[int] = []
    mcc_pred: list[int] = []
    det_records: list[dict] = []
    t0 = time.time()

    md = test_dataset.metadata.reset_index(drop=True)

    for k, idx in enumerate(indices):
        try:
            wf, _ = test_dataset.get_sample(int(idx))
        except Exception as exc:
            # Log at DEBUG so a 100% n_failed run still leaves evidence
            # in the bench.log file when run with --log-level DEBUG.
            logger.debug("get_sample failed on idx %d: %s: %s",
                         idx, type(exc).__name__, exc)
            n_failed += 1
            continue
        meta = md.iloc[int(idx)]
        true_picks = parse_true_picks(meta)
        if not true_picks:
            continue

        stream = trace_to_stream(
            wf, meta, cfg.sampling_rate, components=cfg.component_order,
            bandpass=cfg.bandpass,
        )
        try:
            output = model.classify(stream, **classify_kwargs)
        except Exception as exc:
            logger.debug("classify failed on idx %d: %s", idx, exc)
            n_failed += 1
            continue

        picks = list(getattr(output, "picks", []) or [])
        detections = list(getattr(output, "detections", []) or [])
        matched, unmatched = match_picks(picks, true_picks, cfg.pick_tol_s)

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
            picks, true_picks, cfg.pick_tol_s,
        ):
            mcc_true.append(0 if true_phase == "P" else 1)
            mcc_pred.append(0 if pred_phase == "P" else 1)

        det_summary = evaluate_trace_detections(detections, true_picks)
        if det_summary is not None:
            det_records.append(det_summary)

        n_evaluated += 1
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(indices) - k - 1)
            logger.info("  %d/%d traces (%.1fs elapsed, ETA %.1fs)",
                        k + 1, len(indices), elapsed, eta)

    return _aggregate_summary(stats, mcc_true, mcc_pred,
                              n_evaluated, n_failed, time.time() - t0,
                              det_records=det_records)


def evaluate_model_sweep(
    model, test_dataset: sbd.WaveformDataset,
    indices: np.ndarray, cfg: BenchConfig,
    thresholds: list[float],
) -> dict:
    """One classify() call per trace at the LOWEST sweep threshold; the
    higher threshold values are evaluated by post-hoc filtering on
    ``pick.peak_value``. Mathematically equivalent to re-running classify
    at each threshold (since the trigger-onset region containing each
    peak is a superset of any region detected at a higher threshold).

    Returns ``{threshold: per-threshold-summary}``.
    """
    base_thresh = min(thresholds)
    classify_kwargs = {
        "P_threshold": base_thresh,
        "S_threshold": base_thresh,
        # Sweep detection alongside picks: use the lowest sweep threshold
        # at classify time so the returned detection list is the superset.
        # The per-threshold `[d for d in all_detections if peak_value >= thr]`
        # filter below then yields a real sweep. (Previously this was
        # cfg.detection_threshold = 0.3, making thr<0.3 rows degenerate;
        # see e.g. EQT-RoSE's rose_detection.csv where tp/fn were
        # identical at thresholds 0.05/0.1/0.2/0.3 prior to this fix.)
        "detection_threshold": base_thresh,
    }

    # Per-threshold accumulators.
    per_thr: dict[float, dict] = {
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
    n_evaluated = 0
    n_failed = 0
    t0 = time.time()
    md = test_dataset.metadata.reset_index(drop=True)

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

        stream = trace_to_stream(
            wf, meta, cfg.sampling_rate, components=cfg.component_order,
            bandpass=cfg.bandpass,
        )
        try:
            output = model.classify(stream, **classify_kwargs)
        except Exception as exc:
            logger.debug("classify failed on idx %d: %s", idx, exc)
            n_failed += 1
            continue

        all_picks = list(getattr(output, "picks", []) or [])
        all_detections = list(getattr(output, "detections", []) or [])

        for thr in thresholds:
            picks_t = [
                p for p in all_picks
                if float(getattr(p, "peak_value", 0.0)) >= thr
            ]
            detections_t = [
                d for d in all_detections
                if float(getattr(d, "peak_value", 0.0)) >= thr
            ]
            matched, unmatched = match_picks(picks_t, true_picks, cfg.pick_tol_s)
            stats = per_thr[thr]["stats"]
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
        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(indices) - k - 1)
            logger.info("  %d/%d traces (%.1fs elapsed, ETA %.1fs)",
                        k + 1, len(indices), elapsed, eta)

    elapsed = time.time() - t0
    return {
        thr: _aggregate_summary(
            d["stats"], d["mcc_true"], d["mcc_pred"],
            n_evaluated, n_failed, elapsed,
            det_records=d["det_records"],
        )
        for thr, d in per_thr.items()
    }


# --------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------- #
def load_pretrained(model_class: str, weight: str, device: torch.device
                    ) -> torch.nn.Module:
    cls = getattr(sbm, model_class)
    model = cls.from_pretrained(weight)
    model.to(device).eval()
    return model


def load_custom_phasenet(ckpt_path: str, device: torch.device,
                         inference_norm: str | None = None) -> torch.nn.Module:
    """Load a custom PhaseNet-on-RoSE checkpoint."""
    state = safe_torch_load(ckpt_path, map_location=device)
    # NOTE: unlike EQT, the SeisBench PhaseNet constructor doesn't consult
    # state['config'] — sampling rate and window length are fixed at 100 Hz /
    # 3001 samples by the architecture. Pulling 'config' out is therefore
    # unnecessary here; left as a comment so the asymmetry is documented.
    model = sbm.PhaseNet(
        phases="PSN",
        norm=inference_norm if inference_norm in ("std", "peak") else "peak",
        default_args={"blinding": (200, 200)},
    )
    model.load_state_dict(state["model"])
    if inference_norm is None:
        model.norm = None
    elif inference_norm in ("std", "peak"):
        model.norm = inference_norm
    model.to(device).eval()
    logger.info(
        "loaded custom PhaseNet (epoch=%s, dev_loss=%.4f) from %s "
        "(inference_norm=%s)",
        state.get("epoch"), state.get("dev_loss", float("nan")),
        ckpt_path, inference_norm,
    )
    return model


def load_custom_eqt(ckpt_path: str, device: torch.device,
                    inference_norm: str | None = None) -> torch.nn.Module:
    """Load a custom EQT-on-RoSE checkpoint.

    inference_norm: how the classify pipeline should normalize batches in
    annotate_batch_pre. The bundled eqt_rose.pt checkpoint was trained
    with sbg.Normalize(amp_norm_type="peak") — its stored config has
    norm="peak" — so callers should pass "peak" (this is the default of
    --custom-inference-norm). Pass None only when the checkpoint was
    trained without any Normalize augmentation; mismatching train-time
    and inference-time normalization destroys recall.
    """
    state = safe_torch_load(ckpt_path, map_location=device)
    cfg = state.get("config", {})
    model = sbm.EQTransformer(
        in_samples=int(cfg.get("model_window", 6000)),
        sampling_rate=int(cfg.get("sampling_rate", 100)),
        phases=["P", "S"],
        norm="std",
    )
    model.load_state_dict(state["model"])
    model.norm = inference_norm        # None disables std normalization
    model.to(device).eval()
    logger.info(
        "loaded custom EQT (epoch=%s, dev_loss=%.4f) from %s "
        "(inference_norm=%s)",
        state.get("epoch"), state.get("dev_loss", float("nan")),
        ckpt_path, inference_norm,
    )
    return model


# --------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rose-dir", default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-test", type=int, default=2000,
                    help="Sample N random test traces (0 = use full split).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--p-threshold", type=float, default=0.1)
    ap.add_argument("--s-threshold", type=float, default=0.1)
    ap.add_argument("--detection-threshold", type=float, default=0.3)
    ap.add_argument("--pick-tol-s", type=float, default=0.5,
                    help="Uniform pick tolerance (s). Ignored if "
                         "--pick-tol-p / --pick-tol-s-phase are both set.")
    ap.add_argument("--pick-tol-p", type=float, default=None,
                    help="P-pick tolerance (s). Set together with "
                         "--pick-tol-s-phase to use pick-benchmark "
                         "asymmetric tolerances (0.5 P / 1.0 S).")
    ap.add_argument("--pick-tol-s-phase", type=float, default=None,
                    help="S-pick tolerance (s).")
    ap.add_argument("--sampling-rate", type=int, default=100)
    ap.add_argument("--component-order", default="ZNE")
    ap.add_argument("--bandpass-low", type=float, default=1.0,
                    help="Pre-classify bandpass low-cut Hz. Set <0 to disable.")
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    ap.add_argument("--custom-eqt", default=None,
                    help="Path to user's EQT-RoSE checkpoint (.pt).")
    ap.add_argument("--custom-name", default="EQT-RoSE",
                    help="Display name for the custom EQT checkpoint.")
    ap.add_argument("--custom-inference-norm", default="peak",
                    choices=["none", "std", "peak"],
                    help="Inference normalization for the custom EQT. "
                         "'peak' (default) matches the bundled eqt_rose.pt "
                         "checkpoint, which was trained with sbg.Normalize "
                         "(amp_norm_type='peak'); see the 'norm' field in the "
                         "checkpoint's stored config. Use 'none' only for a "
                         "checkpoint trained with no Normalize augmentation, "
                         "and 'std' for sbg.Normalize(amp_norm_type='std').")
    ap.add_argument("--custom-phasenet", default=None,
                    help="Path to user's PhaseNet-RoSE checkpoint (.pt).")
    ap.add_argument("--custom-phasenet-name", default="PhaseNet-RoSE",
                    help="Display name for the custom PhaseNet checkpoint.")
    ap.add_argument("--custom-phasenet-norm", default="peak",
                    choices=["none", "std", "peak"],
                    help="Inference normalization for the custom PhaseNet.")
    ap.add_argument("--skip-pretrained", action="store_true",
                    help="Skip the default pretrained lineup; only run --custom-eqt.")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Restrict to these model names (subset of the lineup).")
    ap.add_argument("--sweep-thresholds", default=None,
                    help="Comma-separated probability thresholds to sweep "
                         "(e.g. '0.05,0.1,0.2,0.3,0.5,0.7'). When set, the "
                         "fixed --p-threshold/--s-threshold are ignored and a "
                         "per-threshold PR table is produced instead.")
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")

    bandpass = None
    if args.bandpass_low is not None and args.bandpass_high is not None:
        if args.bandpass_low > 0 and args.bandpass_high > 0:
            bandpass = (args.bandpass_low, args.bandpass_high)
    if args.pick_tol_p is not None and args.pick_tol_s_phase is not None:
        pick_tol = {"P": float(args.pick_tol_p),
                    "S": float(args.pick_tol_s_phase)}
    else:
        pick_tol = float(args.pick_tol_s)
    cfg = BenchConfig(
        rose_dir=args.rose_dir, out_dir=args.out_dir,
        num_test=args.num_test, seed=args.seed,
        p_threshold=args.p_threshold, s_threshold=args.s_threshold,
        detection_threshold=args.detection_threshold,
        pick_tol_s=pick_tol,
        sampling_rate=args.sampling_rate,
        component_order=args.component_order,
        bandpass=bandpass,
    )

    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / "bench.log")],
        force=True,
    )
    logger.info("config = %s", json.dumps(cfg.__dict__, indent=2))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = sbd.WaveformDataset(
        path=cfg.rose_dir, sampling_rate=cfg.sampling_rate,
        component_order=cfg.component_order, cache=None,
    )
    test = data.test()
    logger.info("RoSE test split: %d traces", len(test))

    rng = np.random.default_rng(cfg.seed)
    if cfg.num_test and 0 < cfg.num_test < len(test):
        indices = np.sort(rng.choice(len(test), size=cfg.num_test, replace=False))
    else:
        indices = np.arange(len(test))
    logger.info("evaluating %d traces", len(indices))

    # Build the model lineup
    lineup: list[tuple[str, str, str | None, str | None]] = []
    if not args.skip_pretrained:
        for name, klass, w in DEFAULT_PRETRAINED:
            lineup.append((name, klass, w, None))
    if args.custom_eqt:
        lineup.append((args.custom_name, "EQTransformer", None, args.custom_eqt))
    if args.custom_phasenet:
        lineup.append(
            (args.custom_phasenet_name, "PhaseNet", None, args.custom_phasenet)
        )
    if args.only:
        keep = set(args.only)
        lineup = [m for m in lineup if m[0] in keep]
    if not lineup:
        raise SystemExit("Nothing to evaluate (lineup empty).")
    logger.info("lineup: %s", [m[0] for m in lineup])

    sweep_thresholds: list[float] | None = None
    if args.sweep_thresholds:
        sweep_thresholds = sorted(
            {float(x.strip()) for x in args.sweep_thresholds.split(",")
             if x.strip()}
        )
        logger.info("PR sweep mode: thresholds=%s "
                    "(classify will run at min=%g)",
                    sweep_thresholds, sweep_thresholds[0])

    all_results: dict = {}
    for name, klass, weight, ckpt in lineup:
        logger.info("=== %s ===", name)
        try:
            if ckpt is not None:
                if klass == "PhaseNet":
                    inf_norm = (None if args.custom_phasenet_norm == "none"
                                else args.custom_phasenet_norm)
                    model = load_custom_phasenet(
                        ckpt, device, inference_norm=inf_norm,
                    )
                else:
                    inf_norm = (None if args.custom_inference_norm == "none"
                                else args.custom_inference_norm)
                    model = load_custom_eqt(
                        ckpt, device, inference_norm=inf_norm,
                    )
            else:
                model = load_pretrained(klass, weight, device)
        except Exception as exc:
            logger.error("failed to load %s: %s", name, exc)
            all_results[name] = {"error": str(exc)}
            continue

        if sweep_thresholds is not None:
            sweep = evaluate_model_sweep(
                model, test, indices, cfg, sweep_thresholds,
            )
            all_results[name] = {str(t): s for t, s in sweep.items()}
            with (out_dir / f"{name}.json").open("w") as fh:
                json.dump(all_results[name], fh, indent=2, default=str)
        else:
            summary = evaluate_model(model, test, indices, cfg)
            all_results[name] = summary
            with (out_dir / f"{name}.json").open("w") as fh:
                json.dump(summary, fh, indent=2, default=str)
        # Free GPU memory between models
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (out_dir / "all_metrics.json").open("w") as fh:
        json.dump(all_results, fh, indent=2, default=str)

    # Per-phase comparison CSV
    rows = []
    for name, s in all_results.items():
        if "error" in s:
            rows.append({"model": name, "error": s["error"]})
            continue
        if sweep_thresholds is not None:
            # s is keyed by str(threshold) → per-threshold summary
            for thr in sweep_thresholds:
                tdict = s.get(str(thr), {})
                det = tdict.get("detection", {}) or {}
                for phase in ("P", "S"):
                    d = tdict.get(phase, {})
                    rows.append({
                        "model": name,
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
                        "mcc_phase_id": tdict.get("mcc_phase_id"),
                        "mcc_phase_accuracy": tdict.get("mcc_phase_accuracy"),
                        "mcc_n_pairs": tdict.get("mcc_n_pairs"),
                        "mcc_n_phase_confusions": tdict.get(
                            "mcc_n_phase_confusions"),
                        "det_recall": det.get("det_recall"),
                        "det_iou_mean": det.get("det_iou_mean"),
                        "det_per_trace_mean": det.get("det_per_trace_mean"),
                        "det_start_mae_s": det.get("det_start_mae_s"),
                        "det_end_mae_s": det.get("det_end_mae_s"),
                        "n_evaluated": tdict.get("n_evaluated"),
                    })
            continue
        det = s.get("detection", {}) or {}
        for phase in ("P", "S"):
            d = s.get(phase, {})
            rows.append({
                "model": name,
                "phase": phase,
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
    out_csv = "sweep_comparison.csv" if sweep_thresholds else "comparison.csv"
    df.to_csv(out_dir / out_csv, index=False)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)
    pd.set_option("display.float_format", lambda x: f"{x:.4f}" if isinstance(x, float) else x)
    logger.info("\n%s", df.to_string(index=False))


if __name__ == "__main__":
    main()
