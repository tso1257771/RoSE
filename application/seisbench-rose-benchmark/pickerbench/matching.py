"""Pick / detection matching used by the RoSE and STEAD benchmarks.

All scoring is keyed on absolute UTCDateTime objects so matching is
robust to per-model annotation offsets (EQTransformer adds ~5 s of
blinding to the input stream; PhaseNet adds ~2.5 s).

Conventions:
    Pick tolerance       — P=0.5 s, S=1.0 s (RED-PAN paper convention).
    False positive       — depends on the benchmark:
                            RoSE  : counted on dedicated noise traces only
                                    (event traces have label-incomplete
                                    aftershocks/secondary phases).
                            STEAD : canonical pick-benchmark convention
                                    (events + noise pooled).
    Detection-box overlap — Mousavi 2020: window = [P, S + 1.4*(S-P)].
"""
from __future__ import annotations

from sklearn.metrics import matthews_corrcoef


def _phase_tol(tol_s, phase: str) -> float:
    if isinstance(tol_s, dict):
        return float(tol_s.get(phase, 0.5))
    return float(tol_s)


def match_picks(predicted, true_picks: dict, tol_s) -> tuple[dict, list]:
    """Greedy nearest-neighbour same-phase matching.

    Returns (matched, unmatched).
        matched[phase] = (peak_time, residual_seconds, peak_value)
        unmatched      = list of predicted picks that no true pick claimed.
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


def match_picks_cross_phase(predicted, true_picks: dict, tol_s
                            ) -> list[tuple[str, str]]:
    """Cross-phase matching for the phase-identification confusion matrix.

    For each true pick, find the nearest predicted pick of *any* phase
    within tolerance (greedy, P-first, each predicted pick consumed at
    most once). Returns ``[(true_phase, predicted_phase), ...]``.
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


def evaluate_trace_detections(detections, true_picks: dict,
                              factor: float = 1.4) -> dict | None:
    """Evaluate detection-box overlap against the labelled event window.

    True window = [T_P, T_S + factor * (T_S - T_P)] (Mousavi 2020).
    Returns None if the trace lacks both P and S (cannot define truth).
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
        d_start = float(det.start_time - true_start)
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


def _safe_mean(xs):
    return float(sum(xs) / len(xs)) if xs else float("nan")


def aggregate_summary(stats: dict, mcc_true: list[int], mcc_pred: list[int],
                      n_evaluated: int, n_failed: int, elapsed_s: float,
                      det_records: list[dict] | None = None,
                      keep_residuals: bool = True) -> dict:
    """Aggregate per-trace stats into the published per-(model, threshold) summary."""
    summary = {"n_evaluated": n_evaluated, "n_failed": n_failed,
               "elapsed_s": float(elapsed_s)}
    import numpy as np
    for phase, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else float("nan"))
        residuals = list(s["residuals"])
        if residuals:
            arr = np.asarray(residuals, dtype=np.float64)
            mae = float(np.mean(np.abs(arr)))
            rmse = float(np.sqrt(np.mean(arr ** 2)))
            median = float(np.median(arr))
            iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25))
        else:
            mae = rmse = median = iqr = float("nan")

        entry = {"tp": tp, "fp": fp, "fn": fn,
                 "precision": precision, "recall": recall, "f1": f1,
                 "n_residuals": len(residuals),
                 "mae_s": mae, "rmse_s": rmse,
                 "median_s": median, "iqr_s": iqr}
        if keep_residuals:
            entry["residuals_s"] = residuals
        summary[phase] = entry

    if mcc_true:
        n_total = len(mcc_true)
        n_correct = sum(1 for t, p in zip(mcc_true, mcc_pred) if t == p)
        if len(set(mcc_true)) >= 2 and len(set(mcc_pred)) >= 2:
            summary["mcc_phase_id"] = float(matthews_corrcoef(mcc_true, mcc_pred))
        else:
            summary["mcc_phase_id"] = float("nan")
        summary["mcc_n_pairs"] = n_total
        summary["mcc_phase_accuracy"] = n_correct / n_total
        summary["mcc_n_phase_confusions"] = n_total - n_correct
    else:
        summary["mcc_phase_id"] = float("nan")
        summary["mcc_n_pairs"] = 0
        summary["mcc_phase_accuracy"] = float("nan")
        summary["mcc_n_phase_confusions"] = 0

    summary["detection"] = _aggregate_detection(det_records or [])
    return summary


def _aggregate_detection(det_records: list[dict]) -> dict:
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
    iou = [r["best_iou"] for r in detected]
    n_per = [r["n_detections"] for r in det_records]
    start_res = [abs(r["start_res_s"]) for r in detected if r.get("start_res_s") is not None]
    end_res = [abs(r["end_res_s"]) for r in detected if r.get("end_res_s") is not None]
    return {
        "det_recall": len(detected) / n_total,
        "det_n_traces": n_total,
        "det_iou_mean": _safe_mean(iou),
        "det_per_trace_mean": _safe_mean(n_per),
        "det_start_mae_s": _safe_mean(start_res),
        "det_end_mae_s": _safe_mean(end_res),
    }
