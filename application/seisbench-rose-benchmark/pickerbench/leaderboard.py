"""Aggregate per-model JSONs into the publication CSVs.

Two leaderboards: phase picking (per phase × threshold) and trace-level
event-vs-noise detection (single line per threshold).
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

from .residual_stats import residual_statistics


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _residuals_or_fallback(d: dict) -> dict:
    """Prefer raw residuals (saved in current bench JSONs); only fall back
    to coarse aggregate-based estimates if `residuals_s` is missing.

    The fallback is *approximate* — it uses ``std ≈ RMSE`` (assumes
    zero-centered residuals; the bias is undetectable without the mean,
    which we don't store) and ``MAD ≈ 0.5 * IQR`` (exact for symmetric
    distributions; biased high for heavy-tailed picker residuals). The
    fallback is provided only for compatibility with legacy JSONs that
    were written before the residual-saving patch; new runs always store
    raw residuals and exercise the exact path above.
    """
    if d.get("residuals_s"):
        return residual_statistics(d["residuals_s"])
    mae = d.get("mae_s", float("nan"))
    rmse = d.get("rmse_s", float("nan"))
    median = d.get("median_s", float("nan"))
    iqr = d.get("iqr_s", float("nan"))
    # Approximate std: use RMSE as upper bound; if median is known and
    # small, we can tighten via Var = E[r^2] - E[r]^2 with mean ~ median.
    if _isnan(median):
        std = rmse
    else:
        std = math.sqrt(max(0.0, float(rmse) ** 2 - float(median) ** 2))
    mad = 0.5 * float(iqr) if not _isnan(iqr) else float("nan")
    return {"n": d.get("n_residuals", 0), "mae": mae, "median": median,
            "std": std, "mad": mad, "rmse": rmse, "iqr": iqr}


def build_picking_csv(per_model_jsons: dict[str, Path], out_csv: Path,
                      thresholds: list[str]) -> Path:
    """`per_model_jsons` maps display name to JSON path."""
    cols = [
        "model", "threshold",
        "P_TP", "P_FP", "P_FN", "P_precision", "P_recall", "P_F1",
        "P_MAE_s", "P_median_s", "P_std_s", "P_MAD_s", "P_RMSE_s", "P_IQR_s",
        "S_TP", "S_FP", "S_FN", "S_precision", "S_recall", "S_F1",
        "S_MAE_s", "S_median_s", "S_std_s", "S_MAD_s", "S_RMSE_s", "S_IQR_s",
        "phase_id_accuracy", "n_phase_pairs", "n_phase_confusions",
        "n_event_traces", "n_noise_traces",
    ]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, path in per_model_jsons.items():
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            for thr in thresholds:
                t = d.get(thr)
                if t is None:
                    continue
                P, S = t["P"], t["S"]
                Pr = _residuals_or_fallback(P)
                Sr = _residuals_or_fallback(S)
                T1 = t.get("T1", {})
                w.writerow({
                    "model": name, "threshold": thr,
                    "P_TP": P["tp"], "P_FP": P["fp"], "P_FN": P["fn"],
                    "P_precision": f"{P['precision']:.4f}",
                    "P_recall": f"{P['recall']:.4f}",
                    "P_F1": f"{P['f1']:.4f}",
                    "P_MAE_s": f"{Pr['mae']:.4f}",
                    "P_median_s": f"{Pr['median']:+.4f}" if not _isnan(Pr["median"]) else "",
                    "P_std_s": f"{Pr['std']:.4f}" if not _isnan(Pr["std"]) else "",
                    "P_MAD_s": f"{Pr['mad']:.4f}" if not _isnan(Pr["mad"]) else "",
                    "P_RMSE_s": f"{Pr['rmse']:.4f}",
                    "P_IQR_s": f"{Pr['iqr']:.4f}" if not _isnan(Pr["iqr"]) else "",
                    "S_TP": S["tp"], "S_FP": S["fp"], "S_FN": S["fn"],
                    "S_precision": f"{S['precision']:.4f}",
                    "S_recall": f"{S['recall']:.4f}",
                    "S_F1": f"{S['f1']:.4f}",
                    "S_MAE_s": f"{Sr['mae']:.4f}",
                    "S_median_s": f"{Sr['median']:+.4f}" if not _isnan(Sr["median"]) else "",
                    "S_std_s": f"{Sr['std']:.4f}" if not _isnan(Sr["std"]) else "",
                    "S_MAD_s": f"{Sr['mad']:.4f}" if not _isnan(Sr["mad"]) else "",
                    "S_RMSE_s": f"{Sr['rmse']:.4f}",
                    "S_IQR_s": f"{Sr['iqr']:.4f}" if not _isnan(Sr["iqr"]) else "",
                    "phase_id_accuracy": f"{t.get('mcc_phase_accuracy', float('nan')):.4f}",
                    "n_phase_pairs": t.get("mcc_n_pairs", 0),
                    "n_phase_confusions": t.get("mcc_n_phase_confusions", 0),
                    "n_event_traces": T1.get("n_event", t.get("n_evaluated", "")),
                    "n_noise_traces": T1.get("n_noise", ""),
                })
    return out_csv


def build_detection_csv(per_model_jsons: dict, has_det_head: dict,
                        out_csv: Path, thresholds: list[str]) -> Path:
    cols = [
        "model", "threshold", "has_detection_head",
        "n_event", "n_noise",
        "T1_F1", "T1_precision", "T1_recall", "T1_mcc", "T1_AUC",
        "det_recall", "det_iou_mean",
        "det_start_mae_s", "det_end_mae_s", "det_per_trace_mean",
    ]
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, path in per_model_jsons.items():
            if not path.exists():
                continue
            has = has_det_head.get(name, False)
            d = json.loads(path.read_text())
            for thr in thresholds:
                t = d.get(thr)
                if t is None:
                    continue
                T1 = t.get("T1", {}) or {}
                det = t.get("detection", {}) or {}
                w.writerow({
                    "model": name, "threshold": thr,
                    "has_detection_head": has,
                    "n_event": T1.get("n_event"), "n_noise": T1.get("n_noise"),
                    "T1_F1": f"{T1.get('f1', float('nan')):.4f}",
                    "T1_precision": f"{T1.get('precision', float('nan')):.4f}",
                    "T1_recall": f"{T1.get('recall', float('nan')):.4f}",
                    "T1_mcc": f"{T1.get('mcc', float('nan')):.4f}",
                    "T1_AUC": f"{T1.get('auc_det_head', float('nan')):.4f}",
                    "det_recall": (f"{det.get('det_recall', float('nan')):.4f}"
                                   if has else ""),
                    "det_iou_mean": (f"{det.get('det_iou_mean', float('nan')):.4f}"
                                     if has else ""),
                    "det_start_mae_s": (f"{det.get('det_start_mae_s', float('nan')):.4f}"
                                        if has else ""),
                    "det_end_mae_s": (f"{det.get('det_end_mae_s', float('nan')):.4f}"
                                      if has else ""),
                    "det_per_trace_mean": (f"{det.get('det_per_trace_mean', float('nan')):.4f}"
                                           if has else ""),
                })
    return out_csv
