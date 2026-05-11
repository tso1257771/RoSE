"""Build the full-STEAD benchmark leaderboards (picking + detection + residuals).

Inputs (all produced by earlier inference runs):
    bench_stead_full_pytorch/<model>/<model>.json   (8 PyTorch models)
    bench_stead_full/redpan_merged.json             (RED-PAN-60s, 4-shard merged)

Outputs:
    bench_stead_full_picking.csv      (per-phase picking, all 9 models, 6 thresholds)
    bench_stead_full_detection.csv    (trace-level event-vs-noise T1, all 9)
    bench_stead_full_residuals.csv    (MAE / MAD / std per matched-pick set)

STEAD label convention is clean (one labelled event per window with no
unlabelled aftershocks expected), so unlike RoSE we use the canonical
pick-benchmark FP convention:
  * per-phase FP includes both event-trace mis-localization and noise picks
  * T1 FP = noise traces with any pick at threshold

Usage:
    python build_stead_full_benchmark.py [--threshold 0.3]
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
import statistics
from pathlib import Path

DEFAULT_EVAL = Path(os.environ.get("ROSE_EVAL_DIR", "eval"))
THRESHOLDS = ["0.05", "0.1", "0.2", "0.3", "0.5", "0.7"]

# (display name, json path, has_detection_head)
def model_paths(eval_dir: Path) -> list[tuple[str, Path, bool]]:
    pt = eval_dir / "bench_stead_full_pytorch"
    rp = eval_dir / "bench_stead_full"
    return [
        ("EQT-RoSE",       pt / "eqt_rose"       / "eqt_rose.json",       True),
        ("PhaseNet-RoSE",  pt / "phasenet_rose"  / "phasenet_rose.json",  False),
        ("EQT-instance",      pt / "eqt_instance"      / "eqt_instance.json",      True),
        ("PhaseNet-instance", pt / "phasenet_instance" / "phasenet_instance.json", False),
        ("EQT-ethz",          pt / "eqt_ethz"          / "eqt_ethz.json",          True),
        ("PhaseNet-ethz",     pt / "phasenet_ethz"     / "phasenet_ethz.json",     False),
        ("EQT-stead",         pt / "eqt_stead"         / "eqt_stead.json",         True),
        ("PhaseNet-stead",    pt / "phasenet_stead"    / "phasenet_stead.json",    False),
        ("REDPAN-60s",        rp / "redpan_merged.json",                            True),
    ]


def f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if (p + r) > 0 else float("nan")


def residual_stats(res: list[float]) -> dict:
    if not res:
        return {"n": 0, "mae": float("nan"), "median": float("nan"),
                "std": float("nan"), "mad": float("nan"),
                "rmse": float("nan"), "iqr": float("nan")}
    n = len(res)
    mean = sum(res) / n
    abs_res = [abs(x) for x in res]
    mae = sum(abs_res) / n
    rmse = math.sqrt(sum(x*x for x in res) / n)
    median = statistics.median(res)
    abs_dev = sorted(abs(x - median) for x in res)
    mad = abs_dev[n // 2] if n % 2 == 1 else 0.5 * (abs_dev[n//2 - 1] + abs_dev[n//2])
    var = sum((x - mean) ** 2 for x in res) / n
    std = math.sqrt(var)
    sr = sorted(res)
    q25 = sr[int(0.25 * n)]
    q75 = sr[int(0.75 * n)]
    iqr = q75 - q25
    return {"n": n, "mae": mae, "median": median, "std": std,
            "mad": mad, "rmse": rmse, "iqr": iqr}


def _is_nan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


def _residuals_or_fallback(d: dict) -> dict:
    """Prefer raw residuals_s; otherwise reconstruct from aggregates.

    Some bench JSONs (e.g. the 4-shard RP60 merge on STEAD) don't preserve
    raw residuals or even the median/IQR (those don't aggregate cleanly
    across shards). For those we fall back to:
      std = RMSE             (exact when residuals are zero-centered;
                              the bias is undetectable without the median)
      mad = NaN              (no robust scale estimate available without
                              raw residuals or IQR)
    """
    if d.get("residuals_s"):
        return residual_stats(d["residuals_s"])

    mae = d.get("mae_s", float("nan"))
    rmse = d.get("rmse_s", float("nan"))
    median = d.get("median_s", float("nan"))
    iqr = d.get("iqr_s", float("nan"))

    if _is_nan(median):
        std = rmse
    else:
        std = math.sqrt(max(0.0, float(rmse) ** 2 - float(median) ** 2))

    if _is_nan(iqr):
        mad = float("nan")
    else:
        mad = 0.5 * float(iqr)

    return {
        "n": d.get("n_residuals", 0),
        "mae": mae, "median": median if not _is_nan(median) else float("nan"),
        "rmse": rmse, "iqr": iqr if not _is_nan(iqr) else float("nan"),
        "std": std, "mad": mad,
    }


def build_picking_table(eval_dir: Path, models: list) -> Path:
    out = eval_dir / "bench_stead_full_picking.csv"
    cols = [
        "model", "threshold",
        "P_TP", "P_FP", "P_FN", "P_precision", "P_recall", "P_F1",
        "P_MAE", "P_median", "P_std", "P_MAD", "P_RMSE", "P_IQR",
        "S_TP", "S_FP", "S_FN", "S_precision", "S_recall", "S_F1",
        "S_MAE", "S_median", "S_std", "S_MAD", "S_RMSE", "S_IQR",
        "phase_id_accuracy", "n_phase_pairs", "n_phase_confusions",
        "n_event_traces", "n_noise_traces",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, path, _ in models:
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            for thr in THRESHOLDS:
                t = d.get(thr)
                if t is None:
                    continue
                P, S = t["P"], t["S"]
                # exact residual stats (preferred — uses raw residuals_s)
                Pr = _residuals_or_fallback(P)
                Sr = _residuals_or_fallback(S)
                T1 = t.get("T1", {})
                w.writerow({
                    "model": name, "threshold": thr,
                    "P_TP": P["tp"], "P_FP": P["fp"], "P_FN": P["fn"],
                    "P_precision": f"{P['precision']:.4f}",
                    "P_recall":    f"{P['recall']:.4f}",
                    "P_F1":        f"{P['f1']:.4f}",
                    "P_MAE":  f"{Pr['mae']:.4f}",
                    "P_median": f"{Pr['median']:+.4f}",
                    "P_std":  f"{Pr['std']:.4f}",
                    "P_MAD":  f"{Pr['mad']:.4f}",
                    "P_RMSE": f"{Pr['rmse']:.4f}",
                    "P_IQR":  f"{Pr['iqr']:.4f}",
                    "S_TP": S["tp"], "S_FP": S["fp"], "S_FN": S["fn"],
                    "S_precision": f"{S['precision']:.4f}",
                    "S_recall":    f"{S['recall']:.4f}",
                    "S_F1":        f"{S['f1']:.4f}",
                    "S_MAE":  f"{Sr['mae']:.4f}",
                    "S_median": f"{Sr['median']:+.4f}",
                    "S_std":  f"{Sr['std']:.4f}",
                    "S_MAD":  f"{Sr['mad']:.4f}",
                    "S_RMSE": f"{Sr['rmse']:.4f}",
                    "S_IQR":  f"{Sr['iqr']:.4f}",
                    "phase_id_accuracy": f"{t.get('mcc_phase_accuracy', float('nan')):.4f}",
                    "n_phase_pairs": t.get("mcc_n_pairs", 0),
                    "n_phase_confusions": t.get("mcc_n_phase_confusions", 0),
                    "n_event_traces": T1.get("n_event", 103040),
                    "n_noise_traces": T1.get("n_noise", 23526),
                })
    return out


def build_detection_table(eval_dir: Path, models: list) -> Path:
    out = eval_dir / "bench_stead_full_detection.csv"
    cols = [
        "model", "threshold", "has_detection_head",
        "n_event", "n_noise",
        "T1_F1", "T1_precision", "T1_recall", "T1_mcc", "T1_AUC",
        "det_recall", "det_iou_mean",
        "det_start_mae_s", "det_end_mae_s",
        "det_per_trace_mean",
    ]
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for name, path, has_det in models:
            if not path.exists():
                continue
            d = json.loads(path.read_text())
            for thr in THRESHOLDS:
                t = d.get(thr)
                if t is None:
                    continue
                T1 = t.get("T1", {})
                det = t.get("detection", {}) or {}
                w.writerow({
                    "model": name, "threshold": thr,
                    "has_detection_head": has_det,
                    "n_event": T1.get("n_event"),
                    "n_noise": T1.get("n_noise"),
                    "T1_F1":        f"{T1.get('f1', float('nan')):.4f}",
                    "T1_precision": f"{T1.get('precision', float('nan')):.4f}",
                    "T1_recall":    f"{T1.get('recall', float('nan')):.4f}",
                    "T1_mcc":       f"{T1.get('mcc', float('nan')):.4f}",
                    "T1_AUC":       f"{T1.get('auc_det_head', float('nan')):.4f}",
                    "det_recall":   f"{det.get('det_recall', float('nan')):.4f}" if has_det else "",
                    "det_iou_mean": f"{det.get('det_iou_mean', float('nan')):.4f}" if has_det else "",
                    "det_start_mae_s": f"{det.get('det_start_mae_s', float('nan')):.4f}" if has_det else "",
                    "det_end_mae_s":   f"{det.get('det_end_mae_s', float('nan')):.4f}" if has_det else "",
                    "det_per_trace_mean": f"{det.get('det_per_trace_mean', float('nan')):.4f}" if has_det else "",
                })
    return out


def print_picking_headline(csv_path: Path, threshold: str = "0.3") -> None:
    print()
    print("=" * 132)
    print(f"  STEAD-FULL phase picking @ threshold {threshold}")
    print("  126,566 traces (103,040 events + 23,526 noise) — canonical pick-benchmark FP convention.")
    print("=" * 132)
    print(f"  {'model':<22}"
          f"{'P_TP':>8}{'P_FP':>7}{'P_FN':>7}{'P_F1':>7}{'P_MAE':>8}{'P_MAD':>8}{'P_std':>8}"
          f"{'  ':>2}"
          f"{'S_TP':>8}{'S_FP':>7}{'S_FN':>7}{'S_F1':>7}{'S_MAE':>8}{'S_MAD':>8}{'S_std':>8}")
    print("-" * 132)
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["threshold"] == threshold]
    rows.sort(key=lambda r: -(float(r["P_F1"]) + float(r["S_F1"])))
    for r in rows:
        print(f"  {r['model']:<22}"
              f"{r['P_TP']:>8}{r['P_FP']:>7}{r['P_FN']:>7}"
              f"{float(r['P_F1']):>7.3f}"
              f"{float(r['P_MAE']):>8.4f}{float(r['P_MAD']):>8.4f}{float(r['P_std']):>8.4f}"
              f"  "
              f"{r['S_TP']:>8}{r['S_FP']:>7}{r['S_FN']:>7}"
              f"{float(r['S_F1']):>7.3f}"
              f"{float(r['S_MAE']):>8.4f}{float(r['S_MAD']):>8.4f}{float(r['S_std']):>8.4f}")


def print_detection_headline(csv_path: Path, threshold: str = "0.3") -> None:
    print()
    print("=" * 130)
    print(f"  STEAD-FULL earthquake detection (T1 event-vs-noise) @ threshold {threshold}")
    print("=" * 130)
    print(f"  {'model':<22}"
          f"{'T1_F1':>8}{'T1_prec':>9}{'T1_rec':>9}{'T1_MCC':>8}{'T1_AUC':>8}"
          f"  {'detRec':>8}{'IoU':>7}{'startMAE':>10}{'endMAE':>9}")
    print("-" * 130)
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["threshold"] == threshold]
    def k(r):
        try: return -float(r["T1_F1"])
        except: return 0.0
    rows.sort(key=k)
    for r in rows:
        det_str = ""
        if r["has_detection_head"] == "True":
            try:
                det_str = (f"  {float(r['det_recall']):>8.4f}"
                           f"{float(r['det_iou_mean']):>7.3f}"
                           f"{float(r['det_start_mae_s']):>10.3f}"
                           f"{float(r['det_end_mae_s']):>9.3f}")
            except Exception:
                det_str = ""
        else:
            det_str = "  (no detection head)"
        print(f"  {r['model']:<22}"
              f"{float(r['T1_F1']):>8.4f}{float(r['T1_precision']):>9.4f}"
              f"{float(r['T1_recall']):>9.4f}{float(r['T1_mcc']):>8.4f}"
              f"{float(r['T1_AUC']):>8.4f}"
              f"{det_str}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL)
    ap.add_argument("--threshold", default="0.3")
    args = ap.parse_args()

    models = model_paths(args.eval_dir)
    available = [(n, p, h) for n, p, h in models if p.exists()]
    missing = [n for n, p, _ in models if not p.exists()]

    print(f"available: {len(available)}/{len(models)} models")
    for n, p, _ in available:
        print(f"  ✓ {n}")
    if missing:
        print("missing (still inferring or not yet run):")
        for n in missing:
            print(f"  ✗ {n}")

    pick_csv = build_picking_table(args.eval_dir, available)
    det_csv  = build_detection_table(args.eval_dir, available)
    print(f"\nwrote {pick_csv}")
    print(f"wrote {det_csv}")

    print_picking_headline(pick_csv, threshold=args.threshold)
    print_detection_headline(det_csv, threshold=args.threshold)


if __name__ == "__main__":
    main()
