"""Build the final RoSE benchmark leaderboards (picking + detection).

Convention:
    False positives are counted on STEAD NOISE TRACES ONLY. Predicted picks
    on event traces that don't match a labelled phase are kept as a separate
    diagnostic column ("unmatched on event trace") because RoSE event labels
    are incomplete — an unlabelled aftershock looks like a false positive
    under canonical pick-benchmark scoring but is actually a real arrival.

Inputs (all produced by earlier inference runs; no re-inference here):
    bench_rose_full_sweep/sweep_comparison.csv   (8 PyTorch models on RoSE)
    bench_redpan_rose_full/sweep_comparison.csv  (RED-PAN-60s on RoSE)
    bench_noise_fp/<model>.json                  (STEAD-noise picks per model)

Outputs:
    bench_rose_picking_clean.csv      (per-phase picking, all 9 models)
    bench_rose_detection_clean.csv    (trace-level detection, EQT family + RP60)

Usage:
    python benchmark/build_rose_final_benchmark.py [--eval-dir eval]

Default --eval-dir comes from ``$ROSE_EVAL_DIR`` (fallback: ``eval``).
"""
from __future__ import annotations

import argparse
import os
import csv
import json
import math
from pathlib import Path

THRESHOLDS = ["0.05", "0.1", "0.2", "0.3", "0.5", "0.7"]

MODEL_TO_NOISE_KEY = {
    "EQT-RoSE": "eqt_rose",
    "PhaseNet-RoSE": "phasenet_rose",
    "EQT-instance": "eqt_instance",
    "PhaseNet-instance": "phasenet_instance",
    "EQT-ethz": "eqt_ethz",
    "PhaseNet-ethz": "phasenet_ethz",
    "EQT-stead": "eqt_stead",
    "PhaseNet-stead": "phasenet_stead",
    "RED-PAN-60s": "redpan",
}

MODEL_HAS_DETECTION_HEAD = {
    "EQT-RoSE": True, "EQT-instance": True,
    "EQT-ethz": True, "EQT-stead": True,
    "RED-PAN-60s": True,
    "PhaseNet-RoSE": False, "PhaseNet-instance": False,
    "PhaseNet-ethz": False, "PhaseNet-stead": False,
}


def _num(x, default: float = float("nan")) -> float:
    """Parse a CSV cell to float; '' / non-numeric -> ``default`` (NaN).

    The output CSVs intentionally write '' for NaN metrics (e.g. a model that
    evaluated zero traces in a subset run), so the headline printers below must
    not choke on them.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def f1_score(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return float("nan")
    return 2 * precision * recall / (precision + recall)


def matthews_corr(tp: int, fp: int, fn: int, tn: int) -> float:
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denom == 0:
        return float("nan")
    return (tp * tn - fp * fn) / denom


def load_sweep_rows(eval_dir: Path) -> list[dict]:
    rows = []
    missing: list[Path] = []
    for fname in [
        eval_dir / "bench_rose_full_sweep" / "sweep_comparison.csv",
        eval_dir / "bench_redpan_rose_full" / "sweep_comparison.csv",
    ]:
        if not fname.is_file():
            missing.append(fname)
            continue
        with fname.open() as f:
            rows.extend(csv.DictReader(f))
    if missing:
        missing_list = "\n  - ".join(str(p) for p in missing)
        hint = ("Run benchmark/run_inference.py first (or pass "
                "--skip-rose if you only care about the STEAD leg).")
        raise FileNotFoundError(
            "Missing per-model sweep output(s):\n  - "
            f"{missing_list}\n{hint}"
        )
    return rows


def index_sweep(rows: list[dict]) -> dict:
    idx = {}
    for r in rows:
        idx.setdefault((r["model"], r["threshold"]), {})[r["phase"]] = r
    return idx


def build_picking_table(eval_dir: Path, sweep_index: dict) -> Path:
    out = eval_dir / "bench_rose_picking_clean.csv"
    cols = [
        "model", "threshold",
        "P_true_pos", "P_false_neg", "P_false_pos_on_noise",
        "P_precision", "P_recall", "P_F1",
        "P_mae_seconds", "P_median_residual_s", "P_iqr_residual_s",
        "S_true_pos", "S_false_neg", "S_false_pos_on_noise",
        "S_precision", "S_recall", "S_F1",
        "S_mae_seconds", "S_median_residual_s", "S_iqr_residual_s",
        "phase_id_accuracy", "n_phase_pairs", "n_phase_confusions",
        "noise_n_traces", "noise_traces_with_any_pick", "noise_trace_fp_rate",
        "P_unmatched_on_event_trace_diagnostic",
        "S_unmatched_on_event_trace_diagnostic",
        "n_event_traces",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for model, noise_key in MODEL_TO_NOISE_KEY.items():
            noise_path = eval_dir / "bench_noise_fp" / f"{noise_key}.json"
            noise = json.loads(noise_path.read_text())
            for thr in THRESHOLDS:
                phases = sweep_index.get((model, thr))
                if phases is None:
                    continue
                p_row, s_row = phases["P"], phases["S"]
                noise_thr = noise[thr]

                p_tp = int(float(p_row["tp"]))
                s_tp = int(float(s_row["tp"]))
                p_fn = int(float(p_row["fn"]))
                s_fn = int(float(s_row["fn"]))
                p_fp_noise = int(noise_thr["n_picks_P"])
                s_fp_noise = int(noise_thr["n_picks_S"])

                p_prec = p_tp / max(1, p_tp + p_fp_noise)
                s_prec = s_tp / max(1, s_tp + s_fp_noise)
                p_rec = p_tp / max(1, p_tp + p_fn)
                s_rec = s_tp / max(1, s_tp + s_fn)

                writer.writerow({
                    "model": model, "threshold": thr,
                    "P_true_pos": p_tp, "P_false_neg": p_fn,
                    "P_false_pos_on_noise": p_fp_noise,
                    "P_precision": f"{p_prec:.4f}",
                    "P_recall": f"{p_rec:.4f}",
                    "P_F1": f"{f1_score(p_prec, p_rec):.4f}",
                    "P_mae_seconds": p_row["mae_s"],
                    "P_median_residual_s": p_row["median_s"],
                    "P_iqr_residual_s": p_row["iqr_s"],
                    "S_true_pos": s_tp, "S_false_neg": s_fn,
                    "S_false_pos_on_noise": s_fp_noise,
                    "S_precision": f"{s_prec:.4f}",
                    "S_recall": f"{s_rec:.4f}",
                    "S_F1": f"{f1_score(s_prec, s_rec):.4f}",
                    "S_mae_seconds": s_row["mae_s"],
                    "S_median_residual_s": s_row["median_s"],
                    "S_iqr_residual_s": s_row["iqr_s"],
                    "phase_id_accuracy": p_row["mcc_phase_accuracy"],
                    "n_phase_pairs": p_row["mcc_n_pairs"],
                    "n_phase_confusions": p_row["mcc_n_phase_confusions"],
                    "noise_n_traces": noise_thr["n_traces_evaluated"],
                    "noise_traces_with_any_pick": noise_thr["n_traces_with_any_pick"],
                    "noise_trace_fp_rate": f"{noise_thr['trace_level_fp_rate']:.4f}",
                    "P_unmatched_on_event_trace_diagnostic": int(float(p_row["fp"])),
                    "S_unmatched_on_event_trace_diagnostic": int(float(s_row["fp"])),
                    "n_event_traces": p_row["n_evaluated"],
                })
    return out


def build_detection_table(eval_dir: Path, sweep_index: dict) -> Path:
    out = eval_dir / "bench_rose_detection_clean.csv"
    cols = [
        "model", "threshold", "has_detection_head",
        "n_event_traces", "n_noise_traces",
        "true_pos", "false_neg", "false_pos", "true_neg",
        "precision", "recall", "F1", "matthews_corr",
        "iou_mean_on_true_pos",
        "start_time_mae_seconds", "end_time_mae_seconds",
        "boxes_per_trace_mean",
    ]
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for model, noise_key in MODEL_TO_NOISE_KEY.items():
            has_det = MODEL_HAS_DETECTION_HEAD[model]
            noise_path = eval_dir / "bench_noise_fp" / f"{noise_key}.json"
            noise = json.loads(noise_path.read_text())
            for thr in THRESHOLDS:
                phases = sweep_index.get((model, thr))
                if phases is None:
                    continue
                p_row = phases["P"]
                noise_thr = noise[thr]
                n_event = int(float(p_row["n_evaluated"]))
                n_noise = int(noise_thr["n_traces_evaluated"])
                fp = int(noise_thr["n_traces_with_any_pick"])
                tn = n_noise - fp

                if has_det:
                    det_recall = float(p_row["det_recall"]) if p_row["det_recall"] else 0.0
                    tp = int(round(det_recall * n_event))
                    fn = n_event - tp
                    iou_mean = p_row["det_iou_mean"]
                    start_mae = p_row["det_start_mae_s"]
                    end_mae = p_row["det_end_mae_s"]
                    boxes_per = p_row["det_per_trace_mean"]
                    prec = tp / max(1, tp + fp)
                    rec = tp / max(1, tp + fn)
                    f1 = f1_score(prec, rec)
                    mcc = matthews_corr(tp, fp, fn, tn)
                else:
                    tp = fn = 0
                    iou_mean = start_mae = end_mae = boxes_per = ""
                    prec = rec = f1 = mcc = float("nan")

                writer.writerow({
                    "model": model, "threshold": thr,
                    "has_detection_head": has_det,
                    "n_event_traces": n_event, "n_noise_traces": n_noise,
                    "true_pos": tp, "false_neg": fn,
                    "false_pos": fp, "true_neg": tn,
                    "precision": "" if math.isnan(prec) else f"{prec:.4f}",
                    "recall": "" if math.isnan(rec) else f"{rec:.4f}",
                    "F1": "" if math.isnan(f1) else f"{f1:.4f}",
                    "matthews_corr": "" if math.isnan(mcc) else f"{mcc:.4f}",
                    "iou_mean_on_true_pos": iou_mean,
                    "start_time_mae_seconds": start_mae,
                    "end_time_mae_seconds": end_mae,
                    "boxes_per_trace_mean": boxes_per,
                })
    return out


def print_picking_headline(picking_csv: Path, threshold: str = "0.3") -> None:
    print()
    print("=" * 130)
    print(f"  PHASE PICKING leaderboard at threshold = {threshold}")
    print("  False positives counted on STEAD noise traces only.")
    print("=" * 130)
    header = (f"  {'model':<22}"
              f"{'P_TP':>7}{'P_FN':>7}{'P_FP':>6}{'P_prec':>8}{'P_rec':>7}{'P_F1':>7}"
              f"{'  ':>2}"
              f"{'S_TP':>7}{'S_FN':>7}{'S_FP':>6}{'S_prec':>8}{'S_rec':>7}{'S_F1':>7}"
              f"{'  ':>2}"
              f"{'noise_FP%':>10}")
    print(header)
    print("-" * 130)
    with picking_csv.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["threshold"] == threshold]

    def _key(r):
        v = _num(r["P_F1"], 0.0) + _num(r["S_F1"], 0.0)
        return -v
    rows.sort(key=_key)
    for r in rows:
        print(f"  {r['model']:<22}"
              f"{r['P_true_pos']:>7}{r['P_false_neg']:>7}{r['P_false_pos_on_noise']:>6}"
              f"{_num(r['P_precision']):>8.3f}{_num(r['P_recall']):>7.3f}{_num(r['P_F1']):>7.3f}"
              f"  "
              f"{r['S_true_pos']:>7}{r['S_false_neg']:>7}{r['S_false_pos_on_noise']:>6}"
              f"{_num(r['S_precision']):>8.3f}{_num(r['S_recall']):>7.3f}{_num(r['S_F1']):>7.3f}"
              f"  "
              f"{_num(r['noise_trace_fp_rate'])*100:>9.2f}%")


def print_detection_headline(detection_csv: Path, threshold: str = "0.3") -> None:
    print()
    print("=" * 130)
    print(f"  EARTHQUAKE DETECTION leaderboard at threshold = {threshold}")
    print("  TP/FN: event traces with overlapping detection box.")
    print("  FP   : noise traces with any pick at threshold.")
    print("=" * 130)
    header = (f"  {'model':<22}"
              f"{'TP':>7}{'FN':>7}{'FP':>6}{'TN':>7}"
              f"{'precision':>11}{'recall':>9}{'F1':>8}{'matthews':>10}"
              f"{'IoU':>7}{'startMAE':>10}{'endMAE':>9}")
    print(header)
    print("-" * 130)
    with detection_csv.open() as f:
        rows = list(csv.DictReader(f))
    rows = [r for r in rows if r["threshold"] == threshold]

    rows.sort(key=lambda r: -_num(r["F1"], 0.0))
    for r in rows:
        if r["has_detection_head"] == "False":
            print(f"  {r['model']:<22}    (no detection head — not applicable)")
            continue
        print(f"  {r['model']:<22}"
              f"{r['true_pos']:>7}{r['false_neg']:>7}{r['false_pos']:>6}{r['true_neg']:>7}"
              f"{_num(r['precision']):>11.3f}{_num(r['recall']):>9.4f}"
              f"{_num(r['F1']):>8.4f}{_num(r['matthews_corr']):>10.4f}"
              f"{_num(r['iou_mean_on_true_pos']):>7.3f}"
              f"{_num(r['start_time_mae_seconds']):>10.3f}"
              f"{_num(r['end_time_mae_seconds']):>9.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eval-dir", type=Path,
                        default=Path(os.environ.get("ROSE_EVAL_DIR", "eval")),
                        help="Directory containing the four input bench folders.")
    parser.add_argument("--threshold", default="0.3",
                        help="Threshold for the headline print (CSV has all six).")
    args = parser.parse_args()

    eval_dir: Path = args.eval_dir
    sweep_index = index_sweep(load_sweep_rows(eval_dir))

    picking_csv = build_picking_table(eval_dir, sweep_index)
    detection_csv = build_detection_table(eval_dir, sweep_index)

    print(f"wrote {picking_csv}")
    print(f"wrote {detection_csv}")

    print_picking_headline(picking_csv, threshold=args.threshold)
    print_detection_headline(detection_csv, threshold=args.threshold)


if __name__ == "__main__":
    main()
