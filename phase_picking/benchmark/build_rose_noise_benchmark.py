#!/usr/bin/env python3
"""RoSE leaderboard with false positives counted on RoSE noise windows.

The published RoSE tables (results/rose_picking.csv and rose_detection.csv)
count false positives on STEAD's 23,526 noise traces, because RoSE had no
noise of its own when they were made. This script recounts them on RoSE noise
windows: the test-split windows that export_rose_noise.py writes, run through
bench_noise_fp.py, which leaves one <model>.json per model in --noise-run (or
<model>.shard<k>of<N>.json files, which are summed). The event side (TP, FN)
is read from the published CSVs unchanged; the formulas are those of
build_rose_final_benchmark.py.

The test split has 13,317 noise windows, fewer than STEAD's noise traces, so
each model's noise counts are scaled to the STEAD count of the published row
(23,526): precision, F1 and MCC then differ from the published values only in
where the noise comes from. --no-scale keeps the raw counts.
rose_noise_false_picks.csv lists the raw counts next to the STEAD ones.

Outputs, under --out-dir (default: the config's eval_dir); --update-results
also copies them into results/:
    rose_noise_false_picks.csv     noise traces with a pick, and picks per hour, RoSE vs STEAD noise
    rose_picking_rose_noise.csv    the picking leaderboard with FP counted on RoSE noise
    rose_detection_rose_noise.csv  the detection leaderboard with FP counted on RoSE noise

    python phase_picking/benchmark/build_rose_noise_benchmark.py \\
        --noise-run phase_picking/benchmark/eval/bench_noise_fp_rose --update-results
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline import add_common_args, load_config, resolve  # noqa: E402
from build_rose_final_benchmark import (  # noqa: E402
    MODEL_TO_NOISE_KEY, f1_score, matthews_corr)

STEAD_NOISE_DATA_S = 60.0    # a STEAD noise trace is 60 s of data followed by 210 s of zeros


def _fmt(x: float, nd: int = 4) -> str:
    return "" if x != x else f"{x:.{nd}f}"


def load_noise_counts(run_dir: Path, key: str) -> dict:
    """{threshold: {"n", "with_pick", "P", "S"}} from one model's bench_noise_fp output."""
    whole = run_dir / f"{key}.json"
    parts = [whole] if whole.is_file() else sorted(run_dir.glob(f"{key}.shard*of*.json"))
    if not parts:
        raise FileNotFoundError(f"no {key}.json (or shards) in {run_dir}; run bench_noise_fp.py first")
    counts: dict = {}
    for path in parts:
        result = json.loads(path.read_text())
        for thr, block in result.items():
            if not isinstance(block, dict):
                continue
            c = counts.setdefault(str(float(thr)), {"n": 0, "with_pick": 0, "P": 0, "S": 0})
            c["n"] += int(block["n_traces_evaluated"])
            c["with_pick"] += int(block["n_traces_with_any_pick"])
            c["P"] += int(block["n_picks_P"])
            c["S"] += int(block["n_picks_S"])
    if any(c["n"] == 0 for c in counts.values()):
        raise ValueError(f"{key}: no noise trace was evaluated in {run_dir}")
    return counts


def _rows(path: Path) -> dict:
    with path.open() as fh:
        return {(r["model"], str(float(r["threshold"]))): r for r in csv.DictReader(fh)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--noise-run", default=None,
                    help="bench_noise_fp.py output folder for the RoSE noise "
                         "(default: <eval_dir>/bench_noise_fp_rose)")
    ap.add_argument("--window-s", type=float, default=60.0,
                    help="length of the exported RoSE noise windows, for picks per hour (default 60)")
    ap.add_argument("--no-scale", action="store_true",
                    help="keep the raw RoSE noise counts instead of scaling them to the STEAD count")
    ap.add_argument("--results-dir", default=None, help="Override config.results_dir.")
    ap.add_argument("--out-dir", default=None, help="where to write the CSVs (default: eval_dir)")
    ap.add_argument("--update-results", action="store_true",
                    help="also copy the three CSVs into the results/ directory")
    ap.add_argument("--threshold", default=None,
                    help="threshold of the printed table (default: config.display_threshold)")
    args = ap.parse_args()

    cfg = resolve(load_config(args.config), args)
    results_dir = Path(args.results_dir or cfg["results_dir"])
    eval_dir = Path(cfg["eval_dir"])
    noise_run = Path(args.noise_run) if args.noise_run else eval_dir / "bench_noise_fp_rose"
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    show = str(float(args.threshold if args.threshold is not None else cfg["display_threshold"]))

    picking = _rows(results_dir / "rose_picking.csv")
    detection = _rows(results_dir / "rose_detection.csv")
    noise_rows, pick_rows, det_rows = [], [], []
    for model, key in MODEL_TO_NOISE_KEY.items():
        counts = load_noise_counts(noise_run, key)
        for (m, thr), pb in picking.items():
            if m != model:
                continue
            if thr not in counts:
                raise KeyError(f"{key}: threshold {thr} is not in the noise run {noise_run}")
            nz, db = counts[thr], detection[(model, thr)]
            n_stead = int(pb["noise_n_traces"])
            k = 1.0 if args.no_scale else n_stead / nz["n"]
            rose_hours = nz["n"] * args.window_s / 3600
            stead_hours = n_stead * STEAD_NOISE_DATA_S / 3600
            stead_p, stead_s = int(pb["P_false_pos_on_noise"]), int(pb["S_false_pos_on_noise"])
            noise_rows.append({
                "model": model, "threshold": thr,
                "rose_windows": nz["n"], "rose_window_s": args.window_s,
                "rose_windows_with_pick": nz["with_pick"],
                "rose_window_fp_rate": _fmt(nz["with_pick"] / nz["n"]),
                "rose_P_picks": nz["P"], "rose_S_picks": nz["S"],
                "rose_picks_per_hour": _fmt((nz["P"] + nz["S"]) / rose_hours, 3),
                "stead_traces": n_stead,
                "stead_traces_with_pick": int(pb["noise_traces_with_any_pick"]),
                "stead_trace_fp_rate": pb["noise_trace_fp_rate"],
                "stead_P_picks": stead_p, "stead_S_picks": stead_s,
                "stead_picks_per_hour": _fmt((stead_p + stead_s) / stead_hours, 3),
            })
            row = {"model": model, "threshold": thr, "noise_scale": _fmt(k)}
            for ph in "PS":
                tp, fn = int(pb[f"{ph}_true_pos"]), int(pb[f"{ph}_false_neg"])
                fp = k * nz[ph]
                prec, rec = tp / max(1.0, tp + fp), tp / max(1, tp + fn)
                row |= {f"{ph}_true_pos": tp, f"{ph}_false_neg": fn,
                        f"{ph}_false_pos_on_rose_noise": _fmt(fp, 1),
                        f"{ph}_precision": _fmt(prec), f"{ph}_recall": _fmt(rec),
                        f"{ph}_F1": _fmt(f1_score(prec, rec)),
                        f"{ph}_precision_stead_noise": pb[f"{ph}_precision"],
                        f"{ph}_F1_stead_noise": pb[f"{ph}_F1"]}
            pick_rows.append(row)
            n_noise, fp = k * nz["n"], k * nz["with_pick"]
            drow = {"model": model, "threshold": thr,
                    "has_detection_head": db["has_detection_head"],
                    "n_event_traces": db["n_event_traces"], "n_noise_traces": _fmt(n_noise, 1),
                    "true_pos": "", "false_neg": "", "false_pos": _fmt(fp, 1),
                    "true_neg": _fmt(n_noise - fp, 1), "precision": "", "recall": "",
                    "F1": "", "matthews_corr": "",
                    "F1_stead_noise": db["F1"], "matthews_corr_stead_noise": db["matthews_corr"]}
            if db["has_detection_head"] == "True":
                tp, fn = int(db["true_pos"]), int(db["false_neg"])
                prec, rec = tp / max(1.0, tp + fp), tp / max(1, tp + fn)
                drow |= {"true_pos": tp, "false_neg": fn, "precision": _fmt(prec),
                         "recall": _fmt(rec), "F1": _fmt(f1_score(prec, rec)),
                         "matthews_corr": _fmt(matthews_corr(tp, fp, fn, n_noise - fp))}
            det_rows.append(drow)

    outputs = []
    for name, rows in (("rose_noise_false_picks.csv", noise_rows),
                       ("rose_picking_rose_noise.csv", pick_rows),
                       ("rose_detection_rose_noise.csv", det_rows)):
        path = out_dir / name
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs.append(path)
        print(f"wrote {path} ({len(rows)} rows)")

    print(f"\nthreshold {show}: F1 with FP on RoSE noise (on STEAD noise), share of 60-s noise "
          f"traces with a pick (RoSE vs STEAD)")
    print(f"  {'model':<18}{'P F1':>15}{'S F1':>15}{'det F1':>15}{'noise traces with pick':>26}")
    dets = {(r["model"], r["threshold"]): r for r in det_rows}
    for r, nz in zip(pick_rows, noise_rows):
        if r["threshold"] != show:
            continue
        d = dets[(r["model"], show)]
        det = f"{d['F1']} ({d['F1_stead_noise']})" if d["F1"] else "—"
        rates = f"{float(nz['rose_window_fp_rate']):.1%} vs {float(nz['stead_trace_fp_rate']):.1%}"
        print(f"  {r['model']:<18}{r['P_F1']:>8} ({float(r['P_F1_stead_noise']):.3f})"
              f"{r['S_F1']:>8} ({float(r['S_F1_stead_noise']):.3f}){det:>17}{rates:>24}")

    if args.update_results:
        for path in outputs:
            if path.resolve() != (results_dir / path.name).resolve():
                shutil.copy2(path, results_dir / path.name)
                print(f"copied {path.name} -> {results_dir}/")


if __name__ == "__main__":
    main()
