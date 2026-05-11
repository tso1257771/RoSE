#!/usr/bin/env python3
"""Stage (b): aggregate the per-model inference results into the leaderboards.

This is the scoring half of the benchmark. It reads the per-model tally files
that `benchmark/run_inference.py` wrote under `$eval_dir/` and produces the five
leaderboard CSVs (no model inference here):

    rose_picking.csv        — RoSE per-phase precision/recall/F1, residuals, per model × threshold
    rose_detection.csv      — RoSE trace-level event-vs-noise (F1, MCC, AUC, detection box)
    rose_residual_stats.csv — RoSE onset-residual stats (MAE, MAD, RMSE, IQR) at one threshold
    stead_picking.csv       — STEAD per-phase, with the standard events+noise FP convention
    stead_detection.csv     — STEAD trace-level event-vs-noise

Every CSV carries a row for every (model, threshold) over the configured
threshold sweep (the metrics are computed at each threshold from a single
inference pass) — that's the "threshold sweep". `--threshold` only picks which
threshold the human-readable leaderboard tables printed to stdout use.

    python benchmark/build_leaderboard.py                    # writes the CSVs under $eval_dir, prints tables
    python benchmark/build_leaderboard.py --update-results   # also copies them into results/
    python benchmark/build_leaderboard.py --threshold 0.5    # print the 0.5 leaderboard
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline import add_common_args, load_config, py, resolve, run  # noqa: E402

# eval/<name> -> results/<name>
_PUBLISH = {
    "bench_rose_picking_clean.csv":   "rose_picking.csv",
    "bench_rose_detection_clean.csv": "rose_detection.csv",
    "bench_rose_residual_stats.csv":  "rose_residual_stats.csv",
    "bench_stead_full_picking.csv":   "stead_picking.csv",
    "bench_stead_full_detection.csv": "stead_detection.csv",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--threshold", default=None,
                    help="Threshold for the printed leaderboard tables (default: config.display_threshold). "
                         "The output CSVs always contain the full sweep.")
    ap.add_argument("--results-dir", default=None, help="Override config.results_dir.")
    ap.add_argument("--update-results", action="store_true",
                    help="Copy the five leaderboard CSVs into the results/ directory.")
    # accepted-but-ignored so `regenerate_results.sh ... | run_inference.py | build_leaderboard.py`
    # can forward the same arg list to both stages
    ap.add_argument("--num-test", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--jobs", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--skip-rose", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--skip-stead", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    cfg = resolve(load_config(args.config), args)
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    thr = str(args.threshold) if args.threshold is not None else str(cfg["display_threshold"])
    eval_dir = Path(cfg["eval_dir"])
    results_dir = Path(cfg["results_dir"])

    print(f"=== build_leaderboard.py ===  eval_dir={eval_dir}  display_threshold={thr}\n")

    print(">>> RoSE leaderboards")
    run(py("build_rose_final_benchmark.py") + ["--eval-dir", str(eval_dir), "--threshold", thr])
    run(py("build_rose_residual_stats.py") + [
        "--eval-dir", str(eval_dir), "--threshold", thr,
        "--out-csv", str(eval_dir / "bench_rose_residual_stats.csv"),
    ])
    print(">>> STEAD leaderboards")
    run(py("build_stead_full_benchmark.py") + ["--eval-dir", str(eval_dir), "--threshold", thr])

    produced = [eval_dir / src for src in _PUBLISH]
    print(f"\nLeaderboard CSVs (under {eval_dir}/):")
    for p in produced:
        print(f"    {p}  {'(missing!)' if not p.is_file() else ''}")

    if args.update_results:
        results_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n>>> --update-results: copying into {results_dir}/")
        for src, dst in _PUBLISH.items():
            s = eval_dir / src
            if s.is_file():
                shutil.copy2(s, results_dir / dst)
                print(f"    {src}  ->  {results_dir / dst}")
            else:
                print(f"    {src}  MISSING — skipped")
        print(f"\n  Review with:  git diff {results_dir}")


if __name__ == "__main__":
    main()
