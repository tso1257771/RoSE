#!/usr/bin/env bash
# Reproduce the benchmark for the 3 published RoSE checkpoints on RoSE + STEAD.
#
# Usage:
#   bash scripts/reproduce_all.sh \
#       --rose-dir  /path/to/rose \
#       --stead-dir /path/to/STEAD/benchmark_stead \
#       --out-dir   results/runs \
#       [--num-test N]      # smoke test: cap each pool at N traces (0 = full)
#
# Per-model outputs land at <out-dir>/<dataset>/<model>/<model>.json, then are
# aggregated into <out-dir>/<dataset>_picking.csv and <out-dir>/<dataset>_detection.csv.
#
# This is an INDEPENDENT re-score with the release's own (self-contained,
# deliberately simple) pickerbench/ scoring — handy for confirming the
# checkpoints load and pick sanely and for ballpark per-phase precision/recall/
# F1. It is NOT a bit-for-bit reproduction of the committed results/*.csv:
# those come from the main repo's full benchmark suite (benchmark/build_*),
# which uses a richer schema, a separate dedicated-noise pass, and the
# "FP on dedicated-noise traces only" convention for the RoSE pool (RoSE event
# labels are incomplete). STEAD rows track the committed ones closely at full
# scale; RoSE precision/FP differ by construction. The committed CSVs also
# carry off-the-shelf `instance`/`ethz`/`stead` baselines and a
# *_residual_stats.csv, which aren't re-run here. Nothing under results/ is
# overwritten (output goes only under <out-dir>/, default results/runs/).
# Note: --num-test for the STEAD pools takes the first N traces, not a random
# sample, so a small subset can be unrepresentative.
set -euo pipefail

ROSE_DIR=""
STEAD_DIR=""
OUT_DIR="results/runs"
NUM_TEST=0          # 0 = full split/pool; >0 = smoke-test on N traces per pool

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rose-dir)  ROSE_DIR="$2"; shift 2 ;;
    --stead-dir) STEAD_DIR="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
    --num-test)  NUM_TEST="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$ROSE_DIR" || -z "$STEAD_DIR" ]]; then
  echo "Both --rose-dir and --stead-dir are required." >&2
  echo "See data/README.md for download instructions." >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p "$OUT_DIR/rose" "$OUT_DIR/stead"

for MODEL in eqt_rose phasenet_rose redpan_tf60; do
  echo "==> $MODEL on RoSE"
  python benchmarks/run_benchmark.py \
    --model "$MODEL" --dataset rose \
    --rose-dir "$ROSE_DIR" --num-test "$NUM_TEST" \
    --out-dir "$OUT_DIR/rose/$MODEL"
done

for MODEL in eqt_rose phasenet_rose redpan_tf60; do
  echo "==> $MODEL on STEAD"
  python benchmarks/run_benchmark.py \
    --model "$MODEL" --dataset stead \
    --stead-dir "$STEAD_DIR" --num-test "$NUM_TEST" \
    --out-dir "$OUT_DIR/stead/$MODEL"
done

echo "==> aggregating tables"
python benchmarks/build_tables.py rose  "$OUT_DIR/rose"  "$OUT_DIR/rose"
python benchmarks/build_tables.py stead "$OUT_DIR/stead" "$OUT_DIR/stead"

echo "Done. Output CSVs (compare against the committed results/*.csv):"
ls -la "$OUT_DIR"/rose_*.csv "$OUT_DIR"/stead_*.csv
