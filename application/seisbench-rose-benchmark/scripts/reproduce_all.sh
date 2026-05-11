#!/usr/bin/env bash
# Reproduce the benchmark for the 3 published RoSE checkpoints on RoSE + STEAD.
#
# Usage:
#   bash scripts/reproduce_all.sh \
#       --rose-dir  /path/to/rose \
#       --stead-dir /path/to/STEAD/benchmark_stead \
#       --out-dir   results/runs
#
# Per-model outputs land at <out-dir>/<dataset>/<model>/<model>.json, then are
# aggregated into <out-dir>/<dataset>_picking.csv and <out-dir>/<dataset>_detection.csv.
# Compare those against the committed results/<dataset>_picking.csv / _detection.csv
# (the EQT-RoSE / PhaseNet-RoSE / RED-PAN-60s rows should match to floating-point
# precision; the committed files also carry off-the-shelf `instance` / `ethz` /
# `stead` baselines and a *_residual_stats.csv from the main repo's full benchmark,
# which aren't re-run here). Nothing under results/ is overwritten.
set -euo pipefail

ROSE_DIR=""
STEAD_DIR=""
OUT_DIR="results/runs"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rose-dir)  ROSE_DIR="$2"; shift 2 ;;
    --stead-dir) STEAD_DIR="$2"; shift 2 ;;
    --out-dir)   OUT_DIR="$2"; shift 2 ;;
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
    --rose-dir "$ROSE_DIR" \
    --out-dir "$OUT_DIR/rose/$MODEL"
done

for MODEL in eqt_rose phasenet_rose redpan_tf60; do
  echo "==> $MODEL on STEAD"
  python benchmarks/run_benchmark.py \
    --model "$MODEL" --dataset stead \
    --stead-dir "$STEAD_DIR" \
    --out-dir "$OUT_DIR/stead/$MODEL"
done

echo "==> aggregating tables"
python benchmarks/build_tables.py rose  "$OUT_DIR/rose"  "$OUT_DIR/rose"
python benchmarks/build_tables.py stead "$OUT_DIR/stead" "$OUT_DIR/stead"

echo "Done. Output CSVs (compare against the committed results/*.csv):"
ls -la "$OUT_DIR"/rose_*.csv "$OUT_DIR"/stead_*.csv
