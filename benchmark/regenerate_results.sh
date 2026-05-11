#!/usr/bin/env bash
# Regenerate the published benchmark tables — results/*.csv —
# from the RoSE + STEAD test datasets and the three bundled checkpoints.
#
# This is THE pipeline behind the committed results/*.csv.
# Five stages:
#
#   1. bench_pickers_rose.py   8 PyTorch pickers on the RoSE test split
#                              -> $EVAL/bench_rose_full_sweep/sweep_comparison.csv
#   2. bench_redpan_rose.py    RED-PAN-60s on the RoSE test split
#                              -> $EVAL/bench_redpan_rose_full/sweep_comparison.csv
#   3. bench_noise_fp.py       picks on the STEAD noise pool, all 9 models
#                              (RoSE FP convention = FP on dedicated noise only)
#                              -> $EVAL/bench_noise_fp/<model_id>.json
#   4. build_rose_final_benchmark.py + build_rose_residual_stats.py
#                              -> $EVAL/bench_rose_{picking_clean,detection_clean,residual_stats}.csv
#   5. bench_stead_test.py x9  8 PyTorch pickers + RED-PAN-60s on the STEAD test set
#                              -> $EVAL/bench_stead_full_pytorch/<m>/<m>.json
#                                 $EVAL/bench_stead_full/redpan_merged.json
#      build_stead_full_benchmark.py
#                              -> $EVAL/bench_stead_full_{picking,detection,residuals}.csv
#
#   Then (with --update-release) copy the five published CSVs into
#   results/.
#
# The 6 off-the-shelf baselines (EQT / PhaseNet x instance / ethz / stead) are
# pulled via seisbench's `from_pretrained` and cached under ~/.seisbench.
#
# Usage:
#   bash benchmark/regenerate_results.sh \
#       [--rose-dir DIR] [--stead-dir DIR] [--eval-dir DIR] \
#       [--num-test N]         # 0 (default) = full pools (~hours on CPU);
#                              #   >0 = quick smoke run on N traces per pool
#       [--update-release]     # also overwrite results/*.csv
#       [--skip-rose] [--skip-stead]
#
# Defaults: --rose-dir $ROSE_DATA_DIR, --stead-dir $STEAD_DIR,
#           --eval-dir ${ROSE_EVAL_DIR:-eval}.
# Runs with bare `python` — activate an env that can load
# seisbench.models.EQTransformer / PhaseNet AND tensorflow (for RED-PAN-60s).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

ROSE_DIR="${ROSE_DATA_DIR:-}"
STEAD_DIR_="${STEAD_DIR:-}"
EVAL_DIR="${ROSE_EVAL_DIR:-eval}"
NUM_TEST=0
UPDATE_RELEASE=0
DO_ROSE=1
DO_STEAD=1
THRESHOLDS="0.05,0.1,0.2,0.3,0.5,0.7"
RESIDUAL_THR="0.3"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rose-dir)       ROSE_DIR="$2"; shift 2 ;;
    --stead-dir)      STEAD_DIR_="$2"; shift 2 ;;
    --eval-dir)       EVAL_DIR="$2"; shift 2 ;;
    --num-test)       NUM_TEST="$2"; shift 2 ;;
    --update-release) UPDATE_RELEASE=1; shift ;;
    --skip-rose)      DO_ROSE=0; shift ;;
    --skip-stead)     DO_STEAD=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

MODELS_DIR="$REPO_ROOT/models"
EQT_CKPT="$MODELS_DIR/eqt_rose/eqt_rose.pt"
PN_CKPT="$MODELS_DIR/phasenet_rose/phasenet_rose.pt"
RP_HDF5="$MODELS_DIR/redpan_tf60/train.hdf5"
RELEASE_RESULTS="$REPO_ROOT/results"
PT_MODELS="eqt_rose phasenet_rose eqt_instance phasenet_instance eqt_ethz phasenet_ethz eqt_stead phasenet_stead"

[[ -n "$ROSE_DIR"   ]] || { echo "ERROR: --rose-dir or \$ROSE_DATA_DIR is required" >&2; exit 2; }
# bench_noise_fp.py (the RoSE FP convention) needs STEAD too, so STEAD is
# required whenever the RoSE leg runs.
if [[ "$DO_STEAD" == 1 || "$DO_ROSE" == 1 ]]; then
  [[ -n "$STEAD_DIR_" ]] || { echo "ERROR: --stead-dir or \$STEAD_DIR is required" >&2; exit 2; }
fi
for f in "$EQT_CKPT" "$PN_CKPT" "$RP_HDF5"; do
  [[ -f "$f" ]] || { echo "ERROR: missing checkpoint $f" >&2; exit 2; }
done

echo "=== regenerate_results.sh ==="
echo "  rose-dir   = $ROSE_DIR"
echo "  stead-dir  = $STEAD_DIR_"
echo "  eval-dir   = $EVAL_DIR"
echo "  num-test   = $NUM_TEST   (0 = full pools)"
echo "  thresholds = $THRESHOLDS"
echo "  python     = $(command -v python)"
echo
mkdir -p "$EVAL_DIR"

if [[ "$DO_ROSE" == 1 ]]; then
  echo "===> [1/5] bench_pickers_rose.py — 8 PyTorch pickers on the RoSE test split"
  python benchmark/bench_pickers_rose.py \
    --rose-dir "$ROSE_DIR" --out-dir "$EVAL_DIR/bench_rose_full_sweep" \
    --num-test "$NUM_TEST" --sweep-thresholds "$THRESHOLDS" \
    --custom-eqt "$EQT_CKPT" --custom-phasenet "$PN_CKPT"

  echo "===> [2/5] bench_redpan_rose.py — RED-PAN-60s on the RoSE test split"
  python benchmark/bench_redpan_rose.py \
    --model-path "$RP_HDF5" --rose-dir "$ROSE_DIR" \
    --out-dir "$EVAL_DIR/bench_redpan_rose_full" \
    --num-test "$NUM_TEST" --sweep-thresholds "$THRESHOLDS"

  echo "===> [3/5] bench_noise_fp.py — STEAD-noise picks, all 9 models"
  python benchmark/bench_noise_fp.py \
    --stead-dir "$STEAD_DIR_" --out-dir "$EVAL_DIR/bench_noise_fp" \
    --num-noise "$NUM_TEST" --sweep-thresholds "$THRESHOLDS" \
    --eqt-rose-ckpt "$EQT_CKPT" --phasenet-rose-ckpt "$PN_CKPT" --redpan-tf "$RP_HDF5"

  echo "===> [4/5] aggregate the RoSE leaderboards"
  python benchmark/build_rose_final_benchmark.py --eval-dir "$EVAL_DIR"
  python benchmark/build_rose_residual_stats.py --eval-dir "$EVAL_DIR" \
    --threshold "$RESIDUAL_THR" --out-csv "$EVAL_DIR/bench_rose_residual_stats.csv"
fi

if [[ "$DO_STEAD" == 1 ]]; then
  echo "===> [5/5] bench_stead_test.py — 8 PyTorch pickers + RED-PAN-60s on the STEAD test set"
  for M in $PT_MODELS; do
    echo "      ... STEAD: $M"
    python benchmark/bench_stead_test.py \
      --stead-dir "$STEAD_DIR_" --out-dir "$EVAL_DIR/bench_stead_full_pytorch/$M" \
      --num-events "$NUM_TEST" --num-noise "$NUM_TEST" \
      --sweep-thresholds "$THRESHOLDS" --models "$M" \
      --eqt-rose-ckpt "$EQT_CKPT" --phasenet-rose-ckpt "$PN_CKPT"
  done
  echo "      ... STEAD: redpan"
  python benchmark/bench_stead_test.py \
    --stead-dir "$STEAD_DIR_" --out-dir "$EVAL_DIR/bench_stead_full" \
    --num-events "$NUM_TEST" --num-noise "$NUM_TEST" \
    --sweep-thresholds "$THRESHOLDS" --models redpan --redpan-tf "$RP_HDF5"
  # build_stead_full_benchmark.py looks for redpan_merged.json; an unsharded run
  # writes redpan.json (the 'merged' name is just how the original 4-shard run
  # was assembled — a single full run is equivalent).
  cp "$EVAL_DIR/bench_stead_full/redpan.json" "$EVAL_DIR/bench_stead_full/redpan_merged.json"

  echo "===> aggregate the STEAD leaderboards"
  python benchmark/build_stead_full_benchmark.py --eval-dir "$EVAL_DIR" --threshold "$RESIDUAL_THR"
fi

echo
echo "=== generated under $EVAL_DIR ==="
ls -la "$EVAL_DIR"/bench_rose_picking_clean.csv "$EVAL_DIR"/bench_rose_detection_clean.csv \
       "$EVAL_DIR"/bench_rose_residual_stats.csv "$EVAL_DIR"/bench_stead_full_picking.csv \
       "$EVAL_DIR"/bench_stead_full_detection.csv 2>/dev/null || true

if [[ "$UPDATE_RELEASE" == 1 ]]; then
  echo
  echo "=== --update-release: copying into results/ ==="
  cp "$EVAL_DIR/bench_rose_picking_clean.csv"   "$RELEASE_RESULTS/rose_picking.csv"
  cp "$EVAL_DIR/bench_rose_detection_clean.csv" "$RELEASE_RESULTS/rose_detection.csv"
  cp "$EVAL_DIR/bench_rose_residual_stats.csv"  "$RELEASE_RESULTS/rose_residual_stats.csv"
  cp "$EVAL_DIR/bench_stead_full_picking.csv"   "$RELEASE_RESULTS/stead_picking.csv"
  cp "$EVAL_DIR/bench_stead_full_detection.csv" "$RELEASE_RESULTS/stead_detection.csv"
  echo "  done — review with: git diff results/"
fi

echo
echo "Done."
