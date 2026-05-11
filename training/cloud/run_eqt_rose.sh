#!/usr/bin/env bash
# SeisBench EQTransformer fine-tuned on RoSE/ROMPLUS (TWCC, 4xV100 32GB).
# Train + dev only; test held out for final benchmarking.
#
# Recipe (matches the published EQT-RoSE checkpoint):
#   * INIT_WEIGHTS=instance         transfer-learning warm start
#   * Normalize amp_norm_type=peak  matches pick-benchmark + pretrained
#   * SIGMA=10                      tighter Gaussian onset label
#   * LR=1e-4 / EPOCHS=30           fine-tune schedule
#   * BATCH=64 / NUM_WORKERS=2 / CACHE=none  RAM-safe DDP defaults
#
# Detection-label width factor:
#   * DETECTION_FACTOR=0.7
#     RoSE local events have PS-diff ~25-30 s; factor=1.4 makes the labelled
#     event box ~60-72 s long — well past the actual coda. factor=0.7 yields
#     a labelled box from P to S + ~18 s, matching local-coda physics.
#       - det_end_mae_s:      31  → < 5 s
#       - det_iou_mean:       0.58 → > 0.80 (close to EQT-stead 0.83)
#       - phase pick metrics: similar or slightly better
#         (tighter detection focus often helps S-onset accuracy too)
#
# Per-GPU batch 64 → effective batch 256 across 4 GPUs.

set -euo pipefail
cd "$(dirname "$0")/.."

INIT_WEIGHTS="${INIT_WEIGHTS:-instance}"

# Pre-fetch pretrained weights single-process so DDP ranks find them cached.
if [[ "$INIT_WEIGHTS" != "scratch" ]]; then
  echo "[pre-fetch] EQTransformer pretrained weights: ${INIT_WEIGHTS}"
  python -c "import seisbench.models as sbm; sbm.EQTransformer.from_pretrained('${INIT_WEIGHTS}')" \
    || { echo "Pre-fetch failed for EQT/${INIT_WEIGHTS}"; exit 1; }
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  training/train_eqt_rose.py \
  --rose-dir "${ROSE_DIR:-/home/u2140647/RoSE/data/rose}" \
  --out-dir  "${OUT_DIR:-${ROSE_TRAIN_OUT_DIR:-checkpoints}/eqt_rose}" \
  --epochs            "${EPOCHS:-30}" \
  --batch-size        "${BATCH:-64}" \
  --lr                "${LR:-1e-4}" \
  --num-workers       "${NUM_WORKERS:-2}" \
  --sigma             "${SIGMA:-10}" \
  --bandpass-low      "${BANDPASS_LOW:-1.0}" \
  --bandpass-high     "${BANDPASS_HIGH:-45.0}" \
  --cache             "${CACHE:-none}" \
  --init-weights      "${INIT_WEIGHTS}" \
  --norm              "${NORM:-peak}" \
  --detection-factor  "${DETECTION_FACTOR:-0.7}"
