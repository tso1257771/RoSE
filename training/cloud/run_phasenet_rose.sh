#!/usr/bin/env bash
# SeisBench PhaseNet fine-tuned on RoSE/ROMPLUS (TWCC, 4×V100 32GB).
# Train + dev only; test held out for final benchmarking.
#
# Recipe matches pick-benchmark exactly:
#   * INIT_WEIGHTS=instance  - warm-start from sbm.PhaseNet.from_pretrained
#                              ("instance"); a stronger baseline than scratch.
#   * Normalize amp_norm_type="peak" inside the augmentation pipeline.
#   * SIGMA=10               - tighter Gaussian label → better onset MAE.
#   * EPOCHS=30              - fine-tuning converges fast.
#   * LR=1e-4                - standard fine-tuning rate (was 1e-3).
#
# Per-GPU batch 256 → effective batch 1024 across 4 GPUs. Override anything
# via env: NPROC_PER_NODE / NUM_WORKERS / EPOCHS / BATCH / LR / CACHE /
# SIGMA / INIT_WEIGHTS.

set -euo pipefail
cd "$(dirname "$0")/.."

INIT_WEIGHTS="${INIT_WEIGHTS:-instance}"

# Pre-fetch pretrained weights in a single non-DDP process so all four
# torchrun ranks find them cached. Without this, ranks race on the
# download and Nth process aborts with "Found partial instance".
if [[ "$INIT_WEIGHTS" != "scratch" ]]; then
  echo "[pre-fetch] PhaseNet pretrained weights: ${INIT_WEIGHTS}"
  python -c "import seisbench.models as sbm; sbm.PhaseNet.from_pretrained('${INIT_WEIGHTS}')" \
    || { echo "Pre-fetch failed for PhaseNet/${INIT_WEIGHTS}"; exit 1; }
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-4}" \
  training/train_phasenet_rose.py \
  --rose-dir "${ROSE_DIR:-/home/u2140647/RoSE/data/rose}" \
  --out-dir  "${OUT_DIR:-${ROSE_TRAIN_OUT_DIR:-checkpoints}/phasenet_rose}" \
  --epochs        "${EPOCHS:-30}" \
  --batch-size    "${BATCH:-256}" \
  --lr            "${LR:-1e-4}" \
  --num-workers   "${NUM_WORKERS:-2}" \
  --sigma         "${SIGMA:-10}" \
  --cache         "${CACHE:-none}" \
  --init-weights  "${INIT_WEIGHTS}" \
  --norm          "${NORM:-peak}"
