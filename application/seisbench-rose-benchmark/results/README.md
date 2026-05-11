# Pre-computed benchmark results

These CSVs are the canonical benchmark numbers reported in the RoSE picker
paper (TRANSFORM² Deliverable 3.4). They cover nine pickers: the three
checkpoints in `models/` — **EQT-RoSE** and **PhaseNet-RoSE** (fine-tuned on
the RoSE training split from INSTANCE init) and **RED-PAN-60s** (RED-PAN
retrained on Taiwan + STEAD + INSTANCE + RoSE, warm-started from the published
RED-PAN(60 s) weights — see `models/README.md`) — plus six off-the-shelf
SeisBench baselines (EQT / PhaseNet × `instance` / `ethz` / `stead`).
`scripts/reproduce_all.sh` re-scores just the three bundled checkpoints (it
needs the RoSE + STEAD test datasets) with this release's own self-contained
`pickerbench/` code, writing `results/runs/*.csv` — close to, but **not** a
bit-for-bit copy of, the rows here: the committed CSVs come from the main
repo's full benchmark suite (richer schema; "FP on dedicated-noise only" for
the RoSE pool; a separate residual-stats CSV). STEAD rows track closely; RoSE
precision/FP differ by construction. Treat `reproduce_all.sh` as a sanity
check of the checkpoints, not as the source of the published numbers.

## RoSE pool (32 374 events + 23 526 STEAD noise)

* `rose_picking.csv`        — phase-pick TP/FP/FN, precision, recall, F1, MAE,
  median, std, MAD, RMSE, IQR per phase × threshold per model.
  *FP convention*: counted on STEAD-noise traces only (RoSE labels are
  incomplete; unlabelled aftershocks would otherwise appear as FP).
* `rose_detection.csv`      — trace-level event-vs-noise detection T1
  (precision/recall/F1, Matthews correlation, AUC) plus detection-box IoU
  + start-time / end-time MAE for models with detection heads.
* `rose_residual_stats.csv` — residual statistics at threshold 0.30.

## STEAD pool (103 040 events + 23 526 noise)

* `stead_picking.csv`     — same columns as `rose_picking.csv`, but FP follows
  the canonical pick-benchmark convention (events + noise pooled) because
  STEAD labels are clean.
* `stead_detection.csv`   — trace-level T1 + detection-box quality on STEAD.

## Threshold sweep

Each CSV has rows for every (model, threshold) combination over the
six-point sweep `{0.05, 0.10, 0.20, 0.30, 0.50, 0.70}`.

## Headline numbers at threshold 0.30

| Pool  | Best phase F1 (P / S)             | Best detection F1                  |
|---|---|---|
| RoSE  | RED-PAN-60s **0.822 / 0.827**     | EQT-RoSE **0.977** (MCC 0.945)  |
| STEAD | RED-PAN-60s **0.972 / 0.980**     | EQT-stead **0.998** (MCC 0.988) — see RoSE-fine-tuned counterpart EQT-RoSE at 0.991 |
