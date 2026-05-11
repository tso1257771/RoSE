# `benchmark/` — how the published numbers are produced

This directory is the benchmark **pipeline**: the scripts that produce the
committed `results/*.csv` (the numbers reported in the paper). One command runs
the whole thing:

```bash
# defaults: --rose-dir $ROSE_DATA_DIR  --stead-dir $STEAD_DIR  --eval-dir ${ROSE_EVAL_DIR:-eval}
bash benchmark/regenerate_results.sh --update-release            # full run (hours on CPU)
bash benchmark/regenerate_results.sh --num-test 200              # quick smoke run (~10 min), no --update-release
```

`--num-test 0` (the default) uses the full pools. `--update-release` copies the
regenerated CSVs over `results/*.csv` (omit it to leave the committed reference
files untouched and just compare under `$ROSE_EVAL_DIR/`). Run with `python`
pointing at an env that can load `seisbench.models.EQTransformer` / `PhaseNet`
**and** `tensorflow` (for RED-PAN-60s).

> **You must supply the two test datasets yourself — they are not in this repo.**
> - **RoSE** (SeisBench format, ~13 GB for the test split): download the RoSE
>   waveform bundle from Zenodo and mount/symlink it, then
>   `export ROSE_DATA_DIR=/path/to/rose`.
> - **STEAD** test set (`metadata.csv`, `metadata_noise.csv`, `waveforms/*.npy`,
>   `noise_waveforms/*.npy`): obtain STEAD and prepare this layout (see
>   [`benchmark/data/README.md`](data/README.md)), then
>   `export STEAD_DIR=/path/to/STEAD/benchmark_stead`.
>
> STEAD is needed **even for the RoSE leg**: RoSE event labels are incomplete,
> so a predicted pick on a RoSE event trace that doesn't match a catalogue
> phase may be a real (unlabelled) aftershock, not a false positive. The
> benchmark therefore counts **false positives on dedicated STEAD *noise*
> traces only** for the RoSE pool (`bench_noise_fp.py`), keeping unmatched-on-
> event picks as a separate diagnostic column. STEAD's own labels are clean, so
> for the STEAD pool the canonical events+noise FP convention is used.

The 6 off-the-shelf baselines (EQT / PhaseNet × `instance` / `ethz` / `stead`)
are fetched by SeisBench's `from_pretrained` and cached under `~/.seisbench`;
they need network access on first run. The 3 RoSE-trained checkpoints come from
`../models/`.

## Pipeline (what `regenerate_results.sh` chains)

| # | Script | In | Out (under `$EVAL`) |
|---|---|---|---|
| 1 | `bench_pickers_rose.py --custom-eqt … --custom-phasenet …` | RoSE test split | `bench_rose_full_sweep/sweep_comparison.csv` — 8 PyTorch pickers (EQT/PhaseNet × {RoSE, instance, ethz, stead}), per-phase metrics over the 6-threshold sweep |
| 2 | `bench_redpan_rose.py --model-path ../models/redpan_tf60/train.hdf5` | RoSE test split | `bench_redpan_rose_full/sweep_comparison.csv` — RED-PAN-60s, same shape |
| 3 | `bench_noise_fp.py` | STEAD noise pool | `bench_noise_fp/<model_id>.json` × 9 — picks emitted on noise traces per model/threshold (the FP source for the RoSE pool) |
| 4 | `build_rose_final_benchmark.py`; `build_rose_residual_stats.py` | (1)+(2)+(3) | `bench_rose_picking_clean.csv`, `bench_rose_detection_clean.csv`, `bench_rose_residual_stats.csv` |
| 5 | `bench_stead_test.py` × 8 PyTorch + × RED-PAN-60s | STEAD events + noise | `bench_stead_full_pytorch/<m>/<m>.json`, `bench_stead_full/redpan_merged.json` |
| 6 | `build_stead_full_benchmark.py` | (5) | `bench_stead_full_picking.csv`, `bench_stead_full_detection.csv`, `bench_stead_full_residuals.csv` |

`--update-release` then copies (no transformation — same column schema):

```
bench_rose_picking_clean.csv    ->  results/rose_picking.csv
bench_rose_detection_clean.csv  ->  results/rose_detection.csv
bench_rose_residual_stats.csv   ->  results/rose_residual_stats.csv
bench_stead_full_picking.csv    ->  results/stead_picking.csv
bench_stead_full_detection.csv  ->  results/stead_detection.csv
```

(`bench_stead_full_residuals.csv` is generated but not shipped.)

## Running pieces individually

Every script takes `--rose-dir` / `--stead-dir` (or `$ROSE_DATA_DIR` /
`$STEAD_DIR`) and `--num-test N` / `--num-events N --num-noise N` (`0` = full).
Defaults for the bundled checkpoints point at `../models/`. Inference runs once
per trace at the lowest threshold; the threshold sweep is post-hoc, so
re-running at more thresholds needs no re-inference. `bench_stead_test.py` and
`bench_noise_fp.py` support `--shard k --total-shards N` for trivial parallelism
(merge the per-shard JSONs afterward; for RED-PAN on STEAD,
`regenerate_results.sh` just runs a single shard and names the output
`redpan_merged.json`).

Standalone tools here (not run by `regenerate_results.sh`):

* `bench_joint_rose.py` — picking metrics under **detector-gated** operation
  (a pick counts only if it lands inside a triggered detection window) vs the
  un-gated picker-only baseline, for the two models with detection heads
  (EQT-RoSE, RED-PAN-60s); writes `joint_comparison.csv` + `summary.json`.
* `eval_eqt_rose.py` — an **EQT-only** evaluation in the Münchmeyer-2022 /
  TRANSFORM²-Section-3 format (T1 detection F1/AUC, T2 phase-ID MCC, T3 onset
  MAE/RMSE), using SeisBench's fixed-window training generator and a synthesised
  *coda-window* negative for T1 (RoSE has no noise traces). It's a different
  protocol from the main pipeline — which already produces equivalent T1/T2/T3
  numbers (STEAD-noise T1, per-phase MCC, residual stats) for **all 9** models
  via `bench_pickers_rose.py` + `build_*.py` — so `eval_eqt_rose.py` is kept
  only for that specific output format; the `results/*.csv` numbers come from
  the pipeline, not from here.
* `viz_models_rose.py` — record-section comparison plots (catalogue vs model
  picks across stations).
* `build_test_indices.py` — regenerate the pinned `benchmark/data/*_index.csv`.

For just *using* a published picker on your own data, you don't need any of
this — `from rose import load_eqt_rose, load_phasenet_rose, load_redpan_tf60`
(see the top-level README and `models/README.md`).
