# `benchmark/` — how the published numbers are produced

This directory holds the **canonical** benchmark suite: the scripts that
produce `application/seisbench-rose-benchmark/results/*.csv` (the numbers
reported in the paper). One command runs the whole pipeline:

```bash
# defaults: --rose-dir $ROSE_DATA_DIR  --stead-dir $STEAD_DIR  --eval-dir ${ROSE_EVAL_DIR:-eval}
bash benchmark/regenerate_results.sh --update-release            # full run (hours on CPU)
bash benchmark/regenerate_results.sh --num-test 200              # quick smoke run (~10 min), no --update-release
```

`--num-test 0` (the default) uses the full pools. `--update-release` copies
the regenerated CSVs over `application/seisbench-rose-benchmark/results/*.csv`
(omit it to leave the committed reference files untouched and compare under
`$ROSE_EVAL_DIR/`). Run with `python` pointing at an env that can load
`seisbench.models.EQTransformer` / `PhaseNet` **and** `tensorflow` (for
RED-PAN-60s).

> **You must supply the two test datasets yourself — they are not in this repo.**
> - **RoSE** (SeisBench format, ~13 GB for the test split): download the RoSE
>   waveform bundle from Zenodo and mount/symlink it, then `export ROSE_DATA_DIR=/path/to/rose`.
> - **STEAD** test set (`metadata.csv`, `metadata_noise.csv`, `waveforms/*.npy`,
>   `noise_waveforms/*.npy`): obtain STEAD and prepare this layout (see
>   `application/seisbench-rose-benchmark/data/README.md`), then `export STEAD_DIR=/path/to/STEAD/benchmark_stead`.
>
> STEAD is needed **even for the RoSE leg**: RoSE event labels are incomplete,
> so a predicted pick on a RoSE event trace that doesn't match a catalogue
> phase may be a real (unlabelled) aftershock, not a false positive. The
> benchmark therefore counts **false positives on dedicated STEAD *noise*
> traces only** for the RoSE pool (`bench_noise_fp.py`), and keeps
> unmatched-on-event picks as a separate diagnostic column. STEAD's own labels
> are clean, so for the STEAD pool the canonical events+noise FP convention is
> used.

The 6 off-the-shelf baselines (EQT / PhaseNet × `instance` / `ethz` / `stead`)
are fetched by SeisBench's `from_pretrained` and cached under `~/.seisbench`;
they need network access on first run.

## Pipeline (what `regenerate_results.sh` chains)

| # | Script | In | Out (under `$EVAL`) |
|---|---|---|---|
| 1 | `bench_pickers_rose.py --custom-eqt … --custom-phasenet …` | RoSE test split | `bench_rose_full_sweep/sweep_comparison.csv` — 8 PyTorch pickers (EQT/PhaseNet × {RoSE, instance, ethz, stead}), per-phase metrics over the 6-threshold sweep |
| 2 | `bench_redpan_rose.py --model-path …/train.hdf5` | RoSE test split | `bench_redpan_rose_full/sweep_comparison.csv` — RED-PAN-60s, same shape |
| 3 | `bench_noise_fp.py` | STEAD noise pool | `bench_noise_fp/<model_id>.json` × 9 — picks emitted on noise traces per model/threshold (the FP source for the RoSE pool) |
| 4 | `build_rose_final_benchmark.py`; `build_rose_residual_stats.py` | (1)+(2)+(3) | `bench_rose_picking_clean.csv`, `bench_rose_detection_clean.csv`, `bench_rose_residual_stats.csv` |
| 5 | `bench_stead_test.py` × 8 PyTorch + × RED-PAN-60s | STEAD events + noise | `bench_stead_full_pytorch/<m>/<m>.json`, `bench_stead_full/redpan_merged.json` |
| 6 | `build_stead_full_benchmark.py` | (5) | `bench_stead_full_picking.csv`, `bench_stead_full_detection.csv`, `bench_stead_full_residuals.csv` |

`--update-release` then copies (no transformation — same column schema):

```
bench_rose_picking_clean.csv    -> application/seisbench-rose-benchmark/results/rose_picking.csv
bench_rose_detection_clean.csv  -> …/results/rose_detection.csv
bench_rose_residual_stats.csv   -> …/results/rose_residual_stats.csv
bench_stead_full_picking.csv    -> …/results/stead_picking.csv
bench_stead_full_detection.csv  -> …/results/stead_detection.csv
```

(`bench_stead_full_residuals.csv` is generated but not shipped.)

## Running pieces individually

Every script takes `--rose-dir` / `--stead-dir` (or `$ROSE_DATA_DIR` /
`$STEAD_DIR`) and `--num-test N` / `--num-events N --num-noise N` (`0` = full).
Defaults for the bundled checkpoints point at
`application/seisbench-rose-benchmark/models/`. Inference runs once per trace at
the lowest threshold; the threshold sweep is post-hoc, so re-running at more
thresholds needs no re-inference. `bench_stead_test.py` and `bench_noise_fp.py`
support `--shard k --total-shards N` for trivial parallelism (merge the per-shard
JSONs afterward; for RED-PAN on STEAD `regenerate_results.sh` just runs a single
shard and names the output `redpan_merged.json`).

Other tools here: `bench_joint_rose.py` (joint detector+picker on RoSE),
`eval_eqt_rose.py` (the Münchmeyer-2022 / TRANSFORM² Section-3 tasks),
`viz_models_rose.py` (record-section plots), `build_test_indices.py` (regenerate
the pinned `application/.../data/*_index.csv`).

## vs. `application/seisbench-rose-benchmark/scripts/reproduce_all.sh`

That one is a **separate, self-contained, intentionally simpler re-score** that
lives inside the release bundle and does NOT import this repo's `benchmark/`
code — it's a quick "do the checkpoints load and pick sanely" check. Its CSVs
use a different schema and a simpler RoSE FP convention, so they are **not** a
bit-for-bit match for the committed `results/*.csv`. `regenerate_results.sh`
(this directory) is the pipeline that actually reproduces them.
