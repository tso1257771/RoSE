# `phase_picking/benchmark/` — how the published numbers are produced

Two stages:

- **(a) inference** — run every model on every test trace, match its picks to
  the catalogue picks, tally per-model results → `eval/`.
- **(b) leaderboard** — aggregate those tallies into the `phase_picking/results/*.csv` tables.

Two scripts + one config:

```bash
# everything lives in benchmark/config.json — override key fields on the CLI
python phase_picking/benchmark/run_inference.py                       # stage (a)  -> eval/
python phase_picking/benchmark/build_leaderboard.py --update-results  # stage (b)  -> phase_picking/results/*.csv

# or both at once:
bash phase_picking/benchmark/regenerate_results.sh                    # = run_inference.py + build_leaderboard.py --update-results
bash phase_picking/benchmark/regenerate_results.sh --num-test 200     # quick subset
```

`--num-test 0` (the config default) = full test pools (~hours on CPU);
`--num-test N` = a quick N-traces-per-pool subset (for the STEAD pools it takes
the first N, not a random sample). Run with a Python env that has `seisbench` +
`torch` + `tensorflow` (the last for RED-PAN-60s).

> **You must supply the two test datasets yourself — they are not in this repo.**
> - **RoSE** (SeisBench-format, ~13 GB for the test split): download the RoSE
>   bundle from Zenodo → `export ROSE_DATA_DIR=…` (or set `rose_dir` in the config).
> - **STEAD** test set (`metadata.csv` + `metadata_noise.csv` + `waveforms/*.npy`
>   + `noise_waveforms/*.npy` — layout recipe in [`data/README.md`](data/README.md))
>   → `export STEAD_DIR=…` (or `stead_dir` in the config). Required **even for
>   the RoSE leg**: RoSE event labels are incomplete, so RoSE false positives are
>   counted on dedicated STEAD *noise* traces (with unmatched-on-event picks kept
>   as a separate diagnostic column).
>
> The three RoSE-trained checkpoints come from `../models/`; the six off-the-shelf
> EQT / PhaseNet baselines are fetched by SeisBench's `from_pretrained` and cached
> under `~/.seisbench`.

## Config — `benchmark/config.json`

| field | meaning | default |
|---|---|---|
| `rose_dir` / `stead_dir` | dataset paths; `null` → `$ROSE_DATA_DIR` / `$STEAD_DIR` | `null` |
| `eval_dir` | where stage (a) writes per-model results | `eval` (gitignored) |
| `results_dir` | where `build_leaderboard.py --update-results` writes the CSVs | `results` |
| `models_dir` | the bundled checkpoints | `models` |
| `thresholds` | detection/pick threshold sweep (metrics computed at each) | `[0.05, 0.1, 0.2, 0.3, 0.5, 0.7]` |
| `display_threshold` | which threshold the printed leaderboard tables use | `0.3` |
| `num_test` | `0` = full pools; `N` = subset | `0` |
| `bandpass` | pre-inference Butterworth band-pass (Hz) | `[1.0, 45.0]` |
| `pytorch_model_ids` | the 8 PyTorch model ids to run | EQT/PhaseNet × {RoSE, instance, ethz, stead} |

Both stage scripts also take `--config FILE` and overrides: `--rose-dir`,
`--stead-dir`, `--eval-dir`, `--num-test` (run_inference), `--threshold`,
`--results-dir`, `--update-results` (build_leaderboard), `--skip-rose`,
`--skip-stead` — see `--help`.

## What's in here

| file | role |
|---|---|
| **`run_inference.py`** | stage (a) — the one script you run for inference |
| **`build_leaderboard.py`** | stage (b) — the one script you run for the leaderboard |
| **`regenerate_results.sh`** | convenience wrapper: runs (a) then (b) `--update-results` |
| `config.json` | the config both stages read |
| `bench_pickers_rose.py` / `bench_redpan_rose.py` / `bench_noise_fp.py` / `bench_stead_test.py` | the inference *stage implementations* `run_inference.py` calls per (model, dataset) — not run directly |
| `build_rose_final_benchmark.py` / `build_rose_residual_stats.py` / `build_stead_full_benchmark.py` | the aggregation *stage implementations* `build_leaderboard.py` calls (each also prints its threshold-0.30 leaderboard table to stdout) |
| `_pipeline.py` | shared config-loading / subprocess helpers for the two stage scripts |
| `build_test_indices.py` | regenerate the pinned `data/*_index.csv` |
| `data/` | the pinned test-set index CSVs (which traces make up each test set) |
| `eval_eqt_rose.py` | standalone: an EQT-only eval in the Münchmeyer-2022 / Section-3 "3 tasks" format (T1/T2/T3) — *not* part of the pipeline; the pipeline already produces equivalent T1/T2/T3 numbers for all 9 models |
| `bench_joint_rose.py` | standalone: picking metrics under detector-*gated* operation vs the un-gated baseline |
| `viz_models_rose.py` | standalone: record-section comparison plots |

The five leaderboard CSVs and their human-readable tables are documented in
[`../results/README.md`](../results/README.md). For just *using* a published
picker on your own data you need none of this — `from rose import load_eqt_rose,
load_phasenet_rose, load_redpan_tf60` (see the top-level README and
[`../models/README.md`](../models/README.md)).
