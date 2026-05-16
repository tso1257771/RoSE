# AGENTS.md

Operating notes for LLM agents working in the **RoSE** repo — a Romanian
seismic dataset + three published phase pickers (EQT-RoSE, PhaseNet-RoSE,
RED-PAN-60s).

## Layout
- **`rose/`** — the importable package: `RoSE` loader, `convert`, `qc`, `splits.hash_split`, `checkpoint_io.safe_torch_load`, **`pickers.load_eqt_rose` / `load_phasenet_rose` / `load_redpan_tf60`** (also re-exported as `rose.load_*`), and the vendored `redpan_inference/` (TF inference subset)
- **`phase_picking/models/`** — the 3 published checkpoints + `SHA256SUMS` + model cards (`eqt_rose/`, `phasenet_rose/`, `redpan_tf60/train.hdf5`)
- **`phase_picking/results/`** — the pre-computed benchmark CSVs (`{rose,stead}_{picking,detection}.csv`, `rose_residual_stats.csv`) + `README.md`
- **`phase_picking/benchmark/`** — the benchmark *pipeline* (all 9 pickers), two stages + a config: `run_inference.py` (a — per-model inference → `eval/`), `build_leaderboard.py` (b — aggregate → `phase_picking/results/*.csv`), `config.json`. `regenerate_results.sh` runs both. The `bench_*.py` / `build_*.py` are the stage implementations those two scripts call; `phase_picking/benchmark/data/` has the pinned test-set index CSVs. See `phase_picking/benchmark/README.md`.
- **`phase_picking/training/`** — fine-tune SeisBench EQT / PhaseNet on RoSE (DDP, INSTANCE-init)
- **`examples/`** — four numbered tutorials (01 load → 02 INSTANCE demo → 03 ground motion → 04 all-three-pickers inference)
- **`docs/`** — `DATASET.md` (native HDF5) + `SEISBENCH_FORMAT.md` (column reference)

## Conventions
- **Channels**: ZNE in/out (RED-PAN wrapper rotates internally).
- **Normalisation**: applied **internally** by every model — peak for SeisBench, per-window Z-score for RED-PAN; **never pre-normalise**.
- **Preprocessing**: demean + linear-detrend + 1–45 Hz Butterworth band-pass (matches EQT training-time filter).
- **Checkpoint loading**: `from rose.checkpoint_io import safe_torch_load` — never bare `torch.load`.
- **Paths**: env vars `ROSE_DATA_DIR`, `STEAD_DIR`, `ROSE_TRAIN_OUT_DIR`, `ROSE_EVAL_DIR` (`.env.example`); no hardcoded `/home/<user>/…`; print via `_rel(p)`.
- **Model IDs**: `eqt_rose`, `phasenet_rose`, `redpan_tf60` — no `_v3` / `_v2` suffixes.

## Don't commit
`data/`, `eval/`, `checkpoints/`, `outputs/`, `rose_split_index.*` — all gitignored.

## Verify
`pytest tests/` (50 tests; pins the split partition, the SeisBench converter, `get_sample_physical`, the QC helpers, `safe_torch_load`, and the `redpan_inference` utilities). CI runs the same suite on Python 3.10/3.11/3.12 via `.github/workflows/tests.yml`.
