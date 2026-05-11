# AGENTS.md

Operating notes for LLM agents working in the **RoSE** repo — a Romanian
seismic dataset + three published phase pickers (EQT-RoSE, PhaseNet-RoSE,
RED-PAN-60s).

## Layout
- **`rose/`** — importable loader + helpers (`splits.hash_split`, `checkpoint_io.safe_torch_load`)
- **`training/`** — fine-tune SeisBench EQT / PhaseNet on RoSE (DDP, INSTANCE-init)
- **`benchmark/`** — RoSE / STEAD picker scoring; vendored `redpan_inference/` (TF subset)
- **`application/seisbench-rose-benchmark/`** — published release: 3 checkpoints + unified loaders (`benchmarks.models.load_eqt_rose`, `…_phasenet_rose`, `…_redpan_tf60`) + results CSVs + `SHA256SUMS`
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
`pytest tests/` (43 tests; pins the split partition, the SeisBench converter, `get_sample_physical`, the QC helpers, and `safe_torch_load`).
