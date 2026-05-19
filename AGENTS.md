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
- **`seisbench_compat/`** — converter + smoke test that turn `phase_picking/models/{eqt_rose,phasenet_rose}/*.pt` into the SeisBench `from_pretrained` layout (`<model>/rose.{pt,json}.v1`), plus the upstream-submission procedure in its README. Output dir `weights/` is gitignored. Run from repo root with `PYTHONPATH=. python seisbench_compat/{convert,smoke_test}.py`.
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
`data/` (except the versioned `data/Enhanced_ROMPLUS_catalog.csv`), `eval/`, `checkpoints/`, `outputs/`, `rose_split_index.*` — all gitignored.

## Verify
`pytest tests/` (50 tests; pins the split partition, the SeisBench converter, `get_sample_physical`, the QC helpers, `safe_torch_load`, and the `redpan_inference` utilities). CI runs the same suite on Python 3.10/3.11/3.12 via `.github/workflows/tests.yml`.

## Release procedure (`v0.1.x → v0.1.{x+1}`)

Each public release mints (a) a tagged git commit, (b) a GitHub Release with the 77 MB picks CSV attached, and (c) a Zenodo software DOI via the `tso1257771/RoSE` ↔ Zenodo integration. Steps:

1. **Pre-flight gate**
   - `pytest tests/ -q` green; `python -m ruff check rose/ tests/ phase_picking/` clean.
   - Fill or refresh placeholder fields:
     - `CITATION.cff` → `authors[0].family-names` / `given-names` / `orcid`, `date-released:`, `version:`.
     - `.zenodo.json` → `creators[0].name` / `affiliation` / `orcid`.
   - If a paper / dataset DOI is now known, uncomment the `references:` block in `CITATION.cff` and add a matching `related_identifiers` entry in `.zenodo.json` (`relation: isSourceOf`, `scheme: doi`).
   - Bump `version` in `CITATION.cff` and (optionally) in `pyproject.toml`.

2. **Tag + push**
   ```bash
   git tag -a vX.Y.Z -m "RoSE toolkit vX.Y.Z"
   git push origin vX.Y.Z
   ```
   Use annotated tags (`-a`) — lightweight tags don't carry metadata that Zenodo / GitHub Releases want.

3. **GitHub Release**
   - Web UI → Releases → "Draft a new release" → pick the tag.
   - Title: `vX.Y.Z — <short description>`. Body: a few lines paraphrasing the changelog since the previous tag.
   - Drag-and-drop `data/Enhanced_ROMPLUS_picks.csv` (77 MB) into the binary-asset area. Optionally add a `.sha256` sidecar.
   - Click "Publish release".

4. **Zenodo software DOI** (auto, if GitHub→Zenodo integration is on at `zenodo.org/account/settings/github/`)
   - The release event triggers Zenodo to create a draft record using `.zenodo.json`.
   - Open Zenodo → Uploads → review the draft (title, creators, license, related identifiers). Edit anything still wrong, then click "Publish".
   - Copy the new DOI. It becomes immutable.

5. **Cross-reference (only when both data + code DOIs exist)**
   - Edit each Zenodo record's *Related identifiers* field:
     - Software record adds: `IsSourceOf: <data DOI>`, `IsDocumentedBy: <paper DOI>`.
     - Data record adds: `IsDocumentedBy: <software DOI>`, `IsCitedBy: <paper DOI>`.
   - In the next minor doc commit, paste both DOIs into the README "Citation" section and uncomment the `references:` block in `CITATION.cff`.

The toolkit can ship a software DOI before the data Zenodo record exists. The cross-references go in after-the-fact; both records remain stable, only metadata updates.
