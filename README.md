# RoSE — Romanian SEismic Dataset

[![tests](https://github.com/tso1257771/RoSE/actions/workflows/tests.yml/badge.svg)](https://github.com/tso1257771/RoSE/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/tso1257771/RoSE/blob/main/pyproject.toml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20250670.svg)](https://doi.org/10.5281/zenodo.20250670)

> A ROMPLUS-enhanced Romanian seismic dataset for machine learning and
> seismological applications, plus three published phase pickers benchmarked
> on it.

End-to-end glue for the **RoSE Romanian seismic dataset** (2014–2024) in
[SeisBench](https://github.com/seisbench/seisbench) format: ~19,230 events and
~416,000 picks, distributed as yearly chunks of bucketed HDF5 plus per-trace
metadata. Built from the NIEP **ROMPLUS** catalog, relocated with hypoDD3D and
repicked with RED-PAN. *ROMPLUS* names the source bulletin (a registered NIEP
product); *RoSE* names the derived, ML-ready dataset that we publish.

This repo also ships a small ML pipeline: deterministic train/dev/test split,
training scripts that fine-tune `seisbench.models.EQTransformer` and `PhaseNet`
on RoSE, three published phase-picker checkpoints (`phase_picking/models/`) with one-call
loaders (`rose.load_eqt_rose` / `load_phasenet_rose` / `load_redpan_tf60`), the
pre-computed benchmark tables (`phase_picking/results/`), and the benchmark pipeline that
produces them (`phase_picking/benchmark/`).

---

## I want to…

| Goal                                              | Read / run |
|---|---|
| **Load and browse the dataset**                   | [Quickstart](#quickstart) below + `examples/01_load_and_browse.py` |
| Reference the dataset schema                       | [`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md) |
| **Use a published picker on my data**             | [Use the published pickers](#use-the-published-pickers) below + `examples/04_picker_inference.py` + [`phase_picking/models/README.md`](phase_picking/models/README.md) |
| Reproduce the benchmark tables (`phase_picking/results/*.csv`)   | `bash phase_picking/benchmark/regenerate_results.sh` — see [`phase_picking/benchmark/README.md`](phase_picking/benchmark/README.md) (needs the RoSE + STEAD test data) |
| **Fine-tune EQT / PhaseNet on RoSE**              | [Training & benchmarking](#training--benchmarking) below + `phase_picking/training/` |
| Run a single picker on RoSE / STEAD                | `phase_picking/benchmark/bench_pickers_rose.py`, `bench_stead_test.py` |
| Build the SeisBench bundle from the native HDF5    | [`docs/DATASET.md`](docs/DATASET.md) + `rose.convert.convert_all` |

---

## Install

The package lives in `pyproject.toml` (Python ≥ 3.10). Pick the path that fits
your tooling:

```bash
git clone https://github.com/tso1257771/RoSE.git
cd RoSE

# A) uv (fastest, recommended)
uv venv --python 3.11 && source .venv/bin/activate
uv pip install -e ".[cpu]"        # or .[cuda] for GPU work

# B) pip (works everywhere)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# C) Docker (CPU-only image; mount data/ as a volume — see "Docker" below)
docker build -t rose:cpu .
```

Optional extras: `.[bench]` adds `scikit-learn` for the benchmark scripts;
`.[tf]` adds `tensorflow` for the RED-PAN-60s picker; `.[dev]` adds `pytest`
and `ruff`.

The 35 GB published waveform dataset (`data/rose/`) and the 80 MB StationXML
response archive (`data/rose_stationxml/`) are distributed separately on
Zenodo — **not** in this repo. Mount or symlink them at `./data/rose/` and
`./data/rose_stationxml/` after cloning, or set the `ROSE_DATA_DIR` /
`ROSE_STATIONXML_DIR` environment variables.

The two compiled ROMPLUS source tables are versioned here:
`data/Enhanced_ROMPLUS_catalog.csv` (2.9 MB, 19 231 events) is committed
to the repo, and `data/Enhanced_ROMPLUS_picks.csv` (77 MB, 416 063 picks)
ships as a GitHub Release asset on each tagged version. See
[`docs/DATASET.md`](docs/DATASET.md) for the download recipe and the
native HDF5 schema both files feed into.

---

## Quickstart

```python
from rose import RoSE

data = RoSE("/path/to/data/rose", component_order="ZNE")  # PhaseNet/EQT order
print(len(data), "traces")

wf_counts, meta = data.get_sample(0)            # raw counts, shape (3, npts)
wf_phys,   meta = data.get_sample_physical(0)   # M/S or M/S**2; divides by per-trace
                                                # sensitivity, raises on missing response
```

Disk format: **counts**, ZNE, 100 Hz; physical units derive per trace from
`trace_sensitivity_{e,n,z}` columns. ~4 % of traces have no usable response
(see [Provenance](#provenance) for the breakdown). Full schema:
[`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md).

---

## Tutorials (`examples/`)

Four runnable examples, each end-to-end against the published dataset:

1. **`01_load_and_browse.py`** — open the bundle, filter on
   `trace_p_snr_db` / `source_magnitude`, plot a random pick.
2. **`02_eqt_instance_vrancea.py`** — full demo on the *M*<sub>w</sub> 5.8
   Vrancea slab event (2018-10-28, 153 km depth, 68 stations): rebuild an
   ObsPy `Stream` from SeisBench, run `EQTransformer.from_pretrained("instance")`
   for an off-the-shelf-picker comparison, plot a record section with catalog
   and model picks plus residuals.
3. **`03_event_ground_motion.py`** — single-event ground-motion workflow:
   waveform QC (`rose.qc`), Arias-intensity D5–95 coda window, instrument
   response removal from the bundled StationXML, PGA / PGV / PGD per pick
   source. Requires `data/rose_stationxml/`.
4. **`04_picker_inference.py`** — load all three published checkpoints
   (PhaseNet-RoSE, EQT-RoSE, RED-PAN-60s) via the release loaders and run
   them on a few held-out test traces. Input is in **Z, N, E** order;
   each model applies its own internal normalisation (`norm="peak"` for the
   SeisBench checkpoints, per-window Z-score for RED-PAN), so the script
   does NOT pre-normalise — it only demeans, detrends, and (by default)
   1–45 Hz Butterworth band-passes the stream (`--highpass` / `--lowpass`
   to override; pass `0` to disable either leg). Saves one 6-panel PNG per
   trace under `outputs/04_picker_inference/trace_<idx>.png` (Z, N, E
   waveforms on top; PhaseNet, EQTransformer, RED-PAN probability curves
   below, each `ylim=[-0.1, 1.1]`, all sharing the same time axis so
   timestamps line up vertically) plus a per-model residual table on
   stdout. RED-PAN-60s needs `.[tf]` (TensorFlow) — pass `--no-redpan` to
   skip it.

```bash
python examples/01_load_and_browse.py     # outputs/01_load_and_browse.png
python examples/04_picker_inference.py    # outputs/04_picker_inference/trace_*.png
```

---

## Use the published pickers

Three trained checkpoints ship under [`phase_picking/models/`](phase_picking/models/) — EQT-RoSE (1.6 MB),
PhaseNet-RoSE (1.1 MB), RED-PAN-60s (5.8 MB) — each with a one-call loader in
the `rose` package that returns a SeisBench-style object:

```python
from rose import load_eqt_rose          # also load_phasenet_rose, load_redpan_tf60

model = load_eqt_rose()                  # loads phase_picking/models/eqt_rose/eqt_rose.pt (weights_only=True)
out   = model.classify(stream,           # an ObsPy Stream in ZNE order
                       P_threshold=0.3, S_threshold=0.3, detection_threshold=0.3)
print(out.picks, out.detections)
```

`load_redpan_tf60()` needs TensorFlow (the `.[tf]` extra); it reorders ZNE → ENZ
internally. `examples/04_picker_inference.py` runs all three on held-out test
traces; [`phase_picking/models/README.md`](phase_picking/models/README.md) has the per-model cards
(architecture, training recipe, dev loss) and `SHA256SUMS` to verify the
checkpoints.

---

## Benchmark headlines

Threshold 0.30, on the RoSE (Romanian local-earthquake) and STEAD test sets.
Full tables — all 9 models (the 3 bundled checkpoints + off-the-shelf EQT /
PhaseNet baselines), all 6 thresholds, onset residuals, plus the EQT-RoSE
Münchmeyer-Section-3 numbers — are in [`results/README.md`](phase_picking/results/README.md);
the raw CSVs are `phase_picking/results/*.csv`.

| Pool  | Best phase-pick F1 (P / S)        | Best event-detection F1                  |
|---|---|---|
| **RoSE**  | **RED-PAN-60s** 0.822 / 0.782     | **EQT-RoSE** 0.977 (MCC 0.945)           |
| **STEAD** | **RED-PAN-60s** 0.972 / 0.980     | **EQT-stead** 0.998; **EQT-RoSE** 0.991  |

(Produced by `bash phase_picking/benchmark/regenerate_results.sh` →
`phase_picking/results/*.csv`; see [`phase_picking/benchmark/README.md`](phase_picking/benchmark/README.md).)

---

## Training & benchmarking

The ML pipeline behind the published pickers lives under `phase_picking/`.
Path inputs come from env vars (`ROSE_DATA_DIR`, `STEAD_DIR`; see
`.env.example`) or `--rose-dir` / `--stead-dir` flags.

```bash
pip install -e ".[cuda,bench]"          # or ".[cpu,bench]"; .[tf] adds TF for RED-PAN-60s

# Fine-tune EQT / PhaseNet on RoSE (DDP, AMP, Adam, INSTANCE-init)
python phase_picking/training/build_rose_split_index.py
python phase_picking/training/train_eqt_rose.py      --epochs 30 --batch-size 64  --lr 1e-4
python phase_picking/training/train_phasenet_rose.py --epochs 30 --batch-size 256 --lr 1e-4

# Reproduce the benchmark CSVs (all 9 pickers × RoSE/STEAD test sets)
bash phase_picking/benchmark/regenerate_results.sh                # full run
bash phase_picking/benchmark/regenerate_results.sh --num-test 200 # quick subset
```

Splits use `rose.splits.hash_split` (salt `ROMPLUS-singleEQ-v1`) — every
trace of one earthquake stays in the same split, no event-level leakage.
PyTorch checkpoints are loaded via `rose.checkpoint_io.safe_torch_load`
(`weights_only=True`). The RED-PAN `.hdf5` is a Keras model — verify against
`phase_picking/models/SHA256SUMS` before loading from untrusted sources.

Full details: [`phase_picking/benchmark/README.md`](phase_picking/benchmark/README.md)
and [`phase_picking/models/README.md`](phase_picking/models/README.md).

---

## Docker

A reproducible CPU image is provided. The tutorials read the dataset path from
`ROSE_DATA_DIR` (default `/data` inside the image). Mount the dataset and run
any tutorial directly:

```bash
docker build -t rose:cpu .
docker run --rm -it -v $(pwd)/data/rose:/data:ro \
    rose:cpu python 01_load_and_browse.py
```

The 35 GB `data/` directory is excluded from the image (`.dockerignore`) —
always mount it as a volume.

---

## Tests

The unit suite exercises the `rose` package directly (no large dataset needed):

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 50 tests
```

`tests/test_splits.py` pins the deterministic split partition bit-for-bit.
GitHub Actions runs the same suite on Python 3.10 / 3.11 / 3.12 for every
push and PR (the badge at the top reflects current status).

---

## Repo layout

```
RoSE/                              # ── the RoSE dataset + its Python API  (the repo's headline)
├── rose/                          # the importable package: RoSE loader · convert · qc · splits
│                                  #   + the picker loaders (pickers.load_*, checkpoint_io, redpan_inference/)
├── docs/                          # DATASET.md, SEISBENCH_FORMAT.md (schemas)
├── examples/                      # 01, 02, 03, 04 — runnable tutorials
├── tests/                         # pytest unit tests
├── stationxml_sources/sc3ml_niep/ # SeisComP SC3ML → FDSN StationXML helper
└── phase_picking/                 # ── the phase-picking extension (built on the `rose` API + SeisBench)
    ├── README.md
    ├── models/                    #   the 3 published checkpoints + SHA256SUMS + model cards
    ├── results/                   #   the pre-computed benchmark CSVs (the published numbers)
    ├── training/                  #   SeisBench EQT / PhaseNet RoSE fine-tuning + cloud/ launchers
    └── benchmark/                 #   the pipeline: run_inference.py (a) + build_leaderboard.py (b)
                                   #     + config.json + the bench_*/build_* stage scripts + data/ (test indices)
```

(Everything under `data/` is gitignored *except* the compiled ROMPLUS
catalog CSV; `outputs/`, `checkpoints/`, `phase_picking/benchmark/eval/`,
`rose_split_index.*` are gitignored — see `.env.example` /
`phase_picking/benchmark/config.json`.)

---

## Provenance

| Concern | Provenance |
|---|---|
| Event hypocenters | hypoDD3D relocations of the NIEP ROMPLUS catalog |
| Origin time, magnitude | inherited from ROMPLUS (`source_*_raw`) |
| Manual picks | NIEP ROMPLUS bulletins |
| ML-assisted repicks | RED-PAN 60 s, 3 s / 5 s P / S consistency window vs. NLLoc theoretical |
| Pick selection | higher-SNR among manual / RED-PAN; theoretical-only never used |
| Sampling | resampled to 100 Hz on a uniform reference axis with median-pad fill |
| Component order | **ZNE** on disk (canonical SeisBench order) |
| Instrument response | StationXML covers **217 / 226 stations (96 %)**: 152 with full poles+zeros (FDSN-fetched), 65 with overall sensitivity only (NIEP-provided). 9 stations (1 SJ + 8 TU; 0.21 % of traces) have no public response — the metadata still carries `trace_sensitivity_{e,n,z}` for them. |
| Event coverage | **19 228 / 19 230** catalog events have waveforms (two excluded for missing all-three-component recordings: `2014_0001698`, `2020_0000816`). |

---

## Citation

**Toolkit (this repository) — v0.1.0**, [`10.5281/zenodo.20250670`](https://doi.org/10.5281/zenodo.20250670):

```bibtex
@software{liao_rose_2026,
  author    = {Liao, Wu-Yu},
  title     = {{RoSE — Romanian SEismic Dataset Toolkit}},
  year      = 2026,
  version   = {v0.1.0},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20250670},
  url       = {https://doi.org/10.5281/zenodo.20250670}
}
```

GitHub's "Cite this repository" sidebar renders the same info from
[`CITATION.cff`](CITATION.cff).

**Dataset** — the RoSE dataset bundle has its own Zenodo record
(DOI to be added once the data deposit is published). Cite the
**ROMPLUS** source bulletin (NIEP) alongside it.

**Model architectures** the bundled pickers extend: RED-PAN —
Liao et al. 2022; SeisBench — Woollam et al. 2022; PhaseNet —
Zhu & Beroza 2019; EQTransformer — Mousavi et al. 2020. Per-model
cards: [`phase_picking/models/README.md`](phase_picking/models/README.md).

---

## License

This repository is a software toolkit. Everything in it — the `rose`
package, the `phase_picking/{benchmark,training}/` scripts, the three
bundled model weights (`*.pt` / `*.hdf5`), the examples, tests, docs,
configs, and `SHA256SUMS` — is released under the **MIT License**
([`LICENSE`](LICENSE)).

The **RoSE dataset itself** — the SeisBench-format waveforms +
per-trace metadata that the toolkit consumes — is distributed
separately from this repo. Its licensing terms are set by the dataset's
rights holders and stated in the dataset's own record (Zenodo, DOI to
be added at release); see the [Citation](#citation) section below.
Third-party StationXML responses under `data/rose_stationxml/` carry
their providers' terms (FDSN data centres / NIEP).

The dataset is derived from the NIEP **ROMPLUS** bulletin (a registered
NIEP product). Cite ROMPLUS in addition to the dataset record when you
use the data.
