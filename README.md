# RoSE — Romanian SEismic Dataset

[![tests](https://github.com/tso1257771/RoSE/actions/workflows/tests.yml/badge.svg)](https://github.com/tso1257771/RoSE/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/tso1257771/RoSE/blob/main/pyproject.toml)
[![license: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20250669.svg)](https://doi.org/10.5281/zenodo.20250669)

> A Romanian seismic waveform toolkit with enhanced phase labels, three
> trained phase pickers, and a fixed benchmark.

This repository contains software for the **RoSE Romanian seismic dataset**
(2014–2024) in [SeisBench](https://github.com/seisbench/seisbench) format. The
dataset has about 19,230 events and 416,000 picks in yearly waveform chunks
with per-trace metadata, plus 89,693 noise windows of 100 s from the same
stations in chunks `noise{YEAR}` (dataset version 3 and later; see
[`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md#noise-chunks)). Event
parameters through 2022 come from the published
ROMPLUS catalog. The 2023–2024 event fields come from the current NIEP
bulletin. The analyst picks, waveform windows, and response files were supplied
separately for this study. They are not part of ROMPLUS. The data files are
restricted until the RoSE data paper is accepted. The authors plan to release
them under CC BY 4.0 at that time.

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
| **Pick which magnitude to use, Mw or ML**          | [`docs/MAGNITUDES.md`](docs/MAGNITUDES.md) |
| Apply the released magnitude scale to another earthquake | `rose.magnitudes` — see [`magnitudes/README.md`](magnitudes/README.md) |
| Reproduce the magnitude calibration tables         | `ROMANIA_ROOT=… bash magnitudes/regenerate_magnitudes.sh` (needs the driver inputs, which are not published; see [`magnitudes/README.md`](magnitudes/README.md#rerunning-the-drivers)) |
| Deposit or update the Zenodo records                | [`docs/ZENODO.md`](docs/ZENODO.md) |
| Load RoSE weights via `seisbench.models.X.from_pretrained("rose")` | [`seisbench_compat/README.md`](seisbench_compat/README.md) — converter + smoke test + the upstream-submission procedure |

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

The waveform files, StationXML files, and compiled catalog and pick tables are
not part of this repository. They are in the RoSE data record
([`10.5281/zenodo.23153183`](https://doi.org/10.5281/zenodo.23153183)), whose
files are restricted until the RoSE data paper is accepted. The code can read
local copies that the user is authorized to use. See [`docs/DATASET.md`](docs/DATASET.md) for the expected file formats.

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

`RoSE(path)` returns the earthquake traces. To load the noise windows as well,
pass `include_noise=True` and select a class on `trace_category`; for the noise
alone pass `chunks=[f"noise{y}" for y in range(2014, 2025)]`.

Disk format: **counts**, ZNE, 100 Hz; physical units derive per trace from
`trace_sensitivity_{e,n,z}` columns. ~4 % of traces have no usable response
(see [Provenance](#provenance) for the breakdown). Full schema:
[`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md).

---

## Tutorials (`examples/`)

Four runnable examples, each end-to-end against the published dataset:

1. **`01_load_and_browse.py`** — open the bundle, filter on
   `trace_p_snr_db` / `source_magnitude`, plot a random pick.
2. **`02_eqt_instance_vrancea.py`** — full demo on the *M*<sub>w</sub> 5.55
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
internally. To avoid TensorFlow, the bundled checkpoint can be ported to
PyTorch with `scripts/convert_redpan_60s.py` from
[RED-PAN-Motion](https://github.com/tso1257771/RED-PAN-Motion), which
reproduces it to 1.2e-7 (max abs difference, picker and detector). `examples/04_picker_inference.py` runs all three on held-out test
traces; [`phase_picking/models/README.md`](phase_picking/models/README.md) has the per-model cards
(architecture, training recipe, dev loss) and `SHA256SUMS` to verify the
checkpoints.

If you'd rather load the SeisBench checkpoints via the upstream
`from_pretrained` API instead of the bundled `rose.load_*` helpers,
[`seisbench_compat/`](seisbench_compat/README.md) converts the same `.pt`
files into the SeisBench weight format
(`<model>/rose.pt.v1` + `<model>/rose.json.v1`) and round-trips
bit-identically against the helpers above. The same README documents the
upstream-submission procedure if you want to ship the weights to the
SeisBench mirror so `EQTransformer.from_pretrained("rose")` works for
everyone.

---

## Benchmark headlines

Threshold 0.30, on the RoSE (Romanian local-earthquake; dataset version 4) and STEAD test sets.
Full tables — all 9 models (the 3 bundled checkpoints + off-the-shelf EQT /
PhaseNet baselines), all 6 thresholds, onset residuals, plus the EQT-RoSE
Münchmeyer-Section-3 numbers — are in [`results/README.md`](phase_picking/results/README.md);
the raw CSVs are `phase_picking/results/*.csv`.

| Pool  | Best phase-pick F1 (P / S)        | Best event-detection F1                  |
|---|---|---|
| **RoSE**  | **RED-PAN-60s** 0.829 / 0.829     | **EQT-RoSE** 0.976 (MCC 0.947)           |
| **STEAD** | **RED-PAN-60s** 0.974 (P); **PhaseNet-stead** 0.983 (S) | **EQT-stead** 0.998; **EQT-RoSE** 0.991  |

(Produced by `bash phase_picking/benchmark/regenerate_results.sh` →
`phase_picking/results/*.csv`; see [`phase_picking/benchmark/README.md`](phase_picking/benchmark/README.md).)
The RoSE false positives are counted on STEAD noise; `results/README.md`
also counts them on RoSE noise, scores the analysts' picks alone (41% of the
test labels are RED-PAN-60s repicks, and on the analysts' picks PhaseNet-RoSE
and PhaseNet-stead lead), and describes the fix to RED-PAN-60s's padding of
records longer than 60 s; its RoSE and STEAD rows come from the rerun with
that fix.

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
`ROSE_DATA_DIR` (default `/data` inside the image). Mount a local copy that you
are authorized to use and run any tutorial directly:

```bash
docker build -t rose:cpu .
docker run --rm -it -v $(pwd)/data/rose:/data:ro \
    rose:cpu python 01_load_and_browse.py
```

The local `data/` directory is excluded from the image (`.dockerignore`). Mount
it as a volume.

---

## Tests

The unit suite exercises the `rose` package directly (no large dataset needed):

```bash
pip install -e ".[dev]"
pytest tests/ -q          # 130 tests
```

`tests/test_splits.py` pins the deterministic split partition bit-for-bit.
GitHub Actions runs the same suite on Python 3.10 / 3.11 / 3.12 for every
push and PR (the badge at the top reflects current status).

---

## Repo layout

```
RoSE/                              # ── the RoSE software toolkit
├── rose/                          # the importable package: RoSE loader · convert · qc · splits
│   │                              #   + the picker loaders (pickers.load_*, checkpoint_io, redpan_inference/)
│   └── magnitudes/                #   the released Mw and ML scales (+ the two vendored measurement packages)
├── docs/                          # DATASET.md, SEISBENCH_FORMAT.md (schemas), MAGNITUDES.md
├── examples/                      # 01, 02, 03, 04 — runnable tutorials
├── tests/                         # pytest unit tests
├── stationxml_sources/sc3ml_niep/ # SeisComP SC3ML → FDSN StationXML helper
├── magnitudes/                    # ── the magnitude calibration: the tables, the drivers that made them
│   ├── calibration/               #   every coefficient of both scales, with its standard error
│   ├── ml/                        #   local magnitude drivers + README (the method)
│   └── mw/                        #   moment magnitude drivers + README (the method)
├── phase_picking/                 # ── the phase-picking extension (built on the `rose` API + SeisBench)
│   ├── README.md
│   ├── models/                    #   the 3 published checkpoints + SHA256SUMS + model cards
│   ├── results/                   #   the pre-computed benchmark CSVs (the published numbers)
│   ├── training/                  #   SeisBench EQT / PhaseNet RoSE fine-tuning + cloud/ launchers
│   └── benchmark/                 #   the pipeline: run_inference.py (a) + build_leaderboard.py (b)
│                                  #     + config.json + the bench_*/build_* stage scripts + data/ (test indices)
└── seisbench_compat/              # ── bridge to seisbench.models.X.from_pretrained("rose")
    │                              #   convert.py + smoke_test.py + README.md (upstream-submission procedure)
    └── weights/                   #   regenerated artifacts (gitignored): <model>/rose.{pt,json}.v1
```

(Everything under `data/` is gitignored. `outputs/`, `checkpoints/`,
`phase_picking/benchmark/eval/`,
`rose_split_index.*`, `seisbench_compat/weights/` are gitignored — see
`.env.example` / `phase_picking/benchmark/config.json`.)

---

## Provenance

| Concern | Provenance |
|---|---|
| Event hypocenters | relocated: hypoDD for 18,574 events and NonLinLoc with source-specific station terms for 416; the other 240 keep the ROMPLUS (984–2022) or NIEP bulletin (2023–2024) solution, which the `raw_*` columns keep for every event |
| Origin time | ROMPLUS 984–2022 or the current NIEP bulletin (`source_*_raw`) |
| Magnitude | measured for this release (Mw, ML); see [`docs/DATASET.md`](docs/DATASET.md) |
| Analyst picks | supplied separately by Alexandru Mărmureanu for this study; not part of ROMPLUS and not distributed while permission is reviewed |
| ML-assisted repicks | RED-PAN 60 s, searched within ±3 s (P) and ±5 s (S) of the analyst pick, or of the theoretical arrival (NonLinLoc, 3D model) when the analyst phase is missing |
| Pick selection | higher-SNR among manual / RED-PAN; theoretical-only never used |
| Sampling | resampled to 100 Hz on a uniform reference axis with median-pad fill. In dataset versions 1 to 3, 3,067 BH traces resampled from 20 or 50 Hz have one wrong sample at their end; version 4 repairs them; see [`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md#known-issues-in-the-earthquake-traces) for the cause and the repair |
| Component order | **ZNE** on disk (canonical SeisBench order) |
| Instrument response | The StationXML archive in the data record has files for **211 / 226 stations**. A full three-component response is available for 210,545 traces, 144 have a vertical response only, and 8,583 lack a full three-component response. |
| Event coverage | **19 228 / 19 230** catalog events have waveforms (two excluded for missing all-three-component recordings: `2014_0001698`, `2020_0000816`). |

---

## Citation

**Toolkit (this repository) — v0.4.3.** Cite the concept DOI
[`10.5281/zenodo.20250669`](https://doi.org/10.5281/zenodo.20250669), which always
resolves to the newest version, and name the version you used. The v0.4.3 record
is [`10.5281/zenodo.23840686`](https://doi.org/10.5281/zenodo.23840686), the v0.4.2 record
is [`10.5281/zenodo.23790328`](https://doi.org/10.5281/zenodo.23790328), the v0.4.1 record
is [`10.5281/zenodo.23786822`](https://doi.org/10.5281/zenodo.23786822), the v0.4.0 record
is [`10.5281/zenodo.23717128`](https://doi.org/10.5281/zenodo.23717128), the v0.3.0 record
is [`10.5281/zenodo.23157260`](https://doi.org/10.5281/zenodo.23157260), and the v0.1.0
record is [`10.5281/zenodo.20250670`](https://doi.org/10.5281/zenodo.20250670).

```bibtex
@software{liao_rose_2026,
  author    = {Liao, Wu-Yu and Manea, Elena-Florinela and M{\u a}rmureanu, Alexandru and Lee, En-Jui},
  title     = {{RoSE — Romanian SEismic Dataset Toolkit}},
  year      = 2026,
  version   = {v0.4.3},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20250669},
  url       = {https://doi.org/10.5281/zenodo.20250669}
}
```

GitHub's "Cite this repository" sidebar renders the same info from
[`CITATION.cff`](CITATION.cff).

**Dataset.** The RoSE dataset record (concept DOI
[`10.5281/zenodo.23153183`](https://doi.org/10.5281/zenodo.23153183), which
resolves to the newest version; the versions are listed in
[`docs/ZENODO.md`](docs/ZENODO.md#data-record-versions)) has its files
restricted until the RoSE data paper is accepted. The authors plan to release
them under CC BY 4.0 at that time. The ROMPLUS 984--2022 citation applies to
the event catalog fields from that release. It does not apply to phase picks,
waveforms, response metadata, or later NIEP products. Please also cite the
networks that recorded the earthquake waveforms: RO ([`10.7914/SN/RO`](https://doi.org/10.7914/SN/RO)), GE ([`10.14470/TR560404`](https://doi.org/10.14470/TR560404)), BS ([`10.7914/SN/BS`](https://doi.org/10.7914/SN/BS)), MD ([`10.7914/SN/MD`](https://doi.org/10.7914/SN/MD)), SJ ([`10.7914/SN/SJ`](https://doi.org/10.7914/SN/SJ)), HU ([`10.14470/UH028726`](https://doi.org/10.14470/UH028726)), MN ([`10.13127/SD/fBBBtDtd6q`](https://doi.org/10.13127/SD/fBBBtDtd6q)), KO ([`10.7914/SN/KO`](https://doi.org/10.7914/SN/KO)), TU ([`10.7914/SN/TU`](https://doi.org/10.7914/SN/TU)) and CZ ([`10.7914/SN/CZ`](https://doi.org/10.7914/SN/CZ)).

**Model architectures** the bundled pickers extend: RED-PAN —
Liao et al. 2022; SeisBench — Woollam et al. 2022; PhaseNet —
Zhu & Beroza 2019; EQTransformer — Mousavi et al. 2020. Per-model
cards: [`phase_picking/models/README.md`](phase_picking/models/README.md).

**Software** — the RED-PAN 60 s model bundled here as
`redpan_tf60/train.hdf5` is the published `REDPAN_60s_240107` checkpoint.
[RED-PAN-Motion](https://github.com/tso1257771/RED-PAN-Motion) (MIT)
maintains a pure-PyTorch implementation of the same architecture, and its
`scripts/convert_redpan_60s.py` ports a TensorFlow checkpoint to PyTorch and
verifies the result. Converting the bundled checkpoint with that script
reproduces it to 1.2e-7, so a TensorFlow-free path is available. The
amplitude window used for the released magnitudes comes from the same
package.

---

## License

This repository is a software toolkit. Everything in it — the `rose`
package, the `phase_picking/{benchmark,training}/` scripts, the three
bundled model weights (`*.pt` / `*.hdf5`), the examples, tests, docs,
configs, and `SHA256SUMS` — is released under the **MIT License**
([`LICENSE`](LICENSE)).

The RoSE dataset is not included in this repository. Its Zenodo files are
restricted until the RoSE data paper is accepted. The authors plan to release
them under CC BY 4.0 at that time.
The archived v0.1.0 and v0.3.0 code records are also restricted because their
original source packages contained the event catalog. ROMPLUS is the source
event catalog through 2022. It is not the source or license for the phase
picks, waveforms, or StationXML files. Written permission should also state
whether it covers distribution of the trained weights and numerical products
derived from those files.
