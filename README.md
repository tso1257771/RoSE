# RoSE — Romanian SEismic Dataset

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
on RoSE, three published phase-picker checkpoints (`models/`) with one-call
loaders (`rose.load_eqt_rose` / `load_phasenet_rose` / `load_redpan_tf60`), the
pre-computed benchmark tables (`results/`), and the benchmark pipeline that
produces them (`benchmark/`).

---

## I want to…

| Goal                                              | Read / run |
|---|---|
| **Load and browse the dataset**                   | [Quickstart](#quickstart) below + `examples/01_load_and_browse.py` |
| Reference the dataset schema                       | [`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md) |
| **Use a published picker on my data**             | [Use the published pickers](#use-the-published-pickers) below + `examples/04_picker_inference.py` + [`models/README.md`](models/README.md) |
| Reproduce the benchmark tables (`results/*.csv`)   | `bash benchmark/regenerate_results.sh --update-release` — see [`benchmark/README.md`](benchmark/README.md) (needs the RoSE + STEAD test data) |
| **Fine-tune EQT / PhaseNet on RoSE**              | [Training & benchmarking](#training--benchmarking) below + `training/` |
| Run a single picker on RoSE / STEAD                | `benchmark/bench_pickers_rose.py`, `bench_stead_test.py` |
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

---

## Quickstart

```python
from rose import RoSE

data = RoSE("/path/to/data/rose", component_order="ZNE")  # PhaseNet/EQT order
print(len(data), "traces")

wf_counts, meta = data.get_sample(0)            # raw counts, shape (3, npts)
wf_phys,   meta = data.get_sample_physical(0)   # M/S or M/S**2 — divides by per-trace sensitivity
                                                # (raises ValueError if a trace lacks a usable
                                                #  instrument response; see below)
```

The dataset stores **counts** on disk and the per-component sensitivity values
in metadata; `get_sample_physical` does the divide for you and raises
`ValueError` on the ~4 % of traces whose `trace_status_physical` is
`partial_response` or `missing_response` (the [Provenance](#provenance) table
breaks this down — only ~0.2 % come from stations with no public response at
all). Why this design? See
[`docs/SEISBENCH_FORMAT.md#units--instrument-response`](docs/SEISBENCH_FORMAT.md).

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

Three trained checkpoints ship under [`models/`](models/) — EQT-RoSE (1.6 MB),
PhaseNet-RoSE (1.1 MB), RED-PAN-60s (5.8 MB) — each with a one-call loader in
the `rose` package that returns a SeisBench-style object:

```python
from rose import load_eqt_rose          # also load_phasenet_rose, load_redpan_tf60

model = load_eqt_rose()                  # loads models/eqt_rose/eqt_rose.pt (weights_only=True)
out   = model.classify(stream,           # an ObsPy Stream in ZNE order
                       P_threshold=0.3, S_threshold=0.3, detection_threshold=0.3)
print(out.picks, out.detections)
```

`load_redpan_tf60()` needs TensorFlow (the `.[tf]` extra); it reorders ZNE → ENZ
internally. `examples/04_picker_inference.py` runs all three on held-out test
traces; [`models/README.md`](models/README.md) has the per-model cards
(architecture, training recipe, dev loss) and `SHA256SUMS` to verify the
checkpoints.

---

## Training & benchmarking

`training/` and `benchmark/` hold the ML pipeline behind the published pickers.
The scripts read default paths from environment variables (see `.env.example`)
and accept explicit `--rose-dir` / `--stead-dir` / `--out-dir` overrides:

```bash
pip install -e ".[cuda,bench]"              # torch + scikit-learn  (or ".[cpu,bench]")
cp .env.example .env && $EDITOR .env        # set ROSE_DATA_DIR, STEAD_DIR, …
set -a; source .env; set +a                 # export them — the scripts don't auto-load .env
```

(Equivalently, pass `--rose-dir` / `--stead-dir` / `--out-dir` on each command;
without one of these or the matching env var the scripts exit with a clear error.)

* **`training/`** — fine-tune SeisBench `EQTransformer` / `PhaseNet` from
  the INSTANCE pretrained weights on the RoSE training split (DDP, AMP, Adam):

  ```bash
  python training/build_rose_split_index.py     # write the SeisBench `split` column
  python training/train_eqt_rose.py      --epochs 30 --batch-size 64  --lr 1e-4
  python training/train_phasenet_rose.py --epochs 30 --batch-size 256 --lr 1e-4
  ```

  `build_rose_split_index.py` uses RED-PAN's deterministic `hash_split` (salt
  `ROMPLUS-singleEQ-v1`, vendored in `rose/splits.py`) so every trace of one
  earthquake stays in the same split — no event-level leakage. The TWCC
  launcher shells used for the published runs are under `training/cloud/`.

* **`benchmark/`** — the benchmark **pipeline**: scores all 9 pickers (the 3
  RoSE-trained checkpoints in `models/` + 6 off-the-shelf EQT/PhaseNet
  baselines) on the RoSE / STEAD test sets (per-phase precision/recall/F1, MCC,
  residual stats, trace-level event-vs-noise T1) and is what **produces** the
  committed `results/*.csv`. `bench_pickers_rose.py` / `bench_redpan_rose.py` /
  `bench_stead_test.py` / `bench_noise_fp.py` do the inference; `build_*.py`
  aggregate; **`benchmark/regenerate_results.sh` chains the whole thing** (see
  [`benchmark/README.md`](benchmark/README.md) — needs the RoSE + STEAD test
  data):

  ```bash
  bash benchmark/regenerate_results.sh --update-release         # full run (hours on CPU)
  bash benchmark/regenerate_results.sh --num-test 200           # ~10-min smoke run
  ```

  The exact test-set composition is pinned by the index files under
  `benchmark/data/`; regenerate them with
  `python benchmark/build_test_indices.py [--stead-dir $STEAD_DIR]`.

Every `.pt` checkpoint that `rose.pickers` and the `training/` + `benchmark/`
scripts read is loaded via `rose.checkpoint_io.safe_torch_load`, which forces
`torch.load(weights_only=True)` — the restricted unpickler — so loading a
third-party `.pt` cannot trigger the classic pickle-deserialization RCE. (The
RED-PAN-60s `.hdf5` is a Keras model; `tf.keras.models.load_model` can execute
embedded code, so verify it against `models/SHA256SUMS` — see `models/README.md`.)

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
pytest tests/ -q          # 43 tests
```

`tests/test_splits.py` golden-value pin tests guarantee the deterministic
split partition remains bit-for-bit reproducible.

---

## Repo layout

```
RoSE/
├── rose/                          # importable package: RoSE loader, qc, convert,
│                                  #   pickers.load_eqt_rose/…, checkpoint_io, redpan_inference/
├── models/                        # the 3 published checkpoints + SHA256SUMS + model cards
├── results/                       # the pre-computed benchmark CSVs (the published numbers)
├── benchmark/                     # benchmark pipeline (all 9 pickers) — *produces* results/*.csv;
│                                  #   regenerate_results.sh, bench_*.py, build_*.py, data/ (test indices)
├── training/                      # SeisBench EQT / PhaseNet RoSE training
├── examples/                      # 01, 02, 03, 04 — runnable tutorials
├── docs/                          # DATASET.md, SEISBENCH_FORMAT.md (schemas)
├── stationxml_sources/sc3ml_niep/ # SeisComP SC3ML → FDSN StationXML helper
└── tests/                         # pytest unit tests
```

(`data/`, `outputs/`, `checkpoints/`, `eval/`, `rose_split_index.*` are
gitignored — see `.env.example` for the env vars that point at them.)

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

If you use this dataset or the published checkpoints, please cite both the
**ROMPLUS** source bulletin (NIEP) and the **RoSE** dataset paper that
accompanies this compilation. See `docs/DATASET.md` for the native HDF5 schema,
`docs/SEISBENCH_FORMAT.md` for the column reference, and `models/README.md` for
the per-model cards and references (RED-PAN — Liao et al. 2022; SeisBench —
Woollam et al. 2022; PhaseNet — Zhu & Beroza 2019; EQTransformer — Mousavi
et al. 2020).

---

## License

| What | License | File |
|---|---|---|
| **Code** — the `rose` package, `models/` loaders/cards, `benchmark/`, `training/`, `examples/`, scripts, and everything else in this repo | **MIT** | [`LICENSE`](LICENSE) |
| **The RoSE dataset** — the SeisBench-format waveforms + per-trace metadata distributed on Zenodo and mounted at `data/rose/` (not in this repo) | **CC-BY-4.0** | [`LICENSE-DATA`](LICENSE-DATA) |

The dataset is derived from the NIEP **ROMPLUS** bulletin (a registered NIEP
product); cite ROMPLUS as well — see [Citation](#citation) above. Third-party
StationXML responses under `data/rose_stationxml/` carry their providers'
terms (FDSN data centres / NIEP).
