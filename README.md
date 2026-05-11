# RoSE — Romanian Seismic Events

> A ROMPLUS-enhanced Romanian seismic dataset for machine learning and
> seismological applications, plus three published phase pickers benchmarked
> on it.

End-to-end glue for the **RoSE Romanian seismic dataset** (2014–2024) in
[SeisBench](https://github.com/seisbench/seisbench) format: ~19 230 events and
~416 000 picks, distributed as yearly chunks of bucketed HDF5 plus per-trace
metadata. Built from the NIEP **ROMPLUS** catalog, relocated with hypoDD3D and
repicked with RED-PAN. *ROMPLUS* names the source bulletin (a registered NIEP
product); *RoSE* names the derived, ML-ready dataset that we publish.

This repo also ships a small ML pipeline: deterministic train/dev/test split,
training scripts that fine-tune `seisbench.models.EQTransformer` and
`PhaseNet` on RoSE, a benchmark suite that scores them against STEAD, and a
self-contained release bundle (`application/seisbench-rose-benchmark/`) with
the trained checkpoints, unified loaders, and headline-number CSVs.

---

## I want to…

| Goal                                              | Read / run |
|---|---|
| **Load and browse the dataset**                   | [Quickstart](#quickstart) below + `examples/01_load_and_browse.py` |
| Reference the dataset schema                       | [`docs/SEISBENCH_FORMAT.md`](docs/SEISBENCH_FORMAT.md) |
| **Use a published picker on my data**             | [`application/seisbench-rose-benchmark/`](application/seisbench-rose-benchmark/README.md) |
| Reproduce the paper's benchmark numbers            | `bash application/seisbench-rose-benchmark/scripts/reproduce_all.sh` |
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
```

The dataset stores **counts** on disk and the per-component sensitivity values
in metadata; `get_sample_physical` does the divide for you and raises on traces
with no usable response (~4 %). Why this design? See
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
   (EQT-RoSE, PhaseNet-RoSE, RED-PAN-60s) via the release loaders and run
   them on a few held-out test traces; prints a per-model residual table
   and saves an overlay plot. RED-PAN-60s needs `.[tf]` (TensorFlow) — pass
   `--no-redpan` to skip it.

```bash
python examples/01_load_and_browse.py     # outputs/01_load_and_browse.png
python examples/04_picker_inference.py    # outputs/04_picker_inference.png
```

---

## Use the published pickers

`application/seisbench-rose-benchmark/` is a self-contained release with three
trained checkpoints (EQT-RoSE 1.6 MB, PhaseNet-RoSE 1.1 MB, RED-PAN-60s
5.8 MB), unified SeisBench-style loaders, the headline-number CSVs, and a
one-shot reproducer. Smallest path to predictions:

```python
import sys; sys.path.insert(0, "application/seisbench-rose-benchmark")
from benchmarks.models import load_eqt_rose

model = load_eqt_rose()                                    # 1.6 MB checkpoint, weights_only=True
out   = model.classify(stream, P_threshold=0.3, S_threshold=0.3,
                       detection_threshold=0.3)            # ObsPy Stream in
print(out.picks, out.detections)
```

See [`application/seisbench-rose-benchmark/README.md`](application/seisbench-rose-benchmark/README.md)
for the full release docs and `models/README.md` for per-model cards.

---

## Training & benchmarking

Three sibling directories hold the ML pipeline behind the published pickers.
The scripts read default paths from environment variables (see `.env.example`)
and accept explicit `--rose-dir` / `--stead-dir` / `--out-dir` overrides:

```bash
pip install -e ".[cuda,bench]"              # torch + scikit-learn
cp .env.example .env && $EDITOR .env        # set ROSE_DATA_DIR, STEAD_DIR, …
```

* **`training/`** — fine-tune SeisBench `EQTransformer` / `PhaseNet` from
  the INSTANCE pretrained weights on the RoSE training split (DDP, AMP, Adam):

  ```bash
  python training/build_rose_split_index.py     # write the SeisBench `split` column
  python training/train_eqt_rose.py      --epochs 30 --batch-size 128 --lr 1e-4
  python training/train_phasenet_rose.py --epochs 30 --batch-size 256 --lr 1e-4
  ```

  `build_rose_split_index.py` uses RED-PAN's deterministic `hash_split` (salt
  `ROMPLUS-singleEQ-v1`, vendored in `rose/splits.py`) so every trace of one
  earthquake stays in the same split — no event-level leakage. The TWCC
  launcher shells used for the published runs are under `training/cloud/`.

* **`benchmark/`** — score pickers on the RoSE / STEAD test sets
  (per-phase precision/recall/F1, MCC, residual stats, trace-level
  event-vs-noise T1). `bench_pickers_rose.py` is the canonical SeisBench-API
  entry point; `bench_stead_test.py` and `bench_redpan_rose.py` add the
  RED-PAN-60s comparison (needs `.[tf]`). The exact test-set composition is
  pinned by the index files in `application/.../data/`; regenerate them with
  `python benchmark/build_test_indices.py [--stead-dir $STEAD_DIR]`.

* **`application/seisbench-rose-benchmark/`** — the published release (model
  weights + verifiable `SHA256SUMS` + loaders + `pickerbench` scoring +
  `redpan_inference` subset + `results/*.csv`). One-shot reproduction:

  ```bash
  bash application/seisbench-rose-benchmark/scripts/reproduce_all.sh
  ```

All `.pt` checkpoints are loaded via `rose.checkpoint_io.safe_torch_load`,
which forces `torch.load(weights_only=True)` — the restricted unpickler — so
running the scripts on third-party `.pt` files cannot trigger the classic
pickle-deserialization RCE.

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
pytest tests/ -q          # 39 tests
```

`tests/test_splits.py` golden-value pin tests guarantee the deterministic
split partition remains bit-for-bit reproducible.

---

## Repo layout

```
RoSE/
├── rose/                          # importable package (loader + helpers)
├── examples/                      # 01, 03, 04 — runnable tutorials
├── docs/                          # DATASET.md, SEISBENCH_FORMAT.md (schemas)
├── training/                      # SeisBench EQT / PhaseNet RoSE training
├── benchmark/                     # multi-model benchmark suite (RoSE + STEAD)
├── application/seisbench-rose-benchmark/   # published release: weights + loaders + results
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
accompanies this compilation. See `docs/DATASET.md` for the native HDF5
schema and `docs/SEISBENCH_FORMAT.md` for the column reference; the picker
release has its own citation block in
[`application/seisbench-rose-benchmark/README.md#citation`](application/seisbench-rose-benchmark/README.md#citation).
