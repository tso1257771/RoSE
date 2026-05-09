# RoSE — Romanian Seismic Events

> **RoSE: a ROMPLUS-enhanced Romanian seismic dataset for machine-learning and
> seismological foundations.**

End-to-end glue for the **RoSE Romanian seismic dataset** (2014–2024) in
[SeisBench](https://github.com/seisbench/seisbench) format.

The published dataset is built from the NIEP **ROMPLUS** catalog, relocated
with hypoDD3D and repicked with RED-PAN. Throughout this repository
*ROMPLUS* refers to the **source bulletin name** (a registered NIEP product),
while *RoSE* is the name of the **derived, ML-ready dataset** that we publish.

The published RoSE SeisBench bundle (`data/rose/`) contains
~19 230 events and ≈ 416 000 picks (2014–2024), distributed as yearly
chunks of bucketed HDF5 plus per-trace metadata.

---

## Layout

```
seisbench_integration/
├── README.md
├── docs/
│   ├── DATASET.md           # native HDF5 schema (intermediate build)
│   └── SEISBENCH_FORMAT.md  # published RoSE schema + column reference
├── rose/                    # importable package (the "rose" Python module)
│   ├── __init__.py
│   ├── convert.py           # native HDF5  →  SeisBench format
│   ├── dataset.py           # RoSE(WaveformDataset) wrapper
│   └── qc.py                # waveform quality-control primitives
├── examples/                # tutorials targeting the published dataset
│   ├── 01_load_and_browse.py
│   ├── 03_eqt_instance_vrancea.py
│   └── 04_event_ground_motion.py
├── stationxml_sources/      # response-archive build helpers
│   └── sc3ml_niep/
│       └── sc3ml_to_stationxml.py   # SeisComP SC3ML → FDSN StationXML
└── tests/                   # unit tests for the rose package
```

The 35 GB published waveform dataset (`data/rose/`) and the curated
StationXML response archive (`data/rose_stationxml/`, ~80 MB) are
distributed separately on Zenodo and are **not** tracked in this
repository. Mount or symlink them at `./data/rose/` and
`./data/rose_stationxml/` after cloning, or set the `ROSE_DATA_DIR` /
`ROSE_STATIONXML_DIR` environment variables. The
`stationxml_sources/sc3ml_niep/sc3ml_to_stationxml.py` helper is
included for reproducibility of the response-archive build from
upstream SeisComP inventories.

## Install

The package definition lives in `pyproject.toml`. We recommend
[uv](https://github.com/astral-sh/uv) for fast, reproducible installs.

```bash
# 1) get uv (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2) create an isolated env and install the package
cd seisbench_integration
uv venv --python 3.11
source .venv/bin/activate

# Pick ONE torch variant — CPU is enough for tutorials 01 and 03:
uv pip install -e ".[cpu]"
# or, for GPU work:
uv pip install -e ".[cuda]"
```

Pip-only fallback:

```bash
pip install -r requirements.txt
pip install -e .
```

### Docker

A reproducible CPU image is provided. The tutorials read the dataset path
from the `ROSE_DATA_DIR` environment variable (default `/data` inside the
image, default `./data/rose` outside), so you mount the dataset at `/data`
and run any tutorial directly:

```bash
docker build -t rose:cpu .
docker run --rm -it \
    -v $(pwd)/data/rose:/data:ro \
    rose:cpu python 01_load_and_browse.py
```

The 35 GB `data/` directory is intentionally excluded from the image
(`.dockerignore`) — always mount it as a volume.

## Quickstart

```python
from rose import RoSE

# Open the published dataset (yearly chunks + chunks manifest).
data = RoSE(
    "/path/to/data/rose",
    component_order="ZNE",    # PhaseNet/EQTransformer convention
)

print(len(data), "traces")
wf_counts, meta = data.get_sample(0)              # raw counts, shape (3, npts)
wf_phys,   meta = data.get_sample_physical(0)     # M/S or M/S**2, same shape
```

If you have the *native* yearly HDF5 build (the intermediate ROMPLUS
HDF5 produced before the SeisBench layout), `rose.convert.convert_all`
re-builds the SeisBench bundle from it. See `docs/DATASET.md` for the
native schema and `rose/convert.py` for the converter API.

### Why a single counts file, not two HDF5s?

For publication it is wasteful to ship counts and physical units
separately: physical units are exactly `counts / sensitivity_value`, and
the sensitivities live in the StationXML that produced the physical
build. The converter therefore stores **only counts** on disk (smaller,
lossless, integer-valued) and attaches the per-component sensitivity
values plus the response status as metadata columns:

| Column | Meaning |
|---|---|
| `trace_units` | always `"counts"` (what is stored) |
| `trace_unit_physical` | `"M/S"`, `"M/S**2"`, … (target unit after divide) |
| `trace_sensitivity_e/n/z` | divisor for each component (counts / sens) |
| `trace_status_physical` | `ok`, `partial_response`, `missing_response` |
| `trace_missing_response_components` | comma list of components without response |

`RoSE.get_sample_physical(idx)` does the divide for you and raises if the
trace lacks a valid response (~4 % of traces).

## Walkthrough

`examples/` covers the published dataset end-to-end:

1. **`01_load_and_browse.py`** — open via `RoSE`, filter on
   `trace_p_snr_db` / `source_magnitude`, plot a random pick.
2. **`03_eqt_instance_vrancea.py`** — full demo on the $M_w$ 5.8 Vrancea
   slab event (2018-10-28, 153 km depth, 68 stations): reconstruct an
   ObsPy `Stream` from the SeisBench dataset, run
   `EQTransformer.from_pretrained("instance")`, and plot a record section
   with catalog and model picks overlaid plus residual statistics.
3. **`04_event_ground_motion.py`** — single-event ground-motion workflow:
   loads an event, runs waveform QC (`rose.qc`: clipping / dead / gaps /
   spikes / SNR), defines the S-coda window via Arias intensity D5–95,
   removes instrument response from the bundled StationXML, and extracts
   PGA/PGV/PGD per pick source. Requires `data/rose_stationxml/` to be
   mounted alongside the dataset.

## Tests

The unit-test suite exercises the `rose` package directly (no large
dataset required):

```bash
pip install pytest
python -m pytest tests/ -q
```

## Provenance

| Concern | Provenance |
|---|---|
| Event hypocenters | hypoDD3D relocations of the NIEP ROMPLUS catalog |
| Origin time, magnitude | inherited from ROMPLUS (`source_*_raw`) |
| Manual picks | NIEP ROMPLUS bulletins |
| ML-assisted repicks | RED-PAN 60s, 3 s/5 s P/S consistency window vs. theoretical (NLLoc) |
| Pick selection | higher-SNR among manual / RED-PAN; theoretical-only never used |
| Sampling | **always** resampled to 100 Hz on a uniform reference axis with median-pad fill |
| Component order | **ZNE** on disk in `data/rose/` (canonical SeisBench order) — vanilla `seisbench.data.WaveformDataset(path)` works without specifying `component_order` |
| Instrument response | StationXML bundled in `data/rose_stationxml/` covers **217/226 stations (96 %)**: 152 with full poles+zeros (FDSN-fetched, enables `remove_response`), 65 with overall sensitivity only (NIEP-provided). The 9 stations with no public response at all (1 SJ + 8 TU, 0.21 % of traces) still carry per-trace `trace_sensitivity_{e,n,z}` in the metadata. |
| Event coverage | **19 228 / 19 230 catalog events** have waveforms in `data/rose/`. Two catalog events (`2014_0001698`, `2020_0000816`) are excluded because no recording station had all three E/N/Z components in a single band code. |

## Citation

If you use this dataset, please cite both the **ROMPLUS** source bulletin
(NIEP) and the **RoSE** dataset paper that accompanies this compilation. See
`docs/DATASET.md` for the native HDF5 attribute schema and
`docs/SEISBENCH_FORMAT.md` for the RoSE metadata column reference.
