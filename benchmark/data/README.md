# Test datasets

The benchmark runs on two test sets — **RoSE** (Romanian local earthquakes) and
the **STEAD test split** (Mousavi et al., 2019). The raw waveforms are external
and multi-GB, so this directory ships only the **index files** that pin down
exactly which traces make up each test set. Point the benchmark scripts at the
waveforms with `--rose-dir` / `--stead-dir` (or set `ROSE_DATA_DIR` / `STEAD_DIR`).

**TL;DR**:

```bash
# 1. download both datasets (see RoSE waveforms / STEAD waveforms below)
export ROSE_DATA_DIR=/path/to/rose
export STEAD_DIR=/path/to/STEAD/benchmark_stead
# 2. run the benchmark (writes results/*.csv)
bash ../regenerate_results.sh                # or: --num-test 200 for a quick subset
```

## Index files in this directory

| File | Rows | Columns | Description |
|---|---:|---|---|
| `rose_test_index.csv`   | 32 374  | `trace_name, source_id, split, year` | RoSE traces assigned to the **test** split. Derived from the repo-root `rose_split_index.csv` (the full deterministic train/dev/test assignment produced by `training/build_rose_split_index.py`, salt `ROMPLUS-singleEQ-v1`). All rows have `split == test`; `trace_name` is the SeisBench bucket key into `waveforms*.hdf5`. |
| `stead_test_index.csv`  | 103 040 | `trace_name, p_sample, s_sample, p_sample_original, s_sample_original, ps_residual_sec, ps_group` | STEAD **event** test traces with P/S sample positions (in the 27 000-sample, 100 Hz, ENZ window used by `benchmark/bench_stead_test.py`). |
| `stead_noise_index.csv` | 23 526  | `trace_name` | STEAD **noise** test traces — the real-negative pool for trace-level event-vs-noise (T1) metrics. |

These files are versioned with the code so the benchmark composition is fixed
and reproducible; only the waveform arrays themselves need to be fetched.

## RoSE waveforms

* **Format**: SeisBench `WaveformDataset` — yearly `waveforms{YEAR}.hdf5` +
  `metadata{YEAR}.csv` + `metadata.csv` manifest (channels Z, N, E at 100 Hz,
  60 s / 6000-sample windows).
* **Source**: published RoSE bundle (TRANSFORM² consortium release / Zenodo).
* **Build the split column yourself**: after downloading, run
  `python training/build_rose_split_index.py --rose-dir /path/to/rose`. It
  writes the `split` column into each `metadata{YEAR}.csv` (read directly by
  `WaveformDataset.train_dev_test()`) and re-emits `rose_split_index.csv` /
  `rose_split_index.json`. Re-running with the same salt is idempotent.
* Then `export ROSE_DATA_DIR=/path/to/rose` and pass `--rose-dir "$ROSE_DATA_DIR"`.

## STEAD waveforms

* **Format**: a flat directory with the two index CSVs above and per-trace
  `.npy` arrays:

  ```
  <stead-dir>/metadata.csv                       # == stead_test_index.csv
  <stead-dir>/metadata_noise.csv                 # == stead_noise_index.csv
  <stead-dir>/waveforms/<trace_name>.npy         # (27000, 3) ENZ float32 per event
  <stead-dir>/noise_waveforms/<trace_name>.npy   # (27000, 3) ENZ float32 per noise
  ```

  (The two CSVs in `<stead-dir>/` are byte-identical to the ones shipped here —
  copy them in, or symlink.)
* **Source**: STEAD original release — <https://github.com/smousavi05/STEAD>.
  Convert the upstream HDF5 to the flat `.npy` layout above (one array per
  `trace_name`, resampled to 100 Hz, channel order E, N, Z).
* Then `export STEAD_DIR=/path/to/stead/benchmark_stead` and pass
  `--stead-dir "$STEAD_DIR"`.

## Sizes

| Dataset | Traces | Waveform disk | Index file (here) |
|---|---:|---:|---|
| RoSE test    |  32 374 | ~13 GB | `rose_test_index.csv` (1.5 MB) |
| STEAD events | 103 040 | ~30 GB | `stead_test_index.csv` (5.4 MB) |
| STEAD noise  |  23 526 |  ~6 GB | `stead_noise_index.csv` (0.6 MB) |
| **Total**    | **158 940** | **~49 GB** | |

If disk is tight, run a subset with `--num-test N` (caps each pool at `N`
traces — for the RoSE pool it draws a random sample; for the STEAD pools it
takes the *first* `N` traces, so a small slice can be unrepresentative). A
few-thousand-trace subset is enough to spot-check that the checkpoints load
and pick sanely; for the published numbers use the full pools (the committed
`../../results/*.csv`).
