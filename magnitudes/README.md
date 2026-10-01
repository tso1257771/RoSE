# Magnitudes

The two magnitudes released with RoSE, the code that computed them, and the
tables they were computed with.

Using the released magnitudes needs nothing in this directory. They are
columns of [`data/Enhanced_ROMPLUS_catalog.csv`](../data/Enhanced_ROMPLUS_catalog.csv),
described in [`docs/MAGNITUDES.md`](../docs/MAGNITUDES.md). This directory is
here so the values can be checked, and so the same scale can be applied to an
earthquake the catalog does not contain.

| Path | What it holds |
|---|---|
| [`calibration/`](calibration/) | every coefficient of both scales, with its standard error |
| [`ml/`](ml/) | the local magnitude drivers, and [`ml/README.md`](ml/README.md) for the method |
| [`mw/`](mw/) | the moment magnitude drivers, and [`mw/README.md`](mw/README.md) for the method |
| [`regenerate_magnitudes.sh`](regenerate_magnitudes.sh) | runs the drivers in order |
| `_paths.py` | where the drivers read and write |

The importable code is `rose.magnitudes`, not this directory. It reads
`calibration/` and applies the published scales:

```python
from rose.magnitudes import load_calibration, mw_from_ml, neg_log_a0

cal = load_calibration()
neg_log_a0([100.0], [10.0], cal.atten, cal.anchor)   # 2.516, the released scale
mw_from_ml([3.4], "crustal", cal.conversion)          # (Mw, validity flag)
```

`rose/magnitudes/taiwan_ml/` and `rose/magnitudes/redpan_motion/` are copied
unchanged from their own repositories, so a reader can see what was run. Their
`__upstream__` strings give the version.

## The calibration tables

| File | What it is |
|---|---|
| `parameter_table.csv` | every scalar of both scales, with a standard error and the method that produced it |
| `station_terms.csv` | one correction per station and response epoch |
| `conversion_coefficients.csv` | the ML to Mw relation per depth regime, with its validity range |
| `anchor_events.csv` | the events the baseline shift was fitted on |
| `ml_minus_mw_by_bin.csv` | ML minus Mw in bins of Mw |
| `conversion_residuals_by_ml_bin.csv` | conversion residuals in bins of ML, which is what the validated range is read from |
| `clipped_fraction_by_magnitude.csv` | share of amplitudes clipped, against magnitude |
| `usgs_cross_validation.csv`, `Mw_validation.yaml` | comparison with USGS Mww/Mwr, reported and not fitted |
| `fit_report.json`, `conversion_report.json`, `validation_extra.json` | the full run records, including the sensitivity tests |

`tests/test_magnitudes.py` checks these tables against the released catalog.

## Rerunning the drivers

The repository ships the tables, not their inputs: the Wood--Anderson
amplitudes, the spectral fits and the waveform archive are tens of gigabytes.
Rerunning therefore needs a working tree assembled separately, named by
`ROMANIA_ROOT`:

```bash
export ROMANIA_ROOT=/path/to/romania
./magnitudes/regenerate_magnitudes.sh            # the two cheap stages
./magnitudes/regenerate_magnitudes.sh --help     # every stage, in order
```

The tree has this shape. A driver that cannot find its input stops and names
the file.

```
$ROMANIA_ROOT/
    outputs/reloc_results_hypoDD3D/
        Enhanced_ROMPLUS_catalog.csv   hypoDD3D origins              (read by extract_all_events.py)
        Enhanced_ROMPLUS_picks.csv     P and S picks                 (read by extract_all_events.py)
    romania_ml/outputs/
        amplitudes/              Wood--Anderson amplitudes      (extract_wa_amplitudes.py)
        observations.csv         amplitudes with geometry and QC (build_observations.py)
        fit_unanchored/          -log A0 shapes, station terms   (fit_ml.py)
        fit/                     the released ML                 (anchor_ml_to_mw.py)
        conversion/              the ML to Mw relation           (fit_ml_to_mw_conversion.py)
    romania_mw/
        data/all_events/         origin and picks per event      (extract_all_events.py)
        sourcespec/outputs/      one inversion per event         (run_sourcespec.py)
        outputs/
            sourcespec_mw.csv    collected spectral fits
            station_fits.csv.gz  per station fits                (build_station_table.py)
            Mw_catalog.csv       the released Mw                 (build_mw_catalog.py)
    seisbench_integration/data/
        rose/                    the waveform dataset
        rose_stationxml/         instrument responses
    sac/                         the waveform archive
```

Of these, the two that are published are `Mw_catalog.csv` and the contents of
`romania_ml/outputs/fit/`. The data archive listed in the repository root
`README.md` holds them, along with `station_fits.csv.gz` and the collected
spectral fits, which is enough to rerun the `mw`, `anchor` and `conversion`
stages without the waveforms.

## What the drivers will not reproduce exactly

Rerunning gives the published values to about 1 part in 10^7, not bit for bit.
The sparse inversion, the orthogonal distance regression and the bootstraps all
iterate to a tolerance, and their last digits move with the SciPy and NumPy
build. The published tables were written with the versions recorded in
`calibration/fit_report.json` and `calibration/conversion_report.json`. Nothing
at that level reaches a reported magnitude, which is given to two decimals.

The `release` stage does not reproduce
[`data/Enhanced_ROMPLUS_catalog.csv`](../data/Enhanced_ROMPLUS_catalog.csv)
byte for byte. It reads the working tree copy of the release catalog, whose
latitude, longitude and depth are given to 4 decimals. The committed file keeps
those columns from the earlier release, with latitude and longitude to 6
decimals, and takes only the eight magnitude columns from the driver output.
Those columns are identical in the two files.
