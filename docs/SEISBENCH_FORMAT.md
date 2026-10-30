# RoSE in SeisBench format

Reference for the dataset directory produced by
`rose.convert.convert_all`, e.g.

```
data/rose/
    chunks                 text file, one chunk name (year) per line
    metadata2014.csv       SeisBench metadata for chunk "2014"
    waveforms2014.hdf5     HDF5 with bucketed waveform arrays
    metadata2015.csv
    waveforms2015.hdf5
    ...
```

## Waveforms HDF5

```
/data/bucket0    shape (B, 3, W)  float32   B traces, ZNE stacked, length W
/data/bucket1    ...
...
/data_format/component_order        b'ZNE'
/data_format/dimension_order        b'CW'
/data_format/sampling_rate          100.0
/data_format/measurement            b'counts' | b'physical'
/data_format/unit                   b'counts' | b'M/S' | b'M/S**2'
/data_format/instrument_response    b'raw' | b'removed'
```

Each row in `metadata{YEAR}.csv` carries `trace_name = "bucket{i}$row,:3,:N"`,
which SeisBench uses to slice the right window out of the bucket array.

## Metadata columns

### Trace
| Column | Description |
|---|---|
| `trace_name` | SeisBench bucket address (auto) |
| `trace_name_original` | `{event_index}_{NET.STA.LOC.BAND}` (pre-bucket) |
| `trace_chunk` | Chunk name, matches the chunk file: the year, or `noise{YEAR}` for a noise window |
| `trace_start_time` | UTC ISO of sample 0 |
| `trace_sampling_rate_hz` | 100.0 |
| `trace_dt_s` | 0.01 |
| `trace_npts` | Length of the trace on the reference axis |
| `trace_category` | `earthquake_local`, or `noise` for a noise window |
| `trace_completeness` | `ok` |
| `trace_component_order` | `ZNE` (canonical SeisBench order) |

### Picks
| Column | Description |
|---|---|
| `trace_p_arrival_sample`, `trace_s_arrival_sample` | sample index on the trace axis; empty (NaN) when there is no pick (SeisBench reads NaN as "no arrival"; dataset version 1 wrote `-1`, which its labellers read as an onset at sample 0); a negative value is a pick shortly before the trace start |
| `trace_p_arrival_time`, `trace_s_arrival_time` | UTC ISO |
| `trace_p_status`, `trace_s_status` | `manual` or `repick` |
| `trace_p_snr_db`, `trace_s_snr_db` | SNR of the chosen pick |
| `trace_p_in_window`, `trace_s_in_window` | bool |
| `trace_p_residual_s`, `trace_s_residual_s` | (final − theoretical) travel-time residual |
| `trace_p_theoretical_time`, `trace_s_theoretical_time` | NLLoc theoretical arrival, UTC |
| `trace_p_manual_time`, `trace_s_manual_time` | manual catalog pick (if any), UTC |

### Units / instrument response
| Column | Description |
|---|---|
| `trace_units` | what is stored on disk; always `"counts"` |
| `trace_unit_physical` | derived target, `"M/S"` or `"M/S**2"` |
| `trace_sensitivity_e`, `_n`, `_z` | per-component divisor (counts / sens → physical) |
| `trace_status_physical` | `ok`, `partial_response`, `missing_response` |
| `trace_missing_response_components` | comma list of components without response |

Use `RoSE.get_sample_physical(idx)` to divide by sensitivity in one call;
it raises on missing-response traces so you can filter or skip cleanly.

### Path
| Column | Description |
|---|---|
| `path_back_azimuth_deg` | back-azimuth, measured at the station and pointing to the source. This is the sense ObsPy's rotation routines expect. Checked against the coordinates: the released values match the station to source azimuth to 0.000 degrees |
| `path_ep_distance_km` | epicentral distance |
| `path_hyp_distance_km` | hypocentral distance |

### Station
| Column | Description |
|---|---|
| `station_network_code`, `station_code`, `station_location_code` | NET, STA, LOC |
| `station_channel` | Band (e.g. `BH`); per-component letter is appended at read time |
| `station_latitude_deg`, `station_longitude_deg`, `station_elevation_m` | from `RO_station.csv` (NaN if no station CSV was passed to the converter) |
| `station_region` | descriptive (empty string if no station CSV was passed to the converter) |

### Source (event)
| Column | Description |
|---|---|
| `source_id` | `event_index` |
| `source_origin_time` | catalog (relocated) origin time |
| `source_latitude_deg`, `source_longitude_deg`, `source_depth_km` | hypocenter (hypoDD3D) |
| `source_magnitude`, `source_magnitude_type` | preferred magnitude and the scale it is on: `mw` where a moment magnitude was measured, otherwise `ml`, empty if neither |
| `source_magnitude_uncertainty` | `Mw_sigma` when the preferred magnitude is Mw |
| `source_mw`, `source_mw_sigma`, `source_mw_quality`, `source_mw_nstations`, `source_mw_fc_hz` | moment magnitude and its quality fields. `source_mw_nstations` is `-1` when the event has no moment magnitude |
| `source_ml`, `source_ml_nstations`, `source_ml_warning` | local magnitude and its quality fields. The two integer fields are `-1` when the event has no local magnitude |
| `source_ml_romplus`, `source_mw_romplus` | NIEP bulletin magnitudes, for continuity only |
| `source_catalog` | `hypoDD_3D` / `SSST` / `ROMPLUS` |
| `source_gap_deg` | azimuthal gap |
| `source_tres_mae_s`, `source_tres_mad_s` | travel-time residual statistics |
| `source_nsta`, `source_npha` | stations / phases used |
| `source_origin_time_raw` | original (pre-relocation) ROMPLUS time |
| `source_latitude_raw_deg`, `source_longitude_raw_deg`, `source_depth_raw_km` | pre-relocation hypocenter |

## Bucketing

The default `GeometricBucketer` packs traces of similar length into the same
bucket (one HDF5 dataset). This trades a tiny amount of zero-padding for a
large reduction in HDF5 metadata overhead, which is the bottleneck for fast
random reads during training. With `bucket_size=1024` and ~19k traces per
year you end up with on the order of ~30 buckets per chunk.

## Working with chunks

`RoSE(path)` reads the chunk names from the `chunks` file (or, without that
file, from the file names) and concatenates the chunks into one logical
dataset. From dataset version 2 on there are also noise chunks (`noise2014` …
`noise2024`); `RoSE(path)` leaves them out, `RoSE(path, include_noise=True)`
loads both classes, and
`RoSE(path, chunks=[f"noise{y}" for y in range(2014, 2025)])` the noise alone.
Toolkit versions before v0.4.0, and a plain `seisbench.data.WaveformDataset`,
load every chunk; pass them `chunks=[str(y) for y in range(2014, 2025)]` to
keep the noise out. To restrict to a subset of years:

```python
data = RoSE(path)
data.filter(data.metadata["trace_chunk"].isin(["2018", "2019", "2020"]),
            inplace=True)
```

The published metadata carry a `split` column, `train`, `val` or `test`, set
per event by `rose.splits.hash_split` (salt `ROMPLUS-singleEQ-v1`), so all
traces of one earthquake share a split and a noise window has the split of its
reference event. SeisBench calls the validation split `dev`: its own
`WaveformDataset.dev()` finds no trace here, while `RoSE.dev()` and
`RoSE.train_dev_test()` take `val` and `dev` alike. For another split,
overwrite the column, e.g. by year:

```python
year = data.metadata["trace_chunk"].astype(str).str[-4:]   # "2023" for chunks 2023 and noise2023
data.metadata["split"] = "train"                            # replaces the published split
data.metadata.loc[year == "2023", "split"] = "dev"
data.metadata.loc[year == "2024", "split"] = "test"
```

## Conversion notes

* The converter reads only the alignment-finished waveforms produced by
  `create_HDF5.py`; **no** resampling or trimming happens here.
* Components are stacked **ZNE** on disk (canonical SeisBench order). No
  reordering is needed for PhaseNet / EQTransformer / PickBlue inputs.
* The published RoSE dataset stores **counts** only on disk; physical units
  are derivable per trace from the `trace_sensitivity_{e,n,z}` columns and
  the `trace_unit_physical` column.

## Known issues in the earthquake traces

**One wrong sample at the end of 3,067 BH traces.** In dataset versions 1 to
3, one or more components of 3,067 earthquake traces (1.4 %) have one wrong
sample at their end. All are BH traces recorded at 20 or 50 Hz and resampled
to 100 Hz. The noise windows were resampled by linear interpolation and are not
affected.

| End sample | Traces | train | val | test |
|---|---|---|---|---|
| infinite | 928 | 645 | 134 | 149 |
| finite; its step from the previous sample is more than 1,000 times the largest step elsewhere in the trace | 1,982 | 1,382 | 298 | 302 |
| finite; that step is 10 to 1,000 times the largest step | 157 | 112 | 19 | 26 |
| total | 3,067 | 2,139 | 451 | 477 |

These counts come from reading every trace on 2026-10-30. In a random sample
of 3,300 other traces, 9 BH traces had a milder case of the same defect, so
about 600 more traces may have one.

*Cause.* The data build resampled these traces with ObsPy's
`Trace.interpolate(100)`. Its default method, `weighted_average_slopes`, ends
in a C routine that does not check array bounds. When rounding of the absolute
time stamps puts the last output time a few millionths of a sample after the
last input sample, the routine reads one value past the end of its input
arrays. Whatever memory it reads sets the last output sample, which can be
infinite, NaN, very large, or slightly wrong. The build then replaced
non-finite samples with the trace's `np.nanmean`. That mean is itself infinite
when the trace holds an infinity, so the infinities stayed, and NaN samples
became the trace mean. The routine is unchanged in the current ObsPy version.

*Where.* The wrong sample is the last sample of the component. In a few traces
whose component ends in median padding, it is the last sample before the
padding.

*Effect.* Normalisation turns a trace with an infinite sample into NaN, so a
model returns no picks on it. A very large finite sample dominates the
amplitude near the end. After the 1-45 Hz band-pass used in the benchmark, the
nine models find 18 to 44 % fewer of the labels on such traces. In the
published benchmark tables, the 477 affected test traces lower recall by 0.003
to 0.006 for every model, and the order of the models does not change. Training
code that does not check for non-finite values can get a NaN loss when a window
reaches the end of an affected trace.

*Repair.* The function below replaces the wrong sample with the sample before
it. On the 3,067 traces it changes one sample in each of 6,232 components and
leaves no non-finite value. On 3,300 random other traces it changed only 10 BH
components, all with the same end defect.

```python
import numpy as np


def repair_trace_end(x, factor=10.0):
    """Repair the end-sample defect of RoSE dataset versions 1-3 in one (3, npts) trace."""
    x = np.array(x, dtype=np.float64)
    for c in x:
        n = len(c)
        r = n - 1
        while r > 0 and c[r - 1] == c[n - 1]:
            r -= 1  # c[r:] is the constant run at the end
        cand = sorted({i for i in (r - 1, n - 2, n - 1) if i >= 1})
        body = c[:cand[0]]
        body = body[np.isfinite(body)]
        lim = factor * (np.max(np.abs(np.diff(body))) if body.size > 1 else 0.0)
        for i in cand:
            if not np.isfinite(c[i]) or abs(c[i] - c[i - 1]) > lim:
                c[i] = c[i - 1]
    return x
```

Apply it to the output of `dataset.get_waveforms(i)`, or register it as the
first augmentation of a SeisBench generator, before any window is cut:

```python
@generator.augmentation
def repair(state_dict):
    x, meta = state_dict["X"]
    state_dict["X"] = (repair_trace_end(x).astype(x.dtype), meta)
```

## Noise chunks

Dataset version 3 (10.5281/zenodo.23681945) carries 89,693 noise windows in
the same layout, chunk names `noise2014` … `noise2024`
(`metadatanoise{YEAR}.csv`, `waveformsnoise{YEAR}.hdf5`). Version 2 has files
with the same names, but its noise metadata do not match its noise waveforms;
use version 3 or later. Each window is 100 s (10,000 samples, ZNE, counts)
taken from the 200 s of continuous data before the catalog P of a RoSE event
at one of its stations. Two windows per record were candidates: `pre100` ends
2 s before the reference P (the earlier of the repicked and the analyst P), or
at the end of the data when the record stops earlier (68 windows); `pre200`
starts at the record start, about 200 s before the reference P. The two
overlap by about 2 s.

A candidate was dropped when (1) any of four phase pickers (RED-PAN 60 s,
PhaseNet+, and the original EQTransformer and PhaseNet weights through
SeisBench 0.12.6) had a P or S probability peak above 0.1 inside it; (2) a
catalog pick (repicked or analyst, P or S) of another event at the same
station fell inside it or in the 60 s before it; (3) it overlapped the span
from the P arrival to the end of another event's earthquake trace at the same
station; (4) for `pre100`, a picker peak above 0.3 lay in the 30 s before it;
(5) it started within 3 h (6 h for M >= 7) after a large regional earthquake
(Aegean 2014, Amatrice and Norcia 2016, Bodrum-Kos 2017, Durres 2019, Samos
2020, the two Kahramanmaras events and Hatay 2023); (6) it overlapped a kept
window of another event at the same station or an earthquake trace of another
event in a different split. All surviving `pre100` windows were kept, then
`pre200` windows in increasing order of `trace_screen_record_max_score` up to
the size budget: 55,845 `pre100` and 33,848 `pre200` windows. That score is 0
for a record without any picker peak above 0.1, so those records came first,
in event and station order (26,862 of the `pre200` windows); the highest kept
score is 0.152.

The pickers see a 200 s record in windows: EQTransformer leaves the last 4 s
of a `pre100` window and the first 5 s of a `pre200` window unscreened,
PhaseNet the last 1.5 s and the first 2.5 s; the other two pickers cover
them. "Noise" therefore means *no detected local arrival*, not *no
earthquake*: weak events below the pickers' sensitivity can remain, the
false-detection rate of the four screening pickers on this set is zero by
construction, every window lies 2-200 s before a catalogued event (the
sampling follows the seismicity, 2023 included), and the station and band
mix differs from the earthquake traces (more HH, fewer BH).

The metadata columns are those above with `trace_category = "noise"`, the
pick sample and time columns empty, `trace_*_in_window` False, and `path_*`
and `source_*` empty, except the five integer columns `source_nsta`,
`source_npha`, `source_mw_nstations`, `source_ml_nstations` and
`source_ml_warning`, which hold `-1`. The earthquake tables use `-1` for
"none" only in the last three; their `source_nsta` and `source_npha` are never
`-1`. `station_*`, `trace_sensitivity_*`, `trace_unit_physical`,
`trace_status_physical` and `split` are copied from the earthquake trace of
the same event and station (same instrument at the same time), so a noise
window of a test event is in the test split; windows whose
station has no earthquake trace for that event take coordinates and
sensitivities from the StationXML archive. `trace_name_original` is
`{event_index}_{NET.STA.LOC.BAND}_pre100` or `_pre200`. Additional columns:

| Column | Description |
|---|---|
| `trace_window_position` | `last` (`pre100`) or `first` (`pre200`) |
| `trace_reference_event_id` | the RoSE event whose P follows the window |
| `trace_reference_p_time` | the reference P arrival (earlier of repicked and analyst P), UTC |
| `trace_end_to_reference_p_s` | seconds from the last sample to the reference P |
| `trace_screen_rule`, `trace_screen_pick_threshold` | the selection rule and its score threshold (0.1) |
| `trace_screen_record_max_score` | the highest score of the P and S peaks above 0.1 that the four pickers found more than 1 s before the end of the 200 s record; 0 when there is no such peak (55,879 windows) |
| `trace_screen_max_{p,s}_{redpan_60s,phasenetplus,eqtransformer,phasenet}` | per-picker maxima over the 200 s record |

Load the noise alone with `RoSE(path, chunks=[f"noise{y}" for y in range(2014, 2025)])`,
both classes with `RoSE(path, include_noise=True)`, and select on `trace_category`.

Known issues in the noise windows of version 3:

* `station_location_code` is `0.0` instead of `00` in the two IU.ANTO windows
  `2016_0000415_IU.ANTO.00.BH_pre100` and `2022_0000708_IU.ANTO.00.BH_pre100`.
  Their horizontal channels BH1 and BH2 are stored as E and N, without a check
  of their orientation.
* Rule (3) used only earthquake traces with a P pick. Five windows overlap, by
  8-25 s, the part after the S pick of another event's trace at the same
  station that has an S pick but no P pick; each starts 65-82 s after that S
  pick, and all five are in the train split:
  `2014_0001488_RO.CVDA..EH_pre200`, `2014_0001689_RO.ICOR..HH_pre100`,
  `2021_0000870_RO.SURR..HH_pre200`, `2023_0000194_RO.MDVR..BH_pre100` and
  `2023_0004586_RO.DEV..HH_pre100`.
* `trace_status_physical` differs from the earthquake traces of the same
  station and band in 29 windows: 12 (RO.BIR, RO.BUC1, RO.ODBI and RO.TUDR, HH)
  are `missing_response` where those traces have a response, and 17 (GE.PSZ
  BH; GE.TIRR BH, HH and HN) are `ok` where those traces are
  `missing_response`. Which is right was not checked. `station_region` is
  empty in 345 windows.
