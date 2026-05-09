# RoSE native HDF5 schema (ROMPLUS-derived)

This document describes the *intermediate* yearly HDF5 files that
`romania/scripts/create_HDF5.py` writes from the relocated ROMPLUS catalog.
The published dataset (`data/rose/`) is built from these by
`romania/scripts/convert_HDF5_to_seisbench.py`. See `SEISBENCH_FORMAT.md`
for the published-format schema.



Reference for the files at
`outputs/hdf5_romplus/{YEAR}_ROMPLUS_{counts,physical}.h5`, written by
`scripts/create_HDF5.py`.

## Hierarchy

```
/<event_index>                                       group, attrs = catalog row
    /waveforms                                       group
        /<NET.STA.LOC.BAND>                          group, attrs = pick + geometry
            E   shape (npts,) float32   attrs       per-channel metadata
            N   shape (npts,) float32   attrs
            Z   shape (npts,) float32   attrs
```

### `event_index`

`YYYY_NNNNNNN`, e.g. `2014_0000001`. The four-digit prefix matches the file's
`year` and the seven-digit suffix is the original ROMPLUS event id.

### `<NET.STA.LOC.BAND>`

Band code is the first two characters of the channel (e.g. `BH`, `HH`, `EH`).
Only one band per station is kept per event: the strict-band selection from the
catalog SAC archive.

## Event attributes

| Attribute | Type | Description |
|---|---|---|
| `event_index` | str | `YYYY_NNNNNNN` |
| `time` | str | Catalog origin time (relocated, hypoDD3D) |
| `event_time_utc` | str | Same, kept as ISO string |
| `latitude`, `longitude`, `depth` | float | Hypocenter (deg, deg, km) |
| `magnitude` | float | Local magnitude |
| `source` | str | Hypocenter solution: `hypoDD_3D`, `SSST`, `ROMPLUS` |
| `raw_time`, `raw_lat`, `raw_lon`, `raw_depth` | str/float | Original ROMPLUS solution |
| `Nsta`, `Npha` | int | Stations / phases used by the relocation |
| `gap` | float | Azimuthal gap in degrees |
| `TRes_MAE`, `TRes_MAD` | float | Travel-time residual statistics (s) |
| `year` | int | Convenience |

## Station attributes

| Attribute | Type | Description |
|---|---|---|
| `strict_station_id_band` | bool | Whether the band code was kept strict |
| `reference_starttime_utc` | str | Common UTC start across the three components |
| `sampling_rate_hz` | float | Always 100.0 |
| `npts` | int | Trace length on the reference axis |
| `missing_components` | str | Comma list, empty if all three present |
| `status_counts` | str | `ok` |
| `back_azimuth`, `epi_dist`, `hyp_dist` | float | Path geometry to source (deg, km, km) |
| `p_arrival_utc`, `s_arrival_utc` | str | Final pick UTC time |
| `p_arrival_sample`, `s_arrival_sample` | int | Sample index on the reference axis |
| `p_arrival_in_trace`, `s_arrival_in_trace` | bool | Whether the pick falls inside the trace |
| `pick_source_p`, `pick_source_s` | str | `manual` or `repick` (RED-PAN) |
| `final_snr_p`, `final_snr_s` | float | SNR of the chosen pick |
| `p_*`, `s_*` (long form) | various | Verbatim copy of the pick CSV row, prefixed |

The `p_*` / `s_*` long-form attributes preserve the entire pick row from
`Enhanced_ROMPLUS_picks.csv` (e.g. `p_manual_arrival`, `p_theo_arrival`,
`p_TRes`, `p_back_azimuth`, …) so no information is lost.

## Channel datasets

| Attribute | Type | Description |
|---|---|---|
| `channel_code` | str | Source channel, e.g. `BHZ` |
| `trace_starttime_utc` | str | Equal to `reference_starttime_utc` (alignment) |
| `units` | str | `counts` for `_counts.h5`, `M/S`/`M/S**2` for `_physical.h5` |
| `sensitivity_value` | float | Only on physical-units file: divisor that was applied |

## Alignment & resampling

For each station, all three SAC components are trimmed to their common union
window with the per-component median padded into gaps, then linearly
interpolated onto a uniform 100 Hz time axis. This gives a single
`(npts,)` length per station with the same reference start across components,
which is the invariant the SeisBench converter relies on to stack ENZ.

## Counts vs. physical

* `_counts.h5` keeps the raw integer-valued sample counts as `float32`.
* `_physical.h5` divides each component by its `instrument_sensitivity` from
  the StationXML response valid at the trace start time, giving velocity
  (`M/S`) for seismometers and acceleration (`M/S**2`) for accelerometers.
  Stations missing a response get `status_physical = "missing_response"` and
  no physical dataset.
