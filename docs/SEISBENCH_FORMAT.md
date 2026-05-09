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
| `trace_chunk` | Year string, matches the chunk file |
| `trace_start_time` | UTC ISO of sample 0 |
| `trace_sampling_rate_hz` | 100.0 |
| `trace_dt_s` | 0.01 |
| `trace_npts` | Length of the trace on the reference axis |
| `trace_category` | `earthquake_local` |
| `trace_completeness` | `ok` |
| `trace_component_order` | `ZNE` (canonical SeisBench order) |

### Picks
| Column | Description |
|---|---|
| `trace_p_arrival_sample`, `trace_s_arrival_sample` | sample index, `-1` if missing |
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
| `path_back_azimuth_deg` | source → station back-azimuth |
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
| `source_magnitude`, `source_magnitude_type` | M, ml |
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

`RoSE(path)` automatically discovers the `chunks` file and concatenates
all chunks into one logical dataset. To restrict to a subset of years:

```python
data = RoSE(path)
data.filter(data.metadata["trace_chunk"].isin(["2018", "2019", "2020"]),
            inplace=True)
```

For a train/dev/test split, populate a `split` column in the metadata before
training (SeisBench's `train_dev_test()` reads it):

```python
data.metadata["split"] = "train"
data.metadata.loc[data.metadata["trace_chunk"] == "2023", "split"] = "dev"
data.metadata.loc[data.metadata["trace_chunk"] == "2024", "split"] = "test"
```

## Conversion notes

* The converter reads only the alignment-finished waveforms produced by
  `create_HDF5.py`; **no** resampling or trimming happens here.
* Components are stacked **ZNE** on disk (canonical SeisBench order). No
  reordering is needed for PhaseNet / EQTransformer / PickBlue inputs.
* The published RoSE dataset stores **counts** only on disk; physical units
  are derivable per trace from the `trace_sensitivity_{e,n,z}` columns and
  the `trace_unit_physical` column.
