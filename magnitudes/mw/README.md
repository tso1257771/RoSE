# Vrancea moment magnitude (Mw)

Moment magnitude for the Vrancea catalogue (2014–2024) from S-wave spectral inversion with
SourceSpec 1.8. No external magnitude enters the values or their uncertainties, and no
depth correction is applied. USGS moment magnitudes are used for validation only. Earlier
products and the trials that led here are described in [TRIALS.md](TRIALS.md); the local
magnitude is in [`../ml/`](../ml/).

The inversion was run with SourceSpec 1.8, ObsPy 1.5, NumPy 1.26, SciPy 1.13 and
pandas 3.0. The drivers rely on pandas 3 behaviour.
[`sourcespec/requirements-sourcespec.txt`](sourcespec/requirements-sourcespec.txt)
pins the set.

## Inputs

Paths are relative to the working tree named by `ROMANIA_ROOT`, whose shape
[`../README.md`](../README.md) gives. The repository ships the coefficients,
not these inputs.

| Path | Used for |
|---|---|
| `sac/<year>/<num>/` | raw SAC waveforms per event |
| `seisbench_integration/data/rose_stationxml/_merged_for_sourcespec.xml` | merged StationXML for the spectral inversion |
| `outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_picks.csv` | P and S picks for the event bundles |
| `outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_catalog.csv` | hypoDD3D relocations and the catalogue magnitude used only for window design |
| `romania_mw/data/all_events/` | origin and picks per event, built by [`sourcespec/extract_all_events.py`](sourcespec/extract_all_events.py) from the two files above |
| `romania_mw/outputs/usgs_cross_validation.csv` | USGS moment magnitudes, validation only |

## Pipeline

| Step | Driver | Output |
|---|---|---|
| Origin and picks per event | [`sourcespec/extract_all_events.py`](sourcespec/extract_all_events.py) | `romania_mw/data/all_events/<event>/{event.yaml,picks.csv}` |
| Spectral inversion per event | [`sourcespec/run_sourcespec.py`](sourcespec/run_sourcespec.py) (config [`sourcespec/vrancea.conf`](sourcespec/vrancea.conf), window wrapper [`sourcespec/source_spec_redpan_windows.py`](sourcespec/source_spec_redpan_windows.py)) | `romania_mw/sourcespec/outputs/<event>/`, `romania_mw/outputs/sourcespec_mw.csv` |
| Per station spectral fits | [`build_station_table.py`](build_station_table.py) | `romania_mw/outputs/station_fits.csv.gz` |
| Event catalogue, uncertainties, flags, USGS validation | [`build_mw_catalog.py`](build_mw_catalog.py) | `romania_mw/outputs/Mw_catalog.csv`, `Mw_validation.yaml`, `Mw_validation.png` |
| Release catalogue with Mw and ML columns | [`build_release_catalog.py`](build_release_catalog.py) | `romania_mw/outputs/Enhanced_ROMPLUS_catalog_with_magnitudes.csv`, `romania_mw/outputs/MAGNITUDE_COLUMNS.md` |

The published run read `vrancea.conf` and the window wrapper from
`romania_mw/sourcespec/` in the working tree. The shipped `vrancea.conf` is
identical to that copy and the shipped wrapper differs from it in two docstring
lines, so `run_sourcespec.py` now reads the shipped copies. `SS_CONFIG` and
`SS_WRAPPER` name other copies.

[`docs/MAGNITUDES.md`](../../docs/MAGNITUDES.md) is a copy of
`MAGNITUDE_COLUMNS.md` edited by hand, not a driver output. No step reproduces
the repository's [`data/Enhanced_ROMPLUS_catalog.csv`](../../data/Enhanced_ROMPLUS_catalog.csv)
byte for byte either: the release driver reads the working tree copy of the
release catalogue, whose latitude, longitude and depth are given to 4 decimals,
and the committed file keeps those columns from the earlier release, with
latitude and longitude to 6 decimals, and takes only the eight magnitude
columns from the driver output. Those eight columns are identical in the two
files.

## Method

Each station spectrum is corrected for geometric spreading (1/R, hypocentral, stations
≤ 300 km) and fitted with a Brune ω² model in magnitude units,

    Y(f) = Mw − (2/3)·log10[1 + (f/fc)²] − (2/3)·log10(e)·π·f·t*

with Mw, the corner frequency fc and the attenuation t* free per station. The moment scale
follows from 4π·v³·ρ / (free-surface factor × radiation pattern) with the layered Koulakov
Vrancea velocity model, free-surface factor 2.0 and S-wave radiation pattern 0.62, and
Mw = (log10 M0 − 9.1)/1.5. The event value is the weighted mean over stations.

**Windows and bands.** The S window follows the RED-PAN-Motion rule (S−P based nominal
window with amplitude-decay truncation, one window per station), never shorter than a floor
derived from the expected corner frequency, so the low-frequency plateau of large events
stays resolved. The spectral band, pre-filter and channel selection are set per event from
the catalogue ML; ML selects windows and bands only and never enters the Mw value. The noise
window matches the signal length and is scaled rather than truncating the signal.

## Catalogue

`romania_mw/outputs/Mw_catalog.csv`: 19,188 of 19,230 events. The column
`ML_catalog_window_design` is the hypoDD3D catalogue magnitude, carried only because the
spectral windows were designed from it (its `ML_type` is `mw` for 118 events); it is not a
magnitude of this product and no downstream script reads it. Quality A 11,656, B 6,281, C 1,251
(A: ≥ 5 distinct recording sites, few t* at the lower t* bound, fc inside the search range, standard
error < 0.15; B: ≥ 3 sites). A site is counted once even when two of its instruments are
fitted separately, which `Mw_nfits` records.

    Mw_sigma = sqrt( sigma_stat² + sigma_sys(Mw)² )

`sigma_stat` is the station scatter shrunk towards the catalogue median (0.245) and divided
by √n; `sigma_sys` is 0.10 for Mw ≥ 3.8 and 0.18 for Mw ≤ 2, linear in between. It combines
the station-distance systematic (0.10) with the ±0.15 sensitivity of small-event Mw to fc,
t* and weighting choices. Median σ is 0.20 at Mw ≤ 2 and 0.11 above 3.8.

Corner frequency, stress drop and t* are reported only for Mw ≥ 3.5; below that the corner
frequency is band-limited and is not interpreted. On the 192 events with a reported fc, the
log10 fc–Mw slope is −0.32 over all of them and −0.33 for the 44 crustal ones (from the
`fc_Hz` column, i.e. `fc_mean` restricted to Mw ≥ 3.5), against −0.5 for constant stress
drop.

**Known systematic, not corrected.** Within an event, station Mw falls with hypocentral
distance (about −0.5 per decade for slab events, −0.26 to −0.31 crustal; near stations
< 25 km read about +0.12 high). An empirical distance term shifts event Mw by only −0.05
(crustal) to +0.05 (slab) and makes the USGS comparison worse, so it is not applied. Its
size is included in `sigma_sys`, and it is one-signed for small crustal events.

**Comparison with USGS (not fitted).** USGS Mww/Mwr for 17 events: USGS − Mw = +0.12 ± 0.14
(crustal +0.20 on 6 events, slab +0.08 on 11), OLS slope 0.97, 88 % within 2σ. See
`romania_mw/outputs/Mw_validation.yaml`. The offset is significant on this sample (t = 3.6), so it is an
offset against USGS rather than agreement with it, and it is not constant in time: the 7 slab
events up to 2018 give +0.14 and the 4 from 2019 onwards −0.02. The slope is indistinguishable
from 1. The comparison constrains the level between Mw 3.8 and 5.6 only.

**Time dependence, stated not corrected.** Against the local magnitude, intermediate-depth Mw
rises by 0.08 ± 0.02 per decade (events of quality A or B with ML ≥ 2.5). It is not caused by
the changing station mix: a magnitude built only from the 38 station channels that recorded in
every year differs from the published one by 0.00 ± 0.01 per decade, and the drift against ML
survives that restriction unchanged. It is therefore either in the local magnitude or in the
waveform archive behind it, and it is smaller than the per-event uncertainty of 0.11 to 0.20.

## Statistics

Crustal events (depth < 60 km) and intermediate-depth events (≥ 60 km) differ in both
completeness and b-value, so a single cut-off mixes two populations.

| | Mc | b-value | N ≥ Mc |
|---|---|---|---|
| Crustal | 2.0 | 1.51 ± 0.02 | 5,086 |
| Intermediate-depth | 2.6 (b-stability) | 0.86 ± 0.03 | 936 |

Mc is the maximum-curvature value + 0.2 (Woessner & Wiemer 2005), except for
intermediate-depth Mw where the distribution top is flat and the b-stability value is quoted
(Cao & Gao 2002); b is the Aki–Utsu maximum-likelihood estimate with half-bin correction and
Shi & Bolt (1982) uncertainty, bin 0.1. b depends on the cut-off and must be quoted with its
Mc.
