# Vrancea moment magnitude (Mw)

Moment magnitude for the Vrancea catalogue (2014–2024) from S wave spectral inversion with
SourceSpec 1.8. No external magnitude enters the values or their uncertainties, and no
depth correction is applied. USGS moment magnitudes are used for validation only. Earlier
products and the trials that led here are described in [TRIALS.md](TRIALS.md). The local
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

## Steps

| Step | Driver | Output |
|---|---|---|
| Origin and picks per event | [`sourcespec/extract_all_events.py`](sourcespec/extract_all_events.py) | `romania_mw/data/all_events/<event>/{event.yaml,picks.csv}` |
| Spectral inversion per event | [`sourcespec/run_sourcespec.py`](sourcespec/run_sourcespec.py) (config [`sourcespec/vrancea.conf`](sourcespec/vrancea.conf), window wrapper [`sourcespec/source_spec_redpan_windows.py`](sourcespec/source_spec_redpan_windows.py)) | `romania_mw/sourcespec/outputs/<event>/`, `romania_mw/outputs/sourcespec_mw.csv` |
| Per station spectral fits | [`build_station_table.py`](build_station_table.py) | `romania_mw/outputs/station_fits.csv.gz` |
| Event catalogue, uncertainties, flags, USGS validation | [`build_mw_catalog.py`](build_mw_catalog.py) | `romania_mw/outputs/Mw_catalog.csv`, `Mw_validation.yaml`, `Mw_validation.png` |
| Release catalogue with Mw and ML columns | [`build_release_catalog.py`](build_release_catalog.py) | `romania_mw/outputs/Enhanced_ROMPLUS_catalog_with_magnitudes.csv`, `romania_mw/outputs/MAGNITUDE_COLUMNS.md` |
| Window test: Mw from plain SourceSpec windows (below) | [`sourcespec/window_test.py`](sourcespec/window_test.py) | `romania_mw/sourcespec/window_test/{fixed,var}/<event>/`, `romania_mw/outputs/window_test_{sample,comparison}.csv`, `window_test_summary.json`; copied to [`../calibration/`](../calibration/) |

The published run read `vrancea.conf` and the window wrapper from
`romania_mw/sourcespec/` in the working tree. The shipped `vrancea.conf` is
identical to that copy and the shipped wrapper differs from it in two docstring
lines, so `run_sourcespec.py` now reads the shipped copies. `SS_CONFIG` and
`SS_WRAPPER` name other copies.

[`docs/MAGNITUDES.md`](../../docs/MAGNITUDES.md) is a copy of
`MAGNITUDE_COLUMNS.md` edited by hand, not a driver output. No step reproduces
the local combined event table byte for byte. The release driver reads the
working copy of the event table, whose latitude, longitude, and depth are given
to 4 decimals. The final table keeps those columns from the earlier
calculation, with latitude and longitude to 6 decimals, and takes only the
eight magnitude columns from the driver output. Those eight columns are
identical in the two tables.

## Method

The spectral part is SourceSpec 1.8 as published (Satriano 2024, SourceSpec v1.8,
doi:10.5281/zenodo.3688587; equations in its documentation, sections "Theoretical
Background" and "Signal Processing"). Only the time windows are changed, as described below.
For each station:

1. The displacement amplitude spectra of the three components over the S window are
   combined as the root sum of squares, S(f).
2. S(f) is corrected for geometric spreading, G(r) = r (hypocentral distance, stations
   ≤ 300 km), and converted to moment and then magnitude units:

       M0(f) = 4π·(ρ_h·ρ_r)^(1/2)·β_h^(5/2)·β_r^(1/2) · r · S(f) / (F·R_θφ)
       Y(f)  = (2/3)·[log10 M0(f) − 9.1]

   ρ is density and β is S wave velocity, at the hypocentre (h) and at the receiver (r).
   F = 2.0 is the free surface factor and R_θφ = 0.62 the average S wave radiation pattern.
   The constant 9.1 (M0 in N·m) is SourceSpec's. Hanks and Kanamori's 10.7 for M0 in
   dyne·cm corresponds to 9.05 and would give Mw 0.03 higher.
3. Y(f) is fitted with a Brune ω² source and one attenuation term:

       Y(f) = Mw − (2/3)·log10[1 + (f/fc)²] − (2/3)·log10(e)·π·f·t*

   with Mw, the corner frequency fc (0.05–40 Hz) and t* (0.005–0.15 s) free (TNC
   algorithm). Each frequency is weighted by the log of its spectral S/N. The band is
   0.5–35 Hz on broadband, 1–35 Hz on short period and 0.5–40 Hz on accelerometer channels,
   lower for large events (below). A record needs S/N ≥ 1.5 in the time window and on
   average over 0.1–20 Hz.
4. The event Mw is SourceSpec's weighted mean. Station values outside 1.5 × IQR are
   rejected and the rest are weighted by 1/σ² of their fit.

ρ_h and β_h come from the layered Koulakov Vrancea model in
[`sourcespec/vrancea.conf`](sourcespec/vrancea.conf) (β 3.20 km/s at the surface to
4.83 km/s below 70 km, ρ 2500 to 3300 kg/m³). `vs_stations` and `rho_stations` are not set,
so SourceSpec takes ρ_r and β_r from the same model at the station depth. That is the top
layer, 3.20 km/s and 2500 kg/m³, at every station. Compared with using the hypocentre values
at both ends (4π·ρ·β³), the receiver term lowers Mw by (1/3)·log10[ρ_h·β_h / (ρ_r·β_r)]:
≤ 0.01 for sources shallower than 20 km and 0.10 for sources at 60 km or deeper.

**Windows and bands (the change to SourceSpec).**
[`sourcespec/source_spec_redpan_windows.py`](sourcespec/source_spec_redpan_windows.py)
replaces four SourceSpec functions. Everything else runs unchanged.

* S window: starts 1 s before S (or (S−P)/2 before it, if shorter), as in SourceSpec. From S
  it runs to min(S + 1.5·(S−P), P + 80 s) and is shortened where the envelope stops decaying
  (RED-PAN-Motion rule, at least max(2 s, 0.3·(S−P)), one window per station). The whole
  window is then at least the floor clip(9/fc_exp, 5, 30) s. SourceSpec itself uses one
  fixed length, `win_length`.
* Noise window: the same length, ending 1 s before P. If less pre-P data exist, the noise is
  zero padded and scaled by √(n_signal/n_noise). SourceSpec would cut the signal to the noise
  length instead.
* Time domain S/N is taken per sample, because the two windows can differ in length.
* Records beyond 300 km hypocentral distance are skipped.

The floor, band, pre-filter and channel selection are set per event from the catalogue ML
through fc_exp, the corner frequency of a Brune source with a 1 MPa stress drop
([`sourcespec/run_sourcespec.py`](sourcespec/run_sourcespec.py)). The floor exceeds 5 s for
203 events (ML ≥ 3.9), the band is lowered and clipping detection is on for 256 events
(ML ≥ 3.8), and short period EH/SH channels are dropped for 57 events (ML ≥ 4.4). ML never
enters the Mw value.

**Window test.** [`sourcespec/window_test.py`](sourcespec/window_test.py) reruns 302 events
with unmodified SourceSpec and otherwise identical settings: 40 crustal and 40 intermediate
depth events drawn at random (fixed seed) in each class Mw < 2, 2–3 and 3–4, all 59 events
with Mw ≥ 4, and the three USGS comparison events not drawn otherwise. With SourceSpec's
default fixed window (`win_length` = the floor, 5 s for most events), Mw is lower than the
release by a median of 0.05, and by 0.18 for crustal events of Mw 3–4, and USGS − Mw becomes
0.16 ± 0.17 (release 0.12 ± 0.14). With SourceSpec's travel time window
(`variable_win_length_factor` = 0.64, about 1.5·(S−P)), the median difference is at most
0.02 in every class from Mw 2 upward and USGS − Mw is 0.12 ± 0.14. Only intermediate depth
events below Mw 2 differ, by +0.13 at the median. For these weak records the release window
ends where the coda stops decaying, while the travel time window can run on into noise.
`window_test.py run --variant fixed|var` makes the runs (hours, resumable). Stage
`window_test` of [`../regenerate_magnitudes.sh`](../regenerate_magnitudes.sh) rebuilds the
comparison from them in seconds, and stage `tables` copies it to
[`../calibration/`](../calibration/).

## Catalogue

`romania_mw/outputs/Mw_catalog.csv`: 19,188 of 19,230 events. The column
`ML_catalog_window_design` is the hypoDD3D catalogue magnitude, kept only because the
spectral windows were designed from it (its `ML_type` is `mw` for 118 events). It is not a
magnitude of this product and nothing downstream reads it. Quality A 11,713, B 6,243, C 1,232
(A: ≥ 5 distinct recording sites, few t* at the lower t* bound, fc inside the search range, standard
error < 0.15. B: ≥ 3 sites). A site is counted once even when two of its instruments are
fitted separately, which `Mw_nfits` records.

    Mw_sigma = sqrt( sigma_stat² + sigma_sys(Mw)² )

`sigma_stat` is the station scatter shrunk towards a prescribed reference scatter of 0.245,
which is above the median station scatter of the catalogue (0.178), and divided by √n. `sigma_sys` is 0.10 for Mw ≥ 3.8 and 0.18 for Mw ≤ 2, linear in between. It combines
the systematic from the trend with station distance (0.10) with the ±0.15 sensitivity of the
Mw of small events to fc, t* and weighting choices. Median σ is 0.20 at Mw ≤ 2 and 0.11 above 3.8.

Corner frequency, stress drop and t* are reported only for Mw ≥ 3.5. Below that the corner
frequency is limited by the band and is not interpreted. On the 192 events with a reported
fc, the slope of log10 fc against Mw is −0.32 over all of them and −0.33 for the 44 crustal
ones (from the `fc_Hz` column, which is `fc_mean` restricted to Mw ≥ 3.5), against −0.5 for
constant stress drop.

**Known systematic, not corrected.** Within an event, station Mw falls with hypocentral
distance (−0.46 per decade for slab events and −0.24 for crustal ones, and crustal stations
within 25 km read +0.10 high at the median). An empirical distance term shifts event Mw by only −0.05
(crustal) to +0.05 (slab) and makes the USGS comparison worse, so it is not applied. Its
size is included in `sigma_sys`, and it has one sign for small crustal events.

**Comparison with USGS (not fitted).** USGS Mww/Mwr for 17 events: USGS − Mw = +0.12 ± 0.14
(crustal +0.20 on 6 events, slab +0.08 on 11), OLS slope 0.97, 88 % within 2σ. See
[`../calibration/Mw_validation.yaml`](../calibration/Mw_validation.yaml). The offset is significant on this sample (t = 3.6), so it is an
offset against USGS rather than agreement with it, and it is not constant in time: the 7 slab
events up to 2018 give +0.14 and the 4 from 2019 onwards −0.02. The slope is indistinguishable
from 1. The comparison constrains the level between Mw 3.8 and 5.6 only.
