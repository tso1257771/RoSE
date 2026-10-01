# Vrancea local magnitude (ML)

Local magnitude for the Vrancea catalogue (2014–2024), computed from Wood-Anderson
amplitudes with a correction for each station and separate distance corrections for crustal
and intermediate depth events. The baseline constant is anchored to the SourceSpec moment
magnitude at Mw 4.0 in each regime. Everything else is fitted from the
amplitudes alone. Earlier calibrations and the trials that led here are described in
[TRIALS.md](TRIALS.md).

The released coefficients are in [`../calibration/`](../calibration/), and
`rose.magnitudes` applies them. This file describes how they were obtained.

## Inputs

Paths are relative to the working tree named by `ROMANIA_ROOT`, whose shape
[`../README.md`](../README.md) gives. The repository ships the coefficients,
not these inputs.

| Path | Used for |
|---|---|
| `seisbench_integration/data/rose/{metadata,waveforms}YYYY.{csv,hdf5}` | RoSE waveforms (counts) and the metadata of each trace, including the P and S picks |
| `seisbench_integration/data/rose_stationxml/<NET>/<STA>.xml` (and `_merged_for_sourcespec.xml`) | instrument responses and response epochs |
| `seisbench_integration/data/Enhanced_ROMPLUS_catalog.csv` | event origins (hypoDD3D) and the ROMPLUS reference ML |
| `romania_mw/outputs/Mw_catalog.csv` | SourceSpec Mw, its uncertainty, quality and fc, for the anchor and the conversion |
| `romania_mw/outputs/usgs_cross_validation.csv` | USGS Mw, quoted in this README as a comparison. No driver here reads it |

## Steps

| Step | Driver | Output |
|---|---|---|
| Wood-Anderson amplitudes from the RoSE waveforms | [`extract_wa_amplitudes.py`](extract_wa_amplitudes.py) | `romania_ml/outputs/amplitudes/wa_<year>.csv` |
| Observation table, catalogue gates, blow-up handling | [`build_observations.py`](build_observations.py) | `romania_ml/outputs/observations_{fit,all}.csv.gz` |
| Distance correction, station terms, event ML | [`fit_ml.py`](fit_ml.py) | `romania_ml/outputs/fit_unanchored/` |
| Baseline anchor to Mw at Mw 4.0 | [`anchor_ml_to_mw.py`](anchor_ml_to_mw.py) | `romania_ml/outputs/fit/` |
| ML → Mw conversion and validity thresholds | [`fit_ml_to_mw_conversion.py`](fit_ml_to_mw_conversion.py) | `romania_ml/outputs/conversion/` |
| Figures and consistency checks | [`plot_ml.py`](plot_ml.py) | `romania_ml/outputs/fit/fig_*.png`, `validation_extra.json` |

The measurement code is copied unchanged into `rose/magnitudes/taiwan_ml/` (the
Taiwan local magnitude method) and `rose/magnitudes/redpan_motion/` (the
RED-PAN-Motion amplitude window). Run the drivers with
[`../regenerate_magnitudes.sh`](../regenerate_magnitudes.sh).

## Model

    ML = log10 A(mm) + (−log A0)(R) − S_j

`A` is the mean of the two horizontal Wood-Anderson amplitudes, measured zero to peak, in the
RED-PAN-Motion window, `R` the hypocentral distance, `S_j` the term of one station, channel
and response epoch, with the terms summing to zero. Event ML is the median over sites after a
1.5 IQR trim, from at least three sites.

**Distance correction.** Crustal (depth < 60 km): three segments in log10 R with hinges at
70 and 190 km, plus an anelastic term that is not allowed to be negative. Intermediate depth
(≥ 60 km): one segment plus its anelastic term. A depth term was tested and is not needed
(binned residual against depth ≤ 0.009).

| Parameter | Crustal | Intermediate |
|---|---|---|
| slope, R ≤ 70 km | 1.585 ± 0.025 | — |
| slope, 70–190 km | 0 (flat) | — |
| slope, R > 190 km | 1.777 ± 0.133 | — |
| slope (single segment) | — | 0.801 ± 0.127 |
| K (per km) | 0.00256 | 0.00306 |
| −log A0 at 100 km | 2.516 | 2.652 |

**Anchor.** C is the intercept of a Huber fit of (Mw − ML) on (Mw − 4.0), over
events with Mw in [3.5, 4.5], Mw quality A/B, no saturation risk, fc ≥ 1.25 Hz (or unknown)
and at least five stations. A median is not used: the window's mean Mw is about 3.85 and
(ML − Mw) has a slope of about 0.35 per magnitude unit there, so a median would absorb that
trend.

    C_crustal      = −0.484 ± 0.10   (year jackknife, n = 37. The event bootstrap gives 0.038 and understates it)
    C_intermediate = −0.348 ± 0.011  (event bootstrap, n = 135)

The crustal constant depends on one sequence: 2023 supplies 14 of the 37 crustal anchor
events and lies about 0.36 lower in Mw − ML than the other years, so dropping it gives −0.386
while dropping any other year gives −0.482 to −0.506. The systematic uncertainty of the anchor
events' Mw adds a shift common to every event of about ±0.10 (crustal) and ±0.10
(intermediate) to absolute ML. The validity bounds and the flag thresholds are fixed numbers in anchored ML and do not
move with C. The conversion intercept does move with it, by about −0.074 per +0.1 in C.

**What depends on Mw.** ML − Mw is zero at Mw 4.0 by construction and gives no information
there. The −log A0 shape, K, the hinges, the station terms, the residuals within events, the
transport test with one year left out, the b-value, the dML/dMw slope and the saturation
onset are independent of Mw. C, the a-value, the absolute ML level and the conversion
intercept depend on it.

## Quality and validity

The residual standard deviation is 0.258, and 0.209 within events (0.367 without station
terms). There are 371 terms for station, channel and epoch, with standard deviation 0.380.
With one year left out, the scatter within the events of that year falls from 0.361 without
corrections to 0.233 with the corrections fitted on the other years. 18,330 events have an
ML.

Amplitude quality control: clipped observations (a flat top in raw counts at 24 bit full
scale, 3 or more consecutive samples) are excluded from the fit and the event median.
Blow-ups are rejected by a 3·10⁴ mm ceiling and by their residual against the distance
model, so large amplitudes that are real are kept. Components stuck at full scale are
excluded (one horizontal of RO.LELR..HH).

| Quantity | Crustal | Intermediate |
|---|---|---|
| Mc (maximum curvature + 0.2) | 1.42 | 1.75 |
| Conversion fit lower bound, ML | 2.0 | 2.0 |
| Conversion validated to ML | 4.5 | 5.0 |
| Conversion extrapolated to ML | 5.0 | 5.5 |

The conversion lower bound is the lower bound of the conversion fit, not the completeness of
the catalogue. It lies above completeness in both regimes, and below it the catalogue Mw is
itself biased high, so ML must not be converted there.

Columns per event in `romania_ml/outputs/fit/event_ml.csv`: `n_sta`, `n_clipped`, `frac_clipped`,
`n_blowup_rejected`, `n_truncated`, `fc_Hz`, `ml_saturation_risk` (fc < 1.25 Hz, or
frac_clipped > 0.2, or clipped + rejected > 0.2·n_sta, 48 events), `ml_unanchored`,
`anchor_C`, `mw_from_ml` and `mw_from_ml_flag`, which says whether that conversion is usable
for the event: 1,551 in range, 5 extrapolated, 16,726 below the range and 48 not to be
converted.

## Saturation

No saturation is detectable within the catalogue, though the largest bins are thin: on the
anchored scale the median ML − Mw is +0.04 at Mw 4.0–4.5 crustal (n = 8) and +0.18 at
Mw 4.5–5.0 intermediate (n = 8), with +0.32 in the highest intermediate bin (n = 2). The
trend is still upward at the top of the range, but rests on 2–8 events per bin. From the catalogue's own scaling of fc with
Mw, the corner frequency crosses the Wood-Anderson natural frequency (1.25 Hz) at
Mw 4.6 crustal and 5.3 intermediate, and the expected onset of saturation is above Mw ≈ 5
and ≈ 5.5 respectively, reaching slope 0.5 near M 7. Below the anchor dML/dMw is about 1.33
(crustal) and 1.41 (intermediate), the inverse of the conversion slope, close to the
expectation of Deichmann for small earthquakes.

The 1977 M7.4, 1986 M7.1 and 1990 M6.9 Vrancea earthquakes lie in the saturated regime and
predate the calibration data. Their ML must be reported as a saturated local magnitude and
must **not** be converted to Mw with the relations here.

## ML → Mw conversion

    Mw = a + b (ML − 3)

| Regime | a | b | σ | fitted over ML | validated to ML | n |
|---|---|---|---|---|---|---|
| Crustal | 3.170 ± 0.072 | 0.754 ± 0.015 | 0.206 | 2.00–4.87 | 4.5 | 663 |
| Intermediate | 3.290 ± 0.007 | 0.711 ± 0.004 | 0.083 | 2.00–5.43 | 5.0 | 889 |

Orthogonal distance regression with the error of the event mean ML, 0.209/√n_sta, on one axis
and the σ(Mw) of each event on the other. The scatter between stations, 0.209, is not the
error of an event mean: used as such it over-corrects attenuation and raises the crustal slope to 0.87.
The fit set is quality A/B events with at least three stations, ML ≥ 2.0 and no amplitude
quality warning, with no cut on Mw. Bounding the fit set in Mw truncates it at low ML and
produces curvature that is not in the scales.

Uncertainties are from a bootstrap of 2000 replicates that resamples whole years. The
event bootstrap gives ± 0.012 on a and ± 0.016 on b for the crustal relation and understates
them, because 318 of the 663 crustal events belong to the 2023 Gorj sequence. The year
jackknife gives ± 0.036 and ± 0.019.

Residual bin means stay within 0.06 of zero from ML 1.5 to 4.5 crustal and within 0.03 to
ML 5.0 intermediate. A quadratic term is not resolved in either regime (c = −0.018 ± 0.030
crustal, +0.007 ± 0.007 intermediate) and a hinge is not identifiable, so one linear form is
published for both. A separate fit above ML 3 agrees with it: it predicts Mw 3.91 at ML 4
against 3.92 from the published line.

Between ML 4.5 and 5.0 crustal, and 5.0 and 5.5 intermediate, the relation is an extrapolation
supported by three and two events. Above those bounds, and for any event with an amplitude
quality warning, take Mw from the Mw catalogue rather than converting. The `mw_from_ml_flag`
column records this per event.

The crustal level is uncertain by about ± 0.07 beyond the formal error. The 2023 Gorj events
lie 0.16 below the line and the rest of the catalogue 0.11 above it, and the split is largely
geographic: Mw − ML is 0.18 west of 24.5° E against 0.53 east of it. The catalogue cannot say
whether ML is high or Mw low in the west, so the structure is reported and not corrected.

## Published parameters

[`../calibration/parameter_table.csv`](../calibration/parameter_table.csv) lists every published
parameter with its standard error and the method behind it (`se_method`): −log A0 coefficients (year jackknife), C per regime
with separate statistical, systematic and total terms, hinges and fit ranges, Mc per regime,
the conversion validity range and the passband and saturation thresholds.
[`../calibration/station_terms.csv`](../calibration/station_terms.csv) lists the terms for
station, channel and epoch with their observation counts. The stability with one year left
out is reported per year in `romania_ml/outputs/fit_unanchored/loyo_station_transport.csv` of
the working tree, which is not shipped. Software versions, seed, Huber delta and bootstrap
counts are in [`../calibration/fit_report.json`](../calibration/fit_report.json) and
[`../calibration/conversion_report.json`](../calibration/conversion_report.json), together
with the path, sha256 and timestamp of the Mw catalogue used for the anchor.
