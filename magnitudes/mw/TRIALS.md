# Mw: products and calibrations that were tried and rejected

History of the moment-magnitude work, kept separately from the method
description in [README.md](README.md).

> The trials below were run in the working tree, not in this repository. The
> outputs and scripts they name are not published: they are recorded so the
> reasoning behind the released scale is on the record, including what was
> rejected and why. Nothing here is part of the released magnitudes.


## 1. Composite catalogue with a depth correction

The first catalogue combined several methods per event:

* SourceSpec spectral Mw with an empirical depth correction,
  Δ(z) = −0.016 + 0.557/(1 + exp(−(z − 58.2)/4.4)), presented as a slab waveguide effect;
* Grond full-waveform moment tensors for M ≥ 3.5;
* coda-envelope Mw calibrated on 17 USGS events;
* spectral-ratio Mw against nearby reference events;
* ROMPLUS Mw as a fallback;
* a composite `Mw_best` following that priority order.

Rejected for three reasons:

* the depth correction was fitted against catalogue ML, and ROMPLUS Mw is itself derived
  from ML, so its apparent agreement was circular. Crustal and slab events are drawn from
  different magnitude ranges, so a fit against depth alone absorbs the ML–Mw trend: with
  magnitude as a covariate the slab term falls from +0.54 to +0.39, and against the USGS
  events the correction made the split worse (crustal −0.77, slab +0.21, against −0.75 and
  −0.33 uncorrected);
* the bootstrap uncertainty published with it was centred on a different curve than the
  correction applied;
* the composite switched methods at M 3.5, which put a step of about +0.4 into the
  magnitude–frequency distribution.

## 2. Grond moment tensors

466 events, M ≥ 3.5, 1-D Koulakov velocity model, 0.03–0.5 Hz. The scalar magnitudes agreed
with USGS in the mean but the mechanisms did not: against Craiu et al. (2023) the median
Kagan rotation was 79° and fault-type agreement 28 %, and the disagreement did not improve
with event size. The magnitude search range also started at 3.5, so 63 smaller events were
forced up to that bound and the values near it are unreliable. Not used in the published
Mw.

## 3. Coda-envelope and spectral-ratio magnitudes

Coda Mw (2–8 Hz, calibrated on 17 USGS events, RMS 0.26) and spectral-ratio Mw against
nearby reference events both saturate at the noise floor below about M 3 and were biased
high by +0.68 and +0.48 against ROMPLUS Mw. Kept only as exploratory columns at the time;
not used now.

## 4. SourceSpec configuration history

The first full run used short 5 s S windows with the SourceSpec default lower band edge of
0.5 Hz, which cannot resolve the plateau of M ≥ 4.5 events: the fitted corner frequencies of
the largest events were 2–3 Hz, implying stress drops of 240–890 MPa, and Mw was about 0.5
too low at the top of the range. Configurations labelled v2–v4 were run on 17 test events
only. The present configuration lowers the
band, bounds t* to 0.005–0.15 s and fc to 0.05–40 Hz, and takes windows from the
RED-PAN-Motion rule.

Two further experiments were dropped after testing: per-station site corrections derived
from the Grond events (raw, capped and distance-stratified variants) changed the
magnitude–frequency distribution very little and were not adopted; dropping the upsampled
20-sps BH/SH channels for small events cost more stations than the clipping change gained.

## 5. Calibration to USGS

An intermediate version tied the catalogue to the 17 USGS events by a single offset
(Mw = Mw_SourceSpec + 0.12, slope fixed to 1 after the free slope came out 0.98 ± 0.05,
residual std 0.14). It was dropped when the decision was taken that no external magnitude
should enter the fit. The offset now appears only as a validation number.

## 6. ML-based products

`Mw_calibrated_from_ML.csv` and the ML → Mw regressions of that period were diagnostics
only. The published relation between the two scales is the ML → Mw conversion in
[`../calibration/conversion_coefficients.csv`](../calibration/conversion_coefficients.csv),
fitted the other way round and valid over a stated range.
