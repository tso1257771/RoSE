# ML: calibrations that were tried and rejected

History of the local-magnitude work, kept separately from the method
description in [README.md](README.md).

> The trials below were run in the working tree, not in this repository. The
> outputs and scripts they name are not published: they are recorded so the
> reasoning behind the released scale is on the record, including what was
> rejected and why. Nothing here is part of the released magnitudes.


## 1. Taiwan method transferred unchanged

First transfer of the reference method (github.com/tso1257771/taiwan-local-magnitude):
Wood–Anderson amplitudes, one −log A0 of the form n·log10(R/100) + K(R−100) +
dK(R−100)·max(0, h−40)/100, station terms, event ML as a trimmed median, anchor from a
reference Mw.

Rejected: one scale for both depth regimes left crustal and intermediate-depth events about
one magnitude unit apart against ROMPLUS ML. The depth hinge was not resolved (misfit
0.285–0.292 across all hinge depths), because the form lets depth change only the distance
slope, and that change is zero at 100 km — near or below the minimum distance of Vrancea
slab events. The relative level of the two regimes was therefore set by the functional form,
not by the data.

## 2. Separate depth regimes, anchored to the composite Mw

Separate −log A0 per regime (crustal 1.83 / −0.0031, intermediate 0.94 / +0.0028), anchored
to the then-current composite Mw catalogue. Crustal agreement with ROMPLUS improved to
+0.03, but the anchor rested on a Mw catalogue that was itself a composite of methods with a
+0.4 step at M 3.5, so it was replaced.

## 3. Anchored to the uncalibrated SourceSpec Mw

Same fit, anchor taken from the SourceSpec Mw alone (no USGS, Grond or ROMPLUS). Pooled
constant C = 2.547 ± 0.035 from 59 events with Mw ≥ 4.0. Superseded when the crustal
distance form and the anchoring rule were revisited.

## 4. Crustal distance-form tests

Two candidate fixes for the crustal −log A0, whose anelastic term came out negative and
whose residuals fell off beyond 200 km:

* forcing the anelastic term to be non-negative: the optimum sat exactly at zero, the
  within-event scatter worsened (0.211 → 0.217), the crustal and intermediate anchors
  stopped agreeing, and the fall-off beyond 200 km was only halved;
* restricting crustal observations to ≤ 200 km: the negative term became more negative
  (−0.0048), showing the data inside 200 km genuinely require curvature that a single
  slope-plus-attenuation form cannot express.

Neither removed the difference between all-station and 100–200 km-band slopes, which was
traced to the station-distance trend of the Mw side, not to ML. Both were rejected in favour
of the three-segment crustal form now in use.

## 5. Fixed-point anchor

ML anchored by definition instead of by Mw: −log A0 = 3.0 at 100 km (Richter/IASPEI), with
no free constant. Scientifically clean and fully independent of Mw, but it places ML about
0.5 below Mw at M 4–5 and, in practice, corresponds to an implicit reference magnitude of
about M 3. Replaced by the explicit Mw 4.0 anchor so that ML and Mw are reported on one
baseline and saturation above the anchor stays readable. The fixed-point value is kept in
`outputs/fit/report.json` (`fixed_point_before`) so the shift is traceable.

## 6. Amplitude handling that was replaced

* A fixed 1000 mm Wood–Anderson ceiling: it discarded genuine large amplitudes
  (1000–4300 mm at 45–200 km, consistent with their events) and would censor every station
  within about 100 km at ML 6.5. Replaced by a blow-up detector.
* Clipping detection on short-period channels only: flat-topped 24-bit traces on HH and BH
  channels passed as valid, and deconvolution inflates them rather than reducing them.
  Replaced by a flat-top detector on all channels.
* RO.LELR..HH was carried with one horizontal pinned at full scale in 3358 of 3703 traces;
  the fit absorbed it as a station term of about −0.46. The component is now excluded.

## 7. Conversion forms

Quadratic and bilinear forms were tested for both regimes and neither is published. Inside
the fit range the curvature is not resolved (c = −0.018 ± 0.030 crustal, +0.007 ± 0.007
intermediate, year-block bootstrap), the hinge of a bilinear fit has no interior optimum, and
the three forms predict Mw to within 0.04 up to ML 4.5. The published form is linear for both
regimes.

Two earlier choices produced a crustal relation that drifted by about 0.5 magnitude units
across its range, and both were errors in the fit rather than properties of the scales. The
station-level within-event scatter of 0.209 was used as the ML error of an event mean, which
over-corrects attenuation and raises the crustal slope from 0.75 to 0.87 and flips the sign of
the quadratic term. The intermediate fit set was also bounded at Mw ≥ 2.6, which removed 1,036
of the 1,083 events between ML 1.35 and 2.0, keeping only the largest Mw at those ML and
manufacturing the curvature that the quadratic form was then fitted to. Both are corrected in
the published fit, and the lower bound is now stated in ML alone.
