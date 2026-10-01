# Magnitudes in `data/Enhanced_ROMPLUS_catalog.csv`

The catalogue is the RoSE release event list with two measured magnitudes added,
19,188 events with Mw and 18,330 with ML out of 19,230.
The two scales are reported separately. ML is anchored to Mw at Mw 4.0, so at that
magnitude the difference between them is zero by construction and is not a check.

## Mw

SourceSpec 1.8 S-wave spectral inversion (Brune omega-square, t* per station, 1/r spreading, stations <= 300 km hypocentral), RED-PAN-Motion S windows with a magnitude-dependent minimum. No external magnitude enters Mw or its uncertainty, and no depth correction is applied. Compared with USGS Mww/Mwr, which is not fitted: USGS minus Mw = +0.12 +- 0.14 (n = 17), an offset that is significant on that sample and is larger before 2019 than after. Mw_sigma = sqrt(station term^2 + systematic^2), systematic 0.10-0.18.

| Column | Meaning |
|---|---|
| `Mw` | moment magnitude, SourceSpec spectral inversion |
| `Mw_sigma` | uncertainty, station term and systematic combined (1 sigma) |
| `Mw_quality` | `A`, `B` or `C`. A: at least 5 distinct sites, few t* at the bound, corner frequency inside the search range, standard error < 0.15. B: at least 3 sites. C: 1 or 2 sites |
| `Mw_nstations` | distinct recording sites used. The two instruments of one site, BH and HH for example, are fitted separately but count once |
| `Mw_fc_Hz` | corner frequency, in Hz. Reported for Mw >= 3.5 only (192 events). Below that the spectrum is band-limited and fc is not interpreted, so empty means not reported, not unmeasurable |

## ML

Local magnitude from Wood-Anderson amplitudes (mean of the two horizontals) with a correction for each station. The distance correction is separate for crustal events (depth < 60 km, three segments) and intermediate-depth ones. The baseline constant is anchored to SourceSpec Mw at Mw 4.0 in each regime, giving C = -0.48 crustal and -0.35 intermediate-depth, so that -log A0 at 100 km is 2.516 and 2.652. ML minus Mw is zero at Mw 4.0 by construction. Attenuation shape, station terms and b-value are independent of Mw. The absolute level of ML has a common uncertainty of about 0.10 from the anchor.

| Column | Meaning |
|---|---|
| `ML` | local magnitude, Wood-Anderson amplitudes with station corrections |
| `ML_nstations` | stations used, at least 3 |
| `ML_warning` | `0` normal, `1` amplitude quality warning (48 events): amplitudes clipped or rejected at more than 20 per cent of the stations, or corner frequency below 1.25 Hz |

## Which magnitude to use

Use `Mw` wherever the magnitude stands for the size of the source, including
ground-motion work and machine-learning labels. It is proportional to moment
over the whole range, it does not saturate, and it is measured for
19,188 events, 886 of them where no local magnitude could be measured and
4,089 where the NIEP bulletin gives no magnitude at all.

`ML` is reported for continuity with the bulletin and for comparison with other
local-magnitude studies. It runs about 1.33 magnitude units per unit of Mw for
crustal events and 1.41 for intermediate-depth ones, because the corner
frequency of a small earthquake lies above the Wood-Anderson band and the
instrument then reads the flat part of the displacement spectrum. Completeness
and b-value differ between the two scales and cannot be carried from one to
the other. Do not put both scales
into one frequency-magnitude distribution. To move between them, use the
conversion in [`magnitudes/calibration/conversion_coefficients.csv`](../magnitudes/calibration/conversion_coefficients.csv)
within its stated range, or call `rose.magnitudes.mw_from_ml`, which applies it
and flags whether the result is inside that range.

Two limits apply at the ends of the Mw range. Below about Mw 2.5 the corner
frequency is at or beyond the resolvable band, so Mw is biased high by an
amount covered by `Mw_sigma`, which is 0.19 below Mw 2 against 0.10 above
Mw 3.8. Below about Mw 1.0 the smallest events are measured only when
their spectrum clears the noise. At the other end the catalogue holds
59 events at Mw 4.0 and above, 17 at 4.5 and above and 7 at 5.0 and above, so it
does not constrain the behaviour of any model in the range a warning system
exists for.

## NIEP bulletin columns

`ML_ROMPLUS` and `Mw_ROMPLUS` are the NIEP bulletin magnitudes, kept for
continuity with the published catalogue. 4,117 events have neither.
The two are equal in 932 rows, so `Mw_ROMPLUS` is not an independent
moment magnitude. Neither column enters `Mw` or `ML`.

## Usual selections

| Selection | Expression | Events |
|---|---|---|
| usable Mw | `Mw_quality in ("A", "B")` | 17,937 |
| best Mw | `Mw_quality == "A"` | 11,656 |
| usable ML | `ML_warning == 0` | 18,282 |
| events with a corner frequency | `Mw_fc_Hz` not empty | 192 |

An empty magnitude column means the event was not measured on that scale:
886 events have Mw only, 28 have ML only and 14 have neither.
The counting columns and `ML_warning` are read back as floating point when a row
is empty.

## Reproducing a magnitude

Every coefficient of both scales is in
[`magnitudes/calibration/`](../magnitudes/calibration/), and `rose.magnitudes`
reads it:

```python
from rose.magnitudes import load_calibration, neg_log_a0, station_magnitude, mw_from_ml

cal = load_calibration()
cal.atten                      # the five -log A0 coefficients
cal.anchor                     # the baseline shift per depth regime
cal.station_term("BS.BLKB..HH@2012-11-20")

# -log A0 of the released scale at 100 km, which is 3.0 + C, not 3.0
neg_log_a0([100.0], [10.0], cal.atten, cal.anchor)

# a station magnitude from a Wood--Anderson amplitude in mm
station_magnitude([0.0], [80.0], [10.0], [-0.52], cal)

# ML to Mw, with the validity of that conversion
mw_from_ml([3.4], "crustal", cal.conversion)
```

Pass `cal.anchor` whenever the result is meant to be a magnitude. Left out,
`neg_log_a0` gives the unanchored curve the shape was fitted in, which is 3.0
at 100 km and about 0.4 magnitude units above the released scale.

The drivers that produced the tables are in
[`magnitudes/ml/`](../magnitudes/ml/) and [`magnitudes/mw/`](../magnitudes/mw/),
and [`magnitudes/README.md`](../magnitudes/README.md) says what rerunning them
needs. `tests/test_magnitudes.py` checks the tables against this catalog.

Full method, uncertainties and validity ranges:
[`magnitudes/ml/README.md`](../magnitudes/ml/README.md) and
[`magnitudes/mw/README.md`](../magnitudes/mw/README.md). Stress drop and t* per
event are in `Mw_catalog.csv` in the data archive.
