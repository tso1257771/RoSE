"""Released magnitudes of the RoSE catalog, and the code that computed them.

Two magnitudes are released. Mw comes from the S wave displacement spectra of
each earthquake, inverted with SourceSpec. ML is a Wood--Anderson local
magnitude on a distance correction refitted for Romania, with a correction per
station and response epoch, shifted onto the Mw scale.

The published values are in the catalog. This package is here so the values
can be checked and so the same scale can be applied to a new earthquake:

>>> from rose.magnitudes import load_calibration, neg_log_a0
>>> cal = load_calibration()
>>> round(float(neg_log_a0([100.0], [10.0], cal.atten, cal.anchor)[0]), 6)
2.516415

Pass ``cal.anchor`` whenever the result is meant to be a magnitude. Left out,
``neg_log_a0`` returns the unanchored curve the shape was fitted in, which is
3.0 at 100 km and about 0.4 magnitude units above the released scale.

The calibration tables, the drivers that produced them and what each table
means are described in ``docs/MAGNITUDES.md``.

Two vendored subpackages carry the measurement code unchanged from its own
repository, so a reader can see exactly what was run:
:mod:`rose.magnitudes.taiwan_ml` (Wood--Anderson amplitudes and the inversion)
and :mod:`rose.magnitudes.redpan_motion` (response removal and amplitude
extraction). Their ``__upstream__`` strings give the version.
"""

from __future__ import annotations

from .anchor import anchor_set, binned, fit_C, huber_line, mc_bvalue
from .attenuation import DEPTH_SPLIT, FIXED, NAMES, R1, R2, columns, neg_log_a0
from .calibration import Calibration, load_calibration
from .conversion import ML_REF, REGIMES, lin, mw_from_ml, regime_from_depth, run_odr
from .local import MIN_STATIONS, event_magnitude, station_magnitude
from .mw_catalog import mw_quality, mw_sigma

__all__ = [
    # calibration tables
    "Calibration", "load_calibration",
    # ML distance correction
    "DEPTH_SPLIT", "FIXED", "NAMES", "R1", "R2", "columns", "neg_log_a0",
    # ML baseline
    "anchor_set", "binned", "fit_C", "huber_line", "mc_bvalue",
    # ML from an amplitude
    "MIN_STATIONS", "event_magnitude", "station_magnitude",
    # ML to Mw
    "ML_REF", "REGIMES", "lin", "mw_from_ml", "regime_from_depth", "run_odr",
    # Mw uncertainty and quality
    "mw_quality", "mw_sigma",
]
