"""Released magnitudes of the RoSE catalog, and the code that computed them.

Two magnitudes are released. Mw comes from the S wave displacement spectra of
each earthquake, inverted with SourceSpec. ML is a Wood-Anderson local
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

What is exported here is the surface a user of the scales needs: the
calibration tables, the distance correction, a station and an event magnitude
from an amplitude, the ML to Mw conversion, and the Mw uncertainty and quality
rules. The fitting code that produced the tables, the Huber line and anchor
set in :mod:`rose.magnitudes.anchor`, the orthogonal distance regression and
its resampling in :mod:`rose.magnitudes.conversion`, and the design matrix in
:mod:`rose.magnitudes.attenuation`, is importable from those modules.

The calibration tables, the drivers that produced them and what each table
means are described in ``docs/MAGNITUDES.md``.

Two vendored subpackages contain the measurement code unchanged from its own
repository, so a reader can see exactly what was run:
:mod:`rose.magnitudes.taiwan_ml` (Wood-Anderson amplitudes and the inversion)
and :mod:`rose.magnitudes.redpan_motion` (response removal and amplitude
extraction). Their ``__upstream__`` strings name the upstream commit or tag.
"""

from __future__ import annotations

from .attenuation import CRUSTAL_RMAX, DEPTH_SPLIT, FIXED, NAMES, R1, R2, neg_log_a0
from .calibration import Calibration, load_calibration
from .conversion import ML_REF, REGIMES, mw_from_ml, regime_from_depth
from .local import MIN_STATIONS, event_magnitude, site_ids, station_magnitude
from .mw_catalog import mw_quality, mw_sigma

__all__ = [
    # calibration tables, with station_term and station_term_at on Calibration
    "Calibration", "load_calibration",
    # ML distance correction
    "CRUSTAL_RMAX", "DEPTH_SPLIT", "FIXED", "NAMES", "R1", "R2", "neg_log_a0",
    # ML from an amplitude
    "MIN_STATIONS", "event_magnitude", "site_ids", "station_magnitude",
    # ML to Mw
    "ML_REF", "REGIMES", "mw_from_ml", "regime_from_depth",
    # Mw uncertainty and quality
    "mw_quality", "mw_sigma",
]
