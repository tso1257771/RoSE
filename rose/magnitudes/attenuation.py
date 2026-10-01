"""Distance correction of the Romanian local magnitude scale.

``-log A0`` is the term that converts a Wood--Anderson displacement amplitude
into a magnitude at a reference distance. It is piecewise in hypocentral
distance ``R`` and splits on focal depth, because the Vrancea intermediate
depth earthquakes sample a different path than the crustal ones:

* crustal (``h < 60`` km), three segments hinged at ``R = 70`` km and
  ``R = 190`` km, with an anelastic term linear in ``R - 100``,
* intermediate depth (``h >= 60`` km), one geometric segment referenced to
  100 km plus the same form of anelastic term.

The shape is fitted with the scale fixed to Richter's point
``-log A0 = 3.0`` at ``R = 100`` km, so the five coefficients describe only
the curve away from 100 km. The released magnitudes then carry a baseline
shift ``C`` per depth regime that puts them on the Mw scale, which makes the
published correction ``3.0 + C`` at 100 km rather than 3.0. Pass ``anchor`` to
:func:`neg_log_a0` to get that published form. **Omitting it gives the
unanchored curve, which is about 0.4 magnitude units too high.**

Coefficient values live in ``magnitudes/calibration/parameter_table.csv`` and
are read with :func:`rose.magnitudes.calibration.load_calibration`.

Reference: Liao et al., the RoSE data descriptor (see the repository README).
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "DEPTH_SPLIT",
    "R1",
    "R2",
    "FIXED",
    "CRUSTAL_RMAX",
    "NAMES",
    "columns",
    "neg_log_a0",
]

DEPTH_SPLIT = 60.0      # km, crustal / intermediate depth split
R1 = 70.0               # km, first crustal hinge
R2 = 190.0              # km, second crustal hinge
FIXED = 3.0             # -log A0 at R = 100 km (Richter's fixed point)
CRUSTAL_RMAX = 400.0    # km, largest crustal distance the coefficients were fitted over

#: Coefficient names, in the column order :func:`columns` returns.
NAMES = ["n1_crustal", "n3_crustal", "K_crustal", "n_intermediate", "K_intermediate"]


def columns(R, h):
    """Design matrix of the ``-log A0`` shape terms.

    Parameters
    ----------
    R : array_like
        Hypocentral distance in km.
    h : array_like
        Focal depth in km, broadcast against ``R``.

    Returns
    -------
    numpy.ndarray
        ``(len(R), 5)`` matrix whose columns match :data:`NAMES`. Every column
        is zero outside the depth regime it belongs to, so the two regimes
        never mix.
    """
    R = np.asarray(R, float)
    crust = np.asarray(h, float) < DEPTH_SPLIT
    return np.column_stack([
        np.where(crust & (R <= R1), np.log10(np.maximum(R, 1e-6) / R1), 0.0),
        np.where(crust & (R > R2), np.log10(np.maximum(R, 1e-6) / R2), 0.0),
        np.where(crust, R - 100.0, 0.0),
        np.where(~crust, np.log10(np.maximum(R, 1e-6) / 100.0), 0.0),
        np.where(~crust, R - 100.0, 0.0),
    ])


def neg_log_a0(R, h, co, anchor=None):
    """Distance correction ``-log A0`` at distance ``R`` and depth ``h``.

    Parameters
    ----------
    R, h : array_like
        Hypocentral distance and focal depth in km.
    co : mapping
        Coefficient per name in :data:`NAMES`, from
        :attr:`rose.magnitudes.calibration.Calibration.atten`.
    anchor : mapping, optional
        Baseline shift per regime, ``{"crustal": C, "intermediate": C}``, from
        :attr:`rose.magnitudes.calibration.Calibration.anchor`. Pass it to get
        the correction the released magnitudes were computed with. Without it
        the curve is the unanchored one the shape was fitted in, worth 3.0 at
        100 km, which is not the published scale.

    Examples
    --------
    >>> from rose.magnitudes import load_calibration
    >>> cal = load_calibration()
    >>> round(float(neg_log_a0([100.0], [10.0], cal.atten)[0]), 6)        # fitted form
    3.0
    >>> round(float(neg_log_a0([100.0], [10.0], cal.atten, cal.anchor)[0]), 6)
    2.516415
    """
    v = FIXED + columns(R, h) @ np.array([co[n] for n in NAMES])
    if anchor is None:
        return v
    crust = np.asarray(h, float) < DEPTH_SPLIT
    return v + np.where(crust, anchor["crustal"], anchor["intermediate"])
