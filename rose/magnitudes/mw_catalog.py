"""Uncertainty and quality of the moment magnitude.

Mw is a weighted mean over the station channel spectral fits of one
earthquake. Its uncertainty is the scatter of those fits divided by their
number, plus a systematic term, and its quality class records how well the
mean is determined.

The scatter of a handful of stations is itself a poor estimate of the scatter,
so the per event station standard deviation is shrunk towards the value pooled
over the whole catalog, :data:`STATION_STD_PRIOR`. An earthquake recorded at
one site then gets the pooled scatter rather than a standard deviation of
zero. This is the reason the reported uncertainty of a one site earthquake is
large but finite.

The systematic term covers what the station scatter cannot see, mainly the
assumed path and radiation model. It is larger for small earthquakes, whose
corner frequency approaches the upper end of the fitted band.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "STATION_STD_PRIOR",
    "SIG_SYS_SMALL",
    "SIG_SYS_LARGE",
    "MW_SYS_SMALL",
    "MW_SYS_LARGE",
    "MIN_NSTA_B",
    "QUALITY_A_MIN_NSTA",
    "QUALITY_A_MAX_SE",
    "QUALITY_A_MAX_FRAC_TSTAR_LO",
    "SE_FLOOR",
    "FC_BOUNDS",
    "mw_sigma",
    "mw_quality",
]

STATION_STD_PRIOR = 0.245       # station scatter pooled over the catalog
SIG_SYS_SMALL, SIG_SYS_LARGE = 0.18, 0.10
MW_SYS_SMALL, MW_SYS_LARGE = 2.0, 3.8

MIN_NSTA_B = 3                  # fewest sites for class B
QUALITY_A_MIN_NSTA = 5
QUALITY_A_MAX_SE = 0.15
QUALITY_A_MAX_FRAC_TSTAR_LO = 0.3
SE_FLOOR = 0.05
FC_BOUNDS = (0.05, 40.0)        # must match fc_min_max in the SourceSpec config


def mw_sigma(mw, station_std, nsta):
    """Combined uncertainty on Mw, one standard deviation.

    Parameters
    ----------
    mw : array_like
        Moment magnitude. Only the systematic term depends on it.
    station_std : array_like
        Standard deviation of the per station channel Mw of that earthquake.
    nsta : array_like
        Number of distinct sites, which is not the number of fits: the two
        instruments of one site do not sample independent ground motion.

    Returns
    -------
    sigma : numpy.ndarray
        Total uncertainty.
    sigma_sys : numpy.ndarray
        The systematic part of it, reported separately so a user who only
        wants relative sizes can take it back out.
    """
    mw = np.asarray(mw, float)
    n = np.asarray(nsta, float)
    s = np.nan_to_num(np.asarray(station_std, float), nan=0.0)

    s_hat = np.sqrt(((n - 1) * s ** 2 + 2 * STATION_STD_PRIOR ** 2) / (n + 1))
    sig_stat = s_hat / np.sqrt(n)
    sig_sys = np.interp(mw, [MW_SYS_SMALL, MW_SYS_LARGE], [SIG_SYS_SMALL, SIG_SYS_LARGE])

    sigma = np.sqrt(sig_stat ** 2 + sig_sys ** 2)
    missing = ~np.isfinite(mw)
    sigma = np.where(missing, np.nan, sigma)
    sig_sys = np.where(missing, np.nan, sig_sys)
    return sigma, sig_sys


def mw_quality(mw, station_std, nsta, fc_wmean_hz, frac_tstar_lo):
    """Quality class of Mw: ``A``, ``B``, ``C``, or empty where Mw is missing.

    ``fc_wmean_hz`` is the weighted mean corner frequency over the fits of
    that earthquake, the ``fc_wmean`` column of ``sourcespec_mw.csv``. It is
    not the ``fc_Hz`` column of the released catalog, which is reported only
    above Mw 3.5 and would put every smaller earthquake in class B.

    Class A needs at least :data:`QUALITY_A_MIN_NSTA` sites, a standard error
    of the mean below :data:`QUALITY_A_MAX_SE`, a corner frequency inside the
    fitted band rather than against its edge, and fewer than
    :data:`QUALITY_A_MAX_FRAC_TSTAR_LO` of its fits at the lower bound of the
    attenuation parameter. Class B relaxes those to
    :data:`MIN_NSTA_B` sites. Class C is one or two sites.

    The class is close to a count of sites in practice, so it ranks how well
    an earthquake was recorded rather than how large it was. Reading it as a
    magnitude dependent accuracy would be wrong: large earthquakes are
    recorded at more sites, so they fall into class A for that reason.
    """
    mw = np.asarray(mw, float)
    n = np.asarray(nsta, float)
    s = np.asarray(station_std, float)

    se = np.maximum(s / np.sqrt(np.clip(n, 1, None)), SE_FLOOR)
    fc = np.asarray(fc_wmean_hz, float)
    # against the edge of the fitted band the corner is a bound, not a measurement
    fc_ok = (fc >= FC_BOUNDS[0] * 1.02) & (fc <= FC_BOUNDS[1] * 0.98)
    good = (n >= QUALITY_A_MIN_NSTA) & (np.asarray(frac_tstar_lo, float) < QUALITY_A_MAX_FRAC_TSTAR_LO) \
        & fc_ok & (se < QUALITY_A_MAX_SE)

    return np.select([~np.isfinite(mw), good, n >= MIN_NSTA_B], ["", "A", "B"], "C")
