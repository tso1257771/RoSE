"""Computing a local magnitude with the released scale.

A station magnitude is the Wood--Anderson displacement amplitude in
millimetres, corrected for distance and for that station:

    ML_station = log10 A + (-log A0)(R, h) - S_station

with ``-log A0`` anchored, so the result is already on the released scale.

``A`` is the amplitude the released scale was calibrated against: the mean of
the two horizontal components, measured zero to peak, in millimetres of
Wood-Anderson displacement, over the S window that RED-PAN-Motion selects.
A maximum of the two horizontals, or a peak to peak amplitude, gives a
different magnitude with no error raised.
The event magnitude is the median over sites, because a few badly measured
amplitudes should not move it. Sites are the unit, not channels: the two
instruments of one site are reduced to their median first, so a site with two
sensors does not count twice.

Measuring ``A`` from a waveform is not done here. It needs the instrument
response and the Wood--Anderson simulation, which live in the vendored
:mod:`rose.magnitudes.redpan_motion` and
:mod:`rose.magnitudes.taiwan_ml` packages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .attenuation import neg_log_a0 as _neg_log_a0

__all__ = ["MIN_STATIONS", "event_magnitude", "site_ids", "station_magnitude"]

MIN_STATIONS = 3        # fewest sites a released ML is reported from


def site_ids(station_keys):
    """Reduce station keys to the site they belong to.

    A station term is specific to a channel prefix and a response epoch, so
    its key looks like ``BS.BLKB..HH@2012-11-20``. Event magnitudes group by
    site instead, ``NET.STA.LOC``, because the two instruments of one site do
    not sample independent ground motion. Pass the result to
    :func:`event_magnitude` as its ``site`` argument. Giving it the station
    keys instead counts a two sensor site twice.

    >>> list(site_ids(["BS.BLKB..HH@2012-11-20", "BS.BLKB..BH@2012-11-20"]))
    ['BS.BLKB.', 'BS.BLKB.']
    """
    from .taiwan_ml.aggregation import station_location_ids

    return station_location_ids(station_keys)


def station_magnitude(log10_amp_mm, R_km, depth_km, station_term, cal):
    """Local magnitude of one amplitude on the released scale.

    Parameters
    ----------
    log10_amp_mm : array_like
        Base 10 logarithm of the Wood-Anderson displacement amplitude in
        millimetres, which is the unit Richter's scale is defined in. It is
        the mean of the two horizontal components, measured zero to peak.
    R_km, depth_km : array_like
        Hypocentral distance and focal depth in km.
    station_term : array_like
        ``S_station_term`` of the station and response epoch, from
        :meth:`rose.magnitudes.calibration.Calibration.station_term`. A station
        with no fitted term has no calibrated correction, and using 0.0 for it
        reports a magnitude the catalog would not have reported.
    cal : Calibration
        Loaded calibration tables.

    Returns
    -------
    numpy.ndarray
    """
    return (np.asarray(log10_amp_mm, float)
            + _neg_log_a0(R_km, depth_km, cal.atten, cal.anchor)
            - np.asarray(station_term, float))


def event_magnitude(station_ml, site, min_stations=MIN_STATIONS, trim_iqr=True):
    """Event magnitude from the station magnitudes of one earthquake.

    Reduces each site to its median, trims at 1.5 interquartile ranges when
    five or more sites remain, then takes the median of what is left.

    Parameters
    ----------
    station_ml : array_like
        Station magnitudes, from :func:`station_magnitude`.
    site : array_like
        The site each one belongs to. Use :func:`site_ids` to get sites from
        station keys.
    min_stations, trim_iqr
        The released catalog used the defaults.

    Returns
    -------
    ml, n_sta, std
        ``n_sta`` counts the sites the median was taken over, so it is the
        count **after** the trim. This is the ``ML_nstations`` column of the
        released catalog. Below ``min_stations`` sites the catalog reports no
        ML, and this returns ``(nan, n_sta, nan)``.
    """
    df = pd.DataFrame({"site": np.asarray(site, dtype=object),
                       "ml": np.asarray(station_ml, float)}).dropna()
    vals = df.groupby("site", sort=False).ml.median().to_numpy()
    if trim_iqr and len(vals) >= 5:
        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1
        vals = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
    n = len(vals)
    if n < min_stations:
        return float("nan"), n, float("nan")
    return float(np.median(vals)), n, float(np.std(vals, ddof=1)) if n > 1 else 0.0
