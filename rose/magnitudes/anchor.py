"""Baseline shift that puts the local magnitude on the moment magnitude scale.

The distance correction in :mod:`rose.magnitudes.attenuation` fixes the shape
of the scale but leaves its absolute level undetermined, because an additive
constant can move from the event magnitudes into the station terms without
changing any residual. A constant ``C`` per depth regime is therefore fitted so
that ``ML = ML_unanchored + C`` matches Mw over a window of well recorded
moderate earthquakes.

``C`` is a Huber fit of ``Mw - ML_unanchored`` against ``Mw - 4.0``. Taking the
intercept at ``Mw = 4.0`` makes ``C`` the offset at the centre of the anchor
window rather than at ``Mw = 0``, so the slope of the relation does not leak
into it. The slope is reported as a diagnostic and is not applied.

Because the shift is additive, it moves the completeness magnitude by exactly
``C`` and leaves the event set, and so the b value, untouched. :func:`mc_bvalue`
accepts a fixed ``mc`` for that reason.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "WINDOW",
    "M_STAR",
    "HUBER",
    "NBOOT",
    "SEED",
    "huber_line",
    "anchor_set",
    "fit_C",
    "binned",
    "mc_bvalue",
]

WINDOW = (3.5, 4.5)     # Mw range of the anchor window
M_STAR = 4.0            # Mw the intercept is taken at
HUBER = 1.345           # Huber tuning constant, 95% efficiency at the normal
NBOOT = 2000            # bootstrap replicates for the uncertainty on C
SEED = 20260921         # fixed so the published value is reproducible


def huber_line(x, y, delta=HUBER, iters=25, tol=1e-10):
    """Huber IRLS fit of ``y = a + b x``.

    Returns ``(a, b)``. The scale is re estimated from the MAD of the
    residuals at every iteration, so a few badly measured events cannot pull
    the line.
    """
    X = np.column_stack([np.ones(len(x)), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    for _ in range(iters):
        r = y - X @ beta
        s = max(1.4826 * np.median(np.abs(r - np.median(r))), 1e-6)
        w = np.minimum(1.0, delta * s / np.maximum(np.abs(r), 1e-9))
        Xw = X * w[:, None]
        new = np.linalg.solve(Xw.T @ X, Xw.T @ y)
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            break
        beta = new
    return float(beta[0]), float(beta[1])


def anchor_set(ev, window=WINDOW, min_sta=5, min_fc_hz=1.25, quality=("A", "B")):
    """Events the anchor constant is fitted on.

    ``ev`` needs the columns ``Mw``, ``Mw_quality``, ``ml_saturation_risk``,
    ``n_sta`` and ``fc_Hz``. The cuts keep earthquakes that are large enough
    for Mw to be well determined and small enough that the Wood--Anderson
    response has not started to saturate: a corner frequency below 1.25 Hz
    approaches the 1.25 Hz Wood--Anderson corner, where ML stops tracking
    moment. Events with no reported corner frequency are kept, since the
    corner is reported only above Mw 3.5.
    """
    return ev[(ev.Mw.between(*window)) & ev.Mw_quality.isin(list(quality))
              & (~ev.ml_saturation_risk) & (ev.n_sta >= min_sta)
              & ((ev.fc_Hz >= min_fc_hz) | ev.fc_Hz.isna())]


def fit_C(s, m_star=M_STAR, nboot=NBOOT, seed=SEED):
    """Fit the anchor constant on the event table ``s`` from :func:`anchor_set`.

    Returns a dict with ``C``, the diagnostic ``slope``, the bootstrap
    ``sigma`` over events, the event count ``n``, ``mean_Mw`` and ``m_star``.
    """
    x = (s.Mw - m_star).to_numpy(float)
    y = (s.Mw - s.ml).to_numpy(float)
    a, b = huber_line(x, y)
    rng = np.random.default_rng(seed)
    reps = [huber_line(x[i], y[i])[0] for i in
            (rng.integers(0, len(x), len(x)) for _ in range(nboot))]
    return {"C": a, "slope": b, "sigma": float(np.std(reps)), "n": len(s),
            "mean_Mw": float(s.Mw.mean()), "m_star": float(m_star)}


def binned(x, y, bins, name):
    """Count, mean, median and standard deviation of ``y`` in bins of ``x``."""
    g = pd.DataFrame({"bin": pd.cut(x, bins), "y": y}).groupby("bin", observed=True).y
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "std": g.std()}
                        ).reset_index().rename(columns={"bin": name})


def mc_bvalue(m, dm=0.1, mc=None):
    """Maximum curvature completeness magnitude and Aki--Utsu b value.

    ``mc`` fixes the completeness instead of re estimating it, which is how the
    anchored magnitudes are scored: an additive shift moves Mc by exactly the
    shift and must leave the event set, and so b, untouched.
    """
    m = np.asarray(m, float)
    m = m[np.isfinite(m)]
    if mc is None:
        edges = np.arange(np.floor(m.min() * 10) / 10, m.max() + dm, dm)
        h, _ = np.histogram(m, edges)
        mc = float(edges[int(np.argmax(h))] + 0.2)
    mc = float(mc)
    s = m[m >= mc - dm / 2]                       # exact cut: keeps the event set fixed
    b = float(np.log10(np.e) / (s.mean() - (mc - dm / 2)))
    return {"Mc": round(mc, 3), "b": b, "b_sigma": float(b / np.sqrt(len(s))), "n": len(s)}
