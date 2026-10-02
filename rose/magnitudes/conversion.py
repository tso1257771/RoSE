"""Conversion from the local magnitude to the moment magnitude scale.

ML and Mw are not the same quantity. Over this catalog ML rises faster than
Mw, so a single additive offset cannot relate them and a linear relation is
fitted per depth regime:

    Mw = a + b (ML - 3.0)

referenced at ML 3.0 so that ``a`` is the Mw of an ML 3.0 earthquake and is
not an extrapolation to ML 0. The fit is an orthogonal distance regression,
which accounts for the uncertainty on ML as well as on Mw, since neither is
the independent variable. Coefficients and their validity range live in
``magnitudes/calibration/conversion_coefficients.csv``.

The relation is a conversion of last resort. Where a measured Mw exists it
should be used instead, because the conversion adds the scatter of the
relation (0.21 crustal, 0.08 intermediate depth magnitude units) to the
uncertainty already on ML.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .attenuation import DEPTH_SPLIT, REGIMES

# The regression is done by ODRPACK. SciPy deprecated its wrapper,
# ``scipy.odr``, in 1.17 and removes it in 1.19. The ``odrpack`` package is
# the wrapper SciPy points to instead and is the one used here. ``scipy.odr``
# is a fallback for an environment without ``odrpack``, while SciPy still has
# it, so the module imports on either side of that removal. The published
# coefficients were fitted with ``scipy.odr``. ``odrpack`` gives them to
# 1 part in 10^7, the level at which any rerun gives them
# (``magnitudes/README.md``).
# The import is deferred to the first fit. Reading the calibration tables,
# evaluating the distance correction and computing an ML or an Mw uncertainty
# do no regression at all, so none of them should need a regression backend
# installed.
_odrpack = None
_scipy_odr = None
_backend = None


def _load_backend():
    """Import a backend on first use. Returns its name."""
    global _odrpack, _scipy_odr, _backend
    if _backend is not None:
        return _backend
    try:
        import odrpack as _mod
        _odrpack, _backend = _mod, "odrpack"
        return _backend
    except ImportError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from scipy import odr as _mod
    except ImportError as exc:
        raise ImportError(
            "fitting the ML to Mw relation needs the odrpack package, or scipy.odr "
            "from a SciPy before 1.19; neither is installed. Install it with "
            "`pip install odrpack`. Reading the published relation with "
            "rose.magnitudes.mw_from_ml does not need it."
        ) from exc
    _scipy_odr, _backend = _mod, "scipy.odr"
    return _backend

__all__ = [
    "ML_REF",
    "NBOOT",
    "SEED",
    "REGIMES",
    "regime_from_depth",
    "lin",
    "quad",
    "run_odr",
    "bootstrap_events",
    "bootstrap_years",
    "jackknife_years",
    "resid_bins",
    "mw_from_ml",
    "MAX_FAILED_FRACTION",
]

ML_REF = 3.0                                # the relation is referenced here
NBOOT, SEED = 2000, 20260921


def regime_from_depth(depth):
    """Depth regime label, ``crustal`` below :data:`~.attenuation.DEPTH_SPLIT`."""
    d = np.asarray(depth, float)
    out = np.where(d < DEPTH_SPLIT, "crustal", "intermediate").astype(object)
    out[~np.isfinite(d)] = None
    return out if out.ndim else out.item()


def lin(b, x):
    return b[0] + b[1] * (x - ML_REF)


def quad(b, x):
    return b[0] + b[1] * (x - ML_REF) + b[2] * (x - ML_REF) ** 2


def _odr_scipy(f, ml, mw, s_mw, s_ml, beta0):
    """Fit with ``scipy.odr``. Returns the coefficients, success and the reason.

    ``scipy.odr`` never raises for a fit that stopped. It reports the reason in
    ``Output.info``: 1 to 3 are the convergence criteria, 4 is the iteration
    limit, and larger values are questionable results or errors.
    """
    data = _scipy_odr.RealData(ml, mw, sx=s_ml, sy=s_mw)
    out = _scipy_odr.ODR(data, _scipy_odr.Model(f), beta0=beta0).run()
    return out.beta, out.info < 4, f"info {out.info}: " + ", ".join(out.stopreason)


def _odr_odrpack(f, ml, mw, s_mw, s_ml, beta0):
    """Fit with ``odrpack``. Same return as :func:`_odr_scipy`.

    The standard errors become weights, as ``scipy.odr.RealData`` does with
    ``sx`` and ``sy``, and the model takes its arguments the other way round.
    ``OdrResult.success`` is ``info < 4``, the same rule as above.
    """
    ml, mw = np.asarray(ml, float), np.asarray(mw, float)
    out = _odrpack.odr_fit(lambda x, b: f(b, x), ml, mw, beta0,
                           weight_x=1.0 / np.asarray(s_ml, float) ** 2,
                           weight_y=1.0 / np.asarray(s_mw, float) ** 2)
    return out.beta, out.success, f"info {out.info}: {out.stopreason}"


#: Which wrapper does the fit: ``"odrpack"``, or ``"scipy.odr"`` when that is
#: the only one installed.
def _odr_fit(f, ml, mw, s_mw, s_ml, beta0):
    """Fit with whichever backend is available, importing it on first use."""
    fit = _odr_odrpack if _load_backend() == "odrpack" else _odr_scipy
    return fit(f, ml, mw, s_mw, s_ml, beta0)


#: Readable as ``rose.magnitudes.conversion.ODR_BACKEND``. It is resolved by
#: ``__getattr__`` rather than set here, so reading it imports a backend
#: instead of requiring one at import time.
_BACKEND_ATTR = "ODR_BACKEND"


def __getattr__(name):
    if name == _BACKEND_ATTR:
        return _load_backend()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_odr(ml, mw, s_mw, s_ml, quadratic=False):
    """Orthogonal distance regression of Mw on ML with errors on both.

    Raises ``RuntimeError`` when ODRPACK stops without converging, whichever
    wrapper does the call (:data:`ODR_BACKEND` names it).
    """
    f = quad if quadratic else lin
    beta0 = [np.median(mw), 0.7, 0.0] if quadratic else [np.median(mw), 0.7]
    beta, ok, reason = _odr_fit(f, ml, mw, s_mw, s_ml, beta0)
    if not ok:
        raise RuntimeError(f"ODR did not converge ({reason})")
    return np.asarray(beta, float)


#: Largest share of bootstrap replicates that may fail before the spread of
#: the survivors stops describing the spread of the whole.
MAX_FAILED_FRACTION = 0.05


def _reps_to_stats(reps, ml_grid, attempted=None, what="bootstrap"):
    """Spread of the replicate coefficients and of their predictions.

    A replicate whose regression did not converge, or raised, is dropped.
    Dropping is silent in aggregate, so the share dropped is checked here:
    past :data:`MAX_FAILED_FRACTION` the survivors are a selected subset and
    the spread they give understates the real one.
    """
    reps = np.asarray(reps)
    if attempted:
        failed = attempted - len(reps)
        if failed:
            share = failed / attempted
            msg = (f"{what}: {failed} of {attempted} replicates "
                   f"({share:.1%}) did not converge or failed")
            if share > MAX_FAILED_FRACTION:
                raise RuntimeError(
                    msg + ". The spread of the rest is not the spread of the whole, "
                    "so no uncertainty is reported."
                )
            warnings.warn(msg + ". The reported spread is over the rest.", stacklevel=3)
    if len(reps) < 2:
        raise RuntimeError(f"{what}: {len(reps)} replicate(s) succeeded, too few for a spread")
    pred = np.array([lin(b, ml_grid) for b in reps])
    return (reps.std(axis=0), np.cov(reps.T),
            pred.std(axis=0), np.percentile(pred, [2.5, 97.5], axis=0))


def bootstrap_events(ml, mw, s_mw, s_ml, ml_grid, nboot=NBOOT, seed=SEED):
    """Resample events with replacement. Treats the events as independent."""
    rng = np.random.default_rng(seed)
    reps = []
    for _ in range(nboot):
        i = rng.integers(0, len(ml), len(ml))
        try:
            reps.append(run_odr(ml[i], mw[i], s_mw[i], s_ml[i]))
        except Exception:                      # did not converge, or the fit raised
            continue
    return _reps_to_stats(reps, ml_grid, nboot, "event bootstrap")


def bootstrap_years(ml, mw, s_mw, s_ml, year, ml_grid, nboot=NBOOT, seed=SEED):
    """Resample the years with replacement, take every event of each drawn year.

    Keeps events of one year together, so the network configuration of that
    year is drawn or left out as a block. This is the uncertainty reported for
    the coefficients, and it is the larger of the two.
    """
    rng = np.random.default_rng(seed)
    years = np.unique(year)
    idx = {y: np.flatnonzero(year == y) for y in years}
    reps = []
    for _ in range(nboot):
        drawn = rng.choice(years, len(years), replace=True)
        i = np.concatenate([idx[y] for y in drawn])
        try:
            reps.append(run_odr(ml[i], mw[i], s_mw[i], s_ml[i]))
        except Exception:                      # did not converge, or the fit raised
            continue
    return _reps_to_stats(reps, ml_grid, nboot, "year block bootstrap")


def jackknife_years(ml, mw, s_mw, s_ml, year):
    """Refit with one year held out at a time and score the held out events.

    Returns the per year rows and the standard deviation of the coefficients
    over them. A year whose refit does not converge is left out of both, and
    warned about, because the spread is then over fewer years than the data
    has.
    """
    out, skipped = [], []
    for y in np.unique(year):
        m = year != y
        try:
            b = run_odr(ml[m], mw[m], s_mw[m], s_ml[m])
        except Exception:
            skipped.append(int(y))
            continue
        held = mw[~m] - lin(b, ml[~m])
        out.append(dict(year=int(y), n_held_out=int((~m).sum()), a=float(b[0]), b=float(b[1]),
                        held_out_median_residual=float(np.median(held))))
    if skipped:
        warnings.warn(f"jackknife_years: no fit for year(s) {skipped}; "
                      f"the spread is over the remaining {len(out)}", stacklevel=2)
    if len(out) < 2:
        raise RuntimeError(f"jackknife_years: {len(out)} year(s) fitted, too few for a spread")
    arr = np.array([[d["a"], d["b"]] for d in out])
    return out, arr.std(axis=0)


def resid_bins(ml, resid, lo=0.5, hi=6.5, step=0.5):
    """Residual statistics in bins of ML, used to locate the validated range."""
    g = pd.DataFrame({"bin": pd.cut(ml, np.arange(lo, hi + 0.01, step)), "r": resid}).groupby(
        "bin", observed=True).r
    t = pd.DataFrame({"n": g.size(), "mean": g.mean(), "se": g.std() / np.sqrt(g.size()),
                      "median": g.median(), "std": g.std()}).reset_index()
    t["bin"] = t.bin.astype(str)
    return t


def _as_warning_flag(warned, shape):
    """``warned`` as a boolean array, refusing anything that is not 0 or 1.

    ``bool(nan)`` is True, so a missing amplitude warning would otherwise
    condemn the event to ``do_not_convert``. That happens as soon as a caller
    left joins the catalog onto their own event list.
    """
    a = np.asarray(warned)
    if a.dtype == bool:
        return np.broadcast_to(a, shape)
    try:
        f = a.astype(float)
    except (TypeError, ValueError):
        raise TypeError(
            "warned must be boolean or 0/1; got dtype "
            f"{a.dtype} (pandas 'boolean' with pd.NA is not accepted, "
            "fill or drop the missing values first)"
        ) from None
    bad = ~np.isin(f, (0.0, 1.0))
    if bad.any():
        n = int(bad.sum())
        raise ValueError(
            f"warned must be 0 or 1, the amplitude quality warning (ML_warning); "
            f"{n} value(s) are not, the first being {f[bad].ravel()[0]!r}. "
            "A missing value is not a warning: drop or fill those rows."
        )
    return np.broadcast_to(f.astype(bool), shape)


def mw_from_ml(ml, regime, conversion, warned=None):
    """Convert ML to Mw with the fitted relation of its depth regime.

    Parameters
    ----------
    ml : array_like
        Anchored local magnitude.
    regime : str or array_like
        ``"crustal"`` or ``"intermediate"``, scalar or per element. Use
        :func:`regime_from_depth` to get it from focal depth.
    conversion : pandas.DataFrame
        Coefficients indexed by regime, with the columns ``a``, ``b``,
        ``ml_fit_min``, ``validated_ml_max`` and ``extrapolation_ml_max``.
        :func:`rose.magnitudes.calibration.load_calibration` returns the
        published table in this form.
    warned : array_like of bool, optional
        The amplitude quality warning of that event, ``ML_warning`` in the
        released catalog. A warned event is flagged ``do_not_convert``
        whatever its ML, because its ML is the part in doubt.

    Returns
    -------
    mw : numpy.ndarray
        Converted moment magnitude, computed for every finite ML so that a
        caller can see what the relation would give. Read ``flag`` before
        using it.
    flag : numpy.ndarray of str
        The validity of that conversion, with the same four values as the
        ``mw_from_ml_flag`` column of the released catalog:

        ``in_range``
            inside the validated range, where the residual bin means are
            near zero.
        ``below_range``
            below the lower bound of the fit. The catalog Mw is itself
            biased high there, so do not convert.
        ``extrapolated``
            above the validated range, supported by a few events only.
        ``do_not_convert``
            above the extrapolation range, or the event has an amplitude
            quality warning. Take Mw from the Mw catalog instead.

        A missing ML gives ``no_input``, which the released catalog has no
        row for.

    Notes
    -----
    Where a measured Mw exists, use it. The conversion adds the scatter of
    the relation to the uncertainty already on ML.

    Examples
    --------
    >>> from rose.magnitudes import load_calibration
    >>> cal = load_calibration()
    >>> mw, flag = mw_from_ml([1.0, 3.0, 4.8, 9.0], "crustal", cal.conversion)
    >>> [str(f) for f in flag]
    ['below_range', 'in_range', 'extrapolated', 'do_not_convert']
    """
    ml = np.asarray(ml, float)
    scalar = ml.ndim == 0
    ml = np.atleast_1d(ml)
    reg = np.broadcast_to(np.asarray(regime, dtype=object), ml.shape)
    if warned is None:
        warn_flag = np.zeros(ml.shape, bool)
    else:
        warn_flag = _as_warning_flag(warned, ml.shape)

    mw = np.full(ml.shape, np.nan)
    flag = np.full(ml.shape, "no_input", dtype=object)

    # object array: `is not None` is not elementwise, so the comparison stays
    for name in np.unique(reg[reg != None]):  # noqa: E711
        if name not in conversion.index:
            raise KeyError(f"no conversion coefficients for regime {name!r}")
        c = conversion.loc[name]
        m = (reg == name) & np.isfinite(ml)
        if not m.any():
            continue
        here = ml[m]
        mw[m] = c["a"] + c["b"] * (here - ML_REF)
        flag[m] = np.where(warn_flag[m] | (here > c["extrapolation_ml_max"]), "do_not_convert",
                   np.where(here < c["ml_fit_min"], "below_range",
                   np.where(here <= c["validated_ml_max"], "in_range", "extrapolated")))

    flag = flag.astype(str)
    return (float(mw[0]), str(flag[0])) if scalar else (mw, flag)
