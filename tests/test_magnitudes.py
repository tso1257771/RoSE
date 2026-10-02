"""Checks on the released magnitude scales.

The published catalog is the reference. Every test here either reproduces a
released column from the calibration tables or pins a property of the scale
that the data descriptor states.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pathlib

import numpy as np
import pandas as pd
import pytest

import rose.magnitudes
from rose.magnitudes import (
    DEPTH_SPLIT,
    FIXED,
    NAMES,
    REGIMES,
    event_magnitude,
    load_calibration,
    mw_from_ml,
    mw_quality,
    mw_sigma,
    neg_log_a0,
    regime_from_depth,
    station_magnitude,
)
from rose.magnitudes.anchor import huber_line, mc_bvalue
from rose.magnitudes.calibration import Calibration


@pytest.fixture(scope="module")
def cal() -> Calibration:
    return load_calibration()


# --------------------------------------------------------------------------
# importing without the dataset stack


def test_magnitudes_imports_without_seisbench():
    """Someone checking a published magnitude need not install the whole stack.

    rose.magnitudes needs only NumPy, pandas, SciPy and odrpack. Run in a subprocess so
    the blocked imports cannot leak into the rest of the suite.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        from importlib.abc import MetaPathFinder
        BLOCK = {"seisbench", "obspy", "h5py", "torch"}

        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in BLOCK:
                    raise ImportError("No module named %r" % name)
                return None

        sys.meta_path.insert(0, Block())
        from rose.magnitudes import load_calibration, neg_log_a0
        cal = load_calibration()
        print(round(float(neg_log_a0([100.0], [10.0], cal.atten, cal.anchor)[0]), 6))

        import rose
        try:
            rose.RoSE
        except ImportError as exc:
            assert "not installed" in str(exc), exc
        else:
            raise AssertionError("rose.RoSE resolved without seisbench")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "2.516415", out.stdout


def _run_without(blocked, body):
    """Run ``body`` in a subprocess where the modules in ``blocked`` are absent.

    A name blocks itself and its submodules. A DeprecationWarning is an error
    there, so an import that SciPy has deprecated fails rather than passes.
    """
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import sys
        from importlib.abc import MetaPathFinder
        BLOCK = %r

        class Block(MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if any(name == b or name.startswith(b + ".") for b in BLOCK):
                    raise ImportError("No module named %%r" %% name)
                return None

        sys.meta_path.insert(0, Block())
    """ % sorted(blocked)) + textwrap.dedent(body)
    out = subprocess.run([sys.executable, "-W", "error::DeprecationWarning", "-c", code],
                         capture_output=True, text=True,
                         cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


# One synthetic problem, fitted under each wrapper. The published relation
# came from scipy.odr. odrpack gives the same coefficients to 1 part in 10^7,
# so both must print the same five decimals.
_SYNTHETIC_FIT = """
    import sys
    import numpy as np
    from rose.magnitudes.conversion import ODR_BACKEND, run_odr
    rng = np.random.default_rng(0)
    ml = rng.uniform(2.0, 5.0, 200)
    mw = 3.2 + 0.75 * (ml - 3.0) + rng.normal(0, 0.1, 200)
    b = run_odr(ml, mw, np.full(200, 0.1), np.full(200, 0.05))
    print(ODR_BACKEND, round(float(b[0]), 5), round(float(b[1]), 5),
          "scipy.odr" in sys.modules)
"""
_SYNTHETIC_RESULT = "3.19499 0.74365"


def test_conversion_imports_without_scipy_odr():
    """SciPy removes scipy.odr in 1.19. The conversion runs on odrpack instead.

    scipy.odr is blocked the way SeisBench is above, so this is what a user
    on a SciPy without it sees.
    """
    out = _run_without({"scipy.odr"}, _SYNTHETIC_FIT)
    assert out == f"odrpack {_SYNTHETIC_RESULT} False", out


@pytest.mark.skipif(importlib.util.find_spec("scipy.odr") is None,
                    reason="this SciPy no longer has scipy.odr")
def test_conversion_falls_back_to_scipy_odr_without_odrpack():
    """Without odrpack, and while SciPy still has it, scipy.odr does the fit.

    This is the wrapper the published coefficients came from. Its deprecation
    warning must not reach the user: the subprocess turns one into an error.
    """
    out = _run_without({"odrpack"}, _SYNTHETIC_FIT)
    assert out == f"scipy.odr {_SYNTHETIC_RESULT} True", out


def test_run_odr_raises_when_the_fit_did_not_converge(monkeypatch):
    """Neither wrapper raises on its own when ODRPACK stops. run_odr must."""
    from rose.magnitudes import conversion
    monkeypatch.setattr(conversion, "_odr_fit",
                        lambda f, ml, mw, s_mw, s_ml, beta0:
                        (np.asarray(beta0, float), False, "info 4: iteration limit"))
    with pytest.raises(RuntimeError, match="did not converge"):
        conversion.run_odr(np.array([2.0, 3.0, 4.0]), np.array([2.5, 3.2, 3.9]),
                           np.full(3, 0.1), np.full(3, 0.05))


def test_bootstrap_counts_a_replicate_that_did_not_converge(monkeypatch):
    """A fit reported as stopped is a failed replicate, and past 5% the spread
    is refused, whichever wrapper reports it."""
    from rose.magnitudes import conversion
    rng = np.random.default_rng(1)
    ml = rng.uniform(2.0, 5.0, 60)
    mw = 3.2 + 0.75 * (ml - 3.0) + rng.normal(0, 0.1, 60)
    s_mw, s_ml, grid = np.full(60, 0.1), np.full(60, 0.05), np.array([2.5, 3.5])
    real = conversion._odr_fit

    def stopping_every(k):
        calls = [0]

        def fit(*args):
            calls[0] += 1
            beta, ok, reason = real(*args)
            if calls[0] % k == 0:
                return beta, False, "info 4: iteration limit"
            return beta, ok, reason
        return fit

    monkeypatch.setattr(conversion, "_odr_fit", stopping_every(50))      # 2 of 100
    with pytest.warns(UserWarning, match=r"2 of 100 replicates \(2.0%\) did not converge"):
        sd, _, _, _ = conversion.bootstrap_events(ml, mw, s_mw, s_ml, grid, nboot=100, seed=1)
    assert np.all(sd > 0)

    monkeypatch.setattr(conversion, "_odr_fit", stopping_every(10))      # 10 of 100, past 5%
    with pytest.raises(RuntimeError, match="no uncertainty is reported"):
        conversion.bootstrap_events(ml, mw, s_mw, s_ml, grid, nboot=100, seed=1)


# --------------------------------------------------------------------------
# calibration tables


def test_calibration_loads_three_tables(cal):
    assert set(NAMES) <= set(cal.parameters.index)
    assert len(cal.station_terms) > 0
    assert set(cal.conversion.index) == set(REGIMES)
    assert {"value", "se", "se_method"} <= set(cal.parameter_table.columns)


def test_missing_calibration_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="calibration"):
        load_calibration(tmp_path)


def test_unknown_station_raises_unless_default_given(cal):
    with pytest.raises(KeyError, match="no station term"):
        cal.station_term("XX.NOSUCH..HH@1970-01-01")
    assert cal.station_term("XX.NOSUCH..HH@1970-01-01", default=0.0) == 0.0


def test_station_terms_are_centred(cal):
    """The fit has one additive degree of freedom, removed by centring S."""
    assert abs(float(cal.station_terms.S_station_term.mean())) < 0.25


def test_station_term_at_picks_the_epoch_in_force(cal):
    """RO.DRGR..BH has three calibrated epochs; the latest one started is used."""
    epochs = sorted(cal.station_terms.query(
        "network == 'RO' and station == 'DRGR' and channel_prefix == 'BH'").response_epoch)
    assert epochs == ["2010-06-18", "2014-11-24", "2014-11-29"]
    for when, epoch in (("2014-11-25T12:00:00", "2014-11-24"),
                        ("2014-11-29", "2014-11-29"),           # the start day counts
                        ("2020-01-01", "2014-11-29"),
                        ("2010-06-18T00:00:00", "2010-06-18")):
        key = cal.station_key_at("RO", "DRGR", "", "BH", when)
        assert key == f"RO.DRGR..BH@{epoch}", when
        assert cal.station_term_at("RO", "DRGR", "", "BH", when) == cal.station_term(key)
    # accepts a Timestamp as well as a string
    assert cal.station_key_at("RO", "DRGR", "", "BH", pd.Timestamp("2020-01-01")) \
        == "RO.DRGR..BH@2014-11-29"


def test_station_term_at_before_the_first_epoch_is_not_a_term(cal):
    with pytest.raises(KeyError, match="no calibrated epoch"):
        cal.station_term_at("RO", "DRGR", "", "BH", "2009-01-01")
    assert cal.station_term_at("RO", "DRGR", "", "BH", "2009-01-01", default=0.0) == 0.0
    with pytest.raises(KeyError, match="no station term"):
        cal.station_term_at("XX", "NOSUCH", "", "HH", "2015-01-01")


def test_public_api_is_the_user_surface():
    """The fitting internals stay importable from their modules, not from here."""
    assert set(rose.magnitudes.__all__) == {
        "Calibration", "load_calibration",
        "CRUSTAL_RMAX", "DEPTH_SPLIT", "FIXED", "NAMES", "R1", "R2", "neg_log_a0",
        "MIN_STATIONS", "event_magnitude", "site_ids", "station_magnitude",
        "ML_REF", "REGIMES", "mw_from_ml", "regime_from_depth",
        "mw_quality", "mw_sigma",
    }
    for name in ("huber_line", "run_odr", "columns", "fit_C", "anchor_set", "binned",
                 "mc_bvalue", "lin"):
        assert not hasattr(rose.magnitudes, name), name


# --------------------------------------------------------------------------
# distance correction


@pytest.mark.parametrize("depth", [5.0, 59.9, 60.0, 150.0])
def test_richter_fixed_point_at_100_km(cal, depth):
    """-log A0 is 3.0 at 100 km in both regimes, by construction."""
    assert neg_log_a0([100.0], [depth], cal.atten)[0] == pytest.approx(FIXED, abs=1e-12)


def test_correction_increases_with_distance(cal):
    R = np.array([10.0, 50.0, 100.0, 200.0, 400.0])
    for depth in (10.0, 120.0):
        v = neg_log_a0(R, np.full_like(R, depth), cal.atten)
        assert np.all(np.diff(v) > 0), f"not monotonic at depth {depth}"


def test_depth_split_is_a_step(cal):
    """The two regimes are different curves, so the split is a discontinuity."""
    below = neg_log_a0([200.0], [DEPTH_SPLIT - 0.1], cal.atten)[0]
    above = neg_log_a0([200.0], [DEPTH_SPLIT], cal.atten)[0]
    assert below != above


def test_anchored_correction_matches_parameter_table_at_100_km(cal):
    """The published -log A0 at 100 km is 3.0 + C, not 3.0."""
    for regime, depth in (("crustal", 10.0), ("intermediate", 120.0)):
        want = float(cal.parameters[f"neg_log_a0_at_100km_{regime}"])
        got = float(neg_log_a0([100.0], [depth], cal.atten, cal.anchor)[0])
        assert got == pytest.approx(want, abs=1e-9)


def test_anchor_lowers_the_scale(cal):
    """Leaving the anchor out overstates a magnitude; pin how much by."""
    for depth, regime in ((10.0, "crustal"), (120.0, "intermediate")):
        bare = neg_log_a0([150.0], [depth], cal.atten)[0]
        anch = neg_log_a0([150.0], [depth], cal.atten, cal.anchor)[0]
        assert anch - bare == pytest.approx(cal.anchor[regime], abs=1e-12)
        assert anch < bare


# --------------------------------------------------------------------------
# a local magnitude from an amplitude


def test_station_magnitude_is_richters_definition(cal):
    """1 mm at 100 km with no station correction is the anchored fixed point."""
    ml = station_magnitude([0.0], [100.0], [10.0], [0.0], cal)
    assert ml[0] == pytest.approx(
        float(cal.parameters["neg_log_a0_at_100km_crustal"]), abs=1e-9)


def test_station_magnitude_subtracts_the_station_term(cal):
    a = station_magnitude([0.0], [80.0], [10.0], [0.0], cal)[0]
    b = station_magnitude([0.0], [80.0], [10.0], [0.3], cal)[0]
    assert a - b == pytest.approx(0.3, abs=1e-12)


def test_station_magnitude_follows_amplitude_decade_for_decade(cal):
    ml = station_magnitude([0.0, 1.0, 2.0], [80.0] * 3, [10.0] * 3, [0.0] * 3, cal)
    assert np.allclose(np.diff(ml), 1.0)


def test_event_magnitude_matches_the_vendored_aggregator(cal):
    """The catalog's ML came from taiwan_ml.event_ml. Pin both against it.

    `fit_ml.py` used that function purely as the aggregator: it added the
    Romania distance correction into log10_A itself and passed a zeroed
    coefficient dict so the Taiwan attenuation contributes nothing. This test
    does the same, then compares the public API against it on random events
    with two instruments at some sites and outliers the trim has to remove.
    """
    from rose.magnitudes.local import site_ids
    from rose.magnitudes.taiwan_ml.model import event_ml

    ZERO = {"n": 0.0, "K": 0.0, "dK": 0.0, "rref": 100.0,
            "h_break": 40.0, "n_cheb": 0, "depth_term": True}

    rng = np.random.default_rng(3)
    nets, stas = ["RO", "BS"], [f"S{i:02d}" for i in range(14)]
    rows = []
    for ev in range(40):
        for sta in rng.choice(stas, rng.integers(2, 12), replace=False):
            for chan in (["HH"] if rng.random() < 0.6 else ["HH", "BH"]):
                rows.append({
                    "public_id": f"ev{ev:03d}",
                    "station_key": f"{rng.choice(nets)}.{sta}..{chan}@2014-01-01",
                    "log10_A": float(rng.normal(0.0, 0.6)),
                    "R_km": float(rng.uniform(5.0, 400.0)),
                    "depth_km": 8.0,                       # crustal, one anchor applies
                })
    obs = pd.DataFrame(rows)
    obs.loc[obs.sample(frac=0.04, random_state=1).index, "log10_A"] += 4.0   # outliers

    terms = {k: float(v) for k, v in
             zip(obs.station_key.unique(),
                 rng.normal(0.0, 0.3, obs.station_key.nunique()))}

    # what the driver ran: Romania correction into log10_A, Taiwan atten zeroed
    driver_obs = obs.copy()
    driver_obs["log10_A"] = obs.log10_A + neg_log_a0(obs.R_km, obs.depth_km, cal.atten)
    want = event_ml(driver_obs, ZERO, terms,
                    anchor_c=cal.anchor["crustal"]).set_index("public_id")

    # the same thing through the public API
    sml = station_magnitude(obs.log10_A, obs.R_km, obs.depth_km,
                            obs.station_key.map(terms), cal)
    got = {pid: event_magnitude(g.ml, g.site) for pid, g in
           pd.DataFrame({"public_id": obs.public_id,
                         "site": site_ids(obs.station_key),
                         "ml": sml}).groupby("public_id")}

    checked = 0
    for pid, (ml, n, _) in got.items():
        if pid not in want.index:
            assert np.isnan(ml), f"{pid}: reported {ml} where the catalog rule reports none"
            continue
        assert ml == pytest.approx(float(want.at[pid, "ml"]), abs=1e-12), pid
        assert n == int(want.at[pid, "n_sta"]), f"{pid}: site count {n} != {want.at[pid, 'n_sta']}"
        checked += 1
    assert checked >= 30, f"only {checked} events compared"


def test_event_magnitude_needs_three_sites():
    ml, n, sd = event_magnitude([3.0, 3.1], ["A", "B"])
    assert np.isnan(ml) and n == 2 and np.isnan(sd)
    ml, n, _ = event_magnitude([3.0, 3.1, 3.2], ["A", "B", "C"])
    assert ml == pytest.approx(3.1) and n == 3


def test_event_magnitude_counts_sites_not_channels():
    """Two instruments at one site are one site."""
    ml, n, _ = event_magnitude([3.0, 3.4, 3.1, 3.2],
                               ["A.X", "A.X", "B.Y", "C.Z"])
    assert n == 3
    assert ml == pytest.approx(3.2)


def test_site_ids_reduces_station_keys_to_sites():
    from rose.magnitudes.local import site_ids

    got = site_ids(["BS.BLKB..HH@2012-11-20", "BS.BLKB..BH@2012-11-20",
                    "RO.MLR..HH@2014-01-01"])
    assert len(set(got)) == 2


def test_event_magnitude_trims_an_outlier_and_stops_counting_it():
    """n_sta is the count the median was taken over, which is ML_nstations."""
    good = [3.0, 3.1, 3.2, 3.3, 3.4]
    clean, n_clean, _ = event_magnitude(good, list("abcde"))
    with_bad, n, _ = event_magnitude(good + [9.0], list("abcdef"))
    assert n == n_clean == 5                             # the outlier is not counted
    assert with_bad == pytest.approx(clean, abs=1e-12)   # nor does it move the median


def test_event_magnitude_ignores_missing_amplitudes():
    ml, n, _ = event_magnitude([3.0, np.nan, 3.1, 3.2], list("abcd"))
    assert n == 3 and ml == pytest.approx(3.1)


# --------------------------------------------------------------------------
# ML to Mw conversion


def test_conversion_is_referenced_at_ml_3(cal):
    for regime in REGIMES:
        mw, flag = mw_from_ml(3.0, regime, cal.conversion)
        assert mw == pytest.approx(float(cal.conversion.at[regime, "a"]))
        assert flag == "in_range"


def test_lower_bound_is_one_number_in_every_table(cal):
    """The rule bound is the published round number, not the fitted minimum.

    conversion_coefficients.csv records the smallest ML the fit saw, 2.0002,
    which is not the bound the released flags apply. Flagging against it would
    call an ML 2.00 earthquake below_range where the catalog calls it in_range.
    """
    for regime in REGIMES:
        published = float(cal.parameters[f"conversion_fit_lower_bound_ml_{regime}"])
        assert float(cal.conversion.at[regime, "ml_fit_min"]) == published
        assert published == 2.0
        # the range the fit saw is kept, and is above the bound
        assert float(cal.conversion.at[regime, "ml_fit_set_min"]) >= published


def test_the_lower_bound_is_inclusive(cal):
    for regime in REGIMES:
        bound = float(cal.conversion.at[regime, "ml_fit_min"])
        assert mw_from_ml(bound, regime, cal.conversion)[1] == "in_range"
        assert mw_from_ml(bound - 1e-6, regime, cal.conversion)[1] == "below_range"
        # between the bound and the smallest fitted ML, still inside the rule
        mid = (bound + float(cal.conversion.at[regime, "ml_fit_set_min"])) / 2
        assert mw_from_ml(mid, regime, cal.conversion)[1] == "in_range"


def test_conversion_flags_match_the_released_vocabulary(cal):
    """The four values are the ones in the catalog's mw_from_ml_flag column."""
    c = cal.conversion.loc["crustal"]
    ml = [c.ml_fit_min - 0.5, 3.0, c.validated_ml_max + 0.1, c.extrapolation_ml_max + 0.5]
    _, flag = mw_from_ml(ml, "crustal", cal.conversion)
    assert list(flag) == ["below_range", "in_range", "extrapolated", "do_not_convert"]


def test_an_amplitude_warning_blocks_the_conversion(cal):
    """A warned event is do_not_convert however good its ML looks."""
    _, flag = mw_from_ml([3.0, 3.0], "crustal", cal.conversion, warned=[False, True])
    assert list(flag) == ["in_range", "do_not_convert"]


def test_conversion_rejects_unknown_regime(cal):
    with pytest.raises(KeyError, match="no conversion coefficients"):
        mw_from_ml([3.0], "mantle", cal.conversion)


def test_conversion_passes_nan_through(cal):
    mw, flag = mw_from_ml([np.nan], "crustal", cal.conversion)
    assert np.isnan(mw[0]) and flag[0] == "no_input"


def test_ml_rises_faster_than_mw(cal):
    """b < 1 means one unit of Mw costs more than one unit of ML."""
    for regime in REGIMES:
        assert 0.5 < float(cal.conversion.at[regime, "b"]) < 1.0


def test_regime_from_depth():
    got = regime_from_depth([0.0, DEPTH_SPLIT - 1e-9, DEPTH_SPLIT, 200.0, np.nan])
    assert list(got) == ["crustal", "crustal", "intermediate", "intermediate", None]


# --------------------------------------------------------------------------
# Mw uncertainty and quality


def test_mw_sigma_falls_with_station_count():
    n = np.array([1.0, 2.0, 5.0, 20.0, 50.0])
    sigma, _ = mw_sigma(np.full(5, 3.0), np.full(5, 0.245), n)
    assert np.all(np.diff(sigma) < 0)


def test_mw_sigma_has_a_floor_at_the_systematic_term():
    sigma, sys_ = mw_sigma([5.0], [0.2], [500.0])
    assert sigma[0] > sys_[0]
    assert sigma[0] == pytest.approx(sys_[0], abs=0.01)


def test_mw_sigma_single_station_gets_the_pooled_scatter():
    """One site has no measurable scatter; it must not report sigma 0."""
    sigma, _ = mw_sigma([3.0], [np.nan], [1.0])
    assert sigma[0] > 0.2


def test_mw_sigma_is_nan_without_mw():
    sigma, sys_ = mw_sigma([np.nan], [0.2], [5.0])
    assert np.isnan(sigma[0]) and np.isnan(sys_[0])


def test_mw_quality_is_empty_without_mw():
    assert mw_quality([np.nan], [0.1], [9.0], [2.0], [0.0])[0] == ""


def test_mw_quality_classes():
    good = dict(station_std=[0.1], fc_wmean_hz=[2.0], frac_tstar_lo=[0.0])
    assert mw_quality([3.0], nsta=[9.0], **good)[0] == "A"
    assert mw_quality([3.0], nsta=[3.0], **good)[0] == "B"
    assert mw_quality([3.0], nsta=[2.0], **good)[0] == "C"
    # a corner frequency against the edge of the fitted band is a bound
    assert mw_quality([3.0], nsta=[9.0], station_std=[0.1],
                      fc_wmean_hz=[40.0], frac_tstar_lo=[0.0])[0] == "B"


# --------------------------------------------------------------------------
# fitting helpers


def test_huber_line_resists_outliers():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 300)
    y = 1.0 + 0.5 * x
    y[:10] += 20.0
    a, b = huber_line(x, y)
    assert a == pytest.approx(1.0, abs=0.05)
    assert b == pytest.approx(0.5, abs=0.05)


def test_additive_shift_moves_mc_but_not_b():
    """Why the anchor is reported as safe: it is a shift, not a rescaling."""
    rng = np.random.default_rng(1)
    m = 1.5 + rng.exponential(1.0 / (1.0 * np.log(10.0)), 20000)
    base = mc_bvalue(m, mc=2.0)
    shifted = mc_bvalue(m - 0.5, mc=1.5)
    assert shifted["n"] == base["n"]
    assert shifted["b"] == pytest.approx(base["b"], rel=1e-12)
    assert shifted["Mc"] == pytest.approx(base["Mc"] - 0.5)


# --------------------------------------------------------------------------
# the released catalog


@pytest.fixture(scope="module")
def catalog() -> pd.DataFrame:
    from rose.convert import CATALOG_SEARCH_PATHS

    for p in CATALOG_SEARCH_PATHS:
        if p.is_file():
            return pd.read_csv(p)
    pytest.skip("released catalog CSV not present")


def test_released_mw_sigma_is_reproducible():
    """Mw_sigma and Mw_sigma_sys of the released Mw catalog follow from mw_sigma.

    The inputs, Mw_station_std and Mw_nsta, are columns of Mw_catalog.csv in
    the data archive, not of the catalog in this repository, so the check
    runs where the working tree is present.
    """
    root = os.environ.get("ROMANIA_ROOT")
    path = Path(root) / "romania_mw" / "outputs" / "Mw_catalog.csv" if root else None
    if path is None or not path.is_file():
        pytest.skip("Mw_catalog.csv is in the data archive: set ROMANIA_ROOT to check it")
    df = pd.read_csv(path).dropna(subset=["Mw"])
    assert len(df) == 19_188
    sigma, sig_sys = mw_sigma(df.Mw, df.Mw_station_std, df.Mw_nsta)
    assert np.nanmax(np.abs(sigma - df.Mw_sigma)) < 1e-9
    assert np.nanmax(np.abs(sig_sys - df.Mw_sigma_sys)) < 1e-9


def test_released_magnitudes_are_in_range(catalog):
    for col, lo, hi in (("Mw", -1.0, 8.0), ("ML", -1.0, 8.0)):
        if col not in catalog:
            continue
        v = catalog[col].dropna()
        assert v.between(lo, hi).all(), f"{col} outside [{lo}, {hi}]"


def test_released_ml_needs_three_stations(catalog):
    if "ML_nstations" not in catalog:
        pytest.skip("catalog does not carry ML_nstations")
    have = catalog.dropna(subset=["ML"])
    assert (have.ML_nstations >= 3).all()
