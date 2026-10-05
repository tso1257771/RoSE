#!/usr/bin/env python
"""ML -> Mw conversion per depth regime, with validity ranges.

One linear form for both regimes,

    Mw = a + b (ML - 3)

fitted by orthogonal distance regression with the error of the event mean ML,
sigma_within / sqrt(n_sta), on the x axis and the per-event Mw_sigma on the y
axis. The fit set is quality A/B events with at least three stations, ML >= 2.0
and no amplitude quality warning. Events are not selected on Mw, the dependent
variable.

Parameter uncertainty is the year-block bootstrap, because half of the crustal
fit set belongs to one sequence and an event bootstrap treats those events as
independent. The event bootstrap and the year jackknife are reported beside it.

Quadratic and ML >= 3 linear fits are kept as diagnostics only: neither the
curvature nor a hinge is resolved inside the fit range.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import ml_root, work_root  # noqa: E402
from rose.magnitudes.conversion import (  # noqa: E402
    NBOOT,
    ODR_BACKEND,
    SEED,
    bootstrap_events,
    bootstrap_years,
    jackknife_years,
    lin,
    mw_from_ml,
    quad,
    resid_bins,
    run_odr,
)

PROJECT = work_root()
ROOT = ml_root()
FIT = ROOT / "outputs/fit"
OUT = ROOT / "outputs/conversion"
MW = PROJECT / "romania_mw/outputs/Mw_catalog.csv"
CAT = PROJECT / "seisbench_integration/data/Enhanced_ROMPLUS_catalog.csv"

# Anchored ML. Fixed numbers, not tied to the anchor constant C: a published
# validity bound must not move when the anchor moves.
FIT_LOWER_ML = 2.0            # below it the catalogue Mw is itself biased high
VALIDATED_MAX = dict(crustal=4.5, intermediate=5.0)     # residual bins near zero
EXTRAPOLATION_MAX = dict(crustal=5.0, intermediate=5.5)  # a few events only
MW_SIGMA_FLOOR = 0.05
EAST_WEST_LON = 24.5


# lin, quad, run_odr, the two bootstraps, jackknife_years and resid_bins come
# from rose.magnitudes.conversion, which is what applies the published relation.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(FIT / "event_ml.csv")
    anchored = json.loads((FIT / "report.json").read_text())
    C = anchored["C"]
    sigma_within = anchored["residuals_unchanged"]["all"]["within_event_std"]

    mw_sigma = pd.read_csv(MW, usecols=["event_index", "Mw_sigma"]).rename(
        columns={"event_index": "public_id", "Mw_sigma": "Mw_sigma_cat"})
    ev = ev.merge(mw_sigma, on="public_id", how="left")
    ev["Mw_sigma"] = ev.Mw_sigma.fillna(ev.Mw_sigma_cat)
    lon = pd.read_csv(CAT, usecols=["event_index", "longitude"]).rename(
        columns={"event_index": "public_id"})
    ev = ev.merge(lon, on="public_id", how="left")
    ev["year"] = pd.to_datetime(ev.ev_time, format="mixed").dt.year
    ev["warned"] = ev.ml_saturation_risk.fillna(False).astype(bool)

    ml_grid = np.arange(2.0, 5.51, 0.5)
    rep = dict(model="Mw = a + b (ML - 3), one linear form for both regimes",
               sigma_ML_within_event=sigma_within,
               sigma_ML_used="sigma_within / sqrt(n_sta), the error of the event mean",
               n_bootstrap=NBOOT, anchor_C=C, mw_catalogue=anchored["mw_catalogue"],
               selection=(f"regime, Mw quality A/B, n_sta >= 3, ML >= {FIT_LOWER_ML}, no amplitude "
                          "quality warning (ml_saturation_risk). No cut on Mw: selecting on the "
                          "dependent variable truncates the sample and manufactures curvature"),
               note="ML is anchored to SourceSpec Mw at Mw 4.0; refit this conversion whenever "
                    "the Mw catalogue changes")
    rows, resid_rows, fits = [], [], {}

    for r in ("crustal", "intermediate"):
        s = ev[(ev.regime == r) & ev.Mw_quality.isin(["A", "B"]) & ev.n_sta.ge(3)
               & ev.ml.ge(FIT_LOWER_ML) & ~ev.warned].dropna(subset=["ml", "Mw", "Mw_sigma"])
        ml, mw = s.ml.to_numpy(float), s.Mw.to_numpy(float)
        s_mw = s.Mw_sigma.clip(lower=MW_SIGMA_FLOOR).to_numpy(float)
        s_ml = sigma_within / np.sqrt(s.n_sta.to_numpy(float))
        year = s.year.to_numpy(int)

        b = run_odr(ml, mw, s_mw, s_ml)
        resid = mw - lin(b, ml)
        sd_ev, cov_ev, pred_sd_ev, pred_ci_ev = bootstrap_events(ml, mw, s_mw, s_ml, ml_grid)
        sd_yr, cov_yr, pred_sd_yr, pred_ci_yr = bootstrap_years(ml, mw, s_mw, s_ml, year, ml_grid)
        jk, sd_jk = jackknife_years(ml, mw, s_mw, s_ml, year)

        entry = dict(
            n=int(len(s)), ml_range_of_fit=[float(ml.min()), float(ml.max())],
            fit_lower_bound_ml=FIT_LOWER_ML,
            validated_range_ml=[FIT_LOWER_ML, VALIDATED_MAX[r]],
            extrapolation_range_ml=[VALIDATED_MAX[r], EXTRAPOLATION_MAX[r]],
            do_not_convert_above_ml=EXTRAPOLATION_MAX[r],
            coeffs=dict(a=float(b[0]), b=float(b[1])),
            se_year_block=dict(a=float(sd_yr[0]), b=float(sd_yr[1])),
            se_event_bootstrap=dict(a=float(sd_ev[0]), b=float(sd_ev[1])),
            se_year_jackknife=dict(a=float(sd_jk[0]), b=float(sd_jk[1])),
            se_published="year_block", cov_year_block=cov_yr.tolist(),
            resid_std=float(resid.std()),
            resid_nmad=float(1.4826 * np.median(np.abs(resid - np.median(resid)))),
            n_above_ml_3_5=int((ml >= 3.5).sum()), n_above_ml_4=int((ml >= 4.0).sum()),
            n_above_ml_4_5=int((ml >= 4.5).sum()),
            prediction=[dict(ml=float(x), mw=float(lin(b, x)), sd_year_block=float(a),
                             ci95_year_block=[float(lo), float(hi)], sd_event=float(e))
                        for x, a, e, lo, hi in zip(ml_grid, pred_sd_yr, pred_sd_ev,
                                                   pred_ci_yr[0], pred_ci_yr[1])],
            year_jackknife=jk)

        # diagnostics: neither is published
        bq = run_odr(ml, mw, s_mw, s_ml, quadratic=True)
        sq_ev, _, _, _ = bootstrap_events(ml, mw, s_mw, s_ml, ml_grid)
        rng = np.random.default_rng(SEED)
        reps_q = []
        years_u = np.unique(year)
        idx = {y: np.flatnonzero(year == y) for y in years_u}
        for _ in range(NBOOT):
            i = np.concatenate([idx[y] for y in rng.choice(years_u, len(years_u), replace=True)])
            try:
                reps_q.append(run_odr(ml[i], mw[i], s_mw[i], s_ml[i], quadratic=True))
            except Exception:
                continue
        sq_yr = np.asarray(reps_q).std(axis=0)
        m3 = ml >= 3.0
        b3 = run_odr(ml[m3], mw[m3], s_mw[m3], s_ml[m3])
        entry["diagnostics"] = dict(
            quadratic=dict(coeffs=[float(x) for x in bq], se_event=[float(x) for x in sq_ev],
                           se_year_block=[float(x) for x in sq_yr],
                           resid_std=float((mw - quad(bq, ml)).std()),
                           c_resolved_2sigma=bool(abs(bq[2]) > 2 * sq_yr[2]),
                           note="curvature is not resolved; the published relation is linear"),
            linear_ml_ge_3=dict(n=int(m3.sum()), coeffs=[float(x) for x in b3],
                                resid_std=float((mw[m3] - lin(b3, ml[m3])).std()),
                                mw_at_ml_4=float(lin(b3, 4.0)),
                                mw_at_ml_4_primary=float(lin(b, 4.0)),
                                note="agrees with the published line; no separate upper-range "
                                     "relation is needed"))

        # residuals on the fit set and on every quality A/B event of this regime
        t = resid_bins(ml, resid); t.insert(0, "sample", "fit_set"); t.insert(0, "regime", r)
        allab = ev[(ev.regime == r) & ev.Mw_quality.isin(["A", "B"])].dropna(subset=["ml", "Mw"])
        ra = allab.Mw.to_numpy(float) - lin(b, allab.ml.to_numpy(float))
        t2 = resid_bins(allab.ml.to_numpy(float), ra)
        t2.insert(0, "sample", "all_quality_AB"); t2.insert(0, "regime", r)
        resid_rows += [t, t2]
        entry["residual_by_ml_bin_fit_set"] = t.drop(columns=["regime", "sample"]).to_dict("records")
        entry["residual_by_ml_bin_all_AB"] = t2.drop(columns=["regime", "sample"]).to_dict("records")

        if r == "crustal":
            is23 = s.year.eq(2023).to_numpy()
            west = s.longitude.lt(EAST_WEST_LON).to_numpy()
            entry["level_structure"] = dict(
                note=("half of the crustal fit set is the 2023 Gorj sequence, which sits low in "
                      "Mw - ML. The split is largely geographic and is reported, not corrected"),
                n_2023=int(is23.sum()), n_other=int((~is23).sum()),
                median_residual_2023=float(np.median(resid[is23])),
                median_residual_other=float(np.median(resid[~is23])),
                median_Mw_minus_ML_west=float(np.median((mw - ml)[west])),
                median_Mw_minus_ML_east=float(np.median((mw - ml)[~west])),
                east_west_split_longitude=EAST_WEST_LON)

        fits[r] = b
        rows.append(dict(regime=r, n=len(s), a=float(b[0]), b=float(b[1]),
                         a_se_year_block=float(sd_yr[0]), b_se_year_block=float(sd_yr[1]),
                         a_se_event=float(sd_ev[0]), b_se_event=float(sd_ev[1]),
                         resid_std=float(resid.std()),
                         # the rule the flags apply, and separately the range
                         # the fit actually saw. They are not the same number.
                         ml_fit_min=FIT_LOWER_ML,
                         ml_fit_set_min=float(ml.min()), ml_fit_set_max=float(ml.max()),
                         validated_ml_max=VALIDATED_MAX[r],
                         extrapolation_ml_max=EXTRAPOLATION_MAX[r],
                         n_ml_ge_3_5=int((ml >= 3.5).sum()), n_ml_ge_4=int((ml >= 4.0).sum())))
        rep[r] = entry

    pd.DataFrame(rows).to_csv(OUT / "conversion_coefficients.csv", index=False)
    pd.concat(resid_rows).to_csv(OUT / "conversion_residuals_by_ml_bin.csv", index=False)

    # per-event converted Mw and the validity of that conversion. The rule
    # lives in rose.magnitudes.conversion, so applying the relation to a new
    # earthquake flags it exactly as the released catalog is flagged.
    coeffs = pd.DataFrame(
        [{"regime": r, "a": fits[r][0], "b": fits[r][1], "ml_fit_min": FIT_LOWER_ML,
          "validated_ml_max": VALIDATED_MAX[r], "extrapolation_ml_max": EXTRAPOLATION_MAX[r]}
         for r in fits],
    ).set_index("regime")
    ev["mw_from_ml"], ev["mw_from_ml_flag"] = mw_from_ml(
        ev.ml.to_numpy(float), ev.regime.to_numpy(object), coeffs, ev.warned.to_numpy(bool))
    ev.drop(columns=["Mw_sigma_cat", "longitude", "year", "warned", "ml_valid_range_flag"],
            errors="ignore").to_csv(FIT / "event_ml.csv", index=False)

    rep["flag_counts"] = ev.mw_from_ml_flag.value_counts().to_dict()
    rep["flag_counts_by_regime"] = {r: g.mw_from_ml_flag.value_counts().to_dict()
                                    for r, g in ev.groupby("regime")}
    rep["flag_definition"] = dict(
        below_range=f"ML < {FIT_LOWER_ML}: the catalogue Mw is biased high there, do not convert",
        in_range="inside the validated range, residual bin means near zero",
        extrapolated="above the validated range, supported by a few events only",
        do_not_convert="above the extrapolation range, or the event carries an amplitude quality "
                       "warning: take Mw from the Mw catalogue")
    rep["method_notes"] = (
        "Two choices carry the fit. The ML error on the x axis is the error of the event mean, "
        "sigma_within / sqrt(n_sta) with a median of 0.05, not the station-level scatter of 0.209: "
        "the larger value over-corrects attenuation and raises the crustal slope to 0.87. The fit "
        "set is bounded in ML only, because bounding it in Mw truncates the sample at low ML and "
        "produces curvature that is not in the scales")
    rep["slope_discussion"] = (
        "dML/dMw is 1/b, about 1.33 (crustal) and 1.41 (intermediate), close to the Deichmann-type "
        "expectation that ML grows up to 1.5 times faster than Mw for small earthquakes, because "
        "the Wood-Anderson band lies above the corner frequency. The difference between the two "
        "scales is therefore a slope, not an offset, and the anchor fixes only the point Mw 4.0")
    rep["limits"] = (
        "Consistency between crustal ML and Mw above ML 4.5 cannot be tested with 2014-2024 data: "
        "no crustal event free of an amplitude quality warning exceeds ML 4.9. Above the validated "
        "range every such event has a SourceSpec Mw, which should be used directly")
    rep["software"] = dict(python=sys.version.split()[0], numpy=np.__version__,
                           pandas=pd.__version__, scipy=__import__("scipy").__version__)
    if ODR_BACKEND == "odrpack":
        rep["software"]["odrpack"] = __import__("odrpack").__version__
    rep["settings"] = dict(bootstrap_replicates=NBOOT, seed=SEED,
                           odr=f"{ODR_BACKEND} orthogonal distance regression",
                           mw_sigma_floor=MW_SIGMA_FLOOR)

    par_path = FIT / "parameter_table.csv"
    par = pd.read_csv(par_path)
    par = par[~par.parameter.str.startswith(("conversion_", "passband_", "saturated_"))]
    add = []
    for r in ("crustal", "intermediate"):
        e = rep[r]
        add += [dict(parameter=f"conversion_a_{r}", value=e["coeffs"]["a"], se=e["se_year_block"]["a"],
                     se_method="year-block bootstrap", unit="mag",
                     note=f"Mw = a + b (ML - 3), n = {e['n']}, residual std {e['resid_std']:.3f}"),
                dict(parameter=f"conversion_b_{r}", value=e["coeffs"]["b"], se=e["se_year_block"]["b"],
                     se_method="year-block bootstrap", unit="",
                     note="event bootstrap understates it: half of the crustal fit set is one sequence"
                          if r == "crustal" else ""),
                dict(parameter=f"conversion_fit_lower_bound_ml_{r}", value=FIT_LOWER_ML, se=None,
                     se_method="", unit="mag (anchored ML)",
                     note="lower bound of the conversion fit, not Mc; below it the catalogue Mw is biased high"),
                dict(parameter=f"conversion_validated_max_ml_{r}", value=VALIDATED_MAX[r], se=None,
                     se_method="", unit="mag (anchored ML)",
                     note=f"residual bin means near zero to here; n(ML >= 3.5) = {e['n_above_ml_3_5']}"),
                dict(parameter=f"conversion_extrapolation_max_ml_{r}", value=EXTRAPOLATION_MAX[r], se=None,
                     se_method="", unit="mag (anchored ML)",
                     note="above this, report Mw instead of converting")]
    pd.concat([par, pd.DataFrame(add)], ignore_index=True).to_csv(par_path, index=False)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    for r in ("crustal", "intermediate"):
        e = rep[r]
        print(f"{r:13s} Mw = {e['coeffs']['a']:.3f} +- {e['se_year_block']['a']:.3f} + "
              f"({e['coeffs']['b']:.3f} +- {e['se_year_block']['b']:.3f})(ML-3)  "
              f"sigma {e['resid_std']:.3f}  n {e['n']}  ML {e['ml_range_of_fit'][0]:.2f}-"
              f"{e['ml_range_of_fit'][1]:.2f}  validated to {VALIDATED_MAX[r]}")
    print("flags:", rep["flag_counts"])


if __name__ == "__main__":
    main()
