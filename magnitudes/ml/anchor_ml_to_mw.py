#!/usr/bin/env python
"""Anchor ML to SourceSpec Mw at Mw = 4.0 (per depth regime).

Baseline shift only: amplitudes, QC, -log A0 shapes and hinges, station terms,
event-ML rule and the saturation safeguards of the previous fit are untouched.
The scale definition becomes

    -log A0(100 km) = 3.0 + C_regime,     ML_anchored = ML + C_regime

with C the Huber-robust intercept of (Mw - ML) regressed on (Mw - 4.0) over
events with Mw in [3.5, 4.5], Mw quality A/B, no saturation risk, fc >= 1.25 Hz
(or unknown) and at least 5 stations.  A median is not used: the mean Mw of the
window is about 3.85, so a median would carry the trend of (Mw - ML) with Mw.

Outputs: outputs/fit/
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import ml_root, work_root  # noqa: E402
from rose.magnitudes.anchor import (  # noqa: E402
    HUBER,
    M_STAR,
    NBOOT,
    SEED,
    WINDOW,
    anchor_set,
    binned,
    fit_C,
    huber_line,
    mc_bvalue,
)
from rose.magnitudes.conversion import REGIMES as REG  # noqa: E402

PROJECT = work_root()
ROOT = ml_root()
SRC = ROOT / "outputs/fit_unanchored"   # ML before the baseline shift
OUT = ROOT / "outputs/fit"              # published ML
MW = PROJECT / "romania_mw/outputs/Mw_catalog.csv"


# huber_line, anchor_set, fit_C, binned and mc_bvalue come from
# rose.magnitudes.anchor, with the window, M* and the seed.


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = pd.read_csv(SRC / "event_ml.csv")
    src_rep = json.loads((SRC / "report.json").read_text())
    co = src_rep["coeffs"]
    rep = dict(
        basis="outputs/fit_unanchored (amplitudes, QC, -log A0 shapes/hinges, station terms, event-ML rule, "
              "saturation safeguards unchanged); this step only shifts the baseline",
        rule="C = Huber intercept of (Mw - ML) on (Mw - 4.0), Mw in [3.5, 4.5], quality A/B, "
             "no saturation risk, fc >= 1.25 Hz or unknown, n_sta >= 5",
        fixed_point_before=dict(R_km=100.0, neg_log_a0=3.0),
        mw_catalogue=dict(path=str(MW), sha256=hashlib.sha256(MW.read_bytes()).hexdigest(),
                          mtime_utc=datetime.fromtimestamp(MW.stat().st_mtime, timezone.utc).isoformat(),
                          magnitude_column="Mw (SourceSpec)"),
        created_utc=datetime.now(timezone.utc).isoformat(),
        software=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                      scipy=__import__("scipy").__version__),
        settings=dict(huber_delta=HUBER, bootstrap_replicates=NBOOT, seed=SEED,
                      anchor_window=list(WINDOW), m_star=M_STAR))

    s_all = anchor_set(ev)
    anchors = {r: fit_C(s_all[s_all.regime == r]) for r in REG}
    rep["anchor"] = anchors
    rep["anchor_pooled"] = fit_C(s_all)
    rep["anchor_sensitivity_to_M_star"] = {
        r: {f"M*={m}": fit_C(s_all[s_all.regime == r], m) for m in (3.5, 4.0, 4.5)} for r in REG}
    rep["anchor_sensitivity_to_M_star"]["pooled"] = {f"M*={m}": fit_C(s_all, m) for m in (3.5, 4.0, 4.5)}
    rep["anchor_sensitivity_dC_dMstar"] = {
        r: float((rep["anchor_sensitivity_to_M_star"][r]["M*=4.5"]["C"]
                  - rep["anchor_sensitivity_to_M_star"][r]["M*=3.5"]["C"]) / 1.0) for r in REG}
    # leave-one-calendar-year-out anchor stability (item 3 of the review)
    s_all = s_all.assign(year=s_all.ev_time.str[:4].astype(int))
    loyo = {}
    for r in REG:
        sr = s_all[s_all.regime == r]
        per_year = {}
        for y in sorted(sr.year.unique()):
            sub = sr[sr.year != y]
            if len(sub) < 10:
                continue
            f = fit_C(sub)
            per_year[str(y)] = dict(C=f["C"], n_dropped=int((sr.year == y).sum()), n_used=f["n"])
        vals = np.array([v["C"] for v in per_year.values()])
        g = len(vals)
        loyo[r] = dict(per_year=per_year,
                       C_min=float(vals.min()), C_max=float(vals.max()),
                       jackknife_se=float(np.sqrt((g - 1) / g * np.sum((vals - vals.mean()) ** 2))),
                       n_years=g)
    rep["anchor_loyo_C"] = loyo
    rep["anchor_loyo_C"]["interpretation"] = (
        "the crustal constant is sequence-dependent: 2023 supplies a large share of the crustal anchor "
        "events and has a markedly different Mw - ML level, so the year-jackknife spread, not the event "
        "bootstrap, is the honest uncertainty on C_crustal")
    C = {r: anchors[r]["C"] for r in REG}
    rep["C"] = C
    # systematic Mw uncertainty of the anchor events (common-mode on absolute ML)
    sys_sigma = {}
    mw_sys = pd.read_csv(MW, usecols=["event_index", "Mw_sigma_sys"]).rename(columns={"event_index": "public_id"})
    s_sys = s_all.merge(mw_sys, on="public_id", how="left")
    for r in REG:
        med = float(s_sys.loc[s_sys.regime == r, "Mw_sigma_sys"].median())
        stat = loyo[r]["jackknife_se"] if r == "crustal" else anchors[r]["sigma"]
        sys_sigma[r] = dict(C_sigma_stat=float(stat),
                            stat_method="year jackknife" if r == "crustal" else "event bootstrap",
                            C_sigma_sys=med, C_sigma_total=float(np.hypot(stat, med)))
    rep["C_uncertainty_budget"] = sys_sigma
    rep["C_uncertainty_budget"]["note"] = (
        "C_sigma_sys is the median systematic Mw uncertainty of the anchor events; it is a common-mode "
        "shift of absolute ML. The validity bounds and flag thresholds are fixed numbers in "
        "anchored ML and do not move with C. The conversion intercept does move with it, by "
        "about -0.074 per +0.1 in C, so it carries the same common-mode term")
    rep["scale_definition"] = {r: dict(neg_log_a0_at_100km=3.0 + C[r],
                                       formula=f"ML = log10 A(mm) + (-log A0)(R) - S_j, "
                                               f"-log A0(100 km) = {3.0 + C[r]:.4f}") for r in REG}
    s_all.assign(dmw=lambda d: d.Mw - d.ml)[
        ["public_id", "regime", "Mw", "ml", "dmw", "n_sta", "fc_Hz", "Mw_quality"]].to_csv(
        OUT / "anchor_events.csv", index=False)

    # ---- apply the shift ----
    ev["ml_unanchored"] = ev.ml
    ev["anchor_C"] = ev.regime.map(C)
    ev["ml"] = ev.ml_unanchored + ev.anchor_C
    ev.to_csv(OUT / "event_ml.csv", index=False)
    rep["events_with_ml"] = int(len(ev))

    # ---- validation ----
    rep["validation"] = {}
    tabs = []
    for r in REG:
        s = ev[(ev.regime == r) & ev.Mw.notna() & ev.Mw_quality.isin(["A", "B"])]
        t = binned(s.Mw, s.ml - s.Mw, np.arange(1.5, 6.51, 0.5), "Mw")
        t.insert(0, "regime", r)
        tabs.append(t)
        at4 = t[t.Mw.astype(str).str.startswith("(3.5")]
        rep["validation"][r] = {
            "n": int(len(s)),
            "median_ML_minus_Mw": float((s.ml - s.Mw).median()),
            "median_in_Mw_3.5_4.0_bin": float(at4["median"].iloc[0]) if len(at4) else None,
            "slope_ML_on_Mw": float(np.polyfit(s.Mw, s.ml, 1)[0])}
    pd.concat(tabs).to_csv(OUT / "ml_minus_mw_by_bin.csv", index=False)
    # zero-by-construction check at the anchor point
    for r in REG:
        s = anchor_set(ev)
        s = s[s.regime == r]
        x = (s.Mw - M_STAR).to_numpy(float)
        y = (s.ml - s.Mw).to_numpy(float)
        a, b = huber_line(x, y)
        rep["validation"][r]["anchor_check_intercept_at_Mw4"] = a
        rep["validation"][r]["anchor_check_slope"] = b

    # ---- b-value / Mc, unanchored vs anchored ----
    rep["frequency_magnitude"] = {"note": "anchored Mc = unanchored Mc + C on the same event set; "
                                          "b is invariant under the shift by construction"}
    for r in REG:
        s = ev[ev.regime == r]
        un = mc_bvalue(s.ml_unanchored)
        rep["frequency_magnitude"][r] = dict(unanchored=un, anchored=mc_bvalue(s.ml, mc=un["Mc"] + C[r]))
    un_all = mc_bvalue(ev.ml_unanchored)
    rep["frequency_magnitude"]["all_note"] = ("pooled Mc mixes the two regimes, whose shifts differ; "
                                              "use the per-regime values")
    rep["frequency_magnitude"]["all"] = dict(unanchored=un_all)

    # residual statistics are invariant under a per-regime additive constant
    rep["residuals_unchanged"] = src_rep["residuals"]
    rep["loyo_unchanged"] = src_rep["loyo_median_over_years"]
    rep["invariance_note"] = ("station ML and event ML shift by the same C, so station residuals, "
                              "within-event scatter, LOYO transport and the b-value are unchanged; "
                              "Mc, the a-value, absolute ML and the conversion intercept shift")

    comp = {}
    for ref in ("ML_ROMPLUS", "Mw"):
        comp[ref] = {}
        for r in REG:
            s = ev[(ev.regime == r)].dropna(subset=[ref])
            d = s.ml - s[ref]
            comp[ref][r] = dict(n=int(len(s)), median=float(d.median()), std=float(d.std()))
    rep["validation_only_external"] = comp
    big = ev[ev.Mw >= 4.5].sort_values("Mw", ascending=False)
    rep["largest_events"] = big[["public_id", "regime", "Mw", "ml", "ml_unanchored", "n_sta",
                                 "n_clipped", "ml_saturation_risk"]].round(3).to_dict("records")

    # ---- station-term table with epochs, n_obs and LOYO stability ----
    st = pd.read_csv(SRC / "station_terms.csv")
    loyo_terms = pd.read_csv(SRC / "loyo_station_transport.csv")
    st["network"] = st.station_key.str.split(".").str[0]
    st["station"] = st.station_key.str.split(".").str[1]
    st["location"] = st.station_key.str.split(".").str[2]
    st["channel_prefix"] = st.station_key.str.split(".").str[3].str.split("@").str[0]
    st["response_epoch"] = st.station_key.str.split("@").str[1]
    st = st[["station_key", "network", "station", "location", "channel_prefix", "response_epoch",
             "S", "n_obs"]].rename(columns={"S": "S_station_term"})
    st.to_csv(OUT / "station_terms.csv", index=False)
    rep["station_terms"] = dict(n=len(st), S_std=float(st.S_station_term.std()),
                                n_lt20_obs=int((st.n_obs < 20).sum()),
                                loyo_within_event_std_no_correction=float(loyo_terms.within_event_std_no_correction.median()),
                                loyo_within_event_std_train_corrections=float(loyo_terms.within_event_std_train_corrections.median()))

    # ---- documented parameter table ----
    jk = src_rep["coeff_jackknife_se"]
    row = lambda p, v, se=None, m="", u="", n="": dict(parameter=p, value=v, se=se, se_method=m, unit=u, note=n)
    par = [row(k, co[k], jk[k], "year jackknife", "", "-log A0 coefficient (unchanged by the anchor)") for k in co]
    par += [row("R1_hinge_km", 70.0, None, "", "km", "crustal, fixed"),
            row("R2_hinge_km", 190.0, None, "", "km", "crustal, fixed"),
            row("depth_split_km", 60.0, None, "", "km", "crustal / intermediate"),
            row("crustal_distance_cap_km", 400.0, None, "", "km", "fit range"),
            row("neg_log_a0_at_100km_crustal", 3.0 + C["crustal"], sys_sigma["crustal"]["C_sigma_total"],
                "year jackknife + Mw systematic", "log10 mm", "Richter fixed point 3.0 plus anchor C"),
            row("neg_log_a0_at_100km_intermediate", 3.0 + C["intermediate"], sys_sigma["intermediate"]["C_sigma_total"],
                "event bootstrap + Mw systematic", "log10 mm", "Richter fixed point 3.0 plus anchor C"),
            row("C_crustal", C["crustal"], sys_sigma["crustal"]["C_sigma_stat"], "year jackknife", "mag",
                f"n={anchors['crustal']['n']}, slope={anchors['crustal']['slope']:.3f}; sequence-dependent "
                f"(2023 supplies {loyo['crustal']['per_year'].get('2023', {}).get('n_dropped', 0)} of "
                f"{anchors['crustal']['n']} anchor events)"),
            row("C_intermediate", C["intermediate"], sys_sigma["intermediate"]["C_sigma_stat"], "event bootstrap",
                "mag", f"n={anchors['intermediate']['n']}, slope={anchors['intermediate']['slope']:.3f}"),
            row("C_crustal_sigma_stat", sys_sigma["crustal"]["C_sigma_stat"], None, "year jackknife", "mag",
                f"event bootstrap gives {anchors['crustal']['sigma']:.3f} and understates it"),
            row("C_crustal_sigma_sys", sys_sigma["crustal"]["C_sigma_sys"], None, "median Mw_sigma_sys", "mag",
                "common-mode systematic of absolute ML"),
            row("C_crustal_sigma_total", sys_sigma["crustal"]["C_sigma_total"], None, "quadrature", "mag", ""),
            row("C_intermediate_sigma_stat", sys_sigma["intermediate"]["C_sigma_stat"], None, "event bootstrap", "mag", ""),
            row("C_intermediate_sigma_sys", sys_sigma["intermediate"]["C_sigma_sys"], None, "median Mw_sigma_sys", "mag",
                "common-mode systematic of absolute ML"),
            row("C_intermediate_sigma_total", sys_sigma["intermediate"]["C_sigma_total"], None, "quadrature", "mag", ""),
            row("within_event_std", src_rep["residuals"]["all"]["within_event_std"], None, "", "mag",
                "median over events"),
            row("Mc_FMD_crustal", rep["frequency_magnitude"]["crustal"]["anchored"]["Mc"], None,
                "max curvature + 0.2", "mag (anchored ML)",
                f"b = {rep['frequency_magnitude']['crustal']['anchored']['b']:.3f}"),
            row("Mc_FMD_intermediate", rep["frequency_magnitude"]["intermediate"]["anchored"]["Mc"], None,
                "max curvature + 0.2", "mag (anchored ML)",
                f"b = {rep['frequency_magnitude']['intermediate']['anchored']['b']:.3f}"),
            ]
    pd.DataFrame(par).to_csv(OUT / "parameter_table.csv", index=False)
    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float)[:4500])


if __name__ == "__main__":
    main()
