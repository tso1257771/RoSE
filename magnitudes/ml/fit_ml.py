#!/usr/bin/env python
"""Fit the Vrancea local-magnitude scale.

  -log A0: crustal (< 60 km) three segments in log10 R with hinges at 70 and
  190 km plus a non-negative anelastic term; intermediate-depth a single
  segment plus its anelastic term. Huber IRLS with event and station fixed
  effects and a sum-zero station gauge, balanced event weights. Event ML is the
  median over station locations with a 1.5 IQR trim and at least 3 stations.
  Station-term transport is checked leave-one-year-out. No catalogue magnitude
  enters the fit; the baseline constant is set afterwards by anchor_ml_to_mw.py.

Amplitude quality control:
 1. clipped observations (raw-count flat top at 24-bit full scale, at least 3
    consecutive samples) are excluded from the fit and from the event median;
 2. blow-ups are rejected by a 3e4 mm ceiling and by their distance-model
    residual, so genuine large amplitudes are kept;
 3. components stuck at full scale are excluded (one horizontal of RO.LELR..HH).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import ml_root, work_root  # noqa: E402
from rose.magnitudes.attenuation import (  # noqa: E402
    CRUSTAL_RMAX,
    DEPTH_SPLIT,
    FIXED,
    NAMES,
    R1,
    R2,
    columns,
    neg_log_a0,
)
from rose.magnitudes.taiwan_ml.invert import solve  # noqa: E402
from rose.magnitudes.taiwan_ml.model import event_ml  # noqa: E402
from rose.magnitudes.taiwan_ml.station_transport import (  # noqa: E402
    balanced_observation_weights,
    fit_transport_terms,
)

PROJECT = work_root()
ROOT = ml_root()
OUT = ROOT / "outputs/fit_unanchored"
MW_CATALOG = PROJECT / "romania_mw/outputs/Mw_catalog.csv"
DELTA, NIRLS = 1.345, 6
BLOWUP_RESID = 1.5
ZERO = dict(n=0.0, K=0.0, dK=0.0, rref=100.0, h_break=40.0, n_cheb=0, depth_term=True)

# columns, neg_log_a0, NAMES and the hinges come from rose.magnitudes.attenuation,
# so the fit and the released scale cannot drift apart.

def weights(obs):
    return balanced_observation_weights(obs.public_id.astype(str), obs.station_key.astype(str),
                                        np.zeros(len(obs), bool))


def fit(obs):
    ev = pd.Categorical(obs.public_id)
    st = pd.Categorical(obs.station_key)
    ne, ns, no = len(ev.categories), len(st.categories), len(obs)
    A = columns(obs.R_km, obs.depth_km)
    A = -(A - A.mean(axis=0))
    rows = np.repeat(np.arange(no), 2)
    cols = np.empty(2 * no, int)
    cols[0::2] = ev.codes
    cols[1::2] = ne + st.codes
    G = sp.hstack([sp.coo_matrix((np.ones(2 * no), (rows, cols)), shape=(no, ne + ns)),
                   sp.csr_matrix(A)]).tocsr()
    meta = dict(events=list(ev.categories), stations=list(st.categories), atten_names=NAMES, n_e=ne, n_s=ns)
    x, info = solve(G, obs.log10_A.to_numpy(float) + FIXED, meta, base_w=weights(obs),
                    n_irls=NIRLS, huber_delta=DELTA)
    return dict(zip(NAMES, x[ne + ns:].tolist())), pd.Series(x[ne:ne + ns], index=meta["stations"]), info


def event_table(obs, co, S):
    o = obs[obs.station_key.isin(S.index)].copy()
    o["log10_A"] = o.log10_A + neg_log_a0(o.R_km, o.depth_km, co)
    return event_ml(o, ZERO, S, anchor_c=0.0)


def binned(x, y, bins, name):
    g = pd.DataFrame({"bin": pd.cut(x, bins), "y": y}).groupby("bin", observed=True).y
    return pd.DataFrame({"n": g.size(), "mean": g.mean(), "median": g.median(), "std": g.std(),
                         "nmad": g.apply(lambda v: 1.4826 * np.median(np.abs(v - np.median(v))))}
                        ).reset_index().rename(columns={"bin": name})


def slope(x, y):
    ok = np.isfinite(x) & np.isfinite(y)
    return float(np.polyfit(x[ok], y[ok], 1)[0]) if ok.sum() > 2 else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    obs_all = pd.read_csv(ROOT / "outputs/observations_fit.csv.gz", low_memory=False)
    app_all = pd.read_csv(ROOT / "outputs/observations_all.csv.gz", low_memory=False)
    for f in (obs_all, app_all):
        f["regime"] = np.where(f.depth_km >= DEPTH_SPLIT, "intermediate", "crustal")
    keep = lambda f: f[((f.regime == "intermediate") | (f.R_km <= CRUSTAL_RMAX))]  # noqa: E731
    obs_all, app_all = keep(obs_all).reset_index(drop=True), keep(app_all).reset_index(drop=True)
    obs_all["year"] = obs_all.ev_time.str[:4].astype(int)
    rep = dict(fixed_point=dict(R_km=100.0, neg_log_a0=FIXED), hinges=dict(R1=R1, R2=R2),
               blowup_resid_threshold=BLOWUP_RESID,
               clipping=dict(detector="raw-count flat top: peak at 24-bit (or 32-bit) full scale with >= 3 "
                                      "consecutive samples, excluding DC-stuck components",
                             clipped_obs_fit=int(obs_all.clipped.sum()),
                             clipped_obs_all=int(app_all.clipped.sum())),
               stuck_components=dict(
                   single_horizontal_obs_fit=int((obs_all.n_horizontals == 1).sum()),
                   by_station={".".join(k): int(v) for k, v in
                               app_all[app_all.n_horizontals == 1].groupby(["net", "sta", "chan_pre"])
                               .size().sort_values(ascending=False).head(10).items()}))

    # ---- iterate: fit -> reject blow-ups by model residual -> refit ----
    obs = obs_all[~obs_all.clipped].reset_index(drop=True)
    rejected = pd.DataFrame()
    for it in range(2):
        co, S, info = fit(obs)
        ev_int = event_table(obs, co, S)
        o = obs.merge(ev_int[["public_id", "ml"]], on="public_id", how="left")
        o["pred"] = o.ml - neg_log_a0(o.R_km, o.depth_km, co) + o.station_key.map(S)
        bad = (o.log10_A - o.pred) > BLOWUP_RESID
        rep[f"blowups_rejected_pass{it + 1}"] = int(bad.sum())
        if not bad.any():
            break
        rejected = pd.concat([rejected, o[bad]])
        obs = o[~bad].drop(columns=["ml", "pred"]).reset_index(drop=True)
    rejected.to_csv(OUT / "rejected_blowups.csv", index=False)
    rep["blowups_rejected_total"] = int(len(rejected))
    rep["amplitudes_recovered_vs_1e3_cap"] = int(((obs.wa_mm > 1e3)).sum())
    rep["coeffs"] = co
    rep["solver_info"] = info
    rep["fit_population"] = dict(obs=len(obs), events=int(obs.public_id.nunique()),
                                 station_keys=int(obs.station_key.nunique()))

    # ---- jackknife, station terms ----
    jk = []
    for y in sorted(obs.year.unique()):
        cj, _, _ = fit(obs[obs.year != y].reset_index(drop=True))
        jk.append(dict(year=int(y), **cj))
    jk = pd.DataFrame(jk)
    jk.to_csv(OUT / "jackknife_year_coeffs.csv", index=False)
    g = len(jk)
    rep["coeff_jackknife_se"] = {k: float(np.sqrt((g - 1) / g * ((jk[k] - jk[k].mean()) ** 2).sum())) for k in NAMES}

    st = pd.DataFrame({"station_key": S.index, "S": S.values})
    st["n_obs"] = st.station_key.map(obs.station_key.value_counts()).astype(int)
    st.to_csv(OUT / "station_terms.csv", index=False)
    rep["station_terms"] = dict(n=len(st), S_std=float(st.S.std()), S_min=float(st.S.min()), S_max=float(st.S.max()),
                                S_iqr=[float(st.S.quantile(.25)), float(st.S.quantile(.75))],
                                n_lt20_obs=int((st.n_obs < 20).sum()),
                                n_abs_gt_1p5=int((st.S.abs() > 1.5).sum()))

    # ---- residuals ----
    obs["raw"] = obs.log10_A + neg_log_a0(obs.R_km, obs.depth_km, co)
    obs["ml_sta"] = obs.raw - obs.station_key.map(S)
    ev_fit = event_table(obs, co, S)
    obs = obs.merge(ev_fit, on="public_id")
    obs["resid"] = obs.ml_sta - obs.ml
    obs["resid_noS"] = obs.raw - obs.ml
    rep["residuals"] = {}
    tabs = []
    for r, s in [("all", obs)] + list(obs.groupby("regime")):
        rep["residuals"][r] = dict(obs_std=float(s.resid.std()),
                                   obs_nmad=float(1.4826 * np.median(np.abs(s.resid - s.resid.median()))),
                                   within_event_std=float(s.groupby("public_id").resid.std().median()),
                                   within_event_std_noS=float(s.groupby("public_id").resid_noS.std().median()))
        t = binned(s.R_km, s.resid, [0, 25, 50, 75, 100, 150, 200, 250, 300, 400, 600], "R_km")
        t.insert(0, "regime", r)
        tabs.append(t)
    pd.concat(tabs).to_csv(OUT / "residual_vs_distance.csv", index=False)
    binned(obs.ml, obs.resid, [-1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6.5], "ML").to_csv(
        OUT / "residual_vs_magnitude.csv", index=False)

    # ---- application, per-event flags ----
    app = app_all[~app_all.clipped]
    app = app[~app.trace_name.isin(set(rejected.trace_name))]
    ev_all = event_table(app, co, S)
    mw = pd.read_csv(MW_CATALOG, usecols=["event_index", "Mw", "Mw_sigma", "Mw_quality", "fc_Hz"]).rename(
        columns={"event_index": "public_id", "Mw": "Mw"})
    meta = app_all.groupby("public_id").agg(ev_time=("ev_time", "first"), depth_km=("depth_km", "first"),
                                            ML_ROMPLUS=("ML_ROMPLUS", "first"),
                                            n_obs_total=("public_id", "size"),
                                            n_clipped=("clipped", "sum"),
                                            n_truncated=("mag_window_truncated", "sum")).reset_index()
    rej_counts = rejected.groupby("public_id").size().rename("n_blowup_rejected")
    ev_all = (ev_all.merge(meta, on="public_id", how="left")
              .merge(rej_counts, on="public_id", how="left")
              .merge(mw, on="public_id", how="left"))
    ev_all["n_blowup_rejected"] = ev_all.n_blowup_rejected.fillna(0).astype(int)
    ev_all["regime"] = np.where(ev_all.depth_km >= DEPTH_SPLIT, "intermediate", "crustal")
    ev_all["frac_clipped"] = ev_all.n_clipped / ev_all.n_obs_total
    ev_all["ml_saturation_risk"] = (
        (ev_all.fc_Hz < 1.25) | (ev_all.frac_clipped > 0.2)
        | ((ev_all.n_clipped + ev_all.n_blowup_rejected) > 0.2 * ev_all.n_sta))
    thr = np.where(ev_all.regime == "crustal", 5.5, 5.8)   # ML thresholds, refined by the conversion step
    ev_all["ml_valid_range_flag"] = np.where(ev_all.ml >= 6.5, "saturated",
                                             np.where(ev_all.ml >= thr, "passband_limited", "in_range"))
    ev_all.to_csv(OUT / "event_ml.csv", index=False)
    rep["events_with_ml"] = int(len(ev_all))
    big = ev_all[ev_all.Mw >= 4.3].sort_values("Mw", ascending=False)
    rep["largest_events"] = big[["public_id", "regime", "Mw", "ml", "n_sta",
                                 "n_clipped", "n_blowup_rejected"]].round(3).to_dict("records")
    rep["flags_by_magnitude_bin"] = (
        ev_all.assign(bin=pd.cut(ev_all.ml, [-1, 1, 2, 3, 4, 5, 6.5]))
        .groupby("bin", observed=True)
        .apply(lambda s: pd.Series({"n_events": len(s), "mean_frac_clipped": s.frac_clipped.mean(),
                                    "n_saturation_risk": int(s.ml_saturation_risk.sum()),
                                    "n_passband_limited": int((s.ml_valid_range_flag == "passband_limited").sum()),
                                    "n_saturated": int((s.ml_valid_range_flag == "saturated").sum())}))
        .round(4).reset_index().astype({"bin": str}).to_dict("records"))
    comp = {}
    for ref in ("Mw", "ML_ROMPLUS"):
        comp[ref] = {}
        for r, s in [("all", ev_all)] + list(ev_all.groupby("regime")):
            s = s.dropna(subset=[ref])
            dd = s.ml - s[ref]
            comp[ref][r] = dict(n=int(len(s)), median=float(dd.median()), std=float(dd.std()),
                                slope_ml_on_ref=slope(s[ref].to_numpy(), s.ml.to_numpy()))
    rep["reference_comparison"] = comp

    # ---- LOYO ----
    rows = []
    for y in sorted(obs.year.unique()):
        tr, te = obs[obs.year != y], obs[obs.year == y]
        tf = fit_transport_terms(tr.public_id, tr.raw, tr.station_key, weights(tr),
                                 gauge_keys=tr.station_key.unique(), minimum_observations=20)
        te = te[te.station_key.isin(tf.terms.index)]
        out = dict(year=int(y), test_events=int(te.public_id.nunique()))
        for lab, terms in (("no_correction", pd.Series(0.0, index=tf.terms.index)), ("train_corrections", tf.terms)):
            sm = te.raw - te.station_key.map(terms)
            out[f"within_event_std_{lab}"] = float(pd.Series(sm.values).groupby(te.public_id.values).std().median())
        rows.append(out)
    loyo = pd.DataFrame(rows)
    loyo.to_csv(OUT / "loyo_station_transport.csv", index=False)
    rep["loyo_median_over_years"] = {c: float(loyo[c].median()) for c in loyo.columns if c != "year"}

    (OUT / "report.json").write_text(json.dumps(rep, indent=2, default=float))
    print(json.dumps(rep, indent=2, default=float)[:6000])


if __name__ == "__main__":
    main()
