#!/usr/bin/env python
"""Figures and two consistency checks for the ML scale.

Figures:
  fig_ml_minus_mw_anchored.png           ML - Mw vs Mw per regime, with the
                                         fitted trend and the anchor (Mw 4.0, 0)
  fig_station_residual_large_events.png  station residual vs distance, Mw >= 4.5
  fig_clipped_fraction_vs_magnitude.png  clipped and rejected fraction vs ML
Checks (validation_extra.json):
  * median ML - Mw in a bin centred on the anchor (Mw 3.75-4.25)
  * b-value invariance: b on the same event set before and after the shift
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import ml_root  # noqa: E402
from rose.magnitudes.attenuation import neg_log_a0  # noqa: E402

ROOT = ml_root()
FIT = ROOT / "outputs/fit"              # published ML
SRC = ROOT / "outputs/fit_unanchored"   # ML before the baseline shift


def aki_b(m, mc, dm=0.1):
    s = np.asarray(m, float)
    s = s[s >= mc - dm / 2]
    return float(np.log10(np.e) / (s.mean() - (mc - dm / 2))), int(len(s))


def main():
    ev = pd.read_csv(FIT / "event_ml.csv")
    rep = json.loads((FIT / "report.json").read_text())
    C = rep["C"]
    co = json.loads((SRC / "report.json").read_text())["coeffs"]
    st = pd.read_csv(FIT / "station_terms.csv").set_index("station_key").S_station_term

    n_risk_total = int(ev.ml_saturation_risk.sum())
    extra = {"anchor_centred_bin": {}, "b_value_invariance": {}}
    for r in ("crustal", "intermediate"):
        s = ev[(ev.regime == r) & ev.Mw.between(3.75, 4.25) & ev.Mw_quality.isin(["A", "B"])]
        extra["anchor_centred_bin"][r] = dict(n=int(len(s)), median_ML_minus_Mw=float((s.ml - s.Mw).median()))
        a = ev[ev.regime == r]
        mc_un = 1.9 if r == "crustal" else 2.1
        b_un, n_un = aki_b(a.ml_unanchored, mc_un)
        b_an, n_an = aki_b(a.ml, mc_un + C[r])
        extra["b_value_invariance"][r] = dict(mc_unanchored=mc_un, mc_anchored=mc_un + C[r],
                                              b_unanchored=b_un, b_anchored=b_an, n=n_un,
                                              same_event_set=bool(n_un == n_an))
    (FIT / "validation_extra.json").write_text(json.dumps(extra, indent=2))

    # --- figure 1 ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for r, ax in zip(("crustal", "intermediate"), axes):
        s = ev[(ev.regime == r) & ev.Mw.notna() & ev.Mw_quality.isin(["A", "B"])]
        d = s.ml - s.Mw
        ax.scatter(s.Mw, d, s=5, alpha=.2, color="#1f77b4", label="events")
        b = s.groupby(pd.cut(s.Mw, np.arange(1.5, 6.51, 0.5)), observed=True).apply(
            lambda g: pd.Series({"x": g.Mw.median(), "y": (g.ml - g.Mw).median()}))
        ax.plot(b.x, b.y, "ko-", ms=4, label="binned median")
        counts = s.groupby(pd.cut(s.Mw, np.arange(1.5, 6.51, 0.5)), observed=True).size()
        for xi, yi, ni in zip(b.x, b.y, counts.values):
            ax.annotate(f"{ni}", (xi, yi), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6, color="0.3")
        risk = s[s.ml_saturation_risk]
        if len(risk):
            ax.scatter(risk.Mw, risk.ml - risk.Mw, s=45, facecolors="none", edgecolors="darkorange",
                       linewidths=1.2, label=f"saturation-risk flag (n={len(risk)} in panel, {n_risk_total} total)")
        # trend of the anchor fit: (ML - Mw) = -slope_anchor * (Mw - 4), i.e.
        # dML/dMw = 1 - slope_anchor (about 1.4 in both regimes)
        sa = rep["anchor"][r]["slope"]
        xx = np.linspace(2, 6.2, 50)
        ax.plot(xx, -sa * (xx - 4.0), "r--", lw=1.2,
                label=f"anchor trend (dML/dMw = {1 - sa:.2f})")
        ax.plot([4.0], [0.0], "r*", ms=14, label="anchor (Mw 4.0)")
        ax.axhline(0, color="gray", lw=.6)
        ax.axvline(4.0, color="gray", lw=.6, ls=":")
        ax.set_xlabel("Mw (SourceSpec)")
        ax.set_title(f"{r}  (C = {C[r]:+.3f})")
        ax.set_xlim(1.5, 6.5)
        ax.set_ylim(-1.5, 1.5)
    axes[0].set_ylabel("ML - Mw")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIT / "fig_ml_minus_mw_anchored.png", dpi=130)

    # --- figure 2 ---
    obs = pd.read_csv(ROOT / "outputs/observations_fit.csv.gz", low_memory=False)
    big = ev[ev.Mw >= 4.5].public_id
    o = obs[obs.public_id.isin(big) & ~obs.clipped].copy()
    o["regime"] = np.where(o.depth_km >= 60, "intermediate", "crustal")
    o["ml_sta"] = (o.log10_A + neg_log_a0(o.R_km, o.depth_km, co) - o.station_key.map(st)
                   + o.regime.map(C))
    o = o.merge(ev[["public_id", "ml"]], on="public_id")
    o["resid"] = o.ml_sta - o.ml
    fig, ax = plt.subplots(figsize=(7, 4.4))
    for r, c in (("crustal", "#1f77b4"), ("intermediate", "#d62728")):
        s = o[o.regime == r]
        # crustal observations beyond 400 km are outside the fit range and
        # enter neither the fit nor the event ML: show them, but marked as excluded
        used = s if r == "intermediate" else s[s.R_km <= 400]
        drop = s.iloc[0:0] if r == "intermediate" else s[s.R_km > 400]
        ax.scatter(used.R_km, used.resid, s=6, alpha=.35, color=c, label=f"{r} (n={len(used)})")
        if len(drop):
            ax.scatter(drop.R_km, drop.resid, s=22, facecolors="none", edgecolors=c, linewidths=.8,
                       label=f"{r}, excluded R > 400 km (n={len(drop)})")
        edges = [0, 50, 100, 150, 200, 250, 300, 400] if r == "crustal" else [0, 50, 100, 150, 200, 250, 300, 400, 600]
        g = used.groupby(pd.cut(used.R_km, edges), observed=True).resid.median()
        ax.plot([i.mid for i in g.index], g.values, "o-", color=c)
    ax.axhline(0, color="k", lw=.7)
    ax.set_xlabel("Hypocentral distance (km)")
    ax.set_ylabel("Station ML - event ML")
    ax.set_title("Station residuals, events with Mw >= 4.5 (anchored ML)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIT / "fig_station_residual_large_events.png", dpi=130)

    # --- figure 3 ---
    ev["frac_rejected"] = ev.n_blowup_rejected / ev.n_obs_total
    bins = np.arange(-0.5, 6.25, 0.5)
    g = ev.groupby(pd.cut(ev.ml, bins), observed=True)
    tab = pd.DataFrame({"n_events": g.size(), "clipped": g.frac_clipped.mean(),
                        "rejected": g.frac_rejected.mean()}).reset_index()
    x = [i.mid for i in tab.iloc[:, 0]]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, 100 * tab.clipped, "o-", label="clipped observations")
    ax.plot(x, 100 * tab.rejected, "s-", label="blow-up rejected")
    ax.set_xlabel("event ML (anchored)")
    ax.set_ylabel("mean fraction of observations (%)")
    ax.set_title("Clipped and rejected observations vs magnitude")
    ax.legend(fontsize=8)
    ax2 = ax.twinx()
    ax2.bar(x, tab.n_events, width=0.4, alpha=.15, color="gray")
    ax2.set_yscale("log")
    ax2.set_ylabel("events per bin")
    fig.tight_layout()
    fig.savefig(FIT / "fig_clipped_fraction_vs_magnitude.png", dpi=130)
    tab.astype({tab.columns[0]: str}).to_csv(FIT / "clipped_fraction_by_magnitude.csv", index=False)
    print(json.dumps(extra, indent=2))


if __name__ == "__main__":
    main()
