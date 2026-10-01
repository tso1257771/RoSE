"""
Mw catalogue for Vrancea from the SourceSpec run.

No external magnitudes (USGS, Grond, ROMPLUS, catalog ML) enter the Mw values
or their uncertainties:

    Mw = SourceSpec event Mw (weighted mean of station Mw)

USGS Mww/Mwr for 17 events are compared afterwards as VALIDATION only
(written to Mw_validation.yaml); nothing is fitted to them.

Per-event uncertainty (rose.magnitudes.mw_catalog.mw_sigma):
    Mw_sigma = sqrt( sigma_stat^2 + sigma_sys(Mw)^2 )
    sigma_stat    : s_hat / sqrt(n), s_hat^2 = [(n-1) s^2 + 2 S0^2] / (n+1)
                    s = station std of Mw (SourceSpec weighted std), n = stations,
                    S0 = the median within-event station std of the run
                    (shrinks s for events with few stations)
    sigma_sys(Mw) : linear in Mw between two bounds. 0.10 = station-distance
                    systematic (within-event distance trend of station Mw);
                    0.18 = 0.10 combined with the +-0.15 sensitivity of small-event
                    Mw to fc / t* / weighting choices. The distance
                    systematic is one-signed for small crustal events (an empirical
                    distance term would lower crustal Mw < 2 by ~0.15); not applied.

Source parameters (fc_Hz, stress_drop_MPa, t_star_s): station means, reported
only for Mw >= SOURCE_PARAM_MIN_MW (band-limited below).

Quality flag, measurement only (rose.magnitudes.mw_catalog.mw_quality):
    A  5+ sites, frac_tstar_lo < 0.3, fc not at a bound, SE < 0.15
    B  3+ sites
    C  otherwise

Outputs (romania_mw/outputs/):
    Mw_catalog.csv, Mw_validation.yaml, Mw_validation.png
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import mw_root, work_root  # noqa: E402
from rose.magnitudes.attenuation import DEPTH_SPLIT as SLAB_DEPTH_KM  # noqa: E402
from rose.magnitudes.mw_catalog import mw_quality, mw_sigma  # noqa: E402

ROOT = work_root()
MW = mw_root()
OUT = MW / "outputs"
SS_CSV = OUT / "sourcespec_mw.csv"
CATALOG = ROOT / "outputs/reloc_results_hypoDD3D/Enhanced_ROMPLUS_catalog.csv"
SIG_SYS_LARGE, SIG_SYS_SMALL = 0.10, 0.18
MW_SYS_LARGE, MW_SYS_SMALL = 3.8, 2.0
SOURCE_PARAM_MIN_MW = 3.5
STATION_FITS = OUT / "station_fits.csv.gz"


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def build_catalog() -> pd.DataFrame:
    cat = pd.read_csv(CATALOG)[["event_index", "time", "latitude", "longitude", "depth", "ML"]]
    # the hypoDD3D catalogue magnitude; used only to design the spectral windows
    # (ML_type is 'mw' for 118 events), never as a magnitude in this product
    cat = cat.rename(columns={"ML": "ML_catalog_window_design"})
    ss = pd.read_csv(SS_CSV)
    ss = ss[ss["status"] == "ok"].drop(columns=["ML"], errors="ignore")
    df = cat.merge(ss, on="event_index", how="left")

    df["Mw"] = df["Mw_wmean"]
    # Mw_nobs counts station-channel fits: the two instruments of one sensor
    # site (BH and HH, say) are fitted separately but do not sample independent
    # ground motion. Quality and the station term of sigma use distinct sites.
    fits = pd.read_csv(STATION_FITS, dtype={"loc": str})
    fits = fits[~fits["Mw_outlier"]]
    nsta = (fits.assign(site=fits["net"] + "." + fits["sta"])
                .groupby("event_index")["site"].nunique().rename("Mw_nsta"))
    df = df.merge(nsta, on="event_index", how="left")
    df["Mw_nsta"] = df["Mw_nsta"].where(df["Mw"].notna())
    df["Mw_sigma"], df["Mw_sigma_sys"] = mw_sigma(
        df["Mw"], df["Mw_wmean_err"], df["Mw_nsta"])
    df["Mw_quality"] = mw_quality(
        df["Mw"], df["Mw_wmean_err"], df["Mw_nsta"], df["fc_wmean"], df["frac_tstar_lo"])

    small = ~(df["Mw"] >= SOURCE_PARAM_MIN_MW)
    df["fc_Hz"] = df["fc_mean"].mask(small)
    df["stress_drop_MPa"] = df["static_stress_drop_mean"].mask(small)
    df["t_star_s"] = df["t_star_mean"].mask(small)
    df["Mw_nobs"] = df["Mw_nobs"].astype("Int64")
    df["Mw_nsta"] = df["Mw_nsta"].astype("Int64")
    df = df.rename(columns={"Mw_wmean_err": "Mw_station_std", "Mw_nobs": "Mw_nfits"})
    cols = ["event_index", "time", "latitude", "longitude", "depth", "ML_catalog_window_design", "Mw",
            "Mw_sigma", "Mw_sigma_sys",
            "Mw_quality", "Mw_nsta", "Mw_nfits", "Mw_station_std", "fc_Hz", "stress_drop_MPa", "t_star_s",
            "frac_tstar_lo", "frac_tstar_hi"]
    return df[cols]


def validate(df: pd.DataFrame) -> dict:
    """Compare with USGS Mww/Mwr. Reported only; nothing is fitted."""
    u = pd.read_csv(OUT / "usgs_cross_validation.csv")[["event_index", "usgs_Mw"]]
    x = df.merge(u, on="event_index").dropna(subset=["Mw"])
    r = x["usgs_Mw"] - x["Mw"]
    slab = x["depth"] >= SLAB_DEPTH_KM
    b, a = np.polyfit(x["Mw"], x["usgs_Mw"], 1)
    rep = {
        "note": "validation only; USGS magnitudes are not used in Mw or Mw_sigma",
        "n": int(len(x)),
        "usgs_minus_Mw": {"mean": float(r.mean()), "median": float(r.median()), "std": float(r.std(ddof=1))},
        "by_regime": {k: {"n": int(len(g)), "mean": float(g.mean()), "std": float(g.std(ddof=1))}
                      for k, g in r.groupby(slab.map({False: "crustal", True: "slab"}))},
        "ols_usgs_on_Mw": {"slope": float(b), "intercept": float(a)},
        "fraction_within_2sigma": float((r.abs() <= 2 * np.sqrt(x["Mw_sigma"] ** 2 + 0.1 ** 2)).mean()),
    }
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.3))
    ax[0].errorbar(x["Mw"], x["usgs_Mw"], xerr=x["Mw_sigma"], fmt="o", ms=4, color="C0")
    lim = [x["Mw"].min() - 0.3, x["Mw"].max() + 0.3]
    ax[0].plot(lim, lim, "k:", lw=0.8)
    ax[0].set(xlabel="Mw (SourceSpec)", ylabel="USGS Mww / Mwr", title=f"(a) Validation, n={len(x)}")
    ax[1].scatter(x["depth"], r, c=np.where(slab, "C3", "C2"), s=18)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(SLAB_DEPTH_KM, color="grey", ls=":")
    ax[1].set(xlabel="depth (km)", ylabel="USGS − Mw", title="(b) Residual vs depth")
    for a_ in ax:
        a_.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "Mw_validation.png", dpi=150)
    return rep


def main():
    df = build_catalog()
    atomic_write(df, OUT / "Mw_catalog.csv")
    rep = {"method": "Mw = SourceSpec event Mw; no external magnitudes in Mw or Mw_sigma",
           "catalog": {"n_events": int(len(df)), "n_Mw": int(df["Mw"].notna().sum()),
                       "quality_counts": {k: int(v) for k, v in df["Mw_quality"].value_counts().items() if k}},
           "validation_USGS": validate(df)}
    (OUT / "Mw_validation.yaml").write_text(yaml.safe_dump(rep, sort_keys=False))
    print(yaml.safe_dump(rep, sort_keys=False))


if __name__ == "__main__":
    main()
