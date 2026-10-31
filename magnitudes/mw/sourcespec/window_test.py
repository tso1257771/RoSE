"""
Does the S window change Mw? Rerun a sample of events with unmodified
SourceSpec 1.8 and compare event Mw with the released values.

The release replaces SourceSpec's time windows (source_spec_redpan_windows.py).
This test runs the same events through plain SourceSpec, with the same
per-event configuration (outputs/<event>/source_spec.conf), in two variants:

  fixed  SourceSpec's default window: from S - 1 s, fixed length win_length
         (the per-event floor, 5 s for most events). Noise window of the same
         length ending at P - 1 s. SourceSpec 1.8 refuses noise_pre_time =
         None, so it is set to win_length + signal_pre_time, the value None
         stands for. With weighting = noise, SourceSpec cuts the signal to the
         noise length when pre-P data are short.
  var    SourceSpec's travel time window: variable_win_length_factor = 0.64,
         i.e. a window of 0.64 times the S travel time, about 1.5 (S - P) for
         Vp/Vs = 1.74, the nominal RED-PAN length. noise_pre_time = 200 s, so
         the noise window starts at the trace start.

In both, the 300 km hypocentral limit of the release is expressed with
SourceSpec's own option, epi_dist_ranges = 0, sqrt(300^2 - depth^2).

Sample: 40 crustal (depth < 60 km) and 40 intermediate depth events in each
released Mw class < 2, 2-3 and 3-4 (random, fixed seed), all events with
Mw >= 4, and the USGS validation events not drawn already. The sample is
written once to outputs/window_test_sample.csv and reused, so a rerun tests
the same events.

Usage (SourceSpec 1.8 environment, see requirements-sourcespec.txt):
    ROMANIA_ROOT=... python magnitudes/mw/sourcespec/window_test.py run --variant fixed   # hours; resumable
    ROMANIA_ROOT=... python magnitudes/mw/sourcespec/window_test.py run --variant var
    ROMANIA_ROOT=... python magnitudes/mw/sourcespec/window_test.py compare   # seconds, from the saved runs

The result is published in magnitudes/calibration/window_test_summary.json and
window_test_comparison.csv (stage `tables` of regenerate_magnitudes.sh).

Outputs, relative to $ROMANIA_ROOT/romania_mw:
    sourcespec/window_test/{fixed,var}/<event>/          SourceSpec runs
    outputs/window_test_sample.csv                       the events
    outputs/window_test_comparison.csv                   Mw per event, released and both variants
    outputs/window_test_summary.json                     the numbers quoted in the text
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Run from a checkout without installing: ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import work_root  # noqa: E402

ROOT = work_root()
MW_ROOT = ROOT / "romania_mw"
SAC_ROOT = ROOT / "sac"
RELEASED = MW_ROOT / "sourcespec" / "outputs"
# SourceSpec 1.8 runs from its own environment (requirements-sourcespec.txt);
# SS_PYTHON names that interpreter.
PYTHON = Path(os.environ.get("SS_PYTHON") or sys.executable)
RUNS = MW_ROOT / "sourcespec" / "window_test"
OUT = MW_ROOT / "outputs"
SAMPLE_CSV = OUT / "window_test_sample.csv"
COMPARISON_CSV = OUT / "window_test_comparison.csv"
SUMMARY_JSON = OUT / "window_test_summary.json"

VARIANTS = ("fixed", "var")
VAR_FACTOR = 0.64
SEED = 20261020
PER_CLASS = 40
HYPO_DIST_MAX_KM = 300.0
SLAB_DEPTH_KM = 60.0
CLASSES = ["<2", "2-3", "3-4", ">=4"]


def draw_sample():
    cat = pd.read_csv(OUT / "Mw_catalog.csv", dtype={"event_index": str}).dropna(subset=["Mw"])
    cat["regime"] = np.where(cat.depth >= SLAB_DEPTH_KM, "slab", "crustal")
    cat["mclass"] = pd.cut(cat.Mw, [0, 2, 3, 4, 9], labels=CLASSES, right=False)
    rows = []
    for (m, _), g in cat.groupby(["mclass", "regime"], observed=True):
        k = len(g) if m == ">=4" else min(len(g), PER_CLASS)
        rows.append(g.sample(k, random_state=SEED))
    s = pd.concat(rows)
    usgs = pd.read_csv(OUT / "usgs_cross_validation.csv", dtype={"event_index": str})
    extra = cat[cat.event_index.isin(usgs.event_index) & ~cat.event_index.isin(s.event_index)].copy()
    extra["mclass"] = "usgs_extra"
    s = pd.concat([s, extra[s.columns]], ignore_index=True)
    return s[["event_index", "Mw", "depth", "regime", "mclass"]]


def config_text(eid, depth_km, variant):
    lines = (RELEASED / eid / "source_spec.conf").read_text().splitlines()
    kv = {ln.split("=", 1)[0].strip(): ln.split("=", 1)[1].strip() for ln in lines
          if "=" in ln and not ln.lstrip().startswith("#")}
    epi_max = float(np.sqrt(max(HYPO_DIST_MAX_KM**2 - depth_km**2, 1.0)))
    noise_pre = (float(kv["win_length"]) + float(kv["signal_pre_time"])
                 if variant == "fixed" else 200.0)
    out = []
    for ln in lines:
        key = (ln.split("=", 1)[0].strip()
               if "=" in ln and not ln.lstrip().startswith("#") else None)
        if key == "noise_pre_time":
            ln = f"noise_pre_time = {noise_pre:.2f}"
        elif key == "epi_dist_ranges":
            ln = f"epi_dist_ranges = 0, {epi_max:.2f}"
        out.append(ln)
    if variant == "var":
        out.append(f"variable_win_length_factor = {VAR_FACTOR}")
    return "\n".join(out) + "\n"


def run_one(args):
    eid, depth, variant = args
    od = RUNS / variant / eid
    if (od / eid / f"{eid}.ssp.yaml").exists():
        return eid, 0
    od.mkdir(parents=True, exist_ok=True)
    cfg = od / "source_spec.conf"
    cfg.write_text(config_text(eid, depth, variant))
    year, num = eid.split("_")
    cmd = [str(PYTHON), "-m", "sourcespec.source_spec", "-c", str(cfg),
           "-q", str(RELEASED / eid / "event.xml"),
           "-t", str(SAC_ROOT / year / num), "-o", str(od), "-e", eid, "-n", "ROMPLUS"]
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, env=env)
    if p.returncode != 0:
        (od / "stderr.log").write_text(p.stderr)
    return eid, p.returncode


def event_mw(path):
    if not path.exists():
        return {}
    info = yaml.safe_load(path.read_text())
    wm = info["summary_spectral_parameters"]["Mw"]["weighted_mean"]
    return {"Mw": wm["value"], "nobs": wm["nobs"]}


def _stats(x):
    x = pd.Series(x).dropna()
    return {"n": int(len(x)), "median": round(float(x.median()), 3),
            "mean": round(float(x.mean()), 3), "std": round(float(x.std()), 3),
            "frac_abs_gt_0.1": round(float((x.abs() > 0.1).mean()), 2)}


def compare():
    s = pd.read_csv(SAMPLE_CSV, dtype={"event_index": str})
    rows = []
    for r in s.itertuples():
        # no depth: the table is published, and event fields stay out of it
        row = {"event_index": r.event_index, "regime": r.regime, "mclass": r.mclass,
               "Mw_released": r.Mw}
        rel = event_mw(RELEASED / r.event_index / r.event_index / f"{r.event_index}.ssp.yaml")
        row["nobs_released"] = rel.get("nobs")
        for v in VARIANTS:
            x = event_mw(RUNS / v / r.event_index / r.event_index / f"{r.event_index}.ssp.yaml")
            row[f"Mw_{v}"] = x.get("Mw")
            row[f"nobs_{v}"] = x.get("nobs")
        rows.append(row)
    d = pd.DataFrame(rows)
    d.to_csv(COMPARISON_CSV, index=False, float_format="%.4f")

    summary = {"n_events": int(len(d)), "var_factor": VAR_FACTOR, "seed": SEED,
               "difference": "variant Mw minus released Mw", "variants": {}}
    for v in VARIANTS:
        diff = d[f"Mw_{v}"] - d.Mw_released
        block = {"all": _stats(diff), "classes": {}}
        for m in CLASSES:
            for reg in ("crustal", "slab"):
                sel = (d.mclass == m) & (d.regime == reg)
                block["classes"][f"{m} {reg}"] = _stats(diff[sel])
        summary["variants"][v] = block

    usgs = pd.read_csv(OUT / "usgs_cross_validation.csv", dtype={"event_index": str})
    j = d.merge(usgs[["event_index", "usgs_Mw"]], on="event_index")
    summary["usgs_minus_mw"] = {}
    for col in ("Mw_released",) + tuple(f"Mw_{v}" for v in VARIANTS):
        k = j.dropna(subset=[col])
        r = k.usgs_Mw - k[col]
        summary["usgs_minus_mw"][col[3:]] = {
            "n": int(len(r)), "mean": round(float(r.mean()), 3), "std": round(float(r.std()), 3),
            "crustal": round(float(r[k.regime == "crustal"].mean()), 3),
            "slab": round(float(r[k.regime == "slab"].mean()), 3)}
    SUMMARY_JSON.write_text(json.dumps(summary, indent=1) + "\n")

    print(f"{len(d)} events; solved: " + ", ".join(
        f"{v} {int(d[f'Mw_{v}'].notna().sum())}" for v in VARIANTS))
    for v in VARIANTS:
        b = summary["variants"][v]
        print(f"\n{v}: Mw - released, all: median {b['all']['median']:+.3f}, "
              f"mean {b['all']['mean']:+.3f}, std {b['all']['std']:.3f}")
        for k, x in b["classes"].items():
            print(f"  Mw {k:12s} n={x['n']:3d}  median {x['median']:+.3f}  mean {x['mean']:+.3f}  "
                  f"std {x['std']:.3f}  |diff|>0.1: {x['frac_abs_gt_0.1']:.2f}")
    print("\nUSGS - Mw:")
    for k, x in summary["usgs_minus_mw"].items():
        print(f"  {k:9s} n={x['n']:2d}  {x['mean']:+.3f} +- {x['std']:.3f}  "
              f"crustal {x['crustal']:+.3f}  slab {x['slab']:+.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("mode", choices=["run", "compare"])
    ap.add_argument("--variant", choices=VARIANTS, default="fixed")
    ap.add_argument("--jobs", type=int, default=12)
    a = ap.parse_args()
    if a.mode == "run":
        if SAMPLE_CSV.exists():
            s = pd.read_csv(SAMPLE_CSV, dtype={"event_index": str})
        else:
            s = draw_sample()
            s.to_csv(SAMPLE_CSV, index=False)
        print(s.groupby(["mclass", "regime"]).size(), flush=True)
        args = [(e, z, a.variant) for e, z in zip(s.event_index, s.depth)]
        with mp.Pool(a.jobs) as pool:
            for i, (eid, rc) in enumerate(pool.imap_unordered(run_one, args), 1):
                if rc != 0 or i % 50 == 0:
                    print(i, eid, "ok" if rc == 0 else f"failed ({rc})", flush=True)
    compare()
    return 0


if __name__ == "__main__":
    sys.exit(main())
