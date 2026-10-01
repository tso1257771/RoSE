"""
Run SourceSpec over the Vrancea catalogue and collect one Mw per event.

  * Base configuration: vrancea.conf.
  * Per-event overrides derived from the catalogue ML, so that the S window and
    the spectral band resolve the low-frequency plateau of large events:

        M0_exp  = 10**(1.5*ML + 9.1)                                [N m]
        fc_exp  = 0.37 * BETA * (16*DSIGMA / (7*M0_exp))**(1/3)      [Hz]
                  (Brune 1970, k = 0.37, BETA = 3500 m/s, DSIGMA = 1 MPa; a low
                   stress drop gives a low fc_exp, i.e. a conservatively long
                   window and low band)
        win_floor    = clip(9 / fc_exp, 5, 30)                       s
        freq1_{broadb,acc,disp} = clip(fc_exp / 4, 0.05, 0.5)        Hz
        bp_freqmin_{broadb,disp} = max(freq1 / 2, 0.05); bp_freqmin_acc = max(freq1 / 2, 0.10)
        fc_exp < 1 Hz : short-period channels (EH?, SH?) ignored; their 1-Hz
                        sensors cannot constrain the plateau.
        fc_exp > 2 Hz : clipping detection off; raw counts of small events are
                        microseism-dominated and would be misflagged.
        BH?/SH? channels (mostly 20 sps, upsampled) are kept: their roll-off
        biases fc of small events but not Mw.
    ML selects windows and bands only; it never enters the Mw value.
  * Per-trace S window from the RED-PAN-Motion rule (S-P based nominal window
    with amplitude-decay truncation), applied by source_spec_redpan_windows.py
    and never shorter than win_floor. Traces beyond HYPO_DIST_MAX_KM
    hypocentral distance are skipped.
  * Per-event QuakeML is written into the output directory; the event bundles
    are left untouched.
  * Resumable: events already present in the results JSONL are skipped, and
    each finished event is appended immediately.

Usage:
    python run_sourcespec.py --jobs 6                 # all events
    python run_sourcespec.py --events 2016_0000850 --jobs 1
"""
import argparse
import json
import logging
import multiprocessing as mp
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from obspy import Catalog, UTCDateTime
from obspy.core.event import Event, Magnitude, Origin, Pick, WaveformStreamID

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from _paths import work_root  # noqa: E402

ROOT = work_root()
MW_ROOT = ROOT / "romania_mw"
BUNDLES = MW_ROOT / "data" / "all_events"
SAC_ROOT = ROOT / "sac"
INVENTORY = (ROOT / "seisbench_integration" / "data" / "rose_stationxml"
             / "_merged_for_sourcespec.xml")
BASE_CONFIG = MW_ROOT / "sourcespec" / "vrancea.conf"
# SourceSpec 1.8 pins obspy and pandas versions that the rest of this
# repository does not, so it runs from its own environment. SS_PYTHON names
# that interpreter; see sourcespec/requirements-sourcespec.txt.
PYTHON = Path(os.environ.get("SS_PYTHON") or sys.executable)
SOURCE_SPEC = MW_ROOT / "sourcespec" / "source_spec_redpan_windows.py"
OUT_ROOT = MW_ROOT / "sourcespec" / os.environ.get("SS_OUT", "outputs")
RESULTS_JSONL = OUT_ROOT / "results.jsonl"
SUMMARY_CSV = MW_ROOT / "outputs" / os.environ.get("SS_SUMMARY", "sourcespec_mw.csv")

BETA = 3500.0      # m/s, window-design shear velocity
DSIGMA = 1.0e6     # Pa, window-design stress drop
WIN_MIN, WIN_MAX = 5.0, 30.0
F1_MIN, F1_MAX = 0.05, 0.5
FC_DROP_SHORTP = 1.0   # fc_exp < 1 Hz (ML >~ 4.4): drop EH/SH
FC_NO_CLIPCHECK = 2.0  # fc_exp > 2 Hz (ML <~ 3.6): no clipping detection
HYPO_DIST_MAX_KM = 300.0
BP_MIN_BROADB, BP_MIN_ACC = 0.05, 0.10   # pre-filter floors (Hz)

log = logging.getLogger("run_sourcespec")


def event_overrides(ml: float) -> dict:
    """Window / band overrides for one event, derived from catalog ML."""
    m0 = 10 ** (1.5 * ml + 9.1)
    fc_exp = 0.37 * BETA * (16 * DSIGMA / (7 * m0)) ** (1 / 3)
    win = float(np.clip(9.0 / fc_exp, WIN_MIN, WIN_MAX))
    f1 = float(np.clip(fc_exp / 4.0, F1_MIN, F1_MAX))
    ov = {
        "fc_exp": fc_exp,
        "win_length": round(win, 2),
        "freq1_broadb": round(f1, 3),
        "freq1_acc": round(f1, 3),
        "freq1_disp": round(f1, 3),
        "bp_freqmin_broadb": round(max(f1 / 2, BP_MIN_BROADB), 3),
        "bp_freqmin_acc": round(max(f1 / 2, BP_MIN_ACC), 3),
        "bp_freqmin_disp": round(max(f1 / 2, BP_MIN_BROADB), 3),
    }
    # SourceSpec matches ignore_traceids as regexes after escaping "."
    if fc_exp < FC_DROP_SHORTP:
        ov["ignore_traceids"] = r"\w*.\w*.\w*.[ES]H\w"
    if fc_exp > FC_NO_CLIPCHECK:
        ov["clipping_detection_algorithm"] = "none"
    return ov


def write_config(outdir: Path, ov: dict) -> Path:
    """Base config with per-event keys replaced (or appended) and inventory."""
    lines = BASE_CONFIG.read_text().splitlines()
    keys = {k: v for k, v in ov.items() if k != "fc_exp"}
    keys["station_metadata"] = str(INVENTORY)
    seen = set()
    out = []
    for ln in lines:
        k = ln.split("=", 1)[0].strip() if "=" in ln and not ln.lstrip().startswith("#") else None
        if k in keys:
            out.append(f"{k} = {keys[k]}")
            seen.add(k)
        else:
            out.append(ln)
    out.append("\n# --- per-event overrides (run_sourcespec.py) ---")
    out += [f"{k} = {v}" for k, v in keys.items() if k not in seen]
    cfg = outdir / "source_spec.conf"
    cfg.write_text("\n".join(out) + "\n")
    return cfg


def build_quakeml(event_dir: Path, outdir: Path) -> tuple[Path, float]:
    meta = yaml.safe_load((event_dir / "event.yaml").read_text())
    picks = pd.read_csv(event_dir / "picks.csv", low_memory=False)

    ev = Event(resource_id=f"smi:niep/event/{meta['event_id']}")
    origin = Origin(
        resource_id=f"smi:niep/origin/{meta['event_id']}",
        time=UTCDateTime(meta["origin_time"]),
        latitude=meta["latitude_deg"],
        longitude=meta["longitude_deg"],
        depth=meta["depth_km"] * 1000.0,
    )
    ev.origins.append(origin)
    ev.preferred_origin_id = origin.resource_id
    mag = Magnitude(mag=meta["catalog_magnitude"], magnitude_type="ML",
                    origin_id=origin.resource_id)
    ev.magnitudes.append(mag)
    ev.preferred_magnitude_id = mag.resource_id

    for _, r in picks.iterrows():
        arr = next((r[c] for c in ("final_arrival", "manual_arrival", "theo_arrival")
                    if c in r and pd.notna(r[c]) and r[c]), None)
        if arr is None:
            continue
        parts = str(r["station_id"]).split(".")
        if len(parts) < 4:
            continue
        net, sta, loc, band = parts[:4]
        ev.picks.append(Pick(
            resource_id=f"smi:niep/pick/{meta['event_id']}_{sta}_{r['phase_hint']}",
            time=UTCDateTime(str(arr)),
            waveform_id=WaveformStreamID(network_code=net, station_code=sta,
                                         location_code=loc,
                                         channel_code=f"{band}Z"),
            phase_hint=str(r["phase_hint"]).upper()[0],
        ))
    qml = outdir / "event.xml"
    Catalog(events=[ev]).write(str(qml), format="QUAKEML")
    return qml, float(meta["catalog_magnitude"])


def parse_output(outdir: Path, eid: str) -> dict:
    yml = outdir / eid / f"{eid}.ssp.yaml"
    if not yml.exists():
        return {"status": "no_yaml"}
    info = yaml.safe_load(yml.read_text())
    ssp = info.get("summary_spectral_parameters", {}) or {}
    out = {"status": "ok"}
    for key, spec_key in [("Mw", "Mw"), ("Mo", "Mo"), ("fc", "fc"),
                          ("t_star", "t_star"), ("Qo", "Qo"),
                          ("static_stress_drop", "ssd"),
                          ("radiated_energy", "Er")]:
        block = ssp.get(spec_key, {}) or {}
        wm = block.get("weighted_mean") or {}
        m = block.get("mean") or {}
        out[f"{key}_wmean"] = wm.get("value")
        out[f"{key}_wmean_err"] = wm.get("uncertainty")
        out[f"{key}_mean"] = m.get("value")
        out[f"{key}_nobs"] = wm.get("nobs")
    # station-level: fraction of stations whose t* or fc hit a bound
    stations = info.get("station_parameters", {}) or {}
    n = n_tlo = n_thi = n_fchi = 0
    for sp in stations.values():
        t = (sp.get("t_star") or {}).get("value")
        fc = (sp.get("fc") or {}).get("value")
        if t is None:
            continue
        n += 1
        n_tlo += t <= 0.0051
        n_thi += t >= 0.1499
        n_fchi += fc is not None and fc >= 39.9
    out.update(n_station_fits=n, frac_tstar_lo=n_tlo / n if n else None,
               frac_tstar_hi=n_thi / n if n else None,
               frac_fc_hi=n_fchi / n if n else None)
    # window diagnostics from the wrapper's per-trace log lines
    logf = outdir / eid / f"{eid}.ssp.log"
    if logf.exists():
        lines = [ln for ln in logf.read_text().splitlines() if "RED-PAN window" in ln]
        if lines:
            out["n_windows"] = len(lines)
            out["frac_floor_used"] = sum("floor_used=True" in ln for ln in lines) / len(lines)
    return out


def run_one(eid: str) -> dict:
    year, num = eid.split("_")
    sac_dir = SAC_ROOT / year / num
    outdir = OUT_ROOT / eid
    outdir.mkdir(parents=True, exist_ok=True)
    row = {"event_index": eid}
    try:
        qml, ml = build_quakeml(BUNDLES / eid, outdir)
    except Exception as e:  # noqa: BLE001 — record and continue the batch
        return {**row, "status": "qml_build_failed", "error": repr(e)}
    ov = event_overrides(ml)
    row.update(ML=ml, **{f"cfg_{k}": v for k, v in ov.items()})
    if not sac_dir.exists():
        return {**row, "status": "no_sac"}
    cfg = write_config(outdir, ov)
    cmd = [str(PYTHON), str(SOURCE_SPEC), "-c", str(cfg), "-q", str(qml), "-t", str(sac_dir),
           "-o", str(outdir), "-e", eid, "-n", "ROMPLUS"]
    try:
        env = {**os.environ, "SSP_WIN_FLOOR": str(ov["win_length"]),
               "SSP_HYPO_DIST_MAX": str(HYPO_DIST_MAX_KM), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, env=env)
    except subprocess.TimeoutExpired:
        return {**row, "status": "timeout"}
    if proc.returncode != 0:
        (outdir / "stderr.log").write_text(proc.stderr)
        tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else ""
        return {**row, "status": "source_spec_failed", "error": tail}
    row.update(parse_output(outdir, eid))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--events", nargs="*")
    ap.add_argument("--seed", type=int, default=20260917)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.events:
        events = args.events
    else:
        events = sorted(p.name for p in BUNDLES.iterdir())

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = set()
    if RESULTS_JSONL.exists():
        for ln in RESULTS_JSONL.read_text().splitlines():
            r = json.loads(ln)
            if r.get("status") == "ok":
                done.add(r["event_index"])
    todo = [e for e in events if e not in done]
    log.info("%d events requested, %d already ok, %d to run (jobs=%d)",
             len(events), len(events) - len(todo), len(todo), args.jobs)

    with open(RESULTS_JSONL, "a") as fh, mp.Pool(args.jobs) as pool:
        for i, row in enumerate(pool.imap_unordered(run_one, todo), 1):
            fh.write(json.dumps(row, default=float) + "\n")
            fh.flush()
            if i % 50 == 0 or i == len(todo):
                log.info("%d / %d done", i, len(todo))

    # summary: last row per event wins
    rows = [json.loads(ln) for ln in RESULTS_JSONL.read_text().splitlines()]
    df = pd.DataFrame(rows).drop_duplicates("event_index", keep="last")
    tmp = SUMMARY_CSV.with_suffix(".csv.tmp")
    df.sort_values("event_index").to_csv(tmp, index=False)
    os.replace(tmp, SUMMARY_CSV)
    log.info("Wrote %s\n%s", SUMMARY_CSV, df["status"].value_counts())


if __name__ == "__main__":
    main()
