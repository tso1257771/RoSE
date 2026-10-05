#!/usr/bin/env python
"""Wood-Anderson amplitudes from the RoSE waveforms.

The amplitude is the mean of the two horizontal zero-to-peak maxima in mm after
instrument-response removal and Wood-Anderson simulation, measured in the
RED-PAN-Motion window. Per-trace diagnostics are computed on the raw counts of
the two horizontals:
  max_h_e/max_h_n   peak |counts| per horizontal
  run_e/run_n       longest run of consecutive samples at that peak
  nmax_e/nmax_n     total samples at the peak
  stuck_e/stuck_n   peak at 24-bit full scale with at least 0.3*npts samples
                    there (dead or DC-saturated component; excluded from the mean)
  clipped           peak at full scale with a flat top of at least 3 consecutive
                    samples on a usable horizontal (event clipping)
  n_horizontals     horizontals used for the mean (2, or 1 if one is stuck)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from functools import lru_cache
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Run from a checkout without installing: the repository root supplies the
# ``rose`` package, ``magnitudes/`` supplies ``_paths``.
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "magnitudes"))
from obspy import Stream, Trace, UTCDateTime, read_inventory  # noqa: E402
from scipy.signal import butter, sosfiltfilt  # noqa: E402

from _paths import work_root  # noqa: E402
from rose.magnitudes.redpan_motion import amplitudes as rpa  # noqa: E402
from rose.magnitudes.redpan_motion import response as rpr  # noqa: E402

PROJECT = work_root()
ROSE = PROJECT / "seisbench_integration/data/rose"
SXML = PROJECT / "seisbench_integration/data/rose_stationxml"
OUT = PROJECT / "romania_ml/outputs/amplitudes"
FULL_SCALE = 2 ** 23 - 1
FULL_SCALE_32 = 2 ** 31 - 1

POST_S_FACTOR = 1.5      # redpan_CWA._matches_to_dataframe defaults
MAX_AMP_WIN_SEC = 80.0
NOISE_WIN_SEC = 3.0


@lru_cache(maxsize=256)
def load_inv(net: str, sta: str):
    p = SXML / net / f"{sta}.xml"
    if not p.exists():
        return None
    try:
        return read_inventory(str(p))
    except Exception:
        return None


def epoch_info(inv, net, sta, loc, pre, t):
    """(response-epoch start date, native sample rate) of the channel at t."""
    for comp in ("Z", "N", "E", "1", "2"):
        sel = inv.select(network=net, station=sta, channel=pre + comp, time=t)
        for n in sel:
            for s in n:
                for c in s:
                    if loc in ("", c.location_code):
                        return str(c.start_date)[:10], float(c.sample_rate or np.nan)
    return "", np.nan


_H5 = {}


def get_h5(year):
    import h5py
    if year not in _H5:
        _H5[year] = h5py.File(ROSE / f"waveforms{year}.hdf5", "r")
    return _H5[year]


def _runs(x, peak):
    """(longest consecutive run at the peak, total samples at the peak)."""
    at = np.abs(np.abs(x) - peak) <= 0.5
    total = int(at.sum())
    if total == 0:
        return 0, 0
    idx = np.flatnonzero(np.diff(np.concatenate(([0], at.view(np.int8), [0]))))
    return int(np.max(idx[1::2] - idx[0::2])), total


def process(row: dict) -> dict:
    out = dict(trace_name=row["trace_name"], year=row["year"], wa_mm=np.nan,
               snr=np.nan, epoch_start="", status="", clipped=False,
               n_horizontals=2, stuck_e=False, stuck_n=False)
    try:
        p_idx = int(row["trace_p_arrival_sample"])
        s_idx = int(row["trace_s_arrival_sample"])
        if p_idx < 0 or s_idx < 0 or s_idx - p_idx < 1:
            out["status"] = "no_ps"
            return out
        net, sta = row["station_network_code"], row["station_code"]
        loc = "" if pd.isna(row["station_location_code"]) else str(row["station_location_code"])
        pre = row["station_channel"]
        inv = load_inv(net, sta)
        if inv is None:
            out["status"] = "no_inventory"
            return out
        bucket, rest = row["trace_name"].split("$")
        idx, _, npts = rest.split(",")
        idx = int(idx)
        npts = int(npts.replace(":", ""))
        wf = get_h5(row["year"])["data/" + bucket][idx, :, :npts].astype(np.float64)
        # RoSE component order ZNE -> RED-PAN (E, N, Z)
        enz = wf[[2, 1, 0]]
        # --- raw-count clipping / stuck diagnostics on the two horizontals ---
        for lab, k in (("e", 0), ("n", 1)):
            x = enz[k]
            pk = float(np.max(np.abs(x))) if x.size else 0.0
            run, tot = _runs(x, pk) if pk > 0 else (0, 0)
            at_fs = pk >= FULL_SCALE - 2 or pk >= FULL_SCALE_32 - 2
            out[f"max_h_{lab}"] = pk
            out[f"run_{lab}"] = run
            out[f"nmax_{lab}"] = tot
            out[f"stuck_{lab}"] = bool(at_fs and tot >= 0.3 * enz.shape[1])
            out[f"clip_{lab}"] = bool(at_fs and run >= 3 and not out[f"stuck_{lab}"])
        if out["stuck_e"] and out["stuck_n"]:
            out["status"] = "both_horizontals_stuck"
            return out
        if out["stuck_e"] or out["stuck_n"]:
            good = 1 if out["stuck_e"] else 0
            enz = enz.copy()
            enz[1 - good] = enz[good]     # WA mean then equals the usable component
            out["n_horizontals"] = 1
        out["clipped"] = bool((out["clip_e"] and not out["stuck_e"])
                              or (out["clip_n"] and not out["stuck_n"]))
        t0 = UTCDateTime(row["trace_start_time"])
        dt = 1.0 / rpr.FS
        n = enz.shape[1]
        out["epoch_start"], native_sr = epoch_info(inv, net, sta, loc, pre, t0 + s_idx * dt)
        out["native_sr"] = native_sr
        # RoSE stores every channel resampled to 100 Hz. For channels recorded at
        # a lower native rate (20/40 Hz BH), remove_response evaluates the FIR
        # stages above the native Nyquist, where the water level then amplifies
        # interpolation noise (+0.38 log10 units median on 20-Hz BH in the pilot).
        # Deviation from the reference chain: zero-phase 4th-order low-pass at
        # 0.8 x native Nyquist on the raw counts for those channels only.
        out["lowpass_hz"] = np.nan
        if np.isfinite(native_sr) and native_sr < 99.0:
            fc = 0.4 * native_sr
            sos = butter(4, fc, fs=rpr.FS, output="sos")
            enz = sosfiltfilt(sos, enz - enz.mean(axis=1, keepdims=True), axis=1)
            out["lowpass_hz"] = fc

        sp_diff = s_idx - p_idx
        p_amp_npts = max(1, sp_diff)
        s_amp_end = min(s_idx + int(POST_S_FACTOR * sp_diff),
                        p_idx + int(MAX_AMP_WIN_SEC / dt), n)
        s_amp_npts_nominal = max(1, s_amp_end - s_idx)
        # sensitivity-corrected copy for SNR / adaptive window (as redpan_CWA)
        sens = np.array([row["trace_sensitivity_e"], row["trace_sensitivity_n"],
                         row["trace_sensitivity_z"]], float)
        if np.all(np.isfinite(sens)) and np.all(sens != 0):
            # redpan_CWA: demean + bandpass 3-45 Hz, then divide by sensitivity
            snr_wf = Stream([Trace(data=enz[k].copy(), header={"sampling_rate": rpr.FS})
                             for k in range(3)])
            snr_wf.detrend("demean").filter("bandpass", freqmin=3.0, freqmax=45.0)
            phys = np.array([tr.data for tr in snr_wf]) / sens[:, None]
            s_amp_npts = rpa.adaptive_s_window(phys, s_idx, s_amp_npts_nominal, p_idx, dt)
            out["snr"] = rpa.compute_snr(phys, s_idx, s_amp_npts,
                                         max(1, int(NOISE_WIN_SEC / dt)), p_idx)
        else:
            s_amp_npts = s_amp_npts_nominal

        sp_sec = sp_diff * dt
        hyp_proxy = rpr.WADATI_VPVS_FACTOR * sp_sec
        t_post = min(rpr.MAG_WINDOW_TMIN_RATIO * sp_sec + rpr.MAG_WINDOW_ALPHA * hyp_proxy,
                     rpr.MAG_WINDOW_TMAX)
        mag_start = max(0, p_idx - int(rpr.MAG_WINDOW_PRE_P / dt))
        mag_end = min(n, s_idx + int(t_post / dt))
        if mag_end <= mag_start:
            mag_end = min(n, mag_start + max(1, p_amp_npts + s_amp_npts))
        out["mag_window_truncated"] = bool(s_idx + int(t_post / dt) > n)
        wa_mm, _, _ = rpa.full_response_amplitudes(
            enz, p_idx, p_amp_npts, s_idx, s_amp_npts, mag_start, mag_end,
            inv, net, sta, loc, pre, t0)
        out["wa_mm"] = wa_mm
        out["status"] = "ok" if np.isfinite(wa_mm) else "wa_nan"
    except Exception as exc:  # keep going; record reason
        out["status"] = f"error:{type(exc).__name__}"
    return out


COLS = ["trace_name", "trace_start_time", "trace_p_arrival_sample", "trace_s_arrival_sample",
        "station_network_code", "station_code", "station_location_code", "station_channel",
        "trace_sensitivity_z", "trace_sensitivity_n", "trace_sensitivity_e"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    import warnings
    warnings.filterwarnings("ignore")
    for year in a.years:
        dest = OUT / f"wa_{year}{a.tag}.csv"
        if dest.exists():
            print(f"skip {dest} (exists)", flush=True)
            continue
        md = pd.read_csv(ROSE / f"metadata{year}.csv", usecols=COLS, low_memory=False)
        md["year"] = year
        if a.limit:
            md = md.sample(n=min(a.limit, len(md)), random_state=1)
        # sort by bucket/index to keep HDF5 reads sequential within a worker
        rows = md.to_dict("records")
        t = time.time()
        with Pool(a.workers) as pool:
            res = pool.map(process, rows, chunksize=64)
        df = pd.DataFrame(res)
        df.to_csv(dest.with_suffix(".tmp"), index=False)
        os.replace(dest.with_suffix(".tmp"), dest)
        el = time.time() - t
        print(f"{year}: {len(df)} traces in {el:.0f}s ({el/len(df)*a.workers:.3f} cpu-s/trace); "
              f"status={df.status.value_counts().to_dict()}", flush=True)


if __name__ == "__main__":
    main()
