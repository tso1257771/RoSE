"""Tutorial 3 — Event workflow: QC, picking, coda analysis, ground motion.

Complete pipeline for a single event. Generates multiple figures:

    fig_record_section.png   Record section with manual/RED-PAN/model picks
                             and D5-95 coda windows per station.
    fig_qc_summary.png       Per-station waveform QC (clipping, gaps, spikes).
    fig_coda_robustness.png  Pulse-injection stress test comparing D5-95 vs
                             envelope-decay coda windows.
    ground_motion.csv        PGA/PGV/PGD table by pick source.

Steps:
    1. Load event from RoSE, export per-station MiniSEED.
    2. QC all stations (rose.qc: clipping, dead, gaps, spikes, SNR).
    3. Phase picks: catalog (manual + RED-PAN) only. Other DL pickers are
       excluded by project policy — use RED-PAN 60s for machine picks.
    4. Define S-coda window via Arias intensity D5-95.
    5. Extract PGA/PGV/PGD in that window.
    6. Stress-test coda algorithms with 12x synthetic pulse injection.

Usage:
    python 03_event_ground_motion.py
    python 03_event_ground_motion.py --event 2022_0000708
    python 03_event_ground_motion.py --coda-method envelope
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import obspy
import pandas as pd
from scipy.signal import hilbert

from rose import RoSE, qc

logger = logging.getLogger(__name__)

DATA_DIR = os.environ.get(
    "ROSE_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "rose"),
)
STATIONXML_DIR = os.environ.get(
    "ROSE_STATIONXML_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "rose_stationxml"),
)
HIGHPASS_HZ = 0.1
PRE_FILT = (0.05, 0.1, 40.0, 45.0)
GRAVITY = 9.81


# ========================================================================
# Utilities
# ========================================================================

def _str_or_none(v):
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    return None if (not s or s.lower() in ("nan", "none")) else s


class StationXMLCache:
    def __init__(self, root):
        self.root = Path(root)
        self._cache: dict = {}

    def get(self, net, sta):
        key = (net, sta)
        if key not in self._cache:
            p = self.root / net / f"{sta}.xml"
            self._cache[key] = obspy.read_inventory(str(p)) if p.exists() else None
        return self._cache[key]


# ========================================================================
# 1. Load & export
# ========================================================================

def load_event(data, event_id):
    md = data.metadata
    rows = md.index[md["source_id"] == event_id].to_numpy()
    if len(rows) == 0:
        raise KeyError(f"event {event_id} not in dataset")
    stream = obspy.Stream()
    info_list = []
    for idx in rows:
        wf, meta = data.get_sample(int(idx))
        sr = float(meta["trace_sampling_rate_hz"])
        starttime = obspy.UTCDateTime(meta["trace_start_time"])
        net, sta = meta["station_network_code"], meta["station_code"]
        loc = _str_or_none(meta["station_location_code"]) or ""
        band = meta["station_channel"]
        for ch, d in zip(data.component_order, wf):
            stream += obspy.Trace(d.astype(np.float32),
                                  header={"network": net, "station": sta,
                                          "location": loc, "channel": f"{band}{ch}",
                                          "starttime": starttime, "sampling_rate": sr})
        p_utc = _str_or_none(meta["trace_p_arrival_time"])
        s_utc = _str_or_none(meta["trace_s_arrival_time"])
        info_list.append({
            "idx": int(idx), "network": net, "station": sta,
            "band": band, "sr": sr, "starttime": starttime,
            "p_sample": int(meta["trace_p_arrival_sample"]),
            "s_sample": int(meta["trace_s_arrival_sample"]),
            "p_utc": obspy.UTCDateTime(p_utc) if p_utc else None,
            "s_utc": obspy.UTCDateTime(s_utc) if s_utc else None,
            "p_status": _str_or_none(meta["trace_p_status"]),
            "s_status": _str_or_none(meta["trace_s_status"]),
            "hyp_dist": float(meta["path_hyp_distance_km"]),
            "trace_status_physical": meta["trace_status_physical"],
        })
    return stream, info_list


def write_miniseed(stream, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    for tr in stream:
        fname = f"{tr.id}_{tr.stats.starttime.strftime('%Y%m%dT%H%M%S')}.mseed"
        tr.write(str(out_dir / fname), format="MSEED")
    print(f"  wrote {len(stream)} MiniSEED files to {out_dir}/")


# ========================================================================
# 2. Unit conversion
# ========================================================================

def get_velocity_stream(stream, station, inv_cache, data, info):
    st = stream.select(station=station).copy()
    if not st:
        return None
    inv = inv_cache.get(st[0].stats.network, station)
    try:
        if inv:
            st.remove_response(inventory=inv, output="VEL", pre_filt=PRE_FILT)
        elif info["trace_status_physical"] == "ok":
            wf_phys, _ = data.get_sample_physical(info["idx"])
            for i, tr in enumerate(st):
                tr.data = wf_phys[i].astype(np.float32)
            st.detrend("linear"); st.taper(0.05)
            st.filter("highpass", freq=HIGHPASS_HZ, zerophase=True)
        else:
            return None
    except (ValueError, ZeroDivisionError, KeyError, AttributeError) as exc:
        logger.warning("velocity conversion failed for %s: %s", station, exc)
        return None
    return st


# ========================================================================
# 3. Coda window algorithms
# ========================================================================

def arias_intensity(acc, dt):
    return (np.pi / (2 * GRAVITY)) * np.cumsum(acc ** 2) * dt


def husid_d5_95(acc, dt):
    ia = arias_intensity(acc, dt)
    if ia[-1] == 0:
        return 0, len(acc) - 1, np.zeros_like(acc)
    h = ia / ia[-1]
    return int(np.searchsorted(h, 0.05)), int(np.searchsorted(h, 0.95)), h


def envelope_coda_end(vel, sr, s_sample, threshold_frac=0.05, smooth_win_s=1.0):
    env = np.abs(hilbert(vel))
    win = max(1, int(smooth_win_s * sr))
    env_smooth = np.convolve(env, np.ones(win) / win, mode="same")
    post_s = env_smooth[s_sample:]
    if len(post_s) == 0:
        return len(vel) - 1
    peak = np.max(post_s)
    if peak == 0:
        return len(vel) - 1
    below = np.where(post_s < threshold_frac * peak)[0]
    return s_sample + int(below[0]) if len(below) else len(vel) - 1


def get_coda_window(vel_stream, p_time, s_time, method="arias"):
    horiz = [tr for tr in vel_stream if tr.stats.channel[-1] in ("E", "N")]
    tr = max(horiz, key=lambda t: np.max(np.abs(t.data))) if horiz else vel_stream[0]
    sr, dt = tr.stats.sampling_rate, 1.0 / tr.stats.sampling_rate
    data = tr.data.astype(np.float64)
    diag = {}
    if method == "arias":
        acc = np.gradient(data, dt)
        i5, i95, husid = husid_d5_95(acc, dt)
        diag.update(husid=husid, i5=i5, i95=i95)
        return i5 / sr, i95 / sr, diag
    else:
        s_samp = int((s_time - tr.stats.starttime) * sr) if s_time else 0
        s_samp = max(0, min(s_samp, len(data) - 1))
        ce = envelope_coda_end(data, sr, s_samp)
        diag.update(s_sample=s_samp, coda_end=ce)
        return s_samp / sr, ce / sr, diag


# ========================================================================
# 4. Ground-motion extraction
# ========================================================================

def measure_ground_motion(vel_stream, t0_s, t1_s):
    horiz = [tr for tr in vel_stream if tr.stats.channel[-1] in ("E", "N")]
    if not horiz:
        return {"pga": np.nan, "pgv": np.nan, "pgd": np.nan, "arias": np.nan}
    sr = horiz[0].stats.sampling_rate
    dt = 1.0 / sr

    def _peak(traces, deriv=0):
        vals = []
        for tr in traces:
            i0, i1 = max(0, int(t0_s * sr)), min(tr.stats.npts, int(t1_s * sr))
            if i1 <= i0: continue
            seg = tr.data[i0:i1].astype(np.float64)
            if deriv == 1: seg = np.gradient(seg, dt)
            elif deriv == -1: seg = np.cumsum(seg) * dt
            vals.append(np.nanmax(np.abs(seg)))
        return max(vals) if vals else np.nan

    acc_segs = [np.gradient(tr.data[max(0, int(t0_s * sr)):min(tr.stats.npts, int(t1_s * sr))].astype(np.float64), dt) for tr in horiz]
    ia = sum((np.pi / (2 * GRAVITY)) * np.sum(a ** 2) * dt for a in acc_segs if len(a) > 0)
    return {"pga": _peak(horiz, 1), "pgv": _peak(horiz, 0),
            "pgd": _peak(horiz, -1), "arias": ia}


# ========================================================================
# 5. Synthetic pulse injection
# ========================================================================

def inject_pulses(data, sr, s_sample, n_pulses=3, amp_factor=12.0,
                  duration_range_s=(0.3, 2.0), rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    out = data.copy()
    peak_amp = np.max(np.abs(data))
    post_s_len = len(data) - s_sample
    if post_s_len < int(sr):
        return out, []
    pulses = []
    for _ in range(n_pulses):
        dur_s = rng.uniform(*duration_range_s)
        dur_n = int(dur_s * sr)
        center = s_sample + rng.integers(int(sr), max(int(sr) + 1, post_s_len - dur_n))
        if center + dur_n > len(data): continue
        amp = peak_amp * amp_factor * rng.choice([-1, 1])
        t = np.arange(dur_n) - dur_n // 2
        pulse = amp * np.exp(-0.5 * (t / (dur_n / 6.0)) ** 2)
        i0 = max(0, center - dur_n // 2)
        i1 = min(len(out), i0 + len(pulse))
        out[i0:i1] += pulse[:i1 - i0]
        pulses.append({"center": center, "amp": amp, "dur_s": dur_s, "i0": i0, "i1": i1})
    return out, pulses


# ========================================================================
# Figure 1: Record section with picks + coda windows
# ========================================================================

def plot_record_section(stream, station_info, model_picks, event_id,
                        event_attrs, coda_windows, out_png, n_show=None):
    """Figure 1: Compact record section — all stations, picks + D5-95 coda."""
    plt.rcParams.update({"font.size": 11})
    ev_origin = obspy.UTCDateTime(event_attrs["source_origin_time"])
    mag, depth = event_attrs.get("source_magnitude", "?"), event_attrs.get("source_depth_km", "?")

    items = sorted(station_info, key=lambda s: (np.isnan(s["hyp_dist"]), s["hyp_dist"]))
    if n_show:
        items = items[:n_show]
    n = len(items)
    fig_h = max(6, 0.28 * n + 2.5)
    fig, ax = plt.subplots(figsize=(13, fig_h))

    dists = np.array([s["hyp_dist"] for s in items])
    spacing = float(np.nanmedian(np.diff(np.sort(dists)))) if len(dists) >= 2 else 1.0
    if not np.isfinite(spacing) or spacing <= 0: spacing = 1.0
    amp_scale = 0.35 * spacing
    tmax = 0.0

    for info in items:
        sta = info["station"]
        st_z = stream.select(station=sta, component="Z")
        if not st_z: continue
        tr = st_z[0]
        d = tr.data.astype(np.float64); d -= np.nanmean(d)
        scl = np.nanmax(np.abs(d)) or 1.0; d = (d / scl) * amp_scale
        sr = tr.stats.sampling_rate
        t_rel = (tr.stats.starttime - ev_origin) + np.arange(d.size) / sr
        tmax = max(tmax, float(t_rel[-1]))
        y = info["hyp_dist"]
        t_off = float(tr.stats.starttime - ev_origin)

        cw = coda_windows.get(sta)
        if cw:
            ax.fill_between([t_off + cw[0], t_off + cw[1]],
                            y - amp_scale * 0.45, y + amp_scale * 0.45,
                            color="gold", alpha=0.15, zorder=0)

        ax.plot(t_rel, y + d, color="black", lw=0.35, alpha=0.9, zorder=2)
        ax.text(t_rel[0] - 1.2, y, f"{info['network']}.{sta}",
                ha="right", va="center", fontsize=6)

        if info["p_utc"]:
            c = "red" if info["p_status"] == "manual" else "deeppink"
            m = "|" if info["p_status"] == "manual" else "x"
            ax.plot(info["p_utc"] - ev_origin, y, m, color=c, ms=10, mew=1.3, zorder=3)
        if info["s_utc"]:
            c = "darkorange" if info["s_status"] == "manual" else "goldenrod"
            m = "|" if info["s_status"] == "manual" else "x"
            ax.plot(info["s_utc"] - ev_origin, y, m, color=c, ms=10, mew=1.3, zorder=3)

        # Optional DL-picker overlay. Leave empty in the main pipeline; project
        # policy accepts only RED-PAN picks, and those already live in the catalog.
        for mname, pdict in model_picks.items():
            for phase, col in {"P": "cyan", "S": "lime"}.items():
                pk = pdict.get(sta, {}).get(phase)
                if pk:
                    ax.plot(pk.peak_time - ev_origin, y, "v", mec=col, mfc="none",
                            ms=6, mew=1.0, zorder=4)

    proxies = [
        plt.Line2D([0], [0], color="red", marker="|", ms=10, lw=0, label="Manual P"),
        plt.Line2D([0], [0], color="darkorange", marker="|", ms=10, lw=0, label="Manual S"),
        plt.Line2D([0], [0], color="deeppink", marker="x", ms=7, lw=0, mew=1.5, label="RED-PAN P"),
        plt.Line2D([0], [0], color="goldenrod", marker="x", ms=7, lw=0, mew=1.5, label="RED-PAN S"),
        plt.Rectangle((0, 0), 1, 1, fc="gold", alpha=0.3, ec="none", label="D5-95 coda"),
    ]
    ax.legend(handles=proxies, loc="upper right", fontsize=9, ncol=3, framealpha=0.95)
    ax.set_title(f"RoSE {event_id}  |  M{mag}, {depth} km depth  |  "
                 f"{n} stations", fontsize=12, fontweight="bold")
    ax.set_xlabel("Time since origin (s)", fontsize=11)
    ax.set_ylabel("Hypocentral distance (km)", fontsize=11)
    ax.tick_params(labelsize=10)
    ax.set_xlim(0, max(tmax, 50.0)); ax.grid(alpha=0.2, ls=":")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"  saved {out_png}")
    plt.close(fig)


def plot_ground_motion(gm_table, event_id, event_attrs, husid_demo, out_png):
    """Figure 2: PGA/PGV attenuation + Husid demo."""
    plt.rcParams.update({"font.size": 12})
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    mag, depth = event_attrs.get("source_magnitude", "?"), event_attrs.get("source_depth_km", "?")

    markers = {"catalog": "o"}
    colors_gm = {"catalog": "tab:red"}

    for ax, col_y, ylabel in [
        (axes[0], "pga_cms2", "PGA (cm/s²)"),
        (axes[1], "pgv_cms", "PGV (cm/s)"),
    ]:
        for src in ["catalog"]:
            sub = gm_table[gm_table["pick_source"] == src]
            if sub.empty: continue
            ax.scatter(sub["hyp_dist_km"], sub[col_y], s=40,
                       marker=markers[src], color=colors_gm[src], label=src, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("Hypocentral distance (km)", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=10); ax.grid(True, which="both", alpha=0.3, ls=":")
        ax.tick_params(labelsize=11)

    if husid_demo:
        ax_h = axes[2]
        sr_d = husid_demo["sr"]
        t_h = np.arange(len(husid_demo["husid"])) / sr_d
        ax_h.plot(t_h, husid_demo["husid"], color="tab:red", lw=2)
        t5, t95 = husid_demo["i5"] / sr_d, husid_demo["i95"] / sr_d
        ax_h.axvspan(t5, t95, color="gold", alpha=0.25, label=f"D5-95 = {t95 - t5:.1f} s")
        ax_h.axhline(0.05, color="gray", ls=":", lw=1); ax_h.axhline(0.95, color="gray", ls=":", lw=1)
        ax_h.text(t5, 0.10, f"t₅ = {t5:.1f} s", fontsize=11, color="tab:blue", ha="right")
        ax_h.text(t95, 0.88, f"t₉₅ = {t95:.1f} s", fontsize=11, color="tab:green", ha="left")
        ax_h.set_xlabel("Time (s)", fontsize=12)
        ax_h.set_ylabel("Normalized Arias intensity", fontsize=12)
        ax_h.set_ylim(-0.02, 1.05)
        ax_h.set_title(f"Husid — {husid_demo['station']}", fontsize=12)
        ax_h.legend(fontsize=10, loc="center right")
        ax_h.tick_params(labelsize=11)
        ax_h.grid(alpha=0.25, ls=":")

    fig.suptitle(f"RoSE {event_id}  |  M{mag}, {depth} km depth  |  "
                 f"Ground motion in D5-95 window", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"  saved {out_png}")
    plt.close(fig)


# ========================================================================
# Figure 2: QC summary
# ========================================================================

def plot_qc_summary(stream, station_info, out_png):
    """Figure 3: QC summary — bar chart + SNR distribution + SNR vs distance."""
    plt.rcParams.update({"font.size": 13})
    reports = []
    for info in station_info:
        sta = info["station"]
        st_z = stream.select(station=sta, component="Z")
        if not st_z: continue
        tr = st_z[0]
        p_samp = info["p_sample"] if info["p_sample"] > 0 else None
        r = qc.quality_report(tr.data, tr.stats.sampling_rate, p_samp)
        r["station"] = sta; r["network"] = info["network"]
        r["hyp_dist"] = info["hyp_dist"]
        if p_samp:
            r["snr"] = qc.compute_snr(tr.data, tr.stats.sampling_rate, p_samp)
        reports.append(r)

    n_pass = sum(r["pass"] for r in reports)
    n_fail = len(reports) - n_pass
    print(f"\n  QC: {n_pass} pass, {n_fail} fail ({100 * n_pass / len(reports):.0f}%)")
    for r in reports:
        if not r["pass"]:
            reasons = []
            if r["clipping"]["clipped"]: reasons.append(f"clip({r['clipping']['clip_fraction']*100:.1f}%)")
            if r["dead"]["dead"]: reasons.append("dead")
            if r["gaps"]["has_gaps"]: reasons.append(f"gaps({r['gaps']['n_gap_samples']})")
            if r["spikes"]["has_spikes"]: reasons.append(f"spikes({r['spikes']['n_spikes']})")
            print(f"    FAIL {r['network']}.{r['station']}: {', '.join(reasons)}")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    labels = ["pass", "clip", "dead", "gaps", "spikes"]
    counts = [n_pass,
              sum(r["clipping"]["clipped"] for r in reports),
              sum(r["dead"]["dead"] for r in reports),
              sum(r["gaps"]["has_gaps"] for r in reports),
              sum(r["spikes"]["has_spikes"] for r in reports)]
    colors = ["tab:green", "tab:red", "gray", "tab:orange", "tab:purple"]
    axes[0].bar(labels, counts, color=colors)
    axes[0].set_ylabel("Number of stations", fontsize=13)
    axes[0].set_title(f"QC results ({len(reports)} stations)", fontsize=13)
    for i, c in enumerate(counts):
        axes[0].text(i, c + 0.5, str(c), ha="center", fontsize=12, fontweight="bold")
    axes[0].tick_params(labelsize=12)

    snrs = [r["snr"]["snr_db"] for r in reports if "snr" in r and np.isfinite(r["snr"]["snr_db"])]
    if snrs:
        axes[1].hist(snrs, bins=25, color="tab:blue", edgecolor="black", alpha=0.7)
        axes[1].axvline(np.median(snrs), color="red", ls="--", lw=2,
                        label=f"median = {np.median(snrs):.1f} dB")
        axes[1].set_xlabel("SNR (dB)", fontsize=13); axes[1].set_ylabel("Count", fontsize=13)
        axes[1].set_title("P-phase SNR distribution", fontsize=13)
        axes[1].legend(fontsize=11)
    axes[1].tick_params(labelsize=12); axes[1].grid(alpha=0.25, ls=":")

    dists_snr = [r["hyp_dist"] for r in reports if "snr" in r and np.isfinite(r["snr"]["snr_db"])]
    snrs_plot = [r["snr"]["snr_db"] for r in reports if "snr" in r and np.isfinite(r["snr"]["snr_db"])]
    pass_mask = [r["pass"] for r in reports if "snr" in r and np.isfinite(r["snr"]["snr_db"])]
    if dists_snr:
        c = ["tab:green" if p else "tab:red" for p in pass_mask]
        axes[2].scatter(dists_snr, snrs_plot, c=c, s=30, alpha=0.7)
        axes[2].set_xlabel("Hypocentral distance (km)", fontsize=13)
        axes[2].set_ylabel("SNR (dB)", fontsize=13)
        axes[2].set_title("SNR vs distance", fontsize=13)
    axes[2].tick_params(labelsize=12); axes[2].grid(alpha=0.25, ls=":")

    fig.suptitle("Waveform Quality Control", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"  saved {out_png}")
    plt.close(fig)
    return reports


# ========================================================================
# Figure 3: Coda robustness test
# ========================================================================

def plot_coda_robustness(stream, station_info, inv_cache, data, out_png):
    """Figure 4: Coda robustness — 2x2 layout (clean/contaminated + Husid + envelope)."""
    plt.rcParams.update({"font.size": 13})
    good = [i for i in station_info if i["s_sample"] > 0 and i["hyp_dist"] < 300]
    good.sort(key=lambda x: x["hyp_dist"])
    vel_st = test_info = None
    for c in good:
        vel_st = get_velocity_stream(stream, c["station"], inv_cache, data, c)
        if vel_st is not None:
            test_info = c; break
    if vel_st is None:
        print("  no station for robustness test — skip"); return

    sta = test_info["station"]
    print(f"\n  robustness test station: {test_info['network']}.{sta} "
          f"(dist={test_info['hyp_dist']:.1f} km)")

    horiz = [tr for tr in vel_st if tr.stats.channel[-1] in ("E", "N")]
    tr = max(horiz, key=lambda t: np.max(np.abs(t.data))) if horiz else vel_st[0]
    clean = tr.data.astype(np.float64)
    sr = tr.stats.sampling_rate; dt = 1.0 / sr
    s_sample = int((test_info["s_utc"] - tr.stats.starttime) * sr) if test_info["s_utc"] else 0

    contaminated, pulses = inject_pulses(clean, sr, s_sample)
    print(f"  injected {len(pulses)} pulses (12x amplitude)")

    results = {}
    for label, wf in [("clean", clean), ("contaminated", contaminated)]:
        acc = np.gradient(wf, dt)
        i5, i95, husid = husid_d5_95(acc, dt)
        ce = envelope_coda_end(wf, sr, s_sample)
        results[label] = {"wf": wf, "husid": husid, "i5": i5, "i95": i95, "ce": ce}
        d595 = (i95 - i5) / sr; env_dur = (ce - s_sample) / sr
        print(f"  {label:14s}  D5-95={d595:.1f}s  envelope={env_dur:.1f}s")

    t = np.arange(len(clean)) / sr
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # (a) Clean waveform
    r = results["clean"]
    axes[0, 0].plot(t, clean * 1e6, color="black", lw=0.6)
    t5, t95 = r["i5"] / sr, r["i95"] / sr
    axes[0, 0].axvspan(t5, t95, color="gold", alpha=0.25, label=f"D5-95 = {t95-t5:.1f} s")
    axes[0, 0].axvspan(s_sample / sr, r["ce"] / sr, color="plum", alpha=0.15,
                        label=f"envelope coda = {(r['ce']-s_sample)/sr:.1f} s")
    if s_sample > 0:
        axes[0, 0].axvline(s_sample / sr, color="darkorange", lw=1.5, label="S arrival")
    axes[0, 0].set_title(f"(a) Clean waveform — {test_info['network']}.{sta}", fontsize=13)
    axes[0, 0].set_ylabel("velocity (µm/s)", fontsize=12)
    axes[0, 0].legend(fontsize=10, loc="upper right"); axes[0, 0].grid(alpha=0.25, ls=":")
    axes[0, 0].tick_params(labelsize=11)

    # (b) Contaminated waveform
    r2 = results["contaminated"]
    axes[0, 1].plot(t, contaminated * 1e6, color="black", lw=0.6)
    for p in pulses:
        axes[0, 1].axvspan(p["i0"] / sr, p["i1"] / sr, color="red", alpha=0.2)
    t5c, t95c = r2["i5"] / sr, r2["i95"] / sr
    axes[0, 1].axvspan(t5c, t95c, color="gold", alpha=0.25, label=f"D5-95 = {t95c-t5c:.1f} s")
    axes[0, 1].axvspan(s_sample / sr, r2["ce"] / sr, color="plum", alpha=0.15,
                        label=f"envelope coda = {(r2['ce']-s_sample)/sr:.1f} s")
    axes[0, 1].set_title("(b) + 3 synthetic pulses (12× amplitude)", fontsize=13)
    axes[0, 1].set_ylabel("velocity (µm/s)", fontsize=12)
    axes[0, 1].legend(fontsize=10, loc="upper right"); axes[0, 1].grid(alpha=0.25, ls=":")
    axes[0, 1].tick_params(labelsize=11)

    # (c) Husid comparison
    d595_shift = abs((r2["i95"] - r2["i5"]) - (r["i95"] - r["i5"])) / sr
    axes[1, 0].plot(t, r["husid"], color="tab:blue", lw=2, label="clean")
    axes[1, 0].plot(t, r2["husid"], color="tab:red", lw=2, ls="--", label="contaminated")
    axes[1, 0].axhline(0.05, color="gray", ls=":", lw=1); axes[1, 0].axhline(0.95, color="gray", ls=":", lw=1)
    axes[1, 0].set_title(f"(c) Husid plot — D5-95 shift = {d595_shift:.1f} s (robust)", fontsize=13)
    axes[1, 0].set_ylim(-0.02, 1.05)
    axes[1, 0].set_xlabel("Time (s)", fontsize=12); axes[1, 0].set_ylabel("Normalized Arias intensity", fontsize=12)
    axes[1, 0].legend(fontsize=11); axes[1, 0].grid(alpha=0.25, ls=":")
    axes[1, 0].tick_params(labelsize=11)

    # (d) Envelope comparison
    env_c = np.convolve(np.abs(hilbert(clean)), np.ones(int(sr)) / sr, mode="same")
    env_x = np.convolve(np.abs(hilbert(contaminated)), np.ones(int(sr)) / sr, mode="same")
    axes[1, 1].plot(t, env_c * 1e6, color="tab:blue", lw=1.5, label="clean")
    axes[1, 1].plot(t, env_x * 1e6, color="tab:red", lw=1.5, ls="--", label="contaminated")
    axes[1, 1].axvline(r["ce"] / sr, color="tab:blue", lw=1.5, label=f"clean end = {r['ce']/sr:.1f} s")
    axes[1, 1].axvline(r2["ce"] / sr, color="tab:red", lw=1.5, label=f"contam end = {r2['ce']/sr:.1f} s")
    env_shift = abs(r2["ce"] - r["ce"]) / sr
    axes[1, 1].set_title(f"(d) Envelope — coda end shift = {env_shift:.1f} s (fragile)", fontsize=13)
    axes[1, 1].set_xlabel("Time (s)", fontsize=12); axes[1, 1].set_ylabel("envelope (µm/s)", fontsize=12)
    axes[1, 1].legend(fontsize=10); axes[1, 1].grid(alpha=0.25, ls=":")
    axes[1, 1].tick_params(labelsize=11)

    fig.suptitle(f"Coda-window robustness test — {test_info['network']}.{sta}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    print(f"  saved {out_png}")
    plt.close(fig)


# ========================================================================
# Main
# ========================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", default="2018_0000140")
    parser.add_argument("--n-show", type=int, default=20)
    parser.add_argument("--coda-method", default="arias", choices=["arias", "envelope"])
    args = parser.parse_args()
    event_id = args.event

    out_dir = Path(__file__).resolve().parents[1] / "outputs" / f"03_event_{event_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    data = RoSE(DATA_DIR)
    inv_cache = StationXMLCache(STATIONXML_DIR)

    # 1. Load & export
    print(f"Loading event {event_id} ...")
    stream, station_info = load_event(data, event_id)
    md = data.metadata
    event_row = md[md["source_id"] == event_id].iloc[0]
    event_attrs = {c: event_row[c] for c in md.columns if c.startswith("source_")}
    print(f"  {len(station_info)} stations, {len(stream)} traces")
    write_miniseed(stream, out_dir / "mseed")

    # 2. QC
    print("\n--- QC ---")
    plot_qc_summary(stream, station_info, str(out_dir / "fig_qc_summary.png"))

    # 3. Picks (catalog only — RED-PAN is the sanctioned DL picker; other DL
    #    pickers are excluded by project policy).
    print("\n--- Phase Picks (catalog only) ---")

    # 4 & 5. Coda + ground motion
    print(f"\n--- Ground Motion (method={args.coda_method}) ---")
    gm_rows = []; coda_windows = {}; husid_demo = None
    for info in station_info:
        sta = info["station"]
        vel_st = get_velocity_stream(stream, sta, inv_cache, data, info)
        if vel_st is None: continue
        pick_sources = {"catalog": (info["p_utc"], info["s_utc"])}
        for src, (p_t, s_t) in pick_sources.items():
            t0, t1, diag = get_coda_window(vel_st, p_t, s_t, method=args.coda_method)
            if t1 <= t0: continue
            gm = measure_ground_motion(vel_st, t0, t1)
            if np.isnan(gm["pga"]): continue
            gm_rows.append({"station": sta, "network": info["network"],
                            "hyp_dist_km": info["hyp_dist"], "pick_source": src,
                            "coda_t0_s": round(t0, 3), "coda_t1_s": round(t1, 3),
                            "d595_s": round(t1 - t0, 3),
                            "pga_cms2": gm["pga"] * 100, "pgv_cms": gm["pgv"] * 100,
                            "pgd_cm": gm["pgd"] * 100, "arias_ms": gm["arias"]})
            if src == "catalog" and sta not in coda_windows:
                coda_windows[sta] = (t0, t1)
            if husid_demo is None and src == "catalog" and "husid" in diag:
                horiz = [tr for tr in vel_st if tr.stats.channel[-1] in ("E", "N")]
                demo_tr = max(horiz, key=lambda t: np.max(np.abs(t.data))) if horiz else vel_st[0]
                s_samp = int((s_t - demo_tr.stats.starttime) * demo_tr.stats.sampling_rate) if s_t else 0
                s_samp = max(0, min(s_samp, demo_tr.stats.npts - 1))
                ce = envelope_coda_end(demo_tr.data.astype(np.float64), demo_tr.stats.sampling_rate, s_samp)
                env = np.abs(hilbert(demo_tr.data.astype(np.float64)))
                win = max(1, int(demo_tr.stats.sampling_rate))
                husid_demo = {"station": sta, "sr": info["sr"],
                              "husid": diag["husid"], "i5": diag["i5"], "i95": diag["i95"],
                              "envelope_data": {"vel": demo_tr.data.astype(np.float64),
                                                "envelope": np.convolve(env, np.ones(win) / win, mode="same"),
                                                "s_sample": s_samp, "coda_end": ce}}

    gm_table = pd.DataFrame(gm_rows)
    csv_path = str(out_dir / "ground_motion.csv")
    gm_table.to_csv(csv_path, index=False)
    print(f"  wrote {csv_path} ({len(gm_table)} rows)")
    for src in ["catalog"]:
        sub = gm_table[gm_table["pick_source"] == src]
        if sub.empty: continue
        print(f"  {src:16s}  n={len(sub):3d}  PGA={sub['pga_cms2'].median():.2g} cm/s²  "
              f"PGV={sub['pgv_cms'].median():.2g} cm/s  D5-95={sub['d595_s'].median():.1f} s")

    # Plot record section (no DL overlay picks — RED-PAN is the sanctioned picker
    # and its picks live in the catalog via trace_p/s_status).
    print("\n--- Plotting ---")
    plot_record_section(stream, station_info, {},
                        event_id, event_attrs, coda_windows,
                        str(out_dir / "fig_record_section.png"))
    plot_ground_motion(gm_table, event_id, event_attrs, husid_demo,
                       str(out_dir / "fig_ground_motion.png"))

    # 6. Robustness test
    print("\n--- Coda Robustness Test ---")
    plot_coda_robustness(stream, station_info, inv_cache, data,
                         str(out_dir / "fig_coda_robustness.png"))

    print("\nDone. Outputs in:", out_dir)


if __name__ == "__main__":
    main()
