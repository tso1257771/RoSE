"""Tutorial 2 — EQTransformer (INSTANCE weights) on a Vrancea deep slab event.

DEPRECATED (2026-04-17): Project policy is RED-PAN 60s for phase picking.
EQTransformer picks are shown here for *comparison only*; do not use them in
downstream tomography. See RED-PAN/ for the supported picker.

End-to-end workflow using the published SeisBench-format dataset:

    1. Open `data/rose` with `RoSE`.
    2. Pick out event 2018_0000140 (Mw 5.8, ~153 km depth, ~68 stations).
    3. Reconstruct an ObsPy Stream from the bucketed waveforms.
    4. Run `EQTransformer.from_pretrained("instance")`.
    5. Plot a record section with catalog and model picks overlaid; report
       the residual statistics.

This script intentionally avoids the native ROMPLUS-derived HDF5 — everything
goes through the published RoSE SeisBench dataset.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

warnings.warn(
    "02_eqt_instance_vrancea is deprecated: project uses RED-PAN 60s for picking; "
    "EQTransformer shown here for comparison only.",
    DeprecationWarning,
    stacklevel=2,
)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import obspy
import seisbench.models as sbm

from rose import RoSE

DATA_DIR = os.environ.get(
    "ROSE_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "rose"),
)
EVENT_ID = "2018_0000140"
REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_PNG = str(REPO_ROOT / "outputs" / "02_eqt_instance_vrancea.png")


def _rel(p):
    """Format ``p`` for log output, relative to the repo root."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _str_or_none(v):
    """Return v as a string, or None if NaN/empty."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    s = str(v).strip()
    if not s or s.lower() in ("nan", "none"):
        return None
    return s


def load_event_stream(data: RoSE, event_id: str):
    """Reconstruct an ObsPy Stream + per-station catalog meta for one event."""
    md = data.metadata
    rows = md.index[md["source_id"] == event_id].to_numpy()
    if len(rows) == 0:
        raise KeyError(f"event {event_id} not in dataset")

    stream = obspy.Stream()
    sta_meta = {}
    event_attrs = None

    for idx in rows:
        wf, meta = data.get_sample(int(idx))
        if event_attrs is None:
            event_attrs = {
                "time": meta["source_origin_time"],
                "lat": float(meta["source_latitude_deg"]),
                "lon": float(meta["source_longitude_deg"]),
                "depth": float(meta["source_depth_km"]),
                "magnitude": float(meta["source_magnitude"]),
            }
        sr = float(meta["trace_sampling_rate_hz"])
        starttime = obspy.UTCDateTime(meta["trace_start_time"])
        net = meta["station_network_code"]
        sta = meta["station_code"]
        loc = _str_or_none(meta["station_location_code"]) or ""
        band = meta["station_channel"]

        for ch_letter, channel_data in zip(data.component_order, wf):
            tr = obspy.Trace(
                channel_data.astype(np.float32),
                header={
                    "network": net,
                    "station": sta,
                    "location": loc,
                    "channel": f"{band}{ch_letter}",
                    "starttime": starttime,
                    "sampling_rate": sr,
                },
            )
            stream += tr

        sta_meta[sta] = {
            "network": net,
            "ref_start": starttime,
            "sr": sr,
            "p_sample": int(meta["trace_p_arrival_sample"]),
            "s_sample": int(meta["trace_s_arrival_sample"]),
            "p_utc": _str_or_none(meta["trace_p_arrival_time"]),
            "s_utc": _str_or_none(meta["trace_s_arrival_time"]),
            "wf_z_idx": list(data.component_order).index("Z"),
            "wf": wf,
            "epi_dist": float(meta["path_ep_distance_km"]),
            "hyp_dist": float(meta["path_hyp_distance_km"]),
        }
    return stream, sta_meta, event_attrs


def report_residuals(sta_meta, picks, event_attrs):
    ev_origin = obspy.UTCDateTime(event_attrs["time"])
    res_p, res_s = [], []
    for p in picks:
        sta = p.trace_id.split(".")[1]
        m = sta_meta.get(sta)
        if not m:
            continue
        if p.phase == "P" and m["p_utc"]:
            dt = p.peak_time - obspy.UTCDateTime(m["p_utc"])
            if abs(dt) < 5.0:
                res_p.append(dt)
        elif p.phase == "S" and m["s_utc"]:
            dt = p.peak_time - obspy.UTCDateTime(m["s_utc"])
            if abs(dt) < 8.0:
                res_s.append(dt)

    print(f"Event origin: {ev_origin}, M{event_attrs['magnitude']}, {event_attrs['depth']} km")
    print(f"Catalog stations with picks: P={sum(bool(m['p_utc']) for m in sta_meta.values())}  "
          f"S={sum(bool(m['s_utc']) for m in sta_meta.values())}")
    print(f"EQT-INSTANCE picks total: {len(picks)}")
    for label, arr in (("P", res_p), ("S", res_s)):
        if not arr:
            continue
        a = np.array(arr)
        print(
            f"  {label} residuals (model-catalog) [s]: "
            f"n={len(a)}  median={np.median(a):+.3f}  "
            f"MAD={np.median(np.abs(a - np.median(a))):.3f}"
        )


def plot_record_section(sta_meta, event_attrs, picks, n_show=24, save_path=OUT_PNG):
    ev_origin = obspy.UTCDateTime(event_attrs["time"])
    items = sorted(
        sta_meta.items(),
        key=lambda kv: (np.isnan(kv[1]["hyp_dist"]), kv[1]["hyp_dist"]),
    )[:n_show]

    picks_by_sta = {}
    for p in picks:
        picks_by_sta.setdefault(p.trace_id.split(".")[1], []).append(p)

    fig, ax = plt.subplots(figsize=(13, 0.45 * n_show + 3.0))

    dists = np.array([m["hyp_dist"] for _, m in items])
    spacing = float(np.nanmedian(np.diff(np.sort(dists)))) if len(dists) >= 2 else 1.0
    if not np.isfinite(spacing) or spacing <= 0:
        spacing = 1.0
    amp_scale = 0.45 * spacing

    tmax = 0.0
    for sta, m in items:
        wf = m["wf"]
        z = wf[m["wf_z_idx"]].astype(np.float64)
        if z.size == 0:
            continue
        z = z - np.nanmean(z)
        scl = np.nanmax(np.abs(z)) or 1.0
        z = (z / scl) * amp_scale
        sr = m["sr"]
        t_rel = (m["ref_start"] - ev_origin) + np.arange(z.size) / sr
        tmax = max(tmax, float(t_rel[-1]))
        y = m["hyp_dist"]
        ax.plot(t_rel, y + z, color="black", lw=0.55, alpha=0.85)
        ax.text(t_rel[0] - 1.5, y, f"{m['network']}.{sta}",
                ha="right", va="center", fontsize=7)

        if m["p_sample"] >= 0:
            t_p = (m["ref_start"] - ev_origin) + m["p_sample"] / sr
            ax.plot(t_p, y, marker="|", color="magenta", ms=14, mew=1.6, alpha=0.9)
        if m["s_sample"] >= 0:
            t_s = (m["ref_start"] - ev_origin) + m["s_sample"] / sr
            ax.plot(t_s, y, marker="|", color="orange", ms=14, mew=1.6, alpha=0.9)

        for pk in picks_by_sta.get(sta, []):
            t_pk = pk.peak_time - ev_origin
            color = "cyan" if pk.phase == "P" else "lime"
            ax.plot(t_pk, y, marker="v", mec=color, mfc="none", ms=8, mew=1.4)

    proxies = [
        plt.Line2D([0], [0], color="magenta", marker="|", ms=12, lw=0, label="Catalog P"),
        plt.Line2D([0], [0], color="orange", marker="|", ms=12, lw=0, label="Catalog S"),
        plt.Line2D([0], [0], mec="cyan", mfc="none", marker="v", ms=10, lw=0,
                   label="EQT-INSTANCE P"),
        plt.Line2D([0], [0], mec="lime", mfc="none", marker="v", ms=10, lw=0,
                   label="EQT-INSTANCE S"),
    ]
    ax.legend(handles=proxies, loc="upper right", fontsize=9)
    ax.set_title(
        f"Vrancea slab event {EVENT_ID}  |  M{event_attrs['magnitude']:.1f}, "
        f"{event_attrs['depth']:.0f} km depth  |  EQTransformer (INSTANCE)  vs  catalog "
        f"({len(items)} closest stations)"
    )
    ax.set_xlabel("Time since origin (s)")
    ax.set_ylabel("Hypocentral distance (km)")
    ax.set_xlim(0, max(tmax, 100.0))
    ax.grid(alpha=0.25, ls=":")

    fig.tight_layout()
    fig.savefig(save_path, dpi=170)
    print(f"saved {_rel(save_path)}")


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")  # CPU is plenty for one event

    print(f"Opening {DATA_DIR} ...")
    data = RoSE(DATA_DIR)
    print(f"  total traces: {len(data)}, events: {data.metadata['source_id'].nunique()}")

    print(f"Loading {EVENT_ID} ...")
    stream, sta_meta, event_attrs = load_event_stream(data, EVENT_ID)
    print(f"  stations: {len(sta_meta)}, traces: {len(stream)}")

    print("Running EQTransformer (INSTANCE) ...")
    model = sbm.EQTransformer.from_pretrained("instance")
    output = model.classify(
        stream, batch_size=32, P_threshold=0.1, S_threshold=0.1, overlap=3000
    )
    picks = output.picks
    print(f"  picks: {len(picks)}")

    report_residuals(sta_meta, picks, event_attrs)
    plot_record_section(sta_meta, event_attrs, picks, n_show=24)


if __name__ == "__main__":
    main()
