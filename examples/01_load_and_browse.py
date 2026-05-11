"""Tutorial 1 — Load the published SeisBench dataset.

Demonstrates filtering on metadata, accessing waveforms, and plotting a
random pick window.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from rose import RoSE

DATA_DIR = os.environ.get(
    "ROSE_DATA_DIR",
    str(Path(__file__).resolve().parents[1] / "data" / "rose"),
)


def main():
    data = RoSE(DATA_DIR)
    print(data)
    print("traces:", len(data))
    print("years:", sorted(data.metadata["trace_chunk"].unique()))
    print("stations:", data.metadata["station_code"].nunique())

    # Filter: high-SNR P picks within the recording window.
    # `trace_p_in_window` round-trips through CSV as a string ("True"/"False"),
    # so coerce explicitly instead of comparing to a Python bool.
    md = data.metadata
    in_window = md["trace_p_in_window"].astype(str).str.lower().isin(["true", "1"])
    mask = (
        in_window
        & (md["trace_p_snr_db"].astype(float) > 5.0)
        & (md["source_magnitude"].astype(float) >= 2.5)
    )
    keep = md.index[mask].to_numpy()
    print(f"high-SNR M>=2.5 with P in window: {len(keep)}")
    if len(keep) == 0:
        return

    rng = np.random.default_rng(0)
    idx = int(rng.choice(keep))
    wf, meta = data.get_sample(idx)
    print(meta["trace_name"], "shape", wf.shape)

    sr = float(meta["trace_sampling_rate_hz"])
    p = int(meta["trace_p_arrival_sample"])
    s = int(meta["trace_s_arrival_sample"])

    t = np.arange(wf.shape[1]) / sr
    fig, axes = plt.subplots(3, 1, figsize=(11, 6), sharex=True)
    for ax, ch, label in zip(axes, wf, list(data.component_order)):
        ax.plot(t, ch, lw=0.6, color="black")
        if p > 0:
            ax.axvline(p / sr, color="magenta", ls="--", lw=0.8, label="P")
        if s > 0:
            ax.axvline(s / sr, color="orange", ls="--", lw=0.8, label="S")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25, ls=":")
    axes[0].set_title(
        f"{meta['source_id']} | {meta['station_network_code']}.{meta['station_code']} "
        f"| dist={float(meta['path_hyp_distance_km']):.1f} km | M={meta['source_magnitude']}"
    )
    axes[-1].set_xlabel("Time (s) since reference start")
    axes[0].legend(loc="upper right")
    fig.tight_layout()
    REPO_ROOT = Path(__file__).resolve().parents[1]
    out = REPO_ROOT / "outputs" / "01_load_and_browse.png"
    fig.savefig(out, dpi=150)
    try:
        print(f"saved {out.relative_to(REPO_ROOT)}")
    except ValueError:
        print(f"saved {out}")


if __name__ == "__main__":
    main()
