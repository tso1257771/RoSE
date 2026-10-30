"""Export the RoSE noise windows of one split in the layout bench_noise_fp.py reads.

bench_noise_fp.py counts false picks on STEAD's noise set: ``metadata_noise.csv``
(one row per trace, keyed by ``trace_name``) plus ``noise_waveforms/<trace_name>.npy``,
one float32 array of shape (samples, 3) in E, N, Z order. This script writes the
RoSE noise windows of one split (the ``noise{YEAR}`` chunks of dataset version 3
or later, 100 s each) in that layout, so the same script counts false picks on
them. A noise window has the split of its event, so the test-split windows come
from test events and none of the RoSE-trained pickers saw them; how the windows
were chosen is in docs/SEISBENCH_FORMAT.md ("Noise chunks").

By default each window is cut to its middle 60 s (20-80 s): the part that all
four screening pickers covered, and the length of the data in a STEAD noise
trace and of one RED-PAN-60s window. ``--window-s 100`` keeps whole windows.

    python phase_picking/benchmark/export_rose_noise.py --rose-dir $ROSE_DATA_DIR \\
        --out-dir rose_noise_test_60s
    python phase_picking/benchmark/bench_noise_fp.py --noise-dir rose_noise_test_60s \\
        --out-dir phase_picking/benchmark/eval/bench_noise_fp_rose
    python phase_picking/benchmark/build_rose_noise_benchmark.py
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from rose.dataset import NOISE_PREFIX, RoSE, select_chunks  # noqa: E402

SAMPLING_RATE = 100.0
WINDOW_NPTS = 10000          # every RoSE noise window is 100 s at 100 Hz
KEEP_COLUMNS = ["trace_name_original", "trace_name", "trace_chunk", "trace_start_time",
                "trace_window_position", "station_network_code", "station_code",
                "station_location_code", "station_channel", "split"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rose-dir", default=os.environ.get("ROSE_DATA_DIR"),
                    help="RoSE dataset directory (default: $ROSE_DATA_DIR)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", choices=["train", "dev", "test"], default="test")
    ap.add_argument("--window-s", type=float, default=60.0,
                    help="seconds to keep from the middle of each 100-s window (default 60)")
    ap.add_argument("--num-windows", type=int, default=0,
                    help="0 = every window of the split; N = the first N (a quick check)")
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")
    npts = int(round(args.window_s * SAMPLING_RATE))
    if not 0 < npts <= WINDOW_NPTS:
        ap.error("--window-s must be in (0, 100]")
    start = (WINDOW_NPTS - npts) // 2

    chunks = [c for c in select_chunks(args.rose_dir, include_noise=True)
              if c.startswith(NOISE_PREFIX)]
    if not chunks:
        sys.exit(f"{args.rose_dir} has no noise chunks; they come with dataset version 3.")
    noise = RoSE(args.rose_dir, chunks=chunks, component_order="ENZ", dimension_order="NWC")
    data = {"train": noise.train, "dev": noise.dev, "test": noise.test}[args.split]()
    meta = data.metadata.reset_index(drop=True)
    if args.num_windows > 0:
        meta = meta.iloc[:args.num_windows]
    if not (meta["trace_sampling_rate_hz"] == SAMPLING_RATE).all():
        sys.exit("expected every noise window at 100 Hz")
    if meta["trace_name_original"].duplicated().any():
        sys.exit("trace_name_original is not unique; cannot use it as the file name")

    out_dir = Path(args.out_dir)
    wf_dir = out_dir / "noise_waveforms"
    wf_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(data)} noise windows in the {args.split} split; writing {len(meta)} "
          f"x {npts / SAMPLING_RATE:g} s ({start / SAMPLING_RATE:g}-"
          f"{(start + npts) / SAMPLING_RATE:g} s of each window) to {out_dir}", flush=True)
    t0 = time.time()
    for i, name in enumerate(meta["trace_name_original"]):
        wf = data.get_waveforms(i)
        if wf.shape != (WINDOW_NPTS, 3):
            sys.exit(f"{name}: expected ({WINDOW_NPTS}, 3) samples, got {wf.shape}")
        np.save(wf_dir / f"{name}.npy", wf[start:start + npts].astype(np.float32))
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(meta)} ({time.time() - t0:.0f} s)", flush=True)

    noise_df = meta[KEEP_COLUMNS].rename(columns={"trace_name": "rose_trace_name",
                                                  "trace_name_original": "trace_name"})
    noise_df["export_start_s"] = start / SAMPLING_RATE
    noise_df.to_csv(out_dir / "metadata_noise.csv", index=False)
    print(f"wrote {len(noise_df)} windows and metadata_noise.csv in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
