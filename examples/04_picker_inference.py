"""Tutorial 4 — Run all three published RoSE pickers on test traces.

Smallest end-to-end demo of the released checkpoints under
``application/seisbench-rose-benchmark/models/``:

    1. Open `data/rose` with `RoSE`.
    2. Filter to the held-out test split (falls back to the full dataset
       if the `split` column hasn't been materialised yet).
    3. Pick a few well-labelled test traces (both P and S in window).
    4. Reconstruct an ObsPy Stream per trace.
    5. Load EQT-RoSE, PhaseNet-RoSE, and RED-PAN-60s via the release loaders
       (``application/seisbench-rose-benchmark/benchmarks/models.py``).
    6. Run ``model.classify(stream, ...)`` on each model and compare
       the predicted picks to the catalog labels — print a per-model
       residual table, plot a 2×2 panel with all three picker overlays.

RED-PAN-60s requires TensorFlow; if it isn't installed (or you pass
``--no-redpan``) the example runs the two SeisBench pickers only.

Usage:
    python examples/04_picker_inference.py
    python examples/04_picker_inference.py --n-traces 8 --seed 7
    python examples/04_picker_inference.py --device cuda
    python examples/04_picker_inference.py --no-redpan          # PyTorch only

Outputs: ``outputs/04_picker_inference.png`` and a residual summary on stdout.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / "application" / "seisbench-rose-benchmark"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(RELEASE))  # so `from benchmarks.models import ...` resolves

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from obspy import Stream, Trace, UTCDateTime  # noqa: E402

from rose import RoSE  # noqa: E402
from benchmarks.models import (  # noqa: E402
    load_eqt_rose,
    load_phasenet_rose,
    load_redpan_tf60,
)

DATA_DIR = os.environ.get("ROSE_DATA_DIR", str(REPO_ROOT / "data" / "rose"))
MODELS_DIR = RELEASE / "models"
OUT_PNG = REPO_ROOT / "outputs" / "04_picker_inference.png"

# RED-PAN paper convention; see benchmark/ for why the tolerances aren't symmetric.
TOL_P_SEC = 0.5
TOL_S_SEC = 1.0

# Pick threshold used for all three models (matches the published benchmark default).
P_THRESHOLD = 0.3
S_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.3

# Per-model line styles for the picks overlay.
MODEL_STYLES = {
    "EQT-RoSE":      {"ls": "--", "lw": 1.0, "alpha": 0.85},
    "PhaseNet-RoSE": {"ls": ":",  "lw": 1.4, "alpha": 0.85},
    "RED-PAN-60s":   {"ls": "-.", "lw": 1.0, "alpha": 0.85},
}
PHASE_COLORS = {"P": "tab:blue", "S": "tab:red"}


# ---------------------------------------------------------------------------
# Data selection + Stream construction
# ---------------------------------------------------------------------------
def select_test_traces(data: RoSE, n: int, seed: int) -> np.ndarray:
    """Return indices of `n` test-split traces with both P and S in window."""
    md = data.metadata
    if "split" in md.columns:
        eligible = md.index[md["split"] == "test"].to_numpy()
        scope = "test split"
    else:
        eligible = md.index.to_numpy()
        scope = (
            "full dataset (no `split` column found — run "
            "`python training/build_rose_split_index.py` to materialise one)"
        )
    p_in = (
        md.loc[eligible, "trace_p_arrival_sample"]
        .astype(float).between(0, 5999, inclusive="both")
    )
    s_in = (
        md.loc[eligible, "trace_s_arrival_sample"]
        .astype(float).between(0, 5999, inclusive="both")
    )
    eligible = eligible[(p_in & s_in).to_numpy()]
    if len(eligible) == 0:
        raise RuntimeError(f"no eligible traces in {scope}")
    rng = np.random.default_rng(seed)
    n = min(n, len(eligible))
    chosen = rng.choice(eligible, size=n, replace=False)
    print(f"selected {n} of {len(eligible)} candidate traces from the {scope}.")
    return np.sort(chosen)


def trace_to_stream(wf: np.ndarray, meta) -> Stream:
    """Build a 3-component ZNE Stream from a SeisBench sample."""
    sr = float(meta["trace_sampling_rate_hz"])
    starttime = UTCDateTime(meta["trace_start_time"])
    net = str(meta["station_network_code"])
    sta = str(meta["station_code"])
    loc_raw = meta["station_location_code"]
    loc = "" if (loc_raw is None or str(loc_raw).lower() in ("nan", "none")) else str(loc_raw)
    band = str(meta["station_channel"])
    traces = []
    for component, channel_data in zip("ZNE", wf):
        traces.append(
            Trace(
                channel_data.astype(np.float32),
                header={
                    "network": net, "station": sta, "location": loc,
                    "channel": f"{band}{component}",
                    "starttime": starttime, "sampling_rate": sr,
                },
            )
        )
    return Stream(traces)


# ---------------------------------------------------------------------------
# Per-trace inference + matching
# ---------------------------------------------------------------------------
def closest_match(predicted, target_time, tolerance_s):
    """Return the predicted pick closest to ``target_time`` within ``tolerance_s``."""
    best, best_dt = None, None
    for p in predicted:
        dt = float(p.peak_time - target_time)
        if abs(dt) <= tolerance_s and (best_dt is None or abs(dt) < abs(best_dt)):
            best, best_dt = p, dt
    return best, best_dt


def predict(model, stream: Stream) -> tuple[list, list]:
    """Run model.classify and split into (P picks, S picks)."""
    output = model.classify(
        stream,
        P_threshold=P_THRESHOLD, S_threshold=S_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
    )
    p_picks = [p for p in output.picks if str(p.phase).upper() == "P"]
    s_picks = [p for p in output.picks if str(p.phase).upper() == "S"]
    return p_picks, s_picks


def evaluate_one(stream: Stream, meta, models: dict) -> dict:
    """Run every model on one trace and report per-phase residuals + match status."""
    sr = float(meta["trace_sampling_rate_hz"])
    starttime = UTCDateTime(meta["trace_start_time"])
    true_p = starttime + float(meta["trace_p_arrival_sample"]) / sr
    true_s = starttime + float(meta["trace_s_arrival_sample"]) / sr

    per_model = {}
    for name, model in models.items():
        p_picks, s_picks = predict(model, stream)
        _, p_dt = closest_match(p_picks, true_p, TOL_P_SEC)
        _, s_dt = closest_match(s_picks, true_s, TOL_S_SEC)
        per_model[name] = {
            "p_picks": p_picks, "s_picks": s_picks,
            "p_dt": p_dt, "s_dt": s_dt,
        }
    return {
        "true_p": true_p, "true_s": true_s, "starttime": starttime,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_panel(ax, stream: Stream, meta, result: dict, show_legend: bool):
    sr = float(meta["trace_sampling_rate_hz"])
    z = stream.select(channel="*Z")[0].data.astype(float)
    z = z / (np.max(np.abs(z)) + 1e-9)
    t = np.arange(len(z)) / sr
    ax.plot(t, z, color="black", lw=0.6)

    starttime = result["starttime"]
    # ground-truth picks (solid, thick)
    ax.axvline(result["true_p"] - starttime, color=PHASE_COLORS["P"], lw=1.6, label="catalog P")
    ax.axvline(result["true_s"] - starttime, color=PHASE_COLORS["S"], lw=1.6, label="catalog S")

    # per-model predicted picks
    for model_name, picks in result["per_model"].items():
        style = MODEL_STYLES[model_name]
        for phase_name, key in (("P", "p_picks"), ("S", "s_picks")):
            for p in picks[key]:
                ax.axvline(
                    p.peak_time - starttime,
                    color=PHASE_COLORS[phase_name],
                    label=f"{model_name} {phase_name}" if show_legend else None,
                    **style,
                )

    title = (
        f"{meta['station_network_code']}.{meta['station_code']}  "
        f"M{float(meta['source_magnitude']):.1f}, "
        f"{float(meta['path_ep_distance_km']):.0f} km"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_xlim(0, len(z) / sr)
    ax.set_yticks([])
    if show_legend:
        # de-duplicate the legend (each model adds 2 entries per panel)
        handles, labels = ax.get_legend_handles_labels()
        seen, uniq = set(), []
        for h, l in zip(handles, labels):
            if l not in seen:
                seen.add(l); uniq.append((h, l))
        ax.legend(*zip(*uniq), loc="upper right", fontsize=7, ncol=2)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rose-dir", default=DATA_DIR,
                    help=f"RoSE SeisBench dataset dir (default: {DATA_DIR}).")
    ap.add_argument("--n-traces", type=int, default=4,
                    help="Number of test traces to evaluate + plot (default: 4).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", help='"cpu" (default) or "cuda".')
    ap.add_argument("--no-redpan", action="store_true",
                    help="Skip RED-PAN-60s (avoids the TensorFlow dependency).")
    ap.add_argument("--out-png", default=str(OUT_PNG))
    args = ap.parse_args()

    print(f"Opening {args.rose_dir} ...")
    data = RoSE(args.rose_dir, component_order="ZNE")
    print(f"  total traces: {len(data)}")

    # --- load all three models (RED-PAN is optional) ---
    print("Loading models from", MODELS_DIR.relative_to(REPO_ROOT))
    models: dict[str, object] = {}
    print(f"  EQT-RoSE      ({(MODELS_DIR / 'eqt_rose' / 'eqt_rose.pt').stat().st_size / 1e6:.1f} MB) ...")
    models["EQT-RoSE"] = load_eqt_rose(device=args.device)
    print(f"  PhaseNet-RoSE ({(MODELS_DIR / 'phasenet_rose' / 'phasenet_rose.pt').stat().st_size / 1e6:.1f} MB) ...")
    models["PhaseNet-RoSE"] = load_phasenet_rose(device=args.device)

    have_tf = importlib.util.find_spec("tensorflow") is not None
    if args.no_redpan:
        print("  RED-PAN-60s   (skipped: --no-redpan)")
    elif not have_tf:
        print("  RED-PAN-60s   (skipped: tensorflow not installed; "
              "`pip install -e \".[tf]\"` to enable)")
    else:
        rp_size = (MODELS_DIR / "redpan_tf60" / "train.hdf5").stat().st_size / 1e6
        print(f"  RED-PAN-60s   ({rp_size:.1f} MB) ...")
        models["RED-PAN-60s"] = load_redpan_tf60()

    # --- pick test traces and run inference ---
    indices = select_test_traces(data, args.n_traces, args.seed)
    n = len(indices)
    ncols = 2 if n > 1 else 1
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.0 * ncols, 2.7 * nrows),
                             squeeze=False)
    axes = axes.ravel()

    print()
    header = f"  {'trace':>7s}   " + "  ".join(
        f"{name:^16s}" for name in models
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    residuals: dict[str, dict[str, list]] = {
        name: {"P": [], "S": []} for name in models
    }
    hits: dict[str, dict[str, int]] = {
        name: {"P": 0, "S": 0} for name in models
    }
    for k, idx in enumerate(indices):
        wf, meta = data.get_sample(int(idx))
        stream = trace_to_stream(wf, meta)
        result = evaluate_one(stream, meta, models)
        plot_panel(axes[k], stream, meta, result, show_legend=(k == 0))

        cells = []
        for name in models:
            r = result["per_model"][name]
            p_str = f"P{r['p_dt']:+.2f}" if r["p_dt"] is not None else "P --"
            s_str = f"S{r['s_dt']:+.2f}" if r["s_dt"] is not None else "S --"
            cells.append(f"{p_str:>7s} {s_str:>7s}")
            for ph, dt in (("P", r["p_dt"]), ("S", r["s_dt"])):
                if dt is not None:
                    residuals[name][ph].append(dt); hits[name][ph] += 1
        print(f"  {idx:7d}   " + "  ".join(f"{c:^16s}" for c in cells))

    # hide any unused axes (when n is odd in a 2-col grid)
    for j in range(n, len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        f"Published RoSE pickers on {n} held-out test trace(s)   "
        f"(thresholds {P_THRESHOLD}/{S_THRESHOLD}/{DETECTION_THRESHOLD},  "
        f"P tol {TOL_P_SEC}s,  S tol {TOL_S_SEC}s)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(args.out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    print(f"\nsaved {out_path}")

    # --- summary table ---
    print("\nSummary (residuals = predicted − catalog, in seconds)")
    print(f"  {'model':<14s}  {'P recall':>8s}  {'P median':>9s}  {'P MAD':>7s}   "
          f"{'S recall':>8s}  {'S median':>9s}  {'S MAD':>7s}")
    print("  " + "-" * 76)
    for name in models:
        cells = [f"{name:<14s}"]
        for ph in ("P", "S"):
            arr = np.array(residuals[name][ph]) if residuals[name][ph] else np.array([])
            cells.append(f"  {hits[name][ph]:>3d}/{n:<3d}")
            if len(arr):
                cells.append(f"  {np.median(arr):>+7.3f}s")
                cells.append(f"  {np.median(np.abs(arr - np.median(arr))):>5.3f}s")
            else:
                cells.append(f"  {'--':>7s}")
                cells.append(f"  {'--':>5s}")
        print("  " + "".join(cells))


if __name__ == "__main__":
    main()
