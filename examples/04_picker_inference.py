"""Tutorial 4 — Run all three published RoSE pickers on test traces.

End-to-end demo of the released checkpoints under
``application/seisbench-rose-benchmark/models/``:

    1. Open `data/rose` with `RoSE`.
    2. Filter to the held-out test split (falls back to the full dataset
       if the `split` column hasn't been materialised yet).
    3. Pick a few well-labelled test traces (both P and S in window).
    4. Reconstruct a 3-component **Z, N, E** ObsPy Stream per trace.
    5. Demean + linear-detrend + (default) **1–45 Hz Butterworth band-pass**
       the stream — matches the published-benchmark pre-filter and the
       1–45 Hz pre-augment filter EQT-RoSE was trained with.
       Each model still applies its OWN amplitude normalisation internally
       (peak for the two SeisBench models, Z-score for RED-PAN); do NOT
       pre-normalise the input. See the long comment block below.
    6. Load PhaseNet-RoSE, EQT-RoSE, and RED-PAN-60s via the release
       loaders (``application/seisbench-rose-benchmark/benchmarks/models.py``).
    7. Run each model and grab raw per-sample probability curves
       (``model.annotate(stream)`` for SeisBench, ``REDPAN.predict(...,
       postprocess=False)`` for RED-PAN), plus picks from
       ``model.classify(stream)``; compare to the catalog labels and
       print a per-model residual table.
    8. For every selected trace, save a 6-panel PNG with a shared time
       axis: Z/N/E waveforms on top, then PhaseNet / EQTransformer /
       RED-PAN probability curves (each ylim ``[-0.1, 1.1]``). Catalog and
       predicted picks are drawn as vertical lines across all panels so
       timestamps line up vertically.

RED-PAN-60s requires TensorFlow; if it isn't installed (or you pass
``--no-redpan``) the example runs the two SeisBench pickers only and the
RED-PAN row is omitted.

Usage:
    python examples/04_picker_inference.py                     # default 1-45 Hz bandpass
    python examples/04_picker_inference.py --highpass 0.5      # bandpass 0.5-45 Hz
    python examples/04_picker_inference.py --lowpass 0         # highpass-only at 1 Hz
    python examples/04_picker_inference.py --highpass 0 --lowpass 0   # raw (demean only)
    python examples/04_picker_inference.py --n-traces 8 --seed 7
    python examples/04_picker_inference.py --device cuda
    python examples/04_picker_inference.py --no-redpan         # PyTorch only

Outputs: ``outputs/04_picker_inference/trace_<idx>.png`` (one per trace) +
a residual summary on stdout.
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


def _rel(p):
    """Format ``p`` for log output, relative to the repo root when possible."""
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from obspy import Stream, Trace, UTCDateTime  # noqa: E402

from rose import RoSE  # noqa: E402
from benchmarks.models import (  # noqa: E402
    load_eqt_rose,
    load_phasenet_rose,
    load_redpan_tf60,
)

DATA_DIR = os.environ.get("ROSE_DATA_DIR", str(REPO_ROOT / "data" / "rose"))
MODELS_DIR = RELEASE / "models"
OUT_DIR = REPO_ROOT / "outputs" / "04_picker_inference"

# RED-PAN paper convention; see benchmark/ for why the tolerances aren't symmetric.
TOL_P_SEC = 0.5
TOL_S_SEC = 1.0

# Pick threshold used for all three models (matches the published benchmark default).
P_THRESHOLD = 0.3
S_THRESHOLD = 0.3
DETECTION_THRESHOLD = 0.3

# ---------------------------------------------------------------------------
# A note on input format and normalisation
# ---------------------------------------------------------------------------
# Channel order:  Z, N, E   (RoSE on-disk + SeisBench convention).
#   - SeisBench EQTransformer / PhaseNet expect ZNE inputs.
#   - RED-PAN-60s expects ENZ; the bundled `_RP60Wrapper.classify()` /
#     `reorder_zne_to_enz()` below rotate the channels for you.
#
# Normalisation is *applied INTERNALLY by every model* — do NOT pre-normalise
# the stream before passing it in:
#   - EQT-RoSE & PhaseNet-RoSE were trained with `norm="peak"` (per-channel
#     demean + divide by max-abs); the loader sets `model.norm = "peak"` and
#     SeisBench applies it inside `model.annotate(...)` / `model.classify(...)`.
#   - RED-PAN-60s uses per-window per-channel Z-score standardisation
#     (demean + divide by std), applied inside `REDPAN.predict(...)`.
#
# We *do* preprocess the input by demean + linear-detrend + (optional)
# Butterworth band-pass via `preprocess()`; the published benchmark
# (`benchmarks/run_benchmark.py` and `bench_pickers_rose.py`) defaults to
# 1-45 Hz bandpass to match the 1-45 Hz pre-augment filter EQT-RoSE was
# trained with — so the same default is used here.
# ---------------------------------------------------------------------------

# Plotting palette: phase = colour, model = line style.
PHASE_COLORS = {
    "P":         "tab:blue",
    "S":         "tab:red",
    "N":         "tab:gray",      # PhaseNet's "noise" channel
    "Detection": "tab:green",     # EQT detection / RED-PAN event mask
}
WAVEFORM_COLOR = "black"
WAVEFORM_LW = 0.4    # waveform Z/N/E trace
CATALOG_LW  = 1.0    # ground-truth (catalog) P/S vertical lines
PRED_LW     = 0.6    # model-predicted P/S vertical lines (dashed)
PROB_LW     = 0.85   # model probability curves
GUIDE_LW    = 0.35   # threshold + zero baselines


# ---------------------------------------------------------------------------
# Data selection + Stream construction
# ---------------------------------------------------------------------------
def select_test_traces(data: RoSE, n: int, seed: int, *,
                       min_snr_db: float = 5.0,
                       min_magnitude: float = 2.5,
                       max_distance_km: float = 200.0,
                       edge_margin_s: float = 5.0) -> np.ndarray:
    """Return indices of `n` *unambiguous* test-split traces.

    Random sampling over the raw test split easily lands on noisy / weak /
    edge-clipped waveforms where you can't visually separate the signal from
    the picks — useless for a tutorial. This helper applies a quality gate
    before sampling so the chosen traces have clear P and S arrivals:

      * test split (catalog-held-out)
      * source magnitude ≥ ``min_magnitude``
      * epicentral distance ≤ ``max_distance_km``  (closer ⇒ stronger signal)
      * trace_p_snr_db AND trace_s_snr_db ≥ ``min_snr_db``
      * both P and S sample positions ≥ ``edge_margin_s`` from the window edges

    Pass any threshold as ``0`` to disable that gate.
    """
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
    n_test = len(eligible)

    sub = md.loc[eligible].copy()
    # cast everything to numeric; missing → NaN → fails the > comparisons.
    for col in ("trace_p_arrival_sample", "trace_s_arrival_sample",
                "trace_p_snr_db", "trace_s_snr_db",
                "source_magnitude", "path_ep_distance_km"):
        sub[col] = pd.to_numeric(sub[col], errors="coerce")

    # 60-s window @ 100 Hz = 6000 samples; require both picks at least
    # `edge_margin_s` from each edge so the model sees full context.
    edge = float(edge_margin_s) * 100.0  # samples
    in_win = lambda col: sub[col].between(edge, 6000 - 1 - edge, inclusive="both")
    mask = in_win("trace_p_arrival_sample") & in_win("trace_s_arrival_sample")
    if min_snr_db > 0:
        mask &= (sub["trace_p_snr_db"] >= min_snr_db) & (sub["trace_s_snr_db"] >= min_snr_db)
    if min_magnitude > 0:
        mask &= sub["source_magnitude"] >= min_magnitude
    if max_distance_km > 0:
        mask &= sub["path_ep_distance_km"] <= max_distance_km
    eligible = eligible[mask.to_numpy()]
    if len(eligible) == 0:
        raise RuntimeError(
            f"no traces in {scope} pass the quality gate "
            f"(SNR≥{min_snr_db} dB, M≥{min_magnitude}, "
            f"epi≤{max_distance_km} km, edge≥{edge_margin_s}s). "
            f"Loosen with --min-snr-db / --min-magnitude / --max-distance-km."
        )
    rng = np.random.default_rng(seed)
    n = min(n, len(eligible))
    chosen = rng.choice(eligible, size=n, replace=False)
    print(
        f"selected {n} of {len(eligible)} clean traces "
        f"({len(eligible)}/{n_test} = {100*len(eligible)/n_test:.1f}% of the {scope} "
        f"pass SNR≥{min_snr_db} dB & M≥{min_magnitude} & epi≤{max_distance_km} km "
        f"& edge≥{edge_margin_s}s)"
    )
    # Print one-line context for each chosen trace so the user knows why
    # the picks are (or aren't) clear.
    for idx in np.sort(chosen):
        r = sub.loc[idx]
        print(
            f"    idx {int(idx):6d}  {r['station_network_code']}.{r['station_code']:5s}  "
            f"M{r['source_magnitude']:.1f}  "
            f"epi {r['path_ep_distance_km']:5.1f} km  "
            f"P-SNR {r['trace_p_snr_db']:5.1f} dB  S-SNR {r['trace_s_snr_db']:5.1f} dB"
        )
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


def reorder_zne_to_enz(stream_zne: Stream) -> Stream:
    """Return a new Stream in E, N, Z order (RED-PAN's training convention)."""
    chans = {tr.stats.channel[-1]: tr for tr in stream_zne}
    if set(chans) != {"Z", "N", "E"}:
        raise ValueError(f"expected Z/N/E channels, got {sorted(chans)}")
    return Stream([chans["E"], chans["N"], chans["Z"]])


def preprocess(stream: Stream,
               highpass_hz: float | None,
               lowpass_hz: float | None) -> Stream:
    """Demean + linear-detrend + (optional) 4-pole zero-phase Butterworth filter.

    Returns a new (deep-copied) Stream so the caller's input isn't mutated.
    Pass ``None`` or ``0`` for either corner to disable that leg of the filter.
    Default for callers in this script is highpass=1.0 Hz / lowpass=45.0 Hz
    (the published-benchmark default; matches EQT-RoSE's training pre-filter).
    """
    s = stream.copy()
    s.detrend("demean").detrend("linear")
    hp = (highpass_hz or 0.0) > 0.0
    lp = (lowpass_hz or 0.0) > 0.0
    if hp and lp:
        s.filter("bandpass", freqmin=highpass_hz, freqmax=lowpass_hz,
                 corners=4, zerophase=True)
    elif hp:
        s.filter("highpass", freq=highpass_hz, corners=4, zerophase=True)
    elif lp:
        s.filter("lowpass", freq=lowpass_hz, corners=4, zerophase=True)
    return s


def filter_label(highpass_hz: float | None, lowpass_hz: float | None) -> str:
    """Short human-readable description of the filter for the plot title."""
    hp = (highpass_hz or 0.0) > 0.0
    lp = (lowpass_hz or 0.0) > 0.0
    if hp and lp:
        return f"bp {highpass_hz:g}–{lowpass_hz:g} Hz"
    if hp:
        return f"hp {highpass_hz:g} Hz"
    if lp:
        return f"lp {lowpass_hz:g} Hz"
    return "no filter (raw + demean/detrend only)"


# ---------------------------------------------------------------------------
# Inference: raw probability curves + picks
# ---------------------------------------------------------------------------
def run_seisbench_curves(model, stream: Stream) -> dict:
    """Return per-channel probability arrays + picks from a SeisBench model."""
    anno = model.annotate(stream)
    starttime = stream[0].stats.starttime
    sr = float(stream[0].stats.sampling_rate)
    curves: dict[str, np.ndarray] = {}
    offsets: dict[str, float] = {}
    for tr in anno:
        # SeisBench annotation channel suffixes:
        #   PhaseNet: <station>_PhaseNet_P / _S / _N
        #   EQT:      <station>_EQTransformer_P / _S / _Detection
        # The annotate Stream's starttime lags the input (blinding +
        # sliding-window stitch); record the offset so curves align in time.
        suffix = tr.stats.channel.split("_")[-1]
        if suffix in {"P", "S", "N", "Detection"}:
            curves[suffix] = np.asarray(tr.data, dtype=float)
            offsets[suffix] = float(tr.stats.starttime - starttime)
    out = model.classify(
        stream,
        P_threshold=P_THRESHOLD, S_threshold=S_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
    )
    p_picks = [p for p in out.picks if str(p.phase).upper() == "P"]
    s_picks = [p for p in out.picks if str(p.phase).upper() == "S"]
    return {
        "curves": curves, "offsets": offsets,
        "p_picks": p_picks, "s_picks": s_picks,
        "starttime": starttime, "sampling_rate": sr,
    }


def run_redpan_curves(wrapper, stream_zne: Stream) -> dict:
    """Same shape as run_seisbench_curves, but for the RED-PAN-60s wrapper."""
    enz = reorder_zne_to_enz(stream_zne)
    p_arr, s_arr, m_arr = wrapper.redpan.predict(enz, postprocess=False)
    starttime = enz[0].stats.starttime
    sr = float(enz[0].stats.sampling_rate)
    out = wrapper.classify(
        stream_zne,
        P_threshold=P_THRESHOLD, S_threshold=S_THRESHOLD,
        detection_threshold=DETECTION_THRESHOLD,
    )
    p_picks = [p for p in out.picks if str(p.phase).upper() == "P"]
    s_picks = [p for p in out.picks if str(p.phase).upper() == "S"]
    return {
        "curves": {"P": np.asarray(p_arr, dtype=float),
                   "S": np.asarray(s_arr, dtype=float),
                   "Detection": np.asarray(m_arr, dtype=float)},
        "offsets": {"P": 0.0, "S": 0.0, "Detection": 0.0},
        "p_picks": p_picks, "s_picks": s_picks,
        "starttime": starttime, "sampling_rate": sr,
    }


def closest_match(predicted, target_time, tolerance_s):
    """Return (best_pick, residual_seconds) within `tolerance_s`, or (None, None)."""
    best, best_dt = None, None
    for p in predicted:
        dt = float(p.peak_time - target_time)
        if abs(dt) <= tolerance_s and (best_dt is None or abs(dt) < abs(best_dt)):
            best, best_dt = p, dt
    return best, best_dt


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_event(stream: Stream, meta, true_p: UTCDateTime, true_s: UTCDateTime,
               model_outputs: dict[str, dict], out_path: Path,
               filter_str: str = "") -> None:
    """One figure per trace: 3 waveform rows + N model-output rows.

    All rows share the same x-axis (seconds since trace start); model rows
    have ylim [-0.1, 1.1] so the probability curves are directly comparable
    and the small overshoot above 1 / undershoot below 0 stays visible.
    """
    sr = float(stream[0].stats.sampling_rate)
    starttime = stream[0].stats.starttime
    npts = stream[0].stats.npts
    t_wf = np.arange(npts) / sr
    t_max = npts / sr

    p_offset_s = float(true_p - starttime)
    s_offset_s = float(true_s - starttime)

    n_model_rows = len(model_outputs)
    n_rows = 3 + n_model_rows
    # Tight layout: small hspace + per-row height ~= 1.0 in keeps the curves
    # large and the inter-panel gap minimal. The bottom xlabel + suptitle
    # eat ~0.6 in of fixed margin combined.
    height_ratios = [1.0] * 3 + [1.05] * n_model_rows
    fig, axes = plt.subplots(
        n_rows, 1, sharex=True,
        figsize=(11.0, 1.0 * n_rows + 0.55),
        gridspec_kw={
            "height_ratios": height_ratios, "hspace": 0.06,
            "left": 0.07, "right": 0.985, "top": 0.945, "bottom": 0.075,
        },
    )

    # --- waveform rows: Z, N, E ---
    for ax, comp in zip(axes[:3], "ZNE"):
        tr = stream.select(channel=f"*{comp}")[0]
        d = tr.data.astype(float)
        ax.plot(t_wf, d, color=WAVEFORM_COLOR, lw=WAVEFORM_LW)
        ax.set_ylabel(comp, fontsize=11, rotation=0, labelpad=12, va="center")
        ax.tick_params(axis="both", labelsize=8)
        amax = max(np.max(np.abs(d)), 1.0)
        ax.set_ylim(-1.08 * amax, 1.08 * amax)
        ax.axhline(0, color="0.85", lw=GUIDE_LW, zorder=0)
        # catalog picks across the waveform
        ax.axvline(p_offset_s, color=PHASE_COLORS["P"], lw=CATALOG_LW, alpha=0.85)
        ax.axvline(s_offset_s, color=PHASE_COLORS["S"], lw=CATALOG_LW, alpha=0.85)

    # Ground-truth legend on the top (Z) row — uses proxy artists so the
    # legend correctly represents both catalog (solid, drawn here) and the
    # model-predicted picks (dashed, drawn on the model rows below).
    pick_legend = [
        Line2D([0], [0], color=PHASE_COLORS["P"], lw=CATALOG_LW,
               alpha=0.85, label="catalog P  (ground truth)"),
        Line2D([0], [0], color=PHASE_COLORS["S"], lw=CATALOG_LW,
               alpha=0.85, label="catalog S  (ground truth)"),
        Line2D([0], [0], color=PHASE_COLORS["P"], lw=PRED_LW, ls="--",
               alpha=0.7, label="model P pick"),
        Line2D([0], [0], color=PHASE_COLORS["S"], lw=PRED_LW, ls="--",
               alpha=0.7, label="model S pick"),
    ]
    axes[0].legend(handles=pick_legend, loc="upper right", fontsize=7,
                   ncol=4, frameon=False, handlelength=1.6, columnspacing=1.0)

    # --- model-output rows ---
    legend_handles_done = False
    for ax, (model_name, result) in zip(axes[3:], model_outputs.items()):
        for ch_name, prob in result["curves"].items():
            offset = result["offsets"].get(ch_name, 0.0)
            t_curve = offset + np.arange(len(prob)) / result["sampling_rate"]
            label = f"{ch_name} prob." if not legend_handles_done else None
            ax.plot(
                t_curve, prob, color=PHASE_COLORS.get(ch_name, "0.4"),
                lw=PROB_LW, alpha=0.95, label=label,
            )
        # threshold guide line
        ax.axhline(P_THRESHOLD, color="0.7", lw=GUIDE_LW, ls=":", zorder=0)
        # catalog picks (ground truth) — also drawn here for vertical alignment
        ax.axvline(p_offset_s, color=PHASE_COLORS["P"], lw=CATALOG_LW, alpha=0.85)
        ax.axvline(s_offset_s, color=PHASE_COLORS["S"], lw=CATALOG_LW, alpha=0.85)
        # model-predicted picks (dashed verticals, phase-coloured)
        for p in result["p_picks"]:
            ax.axvline(
                float(p.peak_time - starttime),
                color=PHASE_COLORS["P"], lw=PRED_LW, ls="--", alpha=0.7,
            )
        for p in result["s_picks"]:
            ax.axvline(
                float(p.peak_time - starttime),
                color=PHASE_COLORS["S"], lw=PRED_LW, ls="--", alpha=0.7,
            )
        ax.set_ylim(-0.1, 1.1)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylabel(model_name, fontsize=10)
        ax.tick_params(axis="both", labelsize=8)
        if not legend_handles_done:
            ax.legend(loc="upper right", fontsize=7, ncol=len(result["curves"]),
                      frameon=False, handlelength=1.6, columnspacing=1.0)
            legend_handles_done = True

    axes[-1].set_xlabel("time (s) since trace start", fontsize=10)
    axes[-1].set_xlim(0, t_max)

    title = (
        f"{meta['station_network_code']}.{meta['station_code']}   "
        f"M{float(meta['source_magnitude']):.1f},  "
        f"epi {float(meta['path_ep_distance_km']):.0f} km,  "
        f"depth {float(meta['source_depth_km']):.0f} km   |   "
        f"start {starttime.strftime('%Y-%m-%dT%H:%M:%S')}   |   "
        f"thresh {P_THRESHOLD}/{S_THRESHOLD}/{DETECTION_THRESHOLD}"
    )
    if filter_str:
        title += f"   |   {filter_str}"
    fig.suptitle(title, fontsize=10, y=0.985)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rose-dir", default=DATA_DIR,
                    help=f"RoSE SeisBench dataset dir "
                         f"(default: $ROSE_DATA_DIR or {_rel(DATA_DIR)}).")
    ap.add_argument("--n-traces", type=int, default=4,
                    help="Number of test traces to evaluate + plot (default: 4).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-snr-db", type=float, default=5.0,
                    help="Reject traces with P or S SNR below this (dB). "
                         "Default 5.0; pass 0 to disable.")
    ap.add_argument("--min-magnitude", type=float, default=2.5,
                    help="Minimum source magnitude (default 2.5; 0 disables).")
    ap.add_argument("--max-distance-km", type=float, default=200.0,
                    help="Maximum epicentral distance (km). Default 200; 0 disables.")
    ap.add_argument("--edge-margin-s", type=float, default=5.0,
                    help="Required margin (seconds) between each pick and the "
                         "60-s window edges, so the model sees full context. "
                         "Default 5.0; pass 0 to disable.")
    ap.add_argument("--device", default="cpu", help='"cpu" (default) or "cuda".')
    ap.add_argument("--no-redpan", action="store_true",
                    help="Skip RED-PAN-60s (avoids the TensorFlow dependency).")
    ap.add_argument("--highpass", type=float, default=1.0,
                    help="Highpass corner (Hz) applied before inference. "
                         "Pass 0 to disable. Default: 1.0 Hz "
                         "(matches the published-benchmark band-pass low corner).")
    ap.add_argument("--lowpass", type=float, default=45.0,
                    help="Lowpass corner (Hz). Combined with --highpass it "
                         "becomes a bandpass. Pass 0 to disable. Default: 45.0 Hz "
                         "(matches the published-benchmark band-pass high corner; "
                         "this matches EQT-RoSE's training-time pre-filter).")
    ap.add_argument("--out-dir", default=str(OUT_DIR),
                    help=f"Directory for per-trace PNGs (default: {_rel(OUT_DIR)}).")
    args = ap.parse_args()

    print(f"Opening {args.rose_dir} ...")
    data = RoSE(args.rose_dir, component_order="ZNE")
    print(f"  total traces: {len(data)}")

    # --- load all three models (RED-PAN is optional) -- order matches plot rows
    print("Loading models from", MODELS_DIR.relative_to(REPO_ROOT))
    models: dict[str, object] = {}

    pn_size = (MODELS_DIR / "phasenet_rose" / "phasenet_rose.pt").stat().st_size / 1e6
    print(f"  PhaseNet-RoSE ({pn_size:.1f} MB) ...")
    models["PhaseNet-RoSE"] = load_phasenet_rose(device=args.device)

    eqt_size = (MODELS_DIR / "eqt_rose" / "eqt_rose.pt").stat().st_size / 1e6
    print(f"  EQT-RoSE      ({eqt_size:.1f} MB) ...")
    models["EQT-RoSE"] = load_eqt_rose(device=args.device)

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
    indices = select_test_traces(
        data, args.n_traces, args.seed,
        min_snr_db=args.min_snr_db,
        min_magnitude=args.min_magnitude,
        max_distance_km=args.max_distance_km,
        edge_margin_s=args.edge_margin_s,
    )
    n = len(indices)

    print()
    header = f"  {'trace':>7s}   " + "  ".join(f"{name:^16s}" for name in models)
    print(header)
    print("  " + "-" * (len(header) - 2))
    residuals: dict[str, dict[str, list]] = {nm: {"P": [], "S": []} for nm in models}
    hits: dict[str, dict[str, int]] = {nm: {"P": 0, "S": 0} for nm in models}

    out_dir = Path(args.out_dir)
    flabel = filter_label(args.highpass, args.lowpass)
    print(f"  preprocessing: detrend(demean+linear) + {flabel}\n")

    for idx in indices:
        wf, meta = data.get_sample(int(idx))
        stream_zne = trace_to_stream(wf, meta)
        # Demean/detrend + (optional) band-pass the input. Each model still
        # applies its OWN normalisation internally (peak for SeisBench,
        # Z-score for RED-PAN) — see the comment block at the top of this file.
        stream_pp = preprocess(stream_zne, args.highpass, args.lowpass)

        sr = float(meta["trace_sampling_rate_hz"])
        starttime = UTCDateTime(meta["trace_start_time"])
        true_p = starttime + float(meta["trace_p_arrival_sample"]) / sr
        true_s = starttime + float(meta["trace_s_arrival_sample"]) / sr

        model_outputs: dict[str, dict] = {}
        cells = []
        for name, m in models.items():
            if name == "RED-PAN-60s":
                result = run_redpan_curves(m, stream_pp)
            else:
                result = run_seisbench_curves(m, stream_pp)
            model_outputs[name] = result

            _, p_dt = closest_match(result["p_picks"], true_p, TOL_P_SEC)
            _, s_dt = closest_match(result["s_picks"], true_s, TOL_S_SEC)
            p_str = f"P{p_dt:+.2f}" if p_dt is not None else "P --"
            s_str = f"S{s_dt:+.2f}" if s_dt is not None else "S --"
            cells.append(f"{p_str:>7s} {s_str:>7s}")
            for ph, dt in (("P", p_dt), ("S", s_dt)):
                if dt is not None:
                    residuals[name][ph].append(dt); hits[name][ph] += 1
        print(f"  {idx:7d}   " + "  ".join(f"{c:^16s}" for c in cells))

        png_path = out_dir / f"trace_{int(idx):07d}.png"
        plot_event(stream_pp, meta, true_p, true_s, model_outputs,
                   png_path, filter_str=flabel)

    print(f"\nsaved {n} PNGs under {_rel(out_dir)}")

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
