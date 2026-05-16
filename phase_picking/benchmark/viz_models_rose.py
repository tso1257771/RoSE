"""Side-by-side visualization of model outputs on RoSE test traces.

For each chosen test trace, runs every model in the lineup and plots:
  * input waveform (Z, N, E)
  * true P / S pick positions (vertical lines)
  * each model's per-sample probability curves (P, S, and detection where
    available)
  * each model's predicted picks (markers above peak threshold)

Models compared:
  - SeisBench pretrained: PhaseNet-instance, EQT-instance
  - Custom SeisBench:     PhaseNet-RoSE,  EQT-RoSE
  - TF RED-PAN:           RED-PAN-60s, with two preprocessing variants:
       a) bandpass 1-45 Hz + internal z-score
       b) raw + internal z-score (probably matches training)

Output: one PNG per trace under --out-dir.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU-only TF (CuDNN broken)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from obspy import UTCDateTime
from scipy.signal import find_peaks

import seisbench.data as sbd
import seisbench.models as sbm

_THIS_DIR = Path(__file__).resolve().parent          # phase_picking/benchmark/
REPO_ROOT = _THIS_DIR.parent.parent          # repo root (phase_picking/benchmark -> ..)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))                    # sibling bench_* + redpan_inference
from rose.checkpoint_io import safe_torch_load  # noqa: E402

from bench_pickers_rose import trace_to_stream, parse_true_picks  # noqa: E402


def load_seisbench_model(klass: str, weight: str | None,
                         ckpt_path: str | None,
                         norm_override: str | None = None):
    if ckpt_path is not None:
        state = safe_torch_load(ckpt_path, map_location="cpu")
        cfg = state.get("config", {})
        if klass == "PhaseNet":
            model = sbm.PhaseNet(
                phases="PSN",
                norm=norm_override or "peak",
                default_args={"blinding": (200, 200)},
            )
        else:
            model = sbm.EQTransformer(
                in_samples=int(cfg.get("model_window", 6000)),
                sampling_rate=int(cfg.get("sampling_rate", 100)),
                phases=["P", "S"],
                norm=norm_override or "peak",
            )
        model.load_state_dict(state["model"])
    else:
        cls = getattr(sbm, "PhaseNet" if klass == "PhaseNet" else "EQTransformer")
        model = cls.from_pretrained(weight)
    if norm_override is not None:
        model.norm = norm_override
    model.to("cpu").eval()
    return model


def load_redpan(model_path: str):
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    for g in gpus:
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    from rose.redpan_inference.core import REDPAN
    model = tf.keras.models.load_model(model_path, compile=False)
    return REDPAN(
        model=model, pred_npts=6000, dt=0.01,
        pred_interval_sec=10.0, batch_size=8,
        use_compiled_infer=False,
    )


def run_seisbench(model, stream):
    """Run model.classify on a stream, return picks + per-sample anno arrays.

    Returns dict with:
      picks: list of (time_offset_s, peak_value, phase) — relative to stream[0]
      curves: dict[str, np.ndarray] of probability traces
      curve_offsets: dict[str, float] — seconds to add to each curve's
        sample-index time axis to align with stream[0].stats.starttime.
        SeisBench's annotate Stream has its own starttime that lags the
        input stream because of blinding + sliding-window stitching
        (PhaseNet ~2.5 s, EQT ~5.0 s); without this offset, plotted
        curves appear shifted to the left of the true picks.
    """
    from seisbench.util import worker_seeding  # noqa: F401  ensure submods loaded

    stream_starttime = stream[0].stats.starttime
    anno = model.annotate(stream)
    out_curves: dict[str, np.ndarray] = {}
    out_offsets: dict[str, float] = {}
    for tr in anno:
        ch = tr.stats.channel
        # SeisBench annotation channel suffixes:
        # PhaseNet: <station>_PhaseNet_P / _S / _N
        # EQT:      <station>_EQTransformer_P / _S / _Detection
        suffix = ch.split("_")[-1]
        if suffix in {"P", "S", "Detection"}:
            out_curves[suffix] = np.asarray(tr.data)
            out_offsets[suffix] = float(tr.stats.starttime - stream_starttime)

    classify_out = model.classify(
        stream, P_threshold=0.05, S_threshold=0.05, detection_threshold=0.05,
    )
    picks = []
    for p in (getattr(classify_out, "picks", []) or []):
        picks.append({
            "phase": str(getattr(p, "phase", "")).upper(),
            "offset_s": float(p.peak_time - stream_starttime),
            "value": float(getattr(p, "peak_value", 0.0)),
        })
    return {
        "picks": picks, "curves": out_curves, "curve_offsets": out_offsets,
        "starttime": stream_starttime,
        "sample_rate": float(stream[0].stats.sampling_rate),
    }


def make_enz_stream(wf_zne: np.ndarray, meta, sampling_rate: float,
                    bandpass: tuple[float, float] | None):
    """RED-PAN expects [E, N, Z] channel order (per its training pipeline,
    redpan/legacy/data_aug.py:207). RoSE WaveformDataset hands us [Z, N, E]
    (component_order='ZNE'), so we reorder the channels then build the
    stream with the matching `components='ENZ'`."""
    wf_enz = wf_zne[[2, 1, 0]]  # [Z,N,E] → [E,N,Z]
    return trace_to_stream(wf_enz, meta, sampling_rate,
                           components="ENZ", bandpass=bandpass)


def run_redpan(redpan, stream):
    p_arr, s_arr, m_arr = redpan.predict(stream, postprocess=False)
    starttime = stream[0].stats.starttime
    sr = float(stream[0].stats.sampling_rate)
    picks = []
    for prob, phase in ((p_arr, "P"), (s_arr, "S")):
        peaks, props = find_peaks(prob, height=0.05, distance=int(sr))
        for idx, h in zip(peaks, props["peak_heights"]):
            picks.append({
                "phase": phase,
                "offset_s": float(idx) / sr,
                "value": float(h),
            })
    return {
        "picks": picks,
        "curves": {"P": p_arr, "S": s_arr, "Detection": m_arr},
        "curve_offsets": {"P": 0.0, "S": 0.0, "Detection": 0.0},
        "starttime": starttime, "sample_rate": sr,
    }


def plot_trace(trace_idx: int, wf: np.ndarray, meta, true_picks: dict,
               sample_rate: float, models_results: list[tuple[str, dict]],
               out_path: Path):
    n_models = len(models_results)
    fig, axes = plt.subplots(n_models + 1, 1, sharex=True,
                             figsize=(14, 1.6 * (n_models + 1)),
                             gridspec_kw={"hspace": 0.05})

    # --- waveform panel -----------------------------------------------------
    ax = axes[0]
    t = np.arange(wf.shape[1]) / sample_rate
    for i, (label, color) in enumerate([("Z", "k"), ("N", "tab:blue"),
                                         ("E", "tab:orange")]):
        x = wf[i].astype(float)
        x = (x - x.mean()) / (x.std() + 1e-9)
        ax.plot(t, x + (2 - i) * 5, lw=0.5, color=color, label=label)
    starttime = UTCDateTime(str(meta["trace_start_time"]))
    for phase, color in (("P", "red"), ("S", "blue")):
        if phase in true_picks:
            ax.axvline(float(true_picks[phase] - starttime),
                       color=color, lw=1.0, alpha=0.6,
                       label=f"true {phase}")
    ax.set_ylabel(f"trace {trace_idx}\nZNE (z-norm)")
    ax.legend(loc="upper right", ncol=5, fontsize=7)
    ax.set_yticks([])

    # --- per-model probability panels ---------------------------------------
    for ax, (model_name, res) in zip(axes[1:], models_results):
        sr_m = res.get("sample_rate", sample_rate)
        offsets = res.get("curve_offsets", {})
        for phase, color in (("Detection", "tab:gray"),
                             ("P", "red"),
                             ("S", "blue")):
            curve = res["curves"].get(phase)
            if curve is None:
                continue
            # Shift each curve by its own starttime offset (PhaseNet ~2.5 s,
            # EQT ~5 s) so the prob peaks line up with the true-pick lines
            # plotted from the input stream's starttime.
            offset = float(offsets.get(phase, 0.0))
            tt = np.arange(len(curve)) / sr_m + offset
            alpha = 0.45 if phase == "Detection" else 0.85
            ax.plot(tt, curve, color=color, lw=0.8, alpha=alpha,
                    label=phase)
        for phase, color in (("P", "red"), ("S", "blue")):
            if phase in true_picks:
                ax.axvline(float(true_picks[phase] - starttime),
                           color=color, lw=0.6, alpha=0.5)
        for p in res["picks"]:
            mk = "x" if p["phase"] == "P" else "+"
            color = "red" if p["phase"] == "P" else "blue"
            ax.scatter(p["offset_s"], p["value"], marker=mk, color=color,
                       s=60, alpha=0.9, zorder=5)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel(model_name, fontsize=8)
        ax.legend(loc="upper right", ncol=3, fontsize=7)

    axes[-1].set_xlabel("time since stream start (s)")
    fig.suptitle(f"RoSE test trace #{trace_idx}  "
                 f"({meta.get('source_id','?')} @ "
                 f"{meta.get('station_network_code','?')}.{meta.get('station_code','?')})",
                 y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rose-dir", default=os.environ.get("ROSE_DATA_DIR"))
    ap.add_argument("--out-dir", default=os.environ.get("ROSE_EVAL_DIR", "eval/viz"))
    ap.add_argument("--n-traces", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--eqt-rose",
        default=str(REPO_ROOT / "models" / "eqt_rose" / "eqt_rose.pt"),
        help="Path to the EQT-RoSE checkpoint.",
    )
    ap.add_argument(
        "--phasenet-rose",
        default=str(REPO_ROOT / "models" / "phasenet_rose" / "phasenet_rose.pt"),
        help="Path to the PhaseNet-RoSE checkpoint.",
    )
    ap.add_argument(
        "--redpan-tf-model",
        default=str(REPO_ROOT / "models" / "redpan_tf60" / "train.hdf5"),
    )
    ap.add_argument("--bandpass-low", type=float, default=1.0)
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    args = ap.parse_args()
    if args.rose_dir is None:
        ap.error("--rose-dir is required (or set the ROSE_DATA_DIR environment variable)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bandpass = (args.bandpass_low, args.bandpass_high)
    if args.bandpass_low <= 0 or args.bandpass_high <= 0:
        bandpass = None

    data = sbd.WaveformDataset(
        path=args.rose_dir, sampling_rate=100,
        component_order="ZNE", cache=None,
    )
    test = data.test()
    md = test.metadata.reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    eligible = md[
        md["trace_p_arrival_sample"].notna() & md["trace_s_arrival_sample"].notna()
    ].index
    indices = rng.choice(eligible, size=args.n_traces, replace=False)
    indices = np.sort(indices)
    print("Selected indices:", indices.tolist())

    # ----- load all models once ---------------------------------------------
    print("loading models...")
    pn_inst = load_seisbench_model("PhaseNet", "instance", None)
    eqt_inst = load_seisbench_model("EQTransformer", "instance", None)
    pn_rose = load_seisbench_model("PhaseNet", None, args.phasenet_rose,
                                   norm_override="peak")
    eqt_rose = load_seisbench_model("EQTransformer", None, args.eqt_rose,
                                    norm_override="peak")
    redpan = load_redpan(args.redpan_tf_model)

    # ----- one figure per trace ---------------------------------------------
    for idx in indices:
        wf, _ = test.get_sample(int(idx))
        meta = md.iloc[int(idx)]
        true_picks = parse_true_picks(meta)
        if not true_picks:
            continue

        # SeisBench-side streams (ZNE, optional bandpass)
        stream_band_zne = trace_to_stream(wf, meta, 100, "ZNE",
                                          bandpass=bandpass)
        # RED-PAN canonical: ENZ-ordered input, no upfront filter
        # (internal z-score happens inside REDPAN.predict).
        stream_enz_raw = make_enz_stream(wf, meta, 100, bandpass=None)

        results = []
        print(f"  trace {idx}: running PhaseNet-instance ...")
        results.append(("PhaseNet-instance",
                        run_seisbench(pn_inst, stream_band_zne)))
        print("             EQT-instance ...")
        results.append(("EQT-instance",
                        run_seisbench(eqt_inst, stream_band_zne)))
        print("             PhaseNet-RoSE ...")
        results.append(("PhaseNet-RoSE",
                        run_seisbench(pn_rose, stream_band_zne)))
        print("             EQT-RoSE ...")
        results.append(("EQT-RoSE",
                        run_seisbench(eqt_rose, stream_band_zne)))
        print("             RED-PAN-60s ...")
        results.append(("RED-PAN-60s",
                        run_redpan(redpan, stream_enz_raw)))

        out_png = out_dir / f"trace_{int(idx):06d}.png"
        plot_trace(int(idx), wf, meta, true_picks, 100.0, results, out_png)
        print(f"  -> {out_png}")

    print(f"done. PNGs under {out_dir}")


if __name__ == "__main__":
    main()
