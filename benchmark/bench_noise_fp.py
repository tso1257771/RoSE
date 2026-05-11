"""Count picks emitted on STEAD-noise per (model, phase, threshold).

Designed to slot into the RoSE benchmark: RoSE has no native noise,
so we use STEAD's 23,526 noise traces as the "no-event" set. For each
model we run the same SeisBench-API inference path and tally:
    n_traces_evaluated
    n_traces_with_any_pick          (T1 false positive rate at trace level)
    total_picks_emitted_above_thr   (per-phase, per-threshold)
    mean / max picks per trace

Combined with the per-event TP/FP/FN already in the RoSE bench, this
yields:
    augmented_precision = TP / (TP + FP_event + FP_noise)
    proper T1 = (event/noise classification by "any pick emitted")
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import torch
from obspy import Stream, Trace, UTCDateTime
from obspy.signal.trigger import trigger_onset
from scipy.signal import find_peaks

import seisbench.models as sbm

_THIS_DIR = Path(__file__).resolve().parent          # benchmark/
REPO_ROOT = _THIS_DIR.parent                          # repo root
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(_THIS_DIR))                    # for `redpan_inference`
from rose.checkpoint_io import safe_torch_load  # noqa: E402


logger = logging.getLogger("bench_noise_fp")

STEAD_DEFAULT = Path(os.environ["STEAD_DIR"]) if "STEAD_DIR" in os.environ else None
SAMPLING_RATE = 100.0
DT = 1.0 / SAMPLING_RATE
SYNTH_T0 = UTCDateTime("2000-01-01")


def array_to_stream(arr_enz: np.ndarray, starttime: UTCDateTime,
                    components_for_model: str) -> Stream:
    if components_for_model == "ENZ":
        data = arr_enz
        chan_order = ("E", "N", "Z")
    elif components_for_model == "ZNE":
        data = arr_enz[:, [2, 1, 0]]
        chan_order = ("Z", "N", "E")
    else:
        raise ValueError(components_for_model)
    traces = []
    for i, comp in enumerate(chan_order):
        tr = Trace(data=data[:, i].astype(np.float32))
        tr.stats.starttime = starttime
        tr.stats.sampling_rate = SAMPLING_RATE
        tr.stats.network = "XX"; tr.stats.station = "STA"
        tr.stats.channel = f"HH{comp}"
        traces.append(tr)
    s = Stream(traces)
    return s


def apply_bandpass(stream: Stream, low: float, high: float) -> Stream:
    stream = stream.copy()
    stream.detrend("demean")
    stream.filter("bandpass", freqmin=low, freqmax=high,
                  corners=4, zerophase=True)
    return stream


def predict_seisbench(model, stream: Stream, base_thr: float):
    out = model.classify(stream, P_threshold=base_thr,
                         S_threshold=base_thr, detection_threshold=base_thr)
    return [
        {"phase": str(getattr(p, "phase", "")).upper(),
         "peak_value": float(getattr(p, "peak_value", 0.0))}
        for p in (getattr(out, "picks", []) or [])
    ]


def predict_redpan(redpan, stream: Stream, base_thr: float, sr: float):
    p_arr, s_arr, _ = redpan.predict(stream, postprocess=False)
    picks = []
    for prob, phase in ((p_arr, "P"), (s_arr, "S")):
        peaks, props = find_peaks(prob, height=base_thr,
                                  distance=int(sr))
        for h in props["peak_heights"]:
            picks.append({"phase": phase, "peak_value": float(h)})
    return picks


def evaluate_noise(model, model_kind, components, noise_df, wf_dir,
                   n_noise, thresholds, bandpass=(1.0, 45.0),
                   per_sample_csv: Path | None = None,
                   model_name: str = "",
                   shard: int = 0, total_shards: int = 1):
    """Evaluate one model on STEAD noise.

    If total_shards > 1, this process handles only indices where
    (i % total_shards) == shard. Per-sample CSV file is suffixed with
    .shard{k}of{N}.csv to avoid concurrent-write corruption.
    """
    rng = np.random.default_rng(42)
    if n_noise > 0 and n_noise < len(noise_df):
        idx = rng.choice(len(noise_df), size=n_noise, replace=False)
    else:
        idx = np.arange(len(noise_df))
    if total_shards > 1:
        idx = idx[shard::total_shards]
        logger.info("[%s] shard %d/%d → %d / %d traces",
                    model_name, shard, total_shards, len(idx),
                    n_noise if n_noise else len(noise_df))
    base_thr = float(min(thresholds))
    sr = SAMPLING_RATE

    per_thr: dict[float, dict] = {
        t: {
            "n_traces_with_any_pick": 0,
            "n_picks_P": 0,
            "n_picks_S": 0,
            "picks_per_trace": [],
        }
        for t in thresholds
    }

    sample_records: list[dict] = []   # per-trace audit rows

    n_eval = 0
    n_failed = 0
    t0 = time.time()
    for k, i in enumerate(idx):
        row = noise_df.iloc[int(i)]
        trace_name = row["trace_name"]
        try:
            arr = np.load(wf_dir / f"{trace_name}.npy")
        except Exception:
            n_failed += 1
            continue
        starttime = SYNTH_T0 + (int(i) + 100000) * 1000.0
        stream = array_to_stream(arr, starttime, components)
        if bandpass is not None:
            stream = apply_bandpass(stream, *bandpass)
        try:
            if model_kind == "redpan":
                picks = predict_redpan(model, stream, base_thr, sr)
            else:
                picks = predict_seisbench(model, stream, base_thr)
        except Exception:
            n_failed += 1
            continue
        n_eval += 1
        max_p = max((p["peak_value"] for p in picks if p["phase"] == "P"),
                    default=0.0)
        max_s = max((p["peak_value"] for p in picks if p["phase"] == "S"),
                    default=0.0)
        sample_row = {
            "model": model_name,
            "trace_name": trace_name,
            "max_p_prob": max_p,
            "max_s_prob": max_s,
        }
        for thr in thresholds:
            picks_t = [p for p in picks if p["peak_value"] >= thr]
            n_p = sum(1 for p in picks_t if p["phase"] == "P")
            n_s = sum(1 for p in picks_t if p["phase"] == "S")
            per_thr[thr]["picks_per_trace"].append(len(picks_t))
            if picks_t:
                per_thr[thr]["n_traces_with_any_pick"] += 1
            per_thr[thr]["n_picks_P"] += n_p
            per_thr[thr]["n_picks_S"] += n_s
            sample_row[f"thr{thr}_n_picks_P"] = n_p
            sample_row[f"thr{thr}_n_picks_S"] = n_s
            sample_row[f"thr{thr}_n_picks_total"] = n_p + n_s
        sample_records.append(sample_row)
        if (k + 1) % 500 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(idx) - k - 1)
            logger.info("  %d/%d (%.0fs elapsed, ETA %.0fs)",
                        k + 1, len(idx), elapsed, eta)

    if per_sample_csv is not None and sample_records:
        df_samples = pd.DataFrame(sample_records)
        # Append mode: header only if file doesn't exist
        write_header = not per_sample_csv.exists()
        df_samples.to_csv(per_sample_csv, mode="a",
                          header=write_header, index=False)

    out: dict = {"n_evaluated": n_eval, "n_failed": n_failed,
                 "elapsed_s": float(time.time() - t0)}
    for thr in thresholds:
        d = per_thr[thr]
        ppt = d["picks_per_trace"]
        out[str(thr)] = {
            "n_traces_evaluated": n_eval,
            "n_traces_with_any_pick": d["n_traces_with_any_pick"],
            "trace_level_fp_rate": (d["n_traces_with_any_pick"] / n_eval
                                    if n_eval else float("nan")),
            "n_picks_total": d["n_picks_P"] + d["n_picks_S"],
            "n_picks_P": d["n_picks_P"],
            "n_picks_S": d["n_picks_S"],
            "mean_picks_per_trace": float(np.mean(ppt)) if ppt else 0.0,
            "max_picks_per_trace": int(np.max(ppt)) if ppt else 0,
            "median_picks_per_trace": float(np.median(ppt)) if ppt else 0.0,
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stead-dir", default=(str(STEAD_DEFAULT) if STEAD_DEFAULT else None))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-noise", type=int, default=0,
                    help="0 = use full 23,526 noise traces.")
    ap.add_argument("--bandpass-low", type=float, default=1.0)
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    ap.add_argument("--sweep-thresholds",
                    default="0.05,0.1,0.2,0.3,0.5,0.7")
    ap.add_argument("--models", nargs="+", default=[
        "eqt_rose","phasenet_rose",
        "eqt_instance","phasenet_instance",
        "eqt_ethz","phasenet_ethz",
        "eqt_stead","phasenet_stead",
        "redpan",
    ])
    ap.add_argument("--eqt-rose-ckpt",
        default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "eqt_rose" / "eqt_rose.pt"))
    ap.add_argument("--phasenet-rose-ckpt",
        default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "phasenet_rose" / "phasenet_rose.pt"))
    ap.add_argument("--redpan-tf",
        default=str(REPO_ROOT / "application" / "seisbench-rose-benchmark" / "models" / "redpan_tf60" / "train.hdf5"))
    ap.add_argument("--tf-threads", type=int, default=2)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--total-shards", type=int, default=1)
    args = ap.parse_args()
    if args.stead_dir is None:
        ap.error("--stead-dir is required (or set the STEAD_DIR environment variable)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(out_dir / "bench.log")],
        force=True,
    )

    bandpass = ((args.bandpass_low, args.bandpass_high)
                if args.bandpass_low > 0 and args.bandpass_high > 0
                else None)
    thresholds = sorted({float(x.strip())
                         for x in args.sweep_thresholds.split(",")
                         if x.strip()})

    stead_dir = Path(args.stead_dir)
    noise_df = pd.read_csv(stead_dir / "metadata_noise.csv", low_memory=False)
    nz_wf_dir = stead_dir / "noise_waveforms"
    n_noise = args.num_noise if args.num_noise > 0 else len(noise_df)
    logger.info("STEAD noise: %d traces total, evaluating %d",
                len(noise_df), min(n_noise, len(noise_df)))

    pretrained_map = {
        "eqt_ethz": (sbm.EQTransformer, "ethz"),
        "eqt_instance": (sbm.EQTransformer, "instance"),
        "eqt_stead": (sbm.EQTransformer, "stead"),
        "phasenet_ethz": (sbm.PhaseNet, "ethz"),
        "phasenet_instance": (sbm.PhaseNet, "instance"),
        "phasenet_stead": (sbm.PhaseNet, "stead"),
    }

    for model_id in args.models:
        logger.info("=== %s ===", model_id)
        try:
            if model_id == "eqt_rose":
                state = safe_torch_load(args.eqt_rose_ckpt, map_location="cpu")
                cfg_ckpt = state.get("config", {})
                m = sbm.EQTransformer(
                    in_samples=int(cfg_ckpt.get("model_window", 6000)),
                    sampling_rate=int(cfg_ckpt.get("sampling_rate", 100)),
                    phases=["P", "S"], norm="peak")
                m.load_state_dict(state["model"]); m.norm = "peak"
                m.to("cpu").eval()
                kind, comps = "seisbench", "ZNE"
            elif model_id == "phasenet_rose":
                state = safe_torch_load(args.phasenet_rose_ckpt, map_location="cpu")
                m = sbm.PhaseNet(phases="PSN", norm="peak",
                                 default_args={"blinding": (200, 200)})
                m.load_state_dict(state["model"]); m.norm = "peak"
                m.to("cpu").eval()
                kind, comps = "seisbench", "ZNE"
            elif model_id in pretrained_map:
                cls, weight = pretrained_map[model_id]
                m = cls.from_pretrained(weight)
                m.to("cpu").eval()
                kind, comps = "seisbench", "ZNE"
            elif model_id == "redpan":
                os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
                import tensorflow as tf
                if args.tf_threads > 0:
                    try:
                        tf.config.threading.set_intra_op_parallelism_threads(
                            args.tf_threads)
                        tf.config.threading.set_inter_op_parallelism_threads(
                            args.tf_threads)
                    except Exception:
                        pass
                from redpan_inference.core import REDPAN
                tf_model = tf.keras.models.load_model(args.redpan_tf,
                                                     compile=False)
                m = REDPAN(model=tf_model, pred_npts=6000, dt=0.01,
                           pred_interval_sec=10.0, batch_size=32,
                           use_compiled_infer=True)
                kind, comps = "redpan", "ENZ"
            else:
                logger.warning("unknown model %s", model_id)
                continue
        except Exception as exc:
            logger.error("load failed: %s", exc)
            continue

        if args.total_shards > 1:
            per_sample_csv = (out_dir
                / f"per_sample_noise.shard{args.shard}of{args.total_shards}.csv")
            json_name = f"{model_id}.shard{args.shard}of{args.total_shards}.json"
        else:
            per_sample_csv = out_dir / "per_sample_noise.csv"
            json_name = f"{model_id}.json"
        result = evaluate_noise(m, kind, comps, noise_df, nz_wf_dir,
                                n_noise, thresholds, bandpass=bandpass,
                                per_sample_csv=per_sample_csv,
                                model_name=model_id,
                                shard=args.shard,
                                total_shards=args.total_shards)
        with (out_dir / json_name).open("w") as fh:
            json.dump(result, fh, indent=2, default=str)
        logger.info("[%s] done: %s evaluated, %.0f s",
                    model_id, result["n_evaluated"], result["elapsed_s"])
        if kind == "seisbench":
            del m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
