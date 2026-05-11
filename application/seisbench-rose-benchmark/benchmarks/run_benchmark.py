"""Run the RoSE / STEAD-test phase-pick benchmark for one model.

Usage:
    # RoSE event traces
    python benchmarks/run_benchmark.py \\
        --dataset rose --rose-dir /path/to/rose \\
        --model eqt_rose \\
        --out-dir results/runs/eqt_rose_rose

    # STEAD test set (events + noise) for one of the three models
    python benchmarks/run_benchmark.py \\
        --dataset stead --stead-dir /path/to/STEAD/benchmark_stead \\
        --model phasenet_rose \\
        --out-dir results/runs/phasenet_rose_stead

    # Just noise traces (false-positive characterisation)
    python benchmarks/run_benchmark.py \\
        --dataset noise --stead-dir /path/to/STEAD/benchmark_stead \\
        --model redpan_tf60 \\
        --out-dir results/runs/redpan_tf60_noise

The output is a per-model JSON keyed by threshold; aggregate the JSONs
into the publication CSVs with `python benchmarks/build_tables.py`.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from obspy import UTCDateTime

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR.parent))   # so `pickerbench` and `redpan_inference` import cleanly

from pickerbench.matching import (
    match_picks, match_picks_cross_phase, evaluate_trace_detections,
    aggregate_summary,
)
from pickerbench.trace_io import waveform_to_stream
from benchmarks.models import LOADERS, HAS_DETECTION_HEAD


THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
PICK_TOL = {"P": 0.5, "S": 1.0}
EVENT_FACTOR = 1.4    # Mousavi 2020 detection-window factor


def parse_true_picks_seisbench(meta) -> dict:
    """Convert SeisBench metadata row → {phase: UTCDateTime}.

    RoSE labels live in `trace_p_arrival_time` / `trace_s_arrival_time`
    as ISO-8601 strings. Some other SeisBench datasets use
    `trace_<P|S>_arrival_sample` instead — both are handled.
    """
    out = {}
    for col, ph in (("trace_p_arrival_time", "P"),
                    ("trace_s_arrival_time", "S")):
        val = meta.get(col)
        if isinstance(val, str) and val.strip() and val != "nan":
            try:
                out[ph] = UTCDateTime(val)
                continue
            except Exception:
                pass
    if out:
        return out
    # fallback: arrival_sample form
    if "trace_start_time" in meta and "trace_sampling_rate_hz" in meta:
        try:
            starttime = UTCDateTime(meta["trace_start_time"])
            sr = float(meta["trace_sampling_rate_hz"])
            for samp_col, ph in (("trace_p_arrival_sample", "P"),
                                  ("trace_s_arrival_sample", "S")):
                if samp_col in meta and pd.notna(meta[samp_col]):
                    out[ph] = starttime + float(meta[samp_col]) / sr
        except Exception:
            pass
    return out


def evaluate_one_model_seisbench(model, model_id: str,
                                 dataset, indices, out_path: Path,
                                 sampling_rate: int = 100,
                                 components: str = "ZNE",
                                 bandpass=(1.0, 45.0)) -> dict:
    """SeisBench-format dataset evaluation, returning the per-threshold summary."""
    logger = logging.getLogger("benchmark")
    has_det = HAS_DETECTION_HEAD[model_id]

    base = min(THRESHOLDS)
    classify_kw = {"P_threshold": base, "S_threshold": base,
                   "detection_threshold": base}

    per_thr = {
        t: {
            "stats": {"P": {"tp": 0, "fp": 0, "fn": 0, "residuals": []},
                      "S": {"tp": 0, "fp": 0, "fn": 0, "residuals": []}},
            "mcc_true": [], "mcc_pred": [], "det_records": [],
            "T1_event_with_pick": 0,
        }
        for t in THRESHOLDS
    }
    n_event = n_noise = 0
    n_failed = 0
    md = dataset.metadata.reset_index(drop=True)
    t0 = time.time()

    for k, idx in enumerate(indices):
        try:
            wf, _ = dataset.get_sample(int(idx))
        except Exception as exc:
            logger.warning("trace %d get_sample failed: %s: %s",
                           int(idx), type(exc).__name__, exc)
            n_failed += 1; continue
        meta = md.iloc[int(idx)]
        true_picks = parse_true_picks_seisbench(meta)
        # Use the metadata's true starttime for the obspy Stream so that
        # absolute UTCDateTime matching against true_picks works.
        try:
            starttime = UTCDateTime(str(meta["trace_start_time"]))
        except Exception:
            starttime = None
        if not true_picks:
            n_noise += 1
            stream = waveform_to_stream(wf, sampling_rate, components,
                                        starttime=starttime,
                                        bandpass=bandpass)
            try:
                out = model.classify(stream, **classify_kw)
            except Exception as exc:
                logger.warning("trace %d classify (noise) failed: %s: %s",
                               int(idx), type(exc).__name__, exc)
                n_failed += 1; continue
            picks_all = list(getattr(out, "picks", []) or [])
            for thr in THRESHOLDS:
                picks_t = [p for p in picks_all
                           if float(getattr(p, "peak_value", 0)) >= thr]
                if picks_t:
                    for p in picks_t:
                        ph = str(getattr(p, "phase", "")).upper()
                        if ph in per_thr[thr]["stats"]:
                            per_thr[thr]["stats"][ph]["fp"] += 1
            continue
        n_event += 1
        stream = waveform_to_stream(wf, sampling_rate, components,
                                    starttime=starttime,
                                    bandpass=bandpass)
        try:
            out = model.classify(stream, **classify_kw)
        except Exception as exc:
            logger.warning("trace %d classify (event) failed: %s: %s",
                           int(idx), type(exc).__name__, exc)
            n_failed += 1; continue
        picks_all = list(getattr(out, "picks", []) or [])
        dets_all = list(getattr(out, "detections", []) or [])

        for thr in THRESHOLDS:
            picks_t = [p for p in picks_all
                       if float(getattr(p, "peak_value", 0)) >= thr]
            dets_t = [d for d in dets_all
                      if float(getattr(d, "peak_value", 0)) >= thr]
            matched, unmatched = match_picks(picks_t, true_picks, PICK_TOL)
            stats = per_thr[thr]["stats"]
            for phase in ("P", "S"):
                if phase in true_picks:
                    if phase in matched:
                        stats[phase]["tp"] += 1
                        stats[phase]["residuals"].append(matched[phase][1])
                    else:
                        stats[phase]["fn"] += 1
            for p in unmatched:
                ph = str(getattr(p, "phase", "")).upper()
                if ph in stats:
                    stats[ph]["fp"] += 1
            for tp, pp in match_picks_cross_phase(picks_t, true_picks, PICK_TOL):
                per_thr[thr]["mcc_true"].append(0 if tp == "P" else 1)
                per_thr[thr]["mcc_pred"].append(0 if pp == "P" else 1)
            if has_det:
                rec = evaluate_trace_detections(dets_t, true_picks, EVENT_FACTOR)
                if rec is not None:
                    per_thr[thr]["det_records"].append(rec)
            if picks_t:
                per_thr[thr]["T1_event_with_pick"] += 1

        if (k + 1) % 200 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(indices) - k - 1)
            logger.info("  %d/%d traces (%.1fs elapsed, ETA %.1fs)",
                        k + 1, len(indices), elapsed, eta)

    # Build summary
    summary = {}
    for thr, s in per_thr.items():
        agg = aggregate_summary(
            s["stats"], s["mcc_true"], s["mcc_pred"],
            n_evaluated=n_event + n_noise, n_failed=n_failed,
            elapsed_s=time.time() - t0, det_records=s["det_records"],
            keep_residuals=True,
        )
        # T1 = trace-level any-pick event-vs-noise
        n_ev_with_pick = s["T1_event_with_pick"]
        n_noise_traces = n_noise
        T1 = {}
        if n_ev_with_pick + n_noise_traces > 0 and n_event > 0:
            n_noise_with_pick = (s["stats"]["P"]["fp"] + s["stats"]["S"]["fp"]
                                 if n_noise else 0)
            # trace-level FP on noise = noise traces with any pick
            # We don't have per-trace count here; use upper bound (count picks).
            # For a proper noise trace count, run with --dataset noise.
            T1 = {"n_event": n_event, "n_noise": n_noise,
                  "f1_pick_based": float("nan"),
                  "definition": "any pick emitted at threshold"}
        agg["T1"] = T1
        summary[str(thr)] = agg

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        json.dump(summary, fh, indent=2, default=str)
    logger.info("wrote %s", out_path)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", required=True, choices=list(LOADERS.keys()))
    ap.add_argument("--dataset", required=True, choices=["rose", "stead", "noise"])
    ap.add_argument("--rose-dir", help="SeisBench-format RoSE dataset directory.")
    ap.add_argument("--stead-dir", help="STEAD test directory with metadata.csv "
                                        "and waveforms/*.npy.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--num-test", type=int, default=0,
                    help="Sample N test traces (0 = use full split).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bandpass-low", type=float, default=1.0)
    ap.add_argument("--bandpass-high", type=float, default=45.0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s :: %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(out_dir / "bench.log")],
                        force=True)
    logger = logging.getLogger("benchmark")
    logger.info("model=%s dataset=%s", args.model, args.dataset)

    bandpass = ((args.bandpass_low, args.bandpass_high)
                if args.bandpass_low > 0 and args.bandpass_high > 0 else None)

    model = LOADERS[args.model]()

    if args.dataset == "rose":
        import seisbench.data as sbd
        if not args.rose_dir:
            raise SystemExit("--rose-dir required for --dataset rose")
        data = sbd.WaveformDataset(path=args.rose_dir, sampling_rate=100,
                                   component_order="ZNE", cache=None)
        test = data.test()
        rng = np.random.default_rng(args.seed)
        if args.num_test and 0 < args.num_test < len(test):
            indices = np.sort(rng.choice(len(test), size=args.num_test,
                                         replace=False))
        else:
            indices = np.arange(len(test))
        evaluate_one_model_seisbench(
            model, args.model, test, indices,
            out_path=out_dir / f"{args.model}.json",
            bandpass=bandpass,
        )
    elif args.dataset in ("stead", "noise"):
        # Use the more elaborate STEAD-format loop: see benchmarks/run_stead.py
        # for the full implementation. Here we delegate to that module.
        from benchmarks.run_stead import run_stead
        run_stead(model, args.model, Path(args.stead_dir),
                  out_dir / f"{args.model}.json",
                  num_events=0 if args.dataset == "noise" else None,
                  num_noise=None,
                  bandpass=bandpass,
                  thresholds=THRESHOLDS, pick_tol=PICK_TOL)
    logger.info("done")


if __name__ == "__main__":
    main()
