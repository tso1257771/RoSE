#!/usr/bin/env python3
"""Stage (a): run every model on the test traces -> per-model results under $eval_dir.

This is the inference half of the benchmark. It runs the four stage scripts
(`bench_pickers_rose.py`, `bench_redpan_rose.py`, `bench_noise_fp.py`,
`bench_stead_test.py`) on the RoSE and STEAD test sets, writing per-model tally
files under `$eval_dir/`. Then run `benchmark/build_leaderboard.py` to turn
those into the leaderboard CSVs.

    python benchmark/run_inference.py                       # uses benchmark/config.json
    python benchmark/run_inference.py --num-test 200        # quick subset
    python benchmark/run_inference.py --rose-dir /path --stead-dir /path
    python benchmark/run_inference.py --skip-stead          # RoSE leg only

Needs the RoSE dataset ($ROSE_DATA_DIR or config.rose_dir) and the STEAD test
set ($STEAD_DIR or config.stead_dir — required even for the RoSE leg, because
RoSE false positives are counted on dedicated STEAD-noise traces). Run with a
Python env that has seisbench + torch + tensorflow.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline import add_common_args, ckpt_paths, load_config, py, resolve, run, thresholds_str  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--num-test", type=int, default=None,
                    help="Override config.num_test (0 = full pools; N = subset).")
    ap.add_argument("--skip-rose", action="store_true")
    ap.add_argument("--skip-stead", action="store_true")
    args = ap.parse_args()

    cfg = resolve(load_config(args.config), args)
    if args.num_test is not None:
        cfg["num_test"] = args.num_test
    n = int(cfg["num_test"])
    thr = thresholds_str(cfg)
    eval_dir = Path(cfg["eval_dir"])
    ck = ckpt_paths(cfg)
    bp_lo, bp_hi = cfg["bandpass"]

    need_rose = not args.skip_rose
    # STEAD is needed for the RoSE leg too (the noise-FP pass = the RoSE FP source)
    need_stead = (not args.skip_stead) or need_rose
    if need_rose and not cfg.get("rose_dir"):
        ap.error("rose_dir not set (config.rose_dir, $ROSE_DATA_DIR, or --rose-dir)")
    if need_stead and not cfg.get("stead_dir"):
        ap.error("stead_dir not set (config.stead_dir, $STEAD_DIR, or --stead-dir)")
    if args.skip_rose and args.skip_stead:
        ap.error("nothing to do — both --skip-rose and --skip-stead given")
    for p in ck.values():
        if not Path(p).is_file():
            ap.error(f"missing checkpoint: {p}")
    eval_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== run_inference.py ===  rose={cfg['rose_dir']}  stead={cfg['stead_dir']}")
    print(f"    eval_dir={eval_dir}  num_test={n} (0=full)  thresholds={thr}")
    print(f"    python={sys.executable}\n")

    if not args.skip_rose:
        print(">>> [1/4] bench_pickers_rose.py — 8 PyTorch pickers on RoSE")
        run(py("bench_pickers_rose.py") + [
            "--rose-dir", cfg["rose_dir"], "--out-dir", str(eval_dir / "bench_rose_full_sweep"),
            "--num-test", str(n), "--sweep-thresholds", thr,
            "--bandpass-low", str(bp_lo), "--bandpass-high", str(bp_hi),
            "--custom-eqt", ck["eqt_rose"], "--custom-phasenet", ck["phasenet_rose"],
        ])
        print(">>> [2/4] bench_redpan_rose.py — RED-PAN-60s on RoSE")
        run(py("bench_redpan_rose.py") + [
            "--model-path", ck["redpan_tf60"], "--rose-dir", cfg["rose_dir"],
            "--out-dir", str(eval_dir / "bench_redpan_rose_full"),
            "--num-test", str(n), "--sweep-thresholds", thr,
            "--bandpass-low", str(bp_lo), "--bandpass-high", str(bp_hi),
        ])
        print(">>> [3/4] bench_noise_fp.py — all 9 models on STEAD noise (RoSE FP source)")
        run(py("bench_noise_fp.py") + [
            "--stead-dir", cfg["stead_dir"], "--out-dir", str(eval_dir / "bench_noise_fp"),
            "--num-noise", str(n), "--sweep-thresholds", thr,
            "--bandpass-low", str(bp_lo), "--bandpass-high", str(bp_hi),
            "--eqt-rose-ckpt", ck["eqt_rose"], "--phasenet-rose-ckpt", ck["phasenet_rose"],
            "--redpan-tf", ck["redpan_tf60"],
        ])

    if not args.skip_stead:
        print(">>> [4/4] bench_stead_test.py — 9 models on STEAD (events + noise)")
        for mid in cfg["pytorch_model_ids"]:
            print(f"      ... {mid}")
            run(py("bench_stead_test.py") + [
                "--stead-dir", cfg["stead_dir"],
                "--out-dir", str(eval_dir / "bench_stead_full_pytorch" / mid),
                "--num-events", str(n), "--num-noise", str(n), "--sweep-thresholds", thr,
                "--bandpass-low", str(bp_lo), "--bandpass-high", str(bp_hi),
                "--models", mid,
                "--eqt-rose-ckpt", ck["eqt_rose"], "--phasenet-rose-ckpt", ck["phasenet_rose"],
            ])
        print("      ... redpan")
        run(py("bench_stead_test.py") + [
            "--stead-dir", cfg["stead_dir"], "--out-dir", str(eval_dir / "bench_stead_full"),
            "--num-events", str(n), "--num-noise", str(n), "--sweep-thresholds", thr,
            "--bandpass-low", str(bp_lo), "--bandpass-high", str(bp_hi),
            "--models", "redpan", "--redpan-tf", ck["redpan_tf60"],
        ])
        # build_stead_full_benchmark.py expects redpan_merged.json (the name is
        # historical — a single full run is equivalent to the old 4-shard merge).
        src = eval_dir / "bench_stead_full" / "redpan.json"
        if src.is_file():
            (eval_dir / "bench_stead_full" / "redpan_merged.json").write_bytes(src.read_bytes())

    print(f"\nDone. Per-model results under {eval_dir}/ — now run:")
    print("    python benchmark/build_leaderboard.py --update-results")


if __name__ == "__main__":
    main()
