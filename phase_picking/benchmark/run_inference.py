#!/usr/bin/env python3
"""Stage (a): run every model on the test traces -> per-model results under $eval_dir.

This is the inference half of the benchmark. It runs the four stage scripts
(`bench_pickers_rose.py`, `bench_redpan_rose.py`, `bench_noise_fp.py`,
`bench_stead_test.py`) on the RoSE and STEAD test sets, writing per-model tally
files under `$eval_dir/`. Then run `benchmark/build_leaderboard.py` to turn
those into the leaderboard CSVs.

    python benchmark/run_inference.py                       # uses benchmark/config.json
    python benchmark/run_inference.py --num-test 200        # quick subset
    python benchmark/run_inference.py --jobs 3              # run the stage scripts concurrently
    python benchmark/run_inference.py --rose-dir /path --stead-dir /path
    python benchmark/run_inference.py --skip-stead          # RoSE leg only

Needs the RoSE dataset ($ROSE_DATA_DIR or config.rose_dir) and the STEAD test
set ($STEAD_DIR or config.stead_dir — required even for the RoSE leg, because
RoSE false positives are counted on dedicated STEAD-noise traces). Run with a
Python env that has seisbench + torch + tensorflow.

`--jobs N` runs the per-(model, dataset) stage scripts concurrently with N
workers, each capped at ~(cores // N) threads so they don't oversubscribe. The
single RED-PAN-on-STEAD task is the long pole and isn't split, so wall time
bottoms out near its runtime — on an 8-core box, `--jobs 3` (≈3 threads/task) is
usually the best balance; `--jobs 8` (1 thread/task) makes that long task slow.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline import (  # noqa: E402
    add_common_args, ckpt_paths, load_config, py, resolve, run_parallel,
    threadcap_env, thresholds_str,
)

_PT_MODELS_ALL = (
    "eqt_rose phasenet_rose eqt_instance phasenet_instance "
    "eqt_ethz phasenet_ethz eqt_stead phasenet_stead"
).split()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--num-test", type=int, default=None,
                    help="Override config.num_test (0 = full pools; N = subset).")
    ap.add_argument("--jobs", type=int, default=1,
                    help="Run the stage scripts concurrently with N workers (default 1 = "
                         "sequential). Each worker is capped at ~(cores // jobs) threads.")
    ap.add_argument("--skip-rose", action="store_true")
    ap.add_argument("--skip-stead", action="store_true")
    args = ap.parse_args()

    cfg = resolve(load_config(args.config), args)
    if args.num_test is not None:
        cfg["num_test"] = args.num_test
    n = str(int(cfg["num_test"]))
    thr = thresholds_str(cfg)
    eval_dir = Path(cfg["eval_dir"])
    ck = ckpt_paths(cfg)
    bp = ["--bandpass-low", str(cfg["bandpass"][0]), "--bandpass-high", str(cfg["bandpass"][1])]
    pt_models = list(cfg.get("pytorch_model_ids") or _PT_MODELS_ALL)
    jobs = max(1, int(args.jobs))
    tpt = max(1, (os.cpu_count() or jobs) // jobs)   # threads per task
    # threadcap_env both caps OMP/MKL/.../TF intra+inter threads AND prepends
    # the pip-bundled cuDNN dir to LD_LIBRARY_PATH so TF finds a libcudnn
    # matching its build version. Apply unconditionally — when jobs==1 the
    # thread cap is a no-op (= 1 worker × cores threads) but the cuDNN path
    # is still important.
    env = threadcap_env(tpt)

    need_rose = not args.skip_rose
    need_stead = (not args.skip_stead) or need_rose   # the RoSE FP convention needs STEAD noise too
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
    print(f"    jobs={jobs}  threads/task={tpt}  python={sys.executable}\n")

    # ------ build the task list: (label, cmd, env) ------------------------------
    # Slowest tasks first, so a small pool picks them up immediately. The three
    # RED-PAN runs (sliding-window TF inference) dominate; the STEAD one is the
    # long pole. The 8 PyTorch-STEAD runs and rose-pickers are comparatively cheap.
    redpan_tasks: list[tuple[str, list[str], dict | None]] = []
    cheap_tasks: list[tuple[str, list[str], dict | None]] = []
    # The two long-pole TF scripts checkpoint their state to <out_dir>/<model>.partial.json
    # every 500 traces and pick up where they left off on relaunch — pass --resume so a
    # killed/crashed run loses minutes, not hours. The cheaper PyTorch passes (<30 min
    # each on GPU) restart from scratch, so they don't bother.
    if not args.skip_stead:
        redpan_tasks.append((
            "stead redpan (long pole)",
            py("bench_stead_test.py") + [
                "--stead-dir", cfg["stead_dir"], "--out-dir", str(eval_dir / "bench_stead_full"),
                "--num-events", n, "--num-noise", n, "--sweep-thresholds", thr, *bp,
                "--models", "redpan", "--redpan-tf", ck["redpan_tf60"],
                "--resume",
            ], env))
    if not args.skip_rose:
        redpan_tasks.append((
            "rose-redpan",
            py("bench_redpan_rose.py") + [
                "--model-path", ck["redpan_tf60"], "--rose-dir", cfg["rose_dir"],
                "--out-dir", str(eval_dir / "bench_redpan_rose_full"),
                "--num-test", n, "--sweep-thresholds", thr, *bp, "--tf-threads", str(tpt),
                "--resume",
            ], env))
        redpan_tasks.append((
            "noise-fp (9 models)",
            py("bench_noise_fp.py") + [
                "--stead-dir", cfg["stead_dir"], "--out-dir", str(eval_dir / "bench_noise_fp"),
                "--num-noise", n, "--sweep-thresholds", thr, *bp,
                "--eqt-rose-ckpt", ck["eqt_rose"], "--phasenet-rose-ckpt", ck["phasenet_rose"],
                "--redpan-tf", ck["redpan_tf60"], "--tf-threads", str(tpt),
            ], env))
        cheap_tasks.append((
            "rose-pickers (8 PyTorch)",
            py("bench_pickers_rose.py") + [
                "--rose-dir", cfg["rose_dir"], "--out-dir", str(eval_dir / "bench_rose_full_sweep"),
                "--num-test", n, "--sweep-thresholds", thr, *bp,
                "--custom-eqt", ck["eqt_rose"], "--custom-phasenet", ck["phasenet_rose"],
            ], env))
    if not args.skip_stead:
        for mid in pt_models:
            cheap_tasks.append((
                f"stead {mid}",
                py("bench_stead_test.py") + [
                    "--stead-dir", cfg["stead_dir"],
                    "--out-dir", str(eval_dir / "bench_stead_full_pytorch" / mid),
                    "--num-events", n, "--num-noise", n, "--sweep-thresholds", thr, *bp,
                    "--models", mid,
                    "--eqt-rose-ckpt", ck["eqt_rose"], "--phasenet-rose-ckpt", ck["phasenet_rose"],
                ], env))
    tasks = redpan_tasks + cheap_tasks

    # ------ run them --------------------------------------------------------------
    run_parallel(tasks, jobs)

    # build_stead_full_benchmark.py expects redpan_merged.json (the name is
    # historical — a single full run is equivalent to the old 4-shard merge).
    if not args.skip_stead:
        src = eval_dir / "bench_stead_full" / "redpan.json"
        if src.is_file():
            (eval_dir / "bench_stead_full" / "redpan_merged.json").write_bytes(src.read_bytes())

    print(f"\nDone. Per-model results under {eval_dir}/ — now run:")
    print("    python benchmark/build_leaderboard.py --update-results")


if __name__ == "__main__":
    main()
