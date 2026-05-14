"""Shared helpers for run_inference.py / build_leaderboard.py.

Loads benchmark/config.json, applies a small set of command-line overrides, and
resolves paths. The benchmark is a two-stage pipeline:

    benchmark/run_inference.py     (a) run every model on the test traces  -> $eval_dir/...
    benchmark/build_leaderboard.py (b) aggregate those into leaderboards    -> $results_dir/*.csv

Both stages are driven by the same config file; everything else in this
directory (bench_*.py, build_*.py) is the stage implementation those two
scripts call — you normally don't run them directly.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "phase_picking" / "benchmark"
DEFAULT_CONFIG = BENCH_DIR / "config.json"


def load_config(path: Path | str | None = None) -> dict:
    path = Path(path) if path else DEFAULT_CONFIG
    cfg = json.loads(path.read_text())
    # fall back to env vars for the two dataset paths
    if not cfg.get("rose_dir"):
        cfg["rose_dir"] = os.environ.get("ROSE_DATA_DIR")
    if not cfg.get("stead_dir"):
        cfg["stead_dir"] = os.environ.get("STEAD_DIR")
    # absolutise the repo-relative paths
    for key in ("eval_dir", "results_dir", "models_dir"):
        cfg[key] = str((REPO_ROOT / cfg[key]).resolve()) if not os.path.isabs(cfg[key]) else cfg[key]
    return cfg


def thresholds_str(cfg: dict) -> str:
    return ",".join(str(t) for t in cfg["thresholds"])


def ckpt_paths(cfg: dict) -> dict:
    m = Path(cfg["models_dir"])
    return {
        "eqt_rose": str(m / "eqt_rose" / "eqt_rose.pt"),
        "phasenet_rose": str(m / "phasenet_rose" / "phasenet_rose.pt"),
        "redpan_tf60": str(m / "redpan_tf60" / "train.hdf5"),
    }


def run(cmd: list[str], *, env: dict | None = None, label: str | None = None) -> None:
    print(f"    [{label}] $ " + " ".join(cmd) if label else "    $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def _bundled_cudnn_lib_dir() -> Path | None:
    """Locate the pip-installed nvidia-cudnn-cu12 libcudnn dir, if present.

    TF 2.16 was built against cuDNN 8.9; on boxes whose system cuDNN is older
    (e.g. 8.8.1 from /usr/local/cuda-11.8/) TF dies with "No DNN in stream
    executor" on every conv1d. The nvidia-cudnn-cu12==8.9.* pip package
    ships a matching ``libcudnn.so.8``; prepending its dir to LD_LIBRARY_PATH
    makes ld.so find the right one first.
    """
    py_lib = Path(sys.executable).resolve().parent.parent / "lib" \
        / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "nvidia" / "cudnn" / "lib"
    if py_lib.is_dir() and any(py_lib.glob("libcudnn*.so*")):
        return py_lib
    return None


def threadcap_env(threads: int) -> dict:
    """Env copy that caps each scientific-stack library to ``threads`` OS threads.

    Used when running several stage scripts concurrently so N workers × T threads
    doesn't oversubscribe the cores (and the bench_*.py also get ``--tf-threads T``).

    Also prepends the pip-bundled cuDNN dir to ``LD_LIBRARY_PATH`` so TF picks
    the right libcudnn at runtime (see ``_bundled_cudnn_lib_dir``); doing it
    here means the bench subprocesses get it via their environ at exec time,
    before Python or any C extension loads.
    """
    e = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
              "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS"):
        e[k] = str(threads)
    cudnn_dir = _bundled_cudnn_lib_dir()
    if cudnn_dir is not None:
        existing = e.get("LD_LIBRARY_PATH", "")
        if str(cudnn_dir) not in existing.split(":"):
            e["LD_LIBRARY_PATH"] = (
                f"{cudnn_dir}:{existing}" if existing else str(cudnn_dir)
            )
    return e


def run_parallel(tasks: list[tuple[str, list[str], dict | None]], jobs: int) -> None:
    """Run ``(label, cmd, env)`` tasks through a ``jobs``-wide thread pool.

    Each worker just blocks in ``subprocess.run`` — the GIL is irrelevant — so
    the actual stage processes run concurrently. First failure cancels the rest.
    """
    if jobs <= 1 or len(tasks) <= 1:
        for label, cmd, env in tasks:
            run(cmd, env=env, label=label)
        return
    lock = threading.Lock()

    def _go(t):
        label, cmd, env = t
        with lock:
            print(f"  >>> start [{label}]", flush=True)
        run(cmd, env=env, label=label)
        with lock:
            print(f"  <<< done  [{label}]", flush=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = {ex.submit(_go, t): t[0] for t in tasks}
        try:
            for fut in concurrent.futures.as_completed(futs):
                fut.result()  # re-raise on failure
        except Exception:
            for f in futs:
                f.cancel()
            raise


def py(script: str) -> list[str]:
    """`[sys.executable, benchmark/<script>]` — runs the stage script with the same interpreter."""
    return [sys.executable, str(BENCH_DIR / script)]


def add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help=f"Config JSON (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)}).")
    ap.add_argument("--rose-dir", default=None, help="Override config.rose_dir.")
    ap.add_argument("--stead-dir", default=None, help="Override config.stead_dir.")
    ap.add_argument("--eval-dir", default=None, help="Override config.eval_dir.")


def resolve(cfg: dict, args: argparse.Namespace) -> dict:
    """Apply the common CLI overrides onto a loaded config dict (in place)."""
    if getattr(args, "rose_dir", None):
        cfg["rose_dir"] = args.rose_dir
    if getattr(args, "stead_dir", None):
        cfg["stead_dir"] = args.stead_dir
    if getattr(args, "eval_dir", None):
        cfg["eval_dir"] = str((REPO_ROOT / args.eval_dir).resolve()) if not os.path.isabs(args.eval_dir) else args.eval_dir
    return cfg
