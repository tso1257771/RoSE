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
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = REPO_ROOT / "benchmark"
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


def run(cmd: list[str]) -> None:
    print("    $ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


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
