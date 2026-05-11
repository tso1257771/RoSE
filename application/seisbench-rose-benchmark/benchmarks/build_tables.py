"""Aggregate per-model JSONs into the publication CSVs.

Examples:
    python benchmarks/build_tables.py rose results/runs results/rose
    python benchmarks/build_tables.py stead results/runs results/stead
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

THIS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(THIS))

from pickerbench.leaderboard import build_picking_csv, build_detection_csv
from benchmarks.models import HAS_DETECTION_HEAD


THRESHOLDS = ["0.05", "0.1", "0.2", "0.3", "0.5", "0.7"]
DEFAULT_NAME_MAP = {
    "eqt_rose_v3":      "EQT-RoSE-v3",
    "phasenet_rose_v2": "PhaseNet-RoSE-v2",
    "redpan_tf60":      "RED-PAN-60s",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("benchmark_name", choices=["rose", "stead"],
                    help="Used in the output filename only.")
    ap.add_argument("runs_dir", type=Path,
                    help="Directory containing <model>/<model>.json files "
                         "produced by run_benchmark.py.")
    ap.add_argument("out_prefix", type=Path,
                    help="Output prefix; appends _picking.csv / _detection.csv.")
    args = ap.parse_args()

    runs = args.runs_dir
    json_paths = {}
    has_det = {}
    for model_id, name in DEFAULT_NAME_MAP.items():
        candidates = [
            runs / model_id / f"{model_id}.json",
            runs / f"{model_id}.json",
        ]
        for p in candidates:
            if p.exists():
                json_paths[name] = p
                has_det[name] = HAS_DETECTION_HEAD.get(model_id, False)
                break
    if not json_paths:
        raise SystemExit(f"no JSON outputs found under {runs}")

    pick_csv = Path(str(args.out_prefix) + "_picking.csv")
    det_csv = Path(str(args.out_prefix) + "_detection.csv")
    build_picking_csv(json_paths, pick_csv, THRESHOLDS)
    build_detection_csv(json_paths, has_det, det_csv, THRESHOLDS)
    print(f"wrote {pick_csv}")
    print(f"wrote {det_csv}")


if __name__ == "__main__":
    main()
