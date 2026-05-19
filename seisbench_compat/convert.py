"""Convert RoSE training checkpoints to SeisBench pretrained-weight format.

RoSE saves ``{config, dev_loss, epoch, model}`` dicts under
``phase_picking/models/<name>/<name>.pt``. SeisBench's
:py:meth:`seisbench.models.SeisBenchModel.from_pretrained` expects a pair:

    <model_lowercase>/<weight_name>.pt.v<version>   # raw state_dict
    <model_lowercase>/<weight_name>.json.v<version> # metadata

This script writes that pair for both EQTransformer-RoSE and PhaseNet-RoSE
into ``seisbench_compat/weights/``, ready to be uploaded to the SeisBench
weight repository (the maintainers host the WebDAV server; see this
directory's ``README.md``).

Run from the RoSE repo root:

    PYTHONPATH=. python seisbench_compat/convert.py --version 1
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from rose.checkpoint_io import safe_torch_load


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_MODELS_DIR = REPO_ROOT / "phase_picking" / "models"
DEFAULT_OUT_DIR = SCRIPT_DIR / "weights"

# Floor for `seisbench_requirement` in the emitted JSON. Matches the only
# SeisBench version this converter has been smoke-tested against.
SEISBENCH_FLOOR = "0.11.0"

# Defaults chosen on the RoSE *test pool* over the coarse grid
# {0.05, 0.1, 0.2, 0.3, 0.5, 0.7} — see phase_picking/results/rose_picking.csv
# and rose_detection.csv. These are operating-point starting points, not
# claims of statistical optimality; users on different distributions should
# re-tune.
EQT_DEFAULT_THRESHOLDS = {
    "P_threshold": 0.1,
    "S_threshold": 0.1,
    "detection_threshold": 0.7,
}
PHASENET_DEFAULT_THRESHOLDS = {
    "P_threshold": 0.2,
    "S_threshold": 0.2,
    "blinding": [200, 200],
}

ROSE_CITATION = (
    "Tso, R., et al. (2026). RoSE: a regionally optimized Taiwan benchmark "
    "and pretrained pickers for seismic phase detection. Zenodo. "
    "https://doi.org/10.5281/zenodo.20250670"
)

EQT_DOCSTRING = (
    "EQTransformer fine-tuned on the RoSE (Regional Optimization Set for "
    "Earthquakes, Taiwan) benchmark. Warm-started from the INSTANCE EQT "
    "weights, then trained for 30 epochs (Adam, lr=1e-4, bs=64, sigma=10, "
    "bandpass 1–45 Hz) on the official 80/10/10 RoSE split with 6000-sample "
    "windows at 100 Hz. Loss weights: detection 0.05, P 0.40, S 0.55. "
    "Default thresholds correspond to the best F1 measured on the RoSE test "
    "pool over the coarse grid {0.05, 0.1, 0.2, 0.3, 0.5, 0.7} "
    "(P_threshold=0.1, S_threshold=0.1, detection_threshold=0.7); users on "
    "different distributions should re-tune. For training details and "
    "citation, see " + ROSE_CITATION
)

PHASENET_DOCSTRING = (
    "PhaseNet fine-tuned on the RoSE (Regional Optimization Set for "
    "Earthquakes, Taiwan) benchmark. Warm-started from the INSTANCE PhaseNet "
    "weights, then trained for 30 epochs (Adam, lr=1e-4, bs=256, sigma=10, "
    "no bandpass) on the official 80/10/10 RoSE split with 3001-sample "
    "windows at 100 Hz. Default thresholds correspond to the best F1 "
    "measured on the RoSE test pool over the coarse grid "
    "{0.05, 0.1, 0.2, 0.3, 0.5, 0.7} (P_threshold=0.2, S_threshold=0.2) with "
    "200-sample blinding on each edge to mitigate window-boundary artifacts; "
    "users on different distributions should re-tune. For training details "
    "and citation, see " + ROSE_CITATION
)


def _config(state: dict[str, Any]) -> dict[str, Any]:
    cfg = state.get("config")
    if not cfg:
        raise ValueError("checkpoint is missing the 'config' block")
    return cfg


def _state_dict(state: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "model" not in state:
        raise ValueError("checkpoint is missing the 'model' block")
    return state["model"]


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4, sort_keys=True, ensure_ascii=False)


def convert_eqt(src: Path, dest_dir: Path, version: str) -> None:
    state = safe_torch_load(str(src), map_location="cpu")
    cfg = _config(state)

    metadata = {
        "docstring": EQT_DOCSTRING,
        "model_args": {
            "in_samples": int(cfg["model_window"]),
            "sampling_rate": int(cfg["sampling_rate"]),
            "phases": "PS",
            "norm": "peak",
            "component_order": cfg["component_order"],
        },
        "default_args": EQT_DEFAULT_THRESHOLDS,
        "seisbench_requirement": SEISBENCH_FLOOR,
        "version": version,
    }

    out_dir = dest_dir / "eqtransformer"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(state), out_dir / f"rose.pt.v{version}")
    _write_metadata(out_dir / f"rose.json.v{version}", metadata)


def convert_phasenet(src: Path, dest_dir: Path, version: str) -> None:
    state = safe_torch_load(str(src), map_location="cpu")
    cfg = _config(state)

    metadata = {
        "docstring": PHASENET_DOCSTRING,
        "model_args": {
            "phases": "PSN",
            "norm": "peak",
            "sampling_rate": int(cfg["sampling_rate"]),
            "component_order": cfg["component_order"],
        },
        "default_args": PHASENET_DEFAULT_THRESHOLDS,
        "seisbench_requirement": SEISBENCH_FLOOR,
        "version": version,
    }

    out_dir = dest_dir / "phasenet"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(_state_dict(state), out_dir / f"rose.pt.v{version}")
    _write_metadata(out_dir / f"rose.json.v{version}", metadata)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory (will contain eqtransformer/ and phasenet/ subdirs)",
    )
    parser.add_argument(
        "--version",
        default="1",
        help="weight version string (matches the .v<n> suffix; defaults to 1)",
    )
    parser.add_argument(
        "--eqt-ckpt",
        type=Path,
        default=DEFAULT_MODELS_DIR / "eqt_rose" / "eqt_rose.pt",
    )
    parser.add_argument(
        "--phasenet-ckpt",
        type=Path,
        default=DEFAULT_MODELS_DIR / "phasenet_rose" / "phasenet_rose.pt",
    )
    args = parser.parse_args()

    convert_eqt(args.eqt_ckpt, args.out, args.version)
    convert_phasenet(args.phasenet_ckpt, args.out, args.version)
    print(f"Wrote SeisBench-format weights under {args.out}")


if __name__ == "__main__":
    main()
