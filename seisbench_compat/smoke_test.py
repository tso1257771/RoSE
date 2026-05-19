"""Smoke-test the converted RoSE weights through SeisBench's from_pretrained.

This script:
  1. Stages ``seisbench_compat/weights/{eqtransformer,phasenet}/rose.*.v1``
     into a throwaway SEISBENCH_CACHE_ROOT (so the production ``~/.seisbench``
     is untouched).
  2. Loads ``EQTransformer.from_pretrained("rose")`` and
     ``PhaseNet.from_pretrained("rose")`` off the staged cache.
  3. Re-loads the same checkpoints via :mod:`rose.pickers` (the legacy
     path) and checks that every tensor in the two state_dicts is
     bit-identical.
  4. Runs a forward pass on dummy input to confirm shape contracts.

Run from the RoSE repo root:

    PYTHONPATH=. python seisbench_compat/smoke_test.py
"""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parent
STAGED = SCRIPT_DIR / "weights"


def stage_cache(tmpdir: Path) -> Path:
    cache = tmpdir / "seisbench"
    (cache / "models" / "v3").mkdir(parents=True)
    for model_dir in ("eqtransformer", "phasenet"):
        dst = cache / "models" / "v3" / model_dir
        shutil.copytree(STAGED / model_dir, dst)
    return cache


def compare_state_dicts(
    a: dict[str, torch.Tensor], b: dict[str, torch.Tensor], name: str
) -> None:
    assert set(a) == set(b), f"{name}: key mismatch"
    for k in a:
        if not torch.equal(a[k], b[k]):
            raise AssertionError(f"{name}: tensor mismatch at key {k}")
    print(f"  state_dict equality OK ({len(a)} tensors)")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cache = stage_cache(tmpdir)
        os.environ["SEISBENCH_CACHE_ROOT"] = str(cache)

        # seisbench/__init__.py reads SEISBENCH_CACHE_ROOT at import time, so
        # nothing above this line may import seisbench (directly OR transitively
        # via `rose.pickers`). torch at the module top is fine; it doesn't
        # touch seisbench.
        import seisbench  # noqa: E402
        import seisbench.models as sbm  # noqa: E402

        from rose.pickers import load_eqt_rose, load_phasenet_rose  # noqa: E402

        print(f"SeisBench {seisbench.__version__}, cache={cache}")

        print("\nEQTransformer.from_pretrained('rose')")
        eqt_sb = sbm.EQTransformer.from_pretrained("rose")
        eqt_rose = load_eqt_rose()
        compare_state_dicts(eqt_sb.state_dict(), eqt_rose.state_dict(), "eqt")
        out = eqt_sb(torch.zeros(1, 3, eqt_sb.in_samples))
        print(f"  forward shape OK -> {tuple(o.shape for o in out) if isinstance(out, tuple) else out.shape}")

        print("\nPhaseNet.from_pretrained('rose')")
        pn_sb = sbm.PhaseNet.from_pretrained("rose")
        pn_rose = load_phasenet_rose()
        compare_state_dicts(pn_sb.state_dict(), pn_rose.state_dict(), "phasenet")
        out = pn_sb(torch.zeros(1, 3, pn_sb.in_samples))
        print(f"  forward shape OK -> {tuple(out.shape)}")

        print("\nall checks passed")


if __name__ == "__main__":
    main()
