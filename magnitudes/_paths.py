"""Where the magnitude drivers read and write.

The repository ships the calibration tables and the code that produced them.
It does not ship their inputs: the Wood-Anderson amplitudes, the spectral
fits and the waveform archive are tens of gigabytes. Rerunning a driver
therefore needs a working tree that is assembled separately, and this module
is the one place that says where it is.

Set ``ROMANIA_ROOT`` to that tree::

    export ROMANIA_ROOT=/path/to/romania
    python magnitudes/ml/fit_ml.py

The tree contains ``romania_ml/`` and ``romania_mw/``, each with an ``outputs/``
directory, next to the waveform archive. ``magnitudes/README.md`` lists which
file each driver needs and which Zenodo archive it comes from.

Nothing here is needed to *use* the released magnitudes. They are in the
catalog, and :mod:`rose.magnitudes` reads the calibration tables from the
repository without any of this.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "REPO",
    "CALIBRATION",
    "work_root",
    "ml_root",
    "mw_root",
    "require",
]

#: The checkout this file lives in.
REPO = Path(__file__).resolve().parent.parent

#: The published calibration tables, always present.
CALIBRATION = REPO / "magnitudes" / "calibration"

_ENV = "ROMANIA_ROOT"
_MESSAGE = (
    "{what} is not available.\n"
    "The magnitude drivers read a working tree that this repository does not\n"
    "ship, because its inputs are tens of gigabytes. Point {env} at it:\n"
    "    export {env}=/path/to/romania\n"
    "See magnitudes/README.md for what the tree has to contain and where the\n"
    "files come from. Using the released magnitudes needs none of this: they\n"
    "are in the catalog, and rose.magnitudes reads the calibration tables\n"
    "from this checkout."
)


def work_root() -> Path:
    """The working tree named by ``ROMANIA_ROOT``.

    Raises
    ------
    RuntimeError
        If the variable is unset or does not point at a directory. There is no
        default on purpose: a default would be one author's disk, and a driver
        that silently wrote somewhere else would be worse than one that stops.
    """
    raw = os.environ.get(_ENV)
    if not raw:
        raise RuntimeError(_MESSAGE.format(what=f"${_ENV}", env=_ENV))
    root = Path(raw).expanduser()
    if not root.is_dir():
        raise RuntimeError(_MESSAGE.format(what=f"${_ENV} ({root})", env=_ENV))
    return root.resolve()


def ml_root() -> Path:
    """``romania_ml/`` inside the working tree."""
    return work_root() / "romania_ml"


def mw_root() -> Path:
    """``romania_mw/`` inside the working tree."""
    return work_root() / "romania_mw"


def require(*paths: Path) -> None:
    """Stop before any work if an input is missing, naming every one of them."""
    missing = [p for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            "missing input(s) for this driver:\n  "
            + "\n  ".join(str(p) for p in missing)
            + "\nSee magnitudes/README.md for where each file comes from."
        )
