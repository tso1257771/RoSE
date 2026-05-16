"""RoSE — Romanian SEismic dataset (SeisBench format) + the published pickers."""

from importlib.metadata import PackageNotFoundError, version as _pkg_version

from .dataset import RoSE
from .convert import convert_year, convert_all
from .pickers import load_eqt_rose, load_phasenet_rose, load_redpan_tf60
from . import qc, splits

try:
    __version__ = _pkg_version("rose-seismic")
except PackageNotFoundError:  # editable install or unpinned env
    __version__ = "0.1.0+dev"

__all__ = [
    "RoSE",
    "convert_year",
    "convert_all",
    "load_eqt_rose",
    "load_phasenet_rose",
    "load_redpan_tf60",
    "qc",
    "splits",
    "__version__",
]
