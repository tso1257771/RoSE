"""RoSE, the Romanian SEismic dataset in SeisBench format, and the published pickers."""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("rose-seismic")
except PackageNotFoundError:  # editable install or unpinned env
    __version__ = "0.1.0+dev"

# Resolved on first use. Loading the dataset and the pickers needs SeisBench,
# ObsPy and h5py, while rose.magnitudes needs only NumPy, pandas and SciPy.
# Importing those eagerly here would make `import rose.magnitudes` fail in an
# environment that has no SeisBench, which is the environment someone checking
# a published magnitude is most likely to be in.
_LAZY = {
    "RoSE": ".dataset",
    "convert_year": ".convert",
    "convert_all": ".convert",
    "load_eqt_rose": ".pickers",
    "load_phasenet_rose": ".pickers",
    "load_redpan_tf60": ".pickers",
    "magnitudes": None,
    "qc": None,
    "splits": None,
}

__all__ = [
    "RoSE",
    "convert_year",
    "convert_all",
    "load_eqt_rose",
    "load_phasenet_rose",
    "load_redpan_tf60",
    "magnitudes",
    "qc",
    "splits",
    "__version__",
]


def __getattr__(name):
    if name not in _LAZY:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    where = _LAZY[name]
    try:
        if where is None:                       # a submodule of its own
            value = import_module(f".{name}", __name__)
        else:
            value = getattr(import_module(where, __name__), name)
    except ImportError as exc:
        raise ImportError(
            f"rose.{name} needs a dependency that is not installed ({exc}). "
            'Install the dataset and picker stack with `pip install "rose-seismic[cpu]"`. '
            "rose.magnitudes does not need it."
        ) from exc
    globals()[name] = value                     # resolve once
    return value


def __dir__():
    return sorted(__all__)
