"""SeisBench dataset wrapper for RoSE.

Usage:
    from rose import RoSE
    data = RoSE("/path/to/data/rose")          # or os.environ["ROSE_DATA_DIR"]
    sample = data.get_sample(0)

Pre-requisite: the dataset directory must already contain SeisBench-format files
produced by ``rose.convert.convert_all`` (i.e. metadata{YEAR}.csv,
waveforms{YEAR}.hdf5, plus a ``chunks`` index file).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from seisbench.data.base import WaveformDataset

NOISE_PREFIX = "noise"        # noise chunks are noise2014 ... noise2024
DEV_SPLITS = ("dev", "val")   # SeisBench's name for the validation split, and the published one


def select_chunks(path: Union[str, Path], *, include_noise: bool = False) -> list[str]:
    """Chunk names of the dataset in ``path``.

    The names come from ``<path>/chunks`` or, without that file, from the
    ``metadata{chunk}.csv`` and ``waveforms{chunk}.hdf5`` file names, as
    SeisBench finds them. From dataset version 2 on there are also noise
    chunks (``noise2014`` … ``noise2024``, ``trace_category == "noise"``).
    They are left out unless ``include_noise`` is true, so a plain
    ``RoSE(path)`` keeps returning the earthquake traces only. Raises
    ``ValueError`` when that leaves no chunk.
    """
    names = WaveformDataset.available_chunks(path)
    if include_noise:
        return names
    quakes = [name for name in names if not name.startswith(NOISE_PREFIX)]
    if not quakes:
        raise ValueError(
            f"{path} has only noise chunks ({', '.join(names)}); "
            "pass include_noise=True to load them."
        )
    return quakes


class RoSE(WaveformDataset):
    """Romanian SEismic dataset in SeisBench format.

    Parameters
    ----------
    path : str or Path
        Directory holding the converted SeisBench chunks.
    component_order : str, optional
        Override the native ``ZNE`` component order if a model needs a
        different one.
    sampling_rate : float, optional
        Resample on read to this rate (Hz). Defaults to native 100 Hz.
    include_noise : bool, optional
        Also load the noise chunks (``noise{YEAR}``, 100 s windows without
        detected arrivals). Use dataset version 3 or later: the noise
        metadata of version 2 do not match its noise waveforms. Off by
        default, so the earthquake traces alone are returned, as with
        dataset version 1. Select one class afterwards, in place, with
        ``data.filter(data.metadata.trace_category == "noise")``.
    chunks : list of str, optional
        Explicit chunk names, e.g. ``[f"noise{y}" for y in range(2014, 2025)]``
        for the noise alone; overrides ``include_noise``.

    The published metadata call the validation split ``val``, SeisBench calls
    it ``dev``; ``dev()`` and ``train_dev_test()`` take both.
    """

    def __init__(
        self,
        path,
        component_order: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        *,
        include_noise: bool = False,
        chunks: Optional[list[str]] = None,
        **kwargs,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {path}. "
                "Run rose.convert.convert_all first."
            )
        if chunks is None:
            chunks = select_chunks(path, include_noise=include_noise)
        super().__init__(
            path=path,
            name="RoSE",
            component_order=component_order,
            sampling_rate=sampling_rate,
            chunks=chunks,
            **kwargs,
        )

    def dev(self):
        """Validation split: the traces whose ``split`` is ``dev`` or ``val``."""
        if "split" not in self.metadata.columns:
            raise ValueError("Split requested but no split defined in metadata")
        return self.filter(self.metadata["split"].isin(DEV_SPLITS).to_numpy(), inplace=False)

    def get_sample_physical(self, idx: int):
        """Return waveforms in physical units (M/S or M/S**2) for trace `idx`.

        The on-disk waveforms are raw counts. This helper divides each
        component by its instrument sensitivity (read from the metadata
        columns ``trace_sensitivity_{e,n,z}``) so the returned array is in
        whatever unit ``trace_unit_physical`` indicates (typically M/S for
        seismometers and M/S**2 for accelerometers).

        Raises ``ValueError`` if the trace has no valid response
        (``trace_status_physical == "missing_response"``).
        """
        import numpy as np

        wf, meta = self.get_sample(idx)
        status = meta.get("trace_status_physical", "unknown")
        if status == "missing_response":
            raise ValueError(
                f"Trace {meta.get('trace_name')} has no instrument response."
            )
        # ZNE is the on-disk component order.
        order = self.component_order or "ZNE"
        sens = np.array(
            [meta.get(f"trace_sensitivity_{c.lower()}") for c in order],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(sens)) or np.any(sens == 0):
            raise ValueError(
                f"Trace {meta.get('trace_name')} has incomplete sensitivity values."
            )
        physical = wf.astype(np.float64) / sens[:, None]
        return physical.astype(np.float32), meta

    @property
    def citation(self) -> str:
        return (
            "RoSE — Romanian SEismic dataset: a ROMPLUS-enhanced Romanian "
            "earthquake dataset for machine-learning and seismological "
            "applications (2014-2024). Source catalog: ROMPLUS (NIEP), "
            "relocated with hypoDD3D and repicked with RED-PAN."
        )
