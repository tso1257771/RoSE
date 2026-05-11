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
from typing import Optional

from seisbench.data.base import WaveformDataset


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
    """

    def __init__(
        self,
        path,
        component_order: Optional[str] = None,
        sampling_rate: Optional[float] = None,
        **kwargs,
    ):
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {path}. "
                "Run rose.convert.convert_all first."
            )
        super().__init__(
            path=path,
            name="RoSE",
            component_order=component_order,
            sampling_rate=sampling_rate,
            **kwargs,
        )

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
        order = self.component_order
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
