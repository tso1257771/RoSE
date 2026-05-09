"""Tests for `rose.RoSE.get_sample_physical` sensitivity divide and guards.

We avoid spinning up a real SeisBench dataset (the parent
`WaveformDataset.__init__` reads HDF5 chunks and validates a `chunks`
manifest, both of which are checked in `test_convert.py`). Instead we
exercise `get_sample_physical` directly by binding the unbound method to
a lightweight stand-in object that mimics the few attributes the method
actually touches.
"""
from __future__ import annotations

import numpy as np
import pytest

from rose.dataset import RoSE


class _StubRoSE:
    """Minimal `RoSE`-shaped object for unit testing `get_sample_physical`.

    Only the attributes referenced by `get_sample_physical` are populated:
    `component_order`, plus a `get_sample` method returning (waveform, meta).
    """

    def __init__(self, waveform: np.ndarray, meta: dict, component_order: str = "ZNE"):
        self._wf = waveform
        self._meta = meta
        self.component_order = component_order

    def get_sample(self, idx: int):
        return self._wf, self._meta


def _physical(stub):
    """Invoke the unbound `RoSE.get_sample_physical` on the stub."""
    return RoSE.get_sample_physical(stub, 0)


def test_get_sample_physical_divides_by_sensitivity():
    counts = np.array(
        [
            [1000.0, 2000.0, 3000.0],   # Z
            [4000.0, 5000.0, 6000.0],   # N
            [7000.0, 8000.0, 9000.0],   # E
        ],
        dtype=np.float32,
    )
    meta = {
        "trace_name": "bucket0$0,:3,:3",
        "trace_status_physical": "ok",
        "trace_unit_physical": "M/S",
        "trace_sensitivity_z": 1000.0,
        "trace_sensitivity_n": 2000.0,
        "trace_sensitivity_e": 4000.0,
    }
    stub = _StubRoSE(counts, meta, component_order="ZNE")
    physical, returned_meta = _physical(stub)

    np.testing.assert_allclose(physical[0], counts[0] / 1000.0)
    np.testing.assert_allclose(physical[1], counts[1] / 2000.0)
    np.testing.assert_allclose(physical[2], counts[2] / 4000.0)
    assert physical.dtype == np.float32
    assert returned_meta is meta


def test_get_sample_physical_respects_component_order():
    """If component_order is 'ENZ', sensitivity lookups must follow that order."""
    counts = np.array(
        [
            [10.0, 20.0],   # E (per the order)
            [30.0, 40.0],   # N
            [50.0, 60.0],   # Z
        ],
        dtype=np.float32,
    )
    meta = {
        "trace_status_physical": "ok",
        "trace_unit_physical": "M/S",
        "trace_sensitivity_e": 10.0,
        "trace_sensitivity_n": 30.0,
        "trace_sensitivity_z": 50.0,
    }
    stub = _StubRoSE(counts, meta, component_order="ENZ")
    physical, _ = _physical(stub)
    np.testing.assert_allclose(physical[0], counts[0] / 10.0)   # E
    np.testing.assert_allclose(physical[1], counts[1] / 30.0)   # N
    np.testing.assert_allclose(physical[2], counts[2] / 50.0)   # Z


def test_get_sample_physical_raises_on_missing_response():
    stub = _StubRoSE(
        np.zeros((3, 4), dtype=np.float32),
        {"trace_name": "bucket0$0", "trace_status_physical": "missing_response"},
    )
    with pytest.raises(ValueError, match="no instrument response"):
        _physical(stub)


def test_get_sample_physical_raises_on_zero_sensitivity():
    meta = {
        "trace_name": "bucket0$0",
        "trace_status_physical": "ok",
        "trace_sensitivity_z": 0.0,
        "trace_sensitivity_n": 1.0,
        "trace_sensitivity_e": 1.0,
    }
    stub = _StubRoSE(np.ones((3, 4), dtype=np.float32), meta)
    with pytest.raises(ValueError, match="incomplete sensitivity"):
        _physical(stub)


def test_get_sample_physical_raises_on_nan_sensitivity():
    meta = {
        "trace_name": "bucket0$0",
        "trace_status_physical": "ok",
        "trace_sensitivity_z": float("nan"),
        "trace_sensitivity_n": 1.0,
        "trace_sensitivity_e": 1.0,
    }
    stub = _StubRoSE(np.ones((3, 4), dtype=np.float32), meta)
    with pytest.raises(ValueError, match="incomplete sensitivity"):
        _physical(stub)
