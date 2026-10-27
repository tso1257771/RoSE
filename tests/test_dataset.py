"""Tests for `rose.RoSE`: the `get_sample_physical` sensitivity divide and
guards, and which chunks a dataset directory loads.

For `get_sample_physical` we avoid spinning up a real SeisBench dataset (the
parent `WaveformDataset.__init__` reads HDF5 chunks and validates a `chunks`
manifest, both of which are checked in `test_convert.py`). Instead we
exercise it directly by binding the unbound method to a lightweight stand-in
object that mimics the few attributes the method actually touches. The chunk
and split tests at the end write tiny datasets with SeisBench's writer.
"""
from __future__ import annotations

import numpy as np
import pytest
from seisbench.data import WaveformDataWriter

from rose.dataset import RoSE, select_chunks


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


def test_select_chunks_leaves_noise_out_by_default(tmp_path):
    (tmp_path / "chunks").write_text("2014\n2015\nnoise2014\nnoise2015\n")
    assert select_chunks(tmp_path) == ["2014", "2015"]
    assert select_chunks(tmp_path, include_noise=True) == ["2014", "2015", "noise2014", "noise2015"]


def test_select_chunks_reads_file_names_without_a_chunks_file(tmp_path):
    for chunk in ("2014", "noise2014"):
        (tmp_path / f"metadata{chunk}.csv").touch()
        (tmp_path / f"waveforms{chunk}.hdf5").touch()
    assert select_chunks(tmp_path) == ["2014"]
    assert select_chunks(tmp_path, include_noise=True) == ["2014", "noise2014"]


def test_select_chunks_with_noise_only(tmp_path):
    (tmp_path / "chunks").write_text("noise2014\n")
    with pytest.raises(ValueError, match="include_noise=True"):
        select_chunks(tmp_path)
    assert select_chunks(tmp_path, include_noise=True) == ["noise2014"]


def test_select_chunks_in_an_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        select_chunks(tmp_path)


def _write_chunk(path, chunk, category, splits):
    """A chunk with one trace per entry of ``splits``."""
    with WaveformDataWriter(path / f"metadata{chunk}.csv", path / f"waveforms{chunk}.hdf5") as writer:
        writer.data_format = {"dimension_order": "CW", "component_order": "ZNE", "sampling_rate": 100.0}
        for i, split in enumerate(splits):
            writer.add_trace({"trace_name": f"{chunk}_{i}", "trace_category": category, "split": split},
                             np.zeros((3, 100), dtype=np.float32))


def test_rose_loads_noise_chunks_only_when_asked(tmp_path):
    _write_chunk(tmp_path, "2099", "earthquake_local", ["train", "val", "test"])
    _write_chunk(tmp_path, "noise2099", "noise", ["train", "test"])
    (tmp_path / "chunks").write_text("2099\nnoise2099\n")

    assert set(RoSE(tmp_path).metadata["trace_category"]) == {"earthquake_local"}
    both = RoSE(tmp_path, include_noise=True)
    assert both.metadata["trace_category"].value_counts().to_dict() == {"earthquake_local": 3, "noise": 2}
    noise = RoSE(tmp_path, chunks=["noise2099"])
    assert len(noise) == 2 and set(noise.metadata["trace_category"]) == {"noise"}


def test_rose_leaves_noise_out_without_a_chunks_file(tmp_path):
    _write_chunk(tmp_path, "2099", "earthquake_local", ["train", "val", "test"])
    _write_chunk(tmp_path, "noise2099", "noise", ["train", "test"])
    assert set(RoSE(tmp_path).metadata["trace_category"]) == {"earthquake_local"}


def test_rose_dev_takes_the_published_val_split(tmp_path):
    """The published metadata call the validation split ``val``; SeisBench looks for ``dev``."""
    _write_chunk(tmp_path, "2099", "earthquake_local", ["train", "train", "val", "dev", "test"])
    train, dev, test = RoSE(tmp_path).train_dev_test()
    assert (len(train), len(dev), len(test)) == (2, 2, 1)
    assert sorted(dev.metadata["split"]) == ["dev", "val"]
