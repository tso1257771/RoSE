"""Tests for `rose.checkpoint_io.safe_torch_load`.

These pin the security-relevant behaviour the README and the picker release
docs advertise: by default the checkpoint is loaded with the restricted
(`weights_only=True`) unpickler, a checkpoint carrying a non-allowlisted
pickled object is *rejected* rather than executed, and the `trusted=True`
escape hatch falls back to the unrestricted loader (and says so in the log).
"""
from __future__ import annotations

import argparse
import logging
import pickle

import pytest

# `rose.checkpoint_io` imports torch lazily (inside the function), so this
# import is safe without torch; the tests below need torch itself, hence the
# importorskip.
from rose.checkpoint_io import safe_torch_load

torch = pytest.importorskip("torch")


def _save(obj, tmp_path, name="ckpt.pt"):
    path = tmp_path / name
    torch.save(obj, path)
    return path


def test_safe_torch_load_roundtrips_wrapped_checkpoint(tmp_path):
    """The shape the training scripts emit: state_dict + scalars + config dict."""
    state_dict = torch.nn.Linear(3, 2).state_dict()
    path = _save(
        {"model": state_dict, "config": {"lr": 1e-4}, "epoch": 7, "dev_loss": 0.123},
        tmp_path,
    )
    loaded = safe_torch_load(path)
    assert set(loaded) == {"model", "config", "epoch", "dev_loss"}
    assert loaded["epoch"] == 7
    assert loaded["config"] == {"lr": 1e-4}
    assert loaded["model"].keys() == state_dict.keys()
    for k in state_dict:
        assert torch.equal(loaded["model"][k], state_dict[k])


def test_safe_torch_load_roundtrips_bare_state_dict(tmp_path):
    state_dict = torch.nn.Linear(4, 1).state_dict()
    path = _save(state_dict, tmp_path)
    loaded = safe_torch_load(path)
    assert loaded.keys() == state_dict.keys()


def test_safe_torch_load_rejects_non_allowlisted_object(tmp_path):
    """A checkpoint carrying an arbitrary pickled object must NOT load by default."""
    path = _save({"weird": argparse.Namespace(a=1)}, tmp_path)
    with pytest.raises(pickle.UnpicklingError):
        safe_torch_load(path)


def test_safe_torch_load_trusted_loads_arbitrary_object_and_warns(tmp_path, caplog):
    """`trusted=True` falls back to the unrestricted loader (and logs a warning)."""
    path = _save({"weird": argparse.Namespace(a=1, b=2)}, tmp_path)
    with caplog.at_level(logging.WARNING, logger="rose.checkpoint_io"):
        loaded = safe_torch_load(path, trusted=True)
    assert loaded["weird"] == argparse.Namespace(a=1, b=2)
    assert any("trusted=True" in r.message for r in caplog.records)
