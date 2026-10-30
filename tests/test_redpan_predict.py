"""Tests for `REDPAN.predict` on records longer than one model window.

Such records are padded with spectrum-matched noise at both ends before the
sliding windows run. These tests pin two guarantees of that path with a
stand-in model, so no checkpoint is needed: repeat calls on the same record
give the same output, and the padding adds no step in amplitude at the end of
the record that the model could take for an onset (a back pad matched to the
start of the record made spurious P picks in the last second of RED-PAN's
predictions). They need TensorFlow, which `rose.redpan_inference.core`
imports; CI installs the package without it and skips them.
"""
from __future__ import annotations

import numpy as np
import pytest
from obspy import Stream, Trace

pytest.importorskip("tensorflow")

from rose.redpan_inference.core import REDPAN  # noqa: E402

SR = 100.0


def _onset_model(x, training=False):
    """Stand-in for the network: P rises with the jump in 1-s power around each sample.

    The output depends smoothly on the input, so different padding noise gives a
    different P curve.
    """
    x = np.asarray(x, dtype=np.float64)
    b, n, _ = x.shape
    k = int(SR)
    power = np.pad((x ** 2).mean(axis=-1), ((0, 0), (1, 0)))
    mean_1s = (np.cumsum(power, axis=1)[:, k:] - np.cumsum(power, axis=1)[:, :-k]) / k
    ratio = np.ones((b, n))
    ratio[:, k:n - k + 1] = (mean_1s[:, k:] + 1e-12) / (mean_1s[:, :n - 2 * k + 1] + 1e-12)
    p = np.clip(np.abs(np.log10(ratio)) / 2.0, 0.0, 1.0)
    return np.stack([p, np.zeros_like(p)], axis=-1).astype(np.float32)


def _stream(data: np.ndarray) -> Stream:
    return Stream([Trace(data=data[:, i].astype(np.float32),
                         header={"sampling_rate": SR, "channel": f"HH{c}"})
                   for i, c in enumerate("ENZ")])


def _record(end: str) -> np.ndarray:
    """120 s of noise with an onset at 30 s; `end` sets what the record ends in."""
    rng = np.random.default_rng(3)
    data = rng.normal(0.0, 1.0, (12000, 3))
    data[3000:] *= 10.0                      # the onset, then loud to the end
    if end == "zeros":                       # like STEAD noise: data, then a zero tail
        data[3000:] = 0.0
        data[:3000] = rng.normal(0.0, 1.0, (3000, 3))
    return data


def _picker(**kwargs) -> REDPAN:
    return REDPAN(model=_onset_model, pred_npts=6000, dt=1.0 / SR,
                  pred_interval_sec=10.0, batch_size=8,
                  use_compiled_infer=False, **kwargs)


def test_predict_long_record_is_repeatable():
    st = _stream(_record("loud"))
    rp = _picker()
    first, second = rp.predict(st), rp.predict(st)
    for a, b in zip(first, second):
        assert np.array_equal(a, b)
    # A new picker with the same seed gives the same output too.
    for a, b in zip(first, _picker().predict(st)):
        assert np.array_equal(a, b)


@pytest.mark.parametrize("end", ["loud", "zeros"])
def test_predict_long_record_has_no_onset_at_the_record_end(end):
    p, s, m = _picker().predict(_stream(_record(end)))
    assert p.shape == s.shape == m.shape == (12000,)
    assert p[2900:3100].max() > 0.5          # the real onset at 30 s is found
    assert p[-300:].max() < 0.2              # nothing in the last 3 s
    assert p[:300].max() < 0.2               # nor in the first 3 s
