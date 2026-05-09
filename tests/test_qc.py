"""Tests for `rose.qc` waveform quality-control primitives."""
from __future__ import annotations

import numpy as np
import pytest

from rose import qc


def test_detect_clipping_clean_trace_is_unflagged():
    rng = np.random.default_rng(0)
    data = rng.standard_normal(2000)
    result = qc.detect_clipping(data)
    assert result["clipped"] == False
    assert result["n_clipped_samples"] == 0
    assert result["clip_fraction"] == 0.0
    assert result["clip_segments"] == []


def test_detect_clipping_flags_long_rail():
    data = np.linspace(-1.0, 1.0, 1000)
    # Rail at +1 for 200 samples
    data[300:500] = 1.0
    result = qc.detect_clipping(data, n_consecutive=5)
    assert result["clipped"] == True
    assert result["n_clipped_samples"] >= 200
    # The detected segment must contain the injected rail
    starts = [s for s, _ in result["clip_segments"]]
    ends = [e for _, e in result["clip_segments"]]
    assert any(s <= 300 and e >= 500 for s, e in zip(starts, ends))


def test_detect_clipping_empty_input():
    result = qc.detect_clipping(np.array([]))
    assert result["clipped"] == False
    assert result["n_clipped_samples"] == 0


def test_detect_clipping_constant_trace_is_fully_clipped():
    result = qc.detect_clipping(np.zeros(100))
    assert result["clipped"] == True
    assert result["clip_fraction"] == 1.0


def test_detect_dead_channel_flags_zero_variance():
    assert qc.detect_dead_channel(np.zeros(500))["dead"] == True
    assert qc.detect_dead_channel(np.full(500, 3.14))["dead"] == True


def test_detect_dead_channel_passes_active_signal():
    rng = np.random.default_rng(1)
    data = rng.standard_normal(500)
    result = qc.detect_dead_channel(data)
    assert result["dead"] == False
    assert result["std"] > 0.1


def test_detect_gaps_zero_run_in_interior():
    rng = np.random.default_rng(2)
    data = rng.standard_normal(2000) * 0.5
    data[600:900] = 0.0
    result = qc.detect_gaps(data, min_gap_samples=50)
    assert result["has_gaps"] == True
    assert result["n_gap_samples"] >= 200


def test_detect_gaps_ignores_edge_zeros():
    # Trailing zeros within margin must NOT be flagged
    data = np.concatenate([np.ones(2000), np.zeros(20)])
    result = qc.detect_gaps(data, min_gap_samples=10)
    assert result["has_gaps"] == False


def test_detect_spikes_isolated_transient():
    rng = np.random.default_rng(3)
    data = rng.standard_normal(2000) * 0.5
    # Inject a single-sample spike well above MAD threshold
    data[1000] = 500.0
    result = qc.detect_spikes(data, mad_threshold=10.0, max_width_samples=3)
    assert result["has_spikes"] == True
    assert 1000 in result["spike_indices"]


def test_detect_spikes_ignores_wide_earthquake_signal():
    rng = np.random.default_rng(4)
    data = rng.standard_normal(2000) * 0.5
    # Wide coherent burst (earthquake-like): exceeds threshold but is wide
    data[800:1000] = 50.0
    result = qc.detect_spikes(data, mad_threshold=10.0, max_width_samples=3)
    assert result["has_spikes"] == False


def test_compute_snr_known_ratio():
    sr = 100.0
    n_pre = int(5 * sr)
    n_post = int(5 * sr)
    rng = np.random.default_rng(5)
    pre = rng.standard_normal(n_pre) * 1.0    # unit RMS noise
    post = rng.standard_normal(n_post) * 10.0  # ~10x RMS signal
    data = np.concatenate([pre, post])
    result = qc.compute_snr(data, sr=sr, phase_sample=n_pre)
    assert result["snr"] == pytest.approx(10.0, rel=0.3)
    assert result["snr_db"] == pytest.approx(20.0, abs=2.0)


def test_compute_snr_returns_nan_when_window_invalid():
    result = qc.compute_snr(np.zeros(10), sr=100.0, phase_sample=0)
    assert np.isnan(result["snr"])


def test_quality_report_pass_on_clean_trace():
    rng = np.random.default_rng(6)
    sr = 100.0
    data = rng.standard_normal(int(60 * sr)) * 0.1
    report = qc.quality_report(data, sr=sr, phase_sample=int(30 * sr))
    assert report["pass"] == True
    assert report["n_issues"] == 0
    assert "snr" in report


def test_quality_report_flags_dead_channel():
    report = qc.quality_report(np.zeros(1000), sr=100.0)
    assert report["pass"] == False
    assert report["dead"]["dead"] == True
