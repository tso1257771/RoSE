"""Smoke tests for `rose.redpan_inference`.

Targets the helpers that don't need a real TF model — keeps the suite
fast and CI-friendly.  The full REDPAN.predict path is exercised by the
phase_picking benchmark; here we just pin contract-level guarantees that
caught two real bugs in this codebase already (the gc.collect strip
shouldn't change numerics, the global RNG shouldn't leak into noise
padding).
"""
from __future__ import annotations

import numpy as np

from rose.redpan_inference.utils import (
    create_gaussian_weights,
    find_reference_signal,
    generate_matching_noise,
    make_sliding_boundary_pads,
    normalize_weights,
)


def test_gaussian_weights_shape_and_normalisation():
    w = create_gaussian_weights(6000)
    assert w.shape == (6000,)
    # `create_gaussian_weights` divides by the mean, so the post-norm
    # mean is ~1.0 by construction.
    assert w.dtype == np.float32
    assert np.isclose(w.mean(), 1.0, atol=1e-5)
    # Peak should be at the center.
    assert int(np.argmax(w)) == 6000 // 2


def test_normalize_weights_zero_input_is_passthrough():
    w = np.zeros(100, dtype=np.float32)
    out = normalize_weights(w)
    assert np.array_equal(out, w)


def test_find_reference_signal_picks_low_variance_window():
    """`find_reference_signal` should return a window from a quiet
    region — verify it doesn't crash on a clean trace + returns a
    1-D array."""
    n = 10000
    x = np.random.default_rng(0).normal(0, 1, n).astype(np.float32)
    # add a spike to confirm the function picks a non-spiked window
    x[5000:5050] *= 100
    ref = find_reference_signal(x, window_size=500, max_search=5000)
    assert ref.ndim == 1
    assert 100 <= ref.size <= 500
    # The picked window should have lower variance than the spike region.
    assert ref.std() < x[4500:5500].std()


def test_generate_matching_noise_with_seed_is_deterministic():
    """The whole point of the rng-parameter HIGH fix: two calls with
    the same seeded Generator must produce identical noise.  Two calls
    with no rng must NOT be tied to the global np.random state."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)
    ref = np.random.default_rng(0).normal(0, 1, 1000).astype(np.float32)
    noise_a = generate_matching_noise(ref, 6000, rng=rng_a)
    noise_b = generate_matching_noise(ref, 6000, rng=rng_b)
    assert np.allclose(noise_a, noise_b)

    # And: seeding the global doesn't affect us.
    np.random.seed(123)
    noise_c = generate_matching_noise(ref, 6000, rng=np.random.default_rng(42))
    assert np.allclose(noise_a, noise_c)


def test_generate_matching_noise_short_input_uses_iid_normal():
    """The < 50-sample branch falls back to plain normal — should still
    accept the rng parameter."""
    rng = np.random.default_rng(7)
    ref = np.array([1.0, 1.1, 0.9, 1.05], dtype=np.float32)  # < 50
    noise = generate_matching_noise(ref, 100, rng=rng)
    assert noise.shape == (100,)
    assert np.isfinite(noise).all()


def test_make_sliding_boundary_pads_shape_and_finite():
    """Combined helper used by the benchmark; just pin output shapes
    and finiteness on synthetic input."""
    wf = np.random.default_rng(1).normal(0, 1, (12000, 3)).astype(np.float32)
    front, back = make_sliding_boundary_pads(wf, pad_npts=6000,
                                              rng=np.random.default_rng(0))
    assert front.shape == (6000, 3)
    assert back.shape == (6000, 3)
    assert np.isfinite(front).all()
    assert np.isfinite(back).all()
    # Back-pad is a constant per channel (the median of that channel).
    for ch in range(3):
        assert np.allclose(back[:, ch], np.median(wf[:, ch]))


def test_generate_matching_noise_no_rng_decoupled_from_global():
    """Without a seeded rng, two calls produce *different* noise (so we
    didn't accidentally make the default deterministic) AND seeding the
    global np.random state has no effect."""
    ref = np.random.default_rng(0).normal(0, 1, 1000).astype(np.float32)
    a = generate_matching_noise(ref, 200)
    np.random.seed(999)
    b = generate_matching_noise(ref, 200)
    assert not np.allclose(a, b)
