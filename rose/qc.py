"""Waveform quality control for the RoSE dataset.

Reusable functions for detecting bad waveforms (clipping, gaps, dead
channels, spikes) and computing trace-level SNR. Each function returns a
dict with the detection result and diagnostic values.
"""

from __future__ import annotations

import numpy as np


def detect_clipping(data: np.ndarray, n_consecutive: int = 5,
                    edge_fraction: float = 0.001) -> dict:
    """Detect clipping: sustained flatlines near the min/max of the trace.

    A trace is clipped when the sensor or digitizer saturates, producing a
    flat rail at the positive or negative extreme.

    Parameters
    ----------
    data : 1-D array
    n_consecutive : int
        Minimum number of consecutive identical samples to flag as clipped.
    edge_fraction : float
        Fraction of the data range that defines "near min/max".

    Returns
    -------
    dict with keys:
        clipped : bool
        n_clipped_samples : int
        clip_fraction : float (0..1)
        clip_segments : list of (start, end) index tuples
    """
    if len(data) == 0:
        return {"clipped": False, "n_clipped_samples": 0,
                "clip_fraction": 0.0, "clip_segments": []}
    dmin, dmax = float(np.nanmin(data)), float(np.nanmax(data))
    rng = dmax - dmin
    if rng == 0:
        return {"clipped": True, "n_clipped_samples": len(data),
                "clip_fraction": 1.0, "clip_segments": [(0, len(data))]}

    threshold = edge_fraction * rng
    near_max = data >= (dmax - threshold)
    near_min = data <= (dmin + threshold)
    rail = near_max | near_min

    segments = []
    n_clip = 0
    i = 0
    while i < len(rail):
        if rail[i]:
            j = i
            while j < len(rail) and rail[j]:
                j += 1
            if (j - i) >= n_consecutive:
                segments.append((i, j))
                n_clip += (j - i)
            i = j
        else:
            i += 1

    return {
        "clipped": n_clip > 0,
        "n_clipped_samples": n_clip,
        "clip_fraction": n_clip / len(data),
        "clip_segments": segments,
    }


def detect_dead_channel(data: np.ndarray, std_threshold: float = 1e-10) -> dict:
    """Detect dead/constant channels (zero or near-zero variance)."""
    if len(data) == 0:
        return {"dead": True, "std": 0.0}
    std = float(np.nanstd(data))
    is_dead = std < std_threshold or np.all(data == data[0])
    return {"dead": is_dead, "std": std}


def detect_gaps(data: np.ndarray, min_gap_samples: int = 10,
                zero_threshold: float = 1e-15) -> dict:
    """Detect zero-filled gaps (instrument dropouts or padding artifacts).

    A gap is a run of near-zero samples in the interior of the trace (not
    counting the first/last few samples which may be legitimate).
    """
    if len(data) == 0:
        return {"has_gaps": False, "n_gap_samples": 0,
                "gap_fraction": 0.0, "gap_segments": []}
    is_zero = np.abs(data) < zero_threshold
    segments = []
    n_gap = 0
    margin = min(50, len(data) // 10)
    interior = is_zero.copy()
    interior[:margin] = False
    if margin > 0:
        interior[-margin:] = False

    i = 0
    while i < len(interior):
        if interior[i]:
            j = i
            while j < len(interior) and interior[j]:
                j += 1
            if (j - i) >= min_gap_samples:
                segments.append((i, j))
                n_gap += (j - i)
            i = j
        else:
            i += 1

    return {
        "has_gaps": n_gap > 0,
        "n_gap_samples": n_gap,
        "gap_fraction": n_gap / len(data),
        "gap_segments": segments,
    }


def detect_spikes(data: np.ndarray, mad_threshold: float = 50.0,
                  max_width_samples: int = 3) -> dict:
    """Detect anomalous instrument spikes via MAD + isolation check.

    A sample is a candidate spike if |x - median| > mad_threshold * MAD.
    Candidates are then filtered: only **isolated** transients (width <=
    ``max_width_samples``) are counted as spikes. This avoids flagging the
    earthquake signal itself (which is wide and coherent) while catching
    single-sample or few-sample digitizer glitches.

    Parameters
    ----------
    mad_threshold : float
        Multiplier on the MAD for outlier detection. Default 50 is
        appropriate for broadband seismic data containing earthquake signals.
    max_width_samples : int
        Maximum contiguous run of outlier samples to be considered a spike.
        Longer runs are earthquake energy, not instrument artefacts.
    """
    if len(data) == 0:
        return {"has_spikes": False, "n_spikes": 0, "spike_fraction": 0.0,
                "spike_indices": np.array([], dtype=int), "mad": 0.0}
    median = float(np.nanmedian(data))
    mad = float(np.nanmedian(np.abs(data - median)))
    if mad == 0:
        return {"has_spikes": False, "n_spikes": 0, "spike_fraction": 0.0,
                "spike_indices": np.array([], dtype=int), "mad": 0.0}
    deviation = np.abs(data - median) / mad
    outlier_mask = deviation > mad_threshold

    spike_indices = []
    i = 0
    while i < len(outlier_mask):
        if outlier_mask[i]:
            j = i
            while j < len(outlier_mask) and outlier_mask[j]:
                j += 1
            if (j - i) <= max_width_samples:
                spike_indices.extend(range(i, j))
            i = j
        else:
            i += 1

    indices = np.array(spike_indices, dtype=int)
    return {
        "has_spikes": len(indices) > 0,
        "n_spikes": len(indices),
        "spike_fraction": len(indices) / len(data),
        "spike_indices": indices,
        "mad": mad,
    }


def compute_snr(data: np.ndarray, sr: float, phase_sample: int,
                noise_window_s: float = 5.0,
                signal_window_s: float = 5.0) -> dict:
    """Compute SNR as the ratio of RMS amplitude in post-arrival vs pre-arrival windows."""
    noise_n = int(noise_window_s * sr)
    signal_n = int(signal_window_s * sr)

    noise_start = max(0, phase_sample - noise_n)
    noise_end = phase_sample
    signal_start = phase_sample
    signal_end = min(len(data), phase_sample + signal_n)

    if noise_end <= noise_start or signal_end <= signal_start:
        return {"snr": np.nan, "snr_db": np.nan, "noise_rms": np.nan, "signal_rms": np.nan}

    noise_rms = float(np.sqrt(np.nanmean(data[noise_start:noise_end] ** 2)))
    signal_rms = float(np.sqrt(np.nanmean(data[signal_start:signal_end] ** 2)))

    if noise_rms == 0:
        snr = np.inf
        snr_db = np.inf
    else:
        snr = signal_rms / noise_rms
        snr_db = 20.0 * np.log10(snr)

    return {"snr": snr, "snr_db": snr_db, "noise_rms": noise_rms, "signal_rms": signal_rms}


def quality_report(data: np.ndarray, sr: float,
                   phase_sample: int | None = None) -> dict:
    """Run all QC checks and return a combined report."""
    report = {
        "clipping": detect_clipping(data),
        "dead": detect_dead_channel(data),
        "gaps": detect_gaps(data),
        "spikes": detect_spikes(data),
    }
    if phase_sample is not None and phase_sample > 0:
        report["snr"] = compute_snr(data, sr, phase_sample)

    n_issues = sum([
        report["clipping"]["clipped"],
        report["dead"]["dead"],
        report["gaps"]["has_gaps"],
        report["spikes"]["has_spikes"],
    ])
    report["pass"] = n_issues == 0
    report["n_issues"] = n_issues
    return report
