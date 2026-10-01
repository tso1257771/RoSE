"""
SourceSpec 1.8 entry point with RED-PAN-Motion S-window lengths.

SourceSpec sets one S-window length per run (win_length, optionally scaled by
travel time). This wrapper replaces
`sourcespec.ssp_process_traces._define_signal_and_noise_windows` so that each
trace gets the window rule of RED-PAN-Motion's amplitude extraction
(RED-PAN-Motion: redpan_motion/picks.py and amplitudes.py. The same code is
copied unchanged into rose/magnitudes/redpan_motion/.):

  1. nominal S window, from the S-P time of that trace:
         S_end = min(S + POST_S_FACTOR * (S-P),  P + MAX_AMP_WIN_SEC)
  2. amplitude-decay truncation (`adaptive_s_window`, vendored verbatim below):
     the envelope after S is scanned and the window is cut before a later
     arrival / noise spike that rises above the decayed S coda. As in
     RED-PAN-Motion it runs once per station on the 3-component array of one
     instrument (HH > BH > EH > SH > HN), so all components and co-located
     instruments share one window.
  3. Mw-specific floor: the window is never shorter than SSP_WIN_FLOOR seconds
     (set per event by run_sourcespec.py from the expected corner
     frequency), so the low-frequency plateau of large events stays resolved.
     The floor is still capped by the trace end.

The noise window ends at P - signal_pre_time and has the signal length,
truncated at the trace start. Two SourceSpec 1.8 behaviours are patched
because pre-P data is often shorter (~29 s) than long S windows:
  * _cut_signal_noise: with weighting = noise SourceSpec would cut the SIGNAL
    to the noise length; instead the noise is zero-padded and scaled by
    sqrt(n_signal / n_noise).
  * _check_sn_ratio: S/N is corrected to RMS per sample.

Traces with hypocentral distance > SSP_HYPO_DIST_MAX km (env, optional) are
skipped (station residuals of slab events drop by ~0.2 beyond 300 km).

Usage: identical to `source_spec` (all CLI args are passed through), e.g.
    SSP_WIN_FLOOR=12.5 python source_spec_redpan_windows.py -c cfg -q ev.xml ...
"""
import logging
import os
import sys

import numpy as np
from scipy.ndimage import uniform_filter1d

import sourcespec.ssp_build_spectra as sbs
import sourcespec.ssp_process_traces as spt

# RED-PAN-Motion constants (redpan_motion/picks.py: extract_picks defaults)
POST_S_FACTOR = 1.5
MAX_AMP_WIN_SEC = 80.0

logger = logging.getLogger("sourcespec.ssp_process_traces")


def adaptive_s_window(wf_np, s_idx, nominal_npts, p_idx, dt,
                      smooth_sec=1.0, min_coda_sec=2.0,
                      min_coda_factor=0.3, spike_ratio=3.0,
                      secondary_ratio=2.0):
    """Adaptively shorten the S-coda amplitude window to avoid noise spikes.

    Vendored verbatim from redpan_motion/amplitudes.py (RED-PAN-Motion).
    """
    total_npts = wf_np.shape[1]
    end_abs = min(s_idx + nominal_npts, total_npts)
    actual_nominal = end_abs - s_idx
    if actual_nominal < 2:
        return max(1, actual_nominal)

    sp_diff = s_idx - p_idx
    earliest = min(
        max(int(min_coda_sec / dt), int(min_coda_factor * sp_diff)),
        actual_nominal,
    )

    p_start = max(0, p_idx)
    variances = [np.var(wf_np[c, s_idx:end_abs]) for c in range(wf_np.shape[0])]
    best_c = int(np.argmax(variances))
    if variances[best_c] == 0:
        return actual_nominal

    seg = wf_np[best_c, p_start:end_abs].astype(np.float64)
    smooth_npts = max(3, int(smooth_sec / dt))
    envelope = uniform_filter1d(np.abs(seg), size=smooth_npts)

    s_rel = s_idx - p_start
    coda_end_rel = s_rel + min(int(10.0 / dt), sp_diff)
    coda_end_rel = min(coda_end_rel, len(envelope))

    main_peak = np.max(envelope[:coda_end_rel]) if coda_end_rel > 0 else 1e-30

    global_peak_rel = int(np.argmax(envelope))
    global_peak = envelope[global_peak_rel]
    if (global_peak_rel > coda_end_rel
            and global_peak > main_peak * secondary_ratio):
        valley_region = envelope[coda_end_rel:global_peak_rel]
        if len(valley_region) > 0:
            valley_rel = coda_end_rel + int(np.argmin(valley_region))
            trunc = max(valley_rel - s_rel, earliest)
            return min(trunc, actual_nominal)

    scan_start = s_rel + earliest
    scan_end = min(s_rel + actual_nominal, len(envelope))
    if scan_end > scan_start + 1:
        region = envelope[scan_start:scan_end]
        running_min = np.minimum.accumulate(region)
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio = region / (running_min + 1e-30)
        below_half = running_min < main_peak * 0.5
        spike_mask = (ratio > spike_ratio) & below_half
        if np.any(spike_mask):
            first_spike = int(np.argmax(spike_mask))
            min_before = int(np.argmin(region[:first_spike + 1]))
            trunc = max(scan_start + min_before - s_rel, earliest)
            return min(trunc, actual_nominal)

    return actual_nominal


_original_define_windows = spt._define_signal_and_noise_windows
_original_process_traces = spt.process_traces
_original_cut_signal_noise = sbs._cut_signal_noise
_original_check_sn_ratio = spt._check_sn_ratio

# Band/instrument preference for the one per-station window
# (raw counts of different instruments are not comparable, so the window is
# computed on the 3 components of a single instrument, as in RED-PAN-Motion)
_INSTRUMENT_PRIORITY = ("HH", "BH", "EH", "SH", "HN", "EN", "HL")
_state = {"stream": None, "windows": {}}


def _process_traces(config, st):
    """Keep a handle on the raw stream so windows can be computed per station."""
    _state["stream"] = st
    _state["windows"] = {}
    return _original_process_traces(config, st)


def _station_adaptive_window(trace, p_time, s_time, nominal_sec):
    """RED-PAN adaptive S window (s), computed once per net.sta.loc on the
    3-component array of the preferred instrument (max-variance channel)."""
    st_all = _state["stream"]
    net, sta, loc, _ = trace.id.split(".")
    key = (net, sta, loc)
    if key in _state["windows"]:
        return _state["windows"][key]
    sel = st_all.select(network=net, station=sta, location=loc) if st_all else None
    comps = []
    if sel:
        bands = {tr.stats.channel[:2] for tr in sel}
        band = next((b for b in _INSTRUMENT_PRIORITY if b in bands), sorted(bands)[0])
        comps = [tr for tr in sel if tr.stats.channel[:2] == band]
    if not comps:
        comps = [trace]
    dt = comps[0].stats.delta
    t0 = max(tr.stats.starttime for tr in comps)
    t1 = min(tr.stats.endtime for tr in comps)
    npts = int(round((t1 - t0) / dt)) + 1
    arr = []
    for tr in comps:
        if abs(tr.stats.delta - dt) > 1e-9:
            continue
        d = tr.copy().detrend("demean").detrend("linear")
        i0 = int(round((t0 - d.stats.starttime) / dt))
        seg = d.data[i0:i0 + npts]
        if len(seg) == npts:
            arr.append(seg)
    p_idx = int(round((p_time - t0) / dt))
    s_idx = int(round((s_time - t0) / dt))
    adaptive_sec = nominal_sec
    if arr and 0 <= p_idx < s_idx < npts:
        n = adaptive_s_window(np.vstack(arr), s_idx,
                              int(round(nominal_sec / dt)), p_idx, dt)
        adaptive_sec = n * dt
    _state["windows"][key] = adaptive_sec
    return adaptive_sec


def _redpan_windows(config, trace):
    hypo_max = os.environ.get("SSP_HYPO_DIST_MAX")
    if hypo_max and trace.stats.hypo_dist > float(hypo_max):
        raise RuntimeError(
            f'{trace.id}: hypocentral distance ({trace.stats.hypo_dist:.1f} km) '
            f'> {float(hypo_max):g} km: skipping trace')
    # SourceSpec's own checks + P window + S1 (incl. its S-P/2 pre-time rule)
    _original_define_windows(config, trace)
    if config.wave_type[0] != 'S':
        return
    st = trace.stats
    p_time = st.arrivals['P'][1]
    s_time = st.arrivals['S'][1]
    s1 = st.arrivals['S1'][1]
    sp = s_time - p_time
    floor = float(os.environ.get("SSP_WIN_FLOOR", config.win_length))

    # 1. nominal window (RED-PAN-Motion rule), measured from the S pick
    s_end = min(s_time + POST_S_FACTOR * sp, p_time + MAX_AMP_WIN_SEC)
    nominal_sec = max(st.delta, s_end - s_time)

    # 2. amplitude-decay truncation, one window per station
    adaptive_sec = _station_adaptive_window(trace, p_time, s_time, nominal_sec)

    # 3. Mw floor, then trace end; window starts at S1 (S - signal_pre_time)
    win = max(adaptive_sec + (s_time - s1), floor)
    s2 = min(st.endtime, s1 + win)
    st.arrivals['S2'] = ('S2', s2)
    win = s2 - s1

    # noise window: up to the same length, ending at P - signal_pre_time.
    # If pre-P data is shorter, _cut_signal_noise below pads and rescales it.
    n2 = p_time - config.signal_pre_time
    n1 = max(st.starttime, n2 - win)
    st.arrivals['N1'] = ('N1', n1)
    st.arrivals['N2'] = ('N2', n2)
    floor_used = adaptive_sec + (s_time - s1) < floor
    logger.info(
        f'{trace.id}: RED-PAN window S-P={sp:.2f}s nominal={nominal_sec:.2f}s '
        f'adaptive={adaptive_sec:.2f}s floor={floor:.2f}s -> S win={win:.2f}s, '
        f'noise win={n2 - n1:.2f}s floor_used={floor_used}')


def _cut_signal_noise(config, trace):
    """Never truncate the signal to a short noise window.

    SourceSpec 1.8 cuts the signal window to the noise length when
    weighting = noise. Instead, zero-pad the noise (SourceSpec's own branch
    for other weightings) and scale it by sqrt(n_signal / n_noise), so that
    the padded noise has the spectral level of stationary noise over the
    full signal length.
    """
    n_noise = int(round(
        (trace.stats.arrivals['N2'][1] - trace.stats.arrivals['N1'][1])
        / trace.stats.delta)) + 1
    weighting = config.weighting
    config.weighting = 'frequency'   # selects the zero-padding branch only
    try:
        trace_signal, trace_noise = _original_cut_signal_noise(config, trace)
    finally:
        config.weighting = weighting
    n_sig = len(trace_signal.data)
    if 0 < n_noise < n_sig:
        trace_noise.data = trace_noise.data * np.sqrt(n_sig / n_noise)
    return trace_signal, trace_noise


def _check_sn_ratio(config, trace):
    """Time-domain S/N from RMS per sample (SourceSpec sums squares over
    windows of unequal length, inflating S/N by sqrt(T_signal/T_noise))."""
    st = trace.stats
    t_sig = st.arrivals['S2'][1] - st.arrivals['S1'][1]
    t_noise = st.arrivals['N2'][1] - st.arrivals['N1'][1]
    _original_check_sn_ratio(config, trace)
    if t_noise <= 0 or t_sig <= t_noise or not hasattr(st, 'sn_ratio'):
        return
    corrected = st.sn_ratio * np.sqrt(t_noise / t_sig)
    logger.info(f'{st.info}: S/N per-sample corrected: {corrected:.1f}')
    st.sn_ratio = corrected
    if not st.ignore and corrected < config.sn_min:
        logger.warning(f'{st.info}: S/N smaller than {config.sn_min:g}: skipping trace')
        st.ignore = True
        st.ignore_reason = 'low S/N'


spt._define_signal_and_noise_windows = _redpan_windows
spt.process_traces = _process_traces
spt._check_sn_ratio = _check_sn_ratio
sbs._cut_signal_noise = _cut_signal_noise


if __name__ == "__main__":
    from sourcespec.source_spec import main
    sys.argv[0] = "source_spec"
    sys.exit(main())
