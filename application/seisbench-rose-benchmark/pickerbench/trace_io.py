"""Convert a (3, n_samples) numpy waveform into an obspy Stream.

SeisBench classify() expects channels in ZNE order. RED-PAN-60s
expects ENZ order (TaiwanCWB convention). The ``components`` argument
controls which order is assumed.
"""
from __future__ import annotations

import numpy as np
from obspy import Stream, Trace, UTCDateTime


def waveform_to_stream(waveform: np.ndarray,
                       sampling_rate: int = 100,
                       components: str = "ZNE",
                       starttime: UTCDateTime | None = None,
                       station: str = "XX",
                       network: str = "XX",
                       bandpass: tuple[float, float] | None = (1.0, 45.0),
                       ) -> Stream:
    """Wrap an ``(n_components, n_samples)`` numpy array as an obspy Stream.

    If `bandpass` is given, applies a 4-pole zero-phase Butterworth band
    between (low, high) Hz before returning. Mean removal and tapering
    are applied first to reduce edge artefacts. Set ``bandpass=None``
    to skip filtering (e.g. when the input is pre-filtered).
    """
    if waveform.ndim != 2 or waveform.shape[0] != 3:
        raise ValueError(f"expected (3, n_samples), got {waveform.shape}")
    if starttime is None:
        starttime = UTCDateTime("2000-01-01T00:00:00")
    components = components.upper()
    if set(components) != set("ZNE"):
        raise ValueError(
            f"components must be a permutation of 'ZNE', got {components!r}"
        )
    traces = []
    for i, ch in enumerate(components):
        tr = Trace(data=np.asarray(waveform[i], dtype=np.float32))
        tr.stats.sampling_rate = sampling_rate
        tr.stats.station = station
        tr.stats.network = network
        tr.stats.channel = "HH" + ch
        tr.stats.starttime = starttime
        traces.append(tr)
    st = Stream(traces=traces)
    st.detrend("demean")
    st.taper(0.05, type="cosine")
    if bandpass is not None and bandpass[0] > 0 and bandpass[1] > 0:
        st.filter("bandpass", freqmin=bandpass[0], freqmax=bandpass[1],
                  corners=4, zerophase=True)
    return st
