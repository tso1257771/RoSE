"""Loaders for the three published RoSE phase pickers.

Identifiers (and the bundled checkpoints under ``phase_picking/models/``):
    eqt_rose      — PyTorch / SeisBench EQTransformer fine-tuned on RoSE
                    (``phase_picking/models/eqt_rose/eqt_rose.pt``)
    phasenet_rose — PyTorch / SeisBench PhaseNet fine-tuned on RoSE
                    (``phase_picking/models/phasenet_rose/phasenet_rose.pt``)
    redpan_tf60   — TensorFlow / Keras RED-PAN-60s (MTAN R2U-Net): retrained
                    outside SeisBench on Taiwan + STEAD + INSTANCE + RoSE,
                    warm-started from the published RED-PAN-60s weights
                    (``phase_picking/models/redpan_tf60/train.hdf5``). See ``phase_picking/models/README.md``
                    for the full recipe.

Each loader returns a SeisBench-style object exposing
``.classify(stream, P_threshold=, S_threshold=[, detection_threshold=])`` → an
object with ``.picks`` (and ``.detections`` for the two models with a detection
head). The obspy ``Stream`` is passed in **ZNE** order (SeisBench convention);
the RED-PAN-60s wrapper reorders to ENZ internally.

PyTorch checkpoints are loaded via :func:`rose.checkpoint_io.safe_torch_load`
(``weights_only=True`` — the restricted unpickler), so loading a third-party
``.pt`` cannot trigger the classic pickle-deserialization RCE.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from obspy import UTCDateTime

from .checkpoint_io import safe_torch_load

# Bundled checkpoints live at <repo>/models/  (rose/pickers.py -> rose/ -> repo)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "phase_picking" / "models"


# --------------------------------------------------------------------- EQT-RoSE
def load_eqt_rose(ckpt: Path | str | None = None, device: str = "cpu"):
    """SeisBench ``EQTransformer`` fine-tuned on RoSE (has a detection head)."""
    import seisbench.models as sbm
    if ckpt is None:
        ckpt = DEFAULT_MODELS_DIR / "eqt_rose" / "eqt_rose.pt"
    state = safe_torch_load(str(ckpt), map_location=device)
    cfg = state.get("config", {})
    model = sbm.EQTransformer(
        in_samples=int(cfg.get("model_window", 6000)),
        sampling_rate=int(cfg.get("sampling_rate", 100)),
        phases=["P", "S"], norm="peak",
    )
    model.load_state_dict(state["model"])
    model.norm = "peak"
    model.to(device).eval()
    return model


# ----------------------------------------------------------------- PhaseNet-RoSE
def load_phasenet_rose(ckpt: Path | str | None = None, device: str = "cpu"):
    """SeisBench ``PhaseNet`` fine-tuned on RoSE (picks only — no detection head)."""
    import seisbench.models as sbm
    if ckpt is None:
        ckpt = DEFAULT_MODELS_DIR / "phasenet_rose" / "phasenet_rose.pt"
    state = safe_torch_load(str(ckpt), map_location=device)
    model = sbm.PhaseNet(
        phases="PSN", norm="peak",
        default_args={"blinding": (200, 200)},
    )
    model.load_state_dict(state["model"])
    model.norm = "peak"
    model.to(device).eval()
    return model


# ------------------------------------------------------------------ RED-PAN-60s
def load_redpan_tf60(weights: Path | str | None = None, batch_size: int = 32):
    """RED-PAN-60s sliding-window inference wrapper (TF/Keras; has a detection head).

    Returns an adapter with a SeisBench-like ``.classify(stream, ...)``
    interface: internally it calls ``REDPAN.predict()`` (raw P/S/mask
    probability arrays), reorders ZNE → ENZ, and post-processes the arrays into
    pick / detection objects via ``scipy.signal.find_peaks`` and
    ``obspy.signal.trigger.trigger_onset``.

    Security: ``tf.keras.models.load_model`` has no ``weights_only=True``
    equivalent — Keras HDF5 files can embed Python objects that execute
    on load. The bundled checkpoint is safe; only point this at HDF5s
    you trust (or construct ``rose.redpan_inference.REDPAN(model=...)``
    yourself after auditing the file's ``model_config``).
    """
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    import tensorflow as tf
    for g in tf.config.list_physical_devices("GPU"):
        try:
            tf.config.experimental.set_memory_growth(g, True)
        except RuntimeError:
            pass
    from .redpan_inference import REDPAN
    if weights is None:
        weights = DEFAULT_MODELS_DIR / "redpan_tf60" / "train.hdf5"
    tf_model = tf.keras.models.load_model(str(weights), compile=False)
    return _RP60Wrapper(REDPAN(model=tf_model, pred_npts=6000, dt=0.01,
                               pred_interval_sec=10.0, batch_size=batch_size,
                               use_compiled_infer=True))


@dataclass
class _Pick:
    phase: str
    peak_time: UTCDateTime
    peak_value: float


@dataclass
class _Detection:
    start_time: UTCDateTime
    end_time: UTCDateTime
    peak_value: float


class _Output:
    def __init__(self, picks, detections):
        self.picks = picks
        self.detections = detections


class _RP60Wrapper:
    """SeisBench-classify-like adapter for RED-PAN-60s.

    ``classify(stream, P_threshold=, S_threshold=, detection_threshold=)`` →
    ``_Output(picks=[...], detections=[...])``. Stream input is in ZNE order
    (SeisBench convention); the wrapper reorders to ENZ before feeding RED-PAN.
    """

    def __init__(self, redpan_obj):
        self.redpan = redpan_obj

    def classify(self, stream, P_threshold=0.3, S_threshold=0.3,
                 detection_threshold=0.3):
        import numpy as np
        from scipy.signal import find_peaks
        from obspy import Stream
        from obspy.signal.trigger import trigger_onset

        traces = {tr.stats.channel[-1]: tr for tr in stream}
        if set(traces) != {"Z", "N", "E"}:
            raise ValueError(
                f"RED-PAN-60s requires Z/N/E channels; got {sorted(traces)}"
            )
        enz = Stream(traces=[traces["E"], traces["N"], traces["Z"]])
        p_arr, s_arr, m_arr = self.redpan.predict(enz, postprocess=False)

        starttime = enz[0].stats.starttime
        dt = enz[0].stats.delta
        sr = int(round(1.0 / dt))

        picks: list[_Pick] = []
        for prob, phase, thr in ((p_arr, "P", P_threshold),
                                 (s_arr, "S", S_threshold)):
            peaks, props = find_peaks(prob, height=thr, distance=sr)
            for idx, height in zip(peaks, props["peak_heights"]):
                picks.append(_Pick(phase=phase,
                                   peak_time=starttime + float(idx) * dt,
                                   peak_value=float(height)))

        detections: list[_Detection] = []
        triggers = trigger_onset(m_arr, detection_threshold, detection_threshold)
        for on_idx, off_idx in triggers:
            seg = m_arr[on_idx: off_idx + 1]
            peak_idx = on_idx + int(np.argmax(seg))
            detections.append(_Detection(
                start_time=starttime + float(on_idx) * dt,
                end_time=starttime + float(off_idx) * dt,
                peak_value=float(m_arr[peak_idx]),
            ))
        return _Output(picks=picks, detections=detections)


LOADERS = {
    "eqt_rose":      load_eqt_rose,
    "phasenet_rose": load_phasenet_rose,
    "redpan_tf60":   load_redpan_tf60,
}

HAS_DETECTION_HEAD = {
    "eqt_rose":      True,
    "phasenet_rose": False,
    "redpan_tf60":   True,
}
