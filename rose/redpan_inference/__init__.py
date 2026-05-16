"""Minimal RED-PAN inference subset (vendored).

Self-contained subset of the upstream `redpan` package
(https://github.com/tso1257771/RED-PAN) sufficient to run RED-PAN-60s
inference on a single trace via the `REDPAN` class. The training,
data-loading, and TF-Keras model-construction modules are not vendored
because the published `phase_picking/models/redpan_tf60/train.hdf5` is the inference target.
Used by `rose.pickers.load_redpan_tf60` and the `phase_picking/benchmark/` scripts.

Usage:
    import tensorflow as tf
    from rose.redpan_inference import REDPAN
    tf_model = tf.keras.models.load_model("phase_picking/models/redpan_tf60/train.hdf5",
                                          compile=False)
    rp = REDPAN(model=tf_model, pred_npts=6000, dt=0.01,
                pred_interval_sec=10.0, batch_size=32,
                use_compiled_infer=True)
    # predict() returns three per-sample probability arrays — convert to
    # discrete picks/detections downstream (e.g. scipy.signal.find_peaks
    # on P_arr/S_arr, obspy.signal.trigger.trigger_onset on M_arr).
    P_arr, S_arr, M_arr = rp.predict(stream)
"""
# PEP 562 lazy-load: importing utils/picks submodules doesn't trigger
# the TensorFlow import that core.py needs.
__all__ = ["REDPAN"]
__version__ = "1.0.0"


def __getattr__(name):
    if name == "REDPAN":
        from .core import REDPAN  # raises ModuleNotFoundError if .[tf] absent
        return REDPAN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted({*globals(), *__all__})
