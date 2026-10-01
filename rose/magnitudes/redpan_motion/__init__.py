"""RED-PAN-Motion amplitude window and Wood-Anderson response (vendored).

Verbatim copy of ``redpan_motion/amplitudes.py`` and ``redpan_motion/response.py``
from RED-PAN-Motion v0.1.3 (https://github.com/tso1257771/RED-PAN-Motion,
MIT-licensed, Copyright (c) 2025 tso1257771). They define the magnitude window of
the released ML (S-P based nominal window, amplitude-decay truncation in
``adaptive_s_window``, the Wadati-proxy window constants), the Wood-Anderson poles
and zeros, and the full-response amplitude path (``full_response_amplitudes``,
``compute_snr``). The same ``adaptive_s_window`` sets the S window of the
SourceSpec Mw inversion; ``magnitudes/mw/sourcespec/source_spec_redpan_windows.py``
carries a copy, because that file runs inside the SourceSpec environment. The
picker, model and training modules are not vendored: the upstream package is not
on PyPI and depends on torch, which RoSE keeps optional. Relicensed under this
repo's MIT LICENSE.

The published ML was measured with a pre-0.1.1 text of these two files. Against
v0.1.3 the differences are docstrings, local variable names, and wider channel
sets; the RoSE observations use only BH, EH, HH and HN, which both texts classify
identically, so v0.1.3 reproduces the released amplitudes.

Used by ``magnitudes/ml/extract_wa_amplitudes.py``. To update: copy the two files
from the upstream tag unchanged and bump ``__upstream__``.
"""
__all__ = ["amplitudes", "response"]
__upstream__ = "RED-PAN-Motion v0.1.3"
