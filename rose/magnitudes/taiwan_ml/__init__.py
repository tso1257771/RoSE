"""Station-corrected local-magnitude inversion (vendored).

Verbatim copy of five modules of the ``taiwan_ml`` package from
taiwan-local-magnitude v3.1.0 (https://github.com/tso1257771/taiwan-local-magnitude,
``src/taiwan_ml/{aggregation,design,invert,model,station_transport}.py``,
MIT-licensed, Copyright (c) 2026 Wu-Yu Liao; that repository licenses its source
code under MIT and its data tables and documentation under CC-BY-4.0, and only
source code is vendored here). They are the parts of the Taiwan method that the
RoSE local magnitude reuses unchanged: the sparse design matrix, the Huber-IRLS
joint inversion for event magnitudes, station terms and attenuation coefficients
with a sum-to-zero station gauge (``invert.solve``), the trimmed-median event
aggregation (``model.event_ml``), and the leave-years-out station-term transport
test (``station_transport``). The Taiwan CLI, configuration, runtime,
reported-magnitude and interpolation modules are not vendored: the upstream
package is not on PyPI, requires Python >= 3.11 and pyarrow, and RoSE supports
3.10. Relicensed under this repo's MIT LICENSE. Comments in the files that name
Taiwan stations (e.g. TW.NLKH) are upstream text and do not apply here. The
Romanian -log A0 form lives in ``rose.magnitudes.attenuation`` and is passed to
``invert.solve`` as extra design columns; ``design.attenuation_columns`` (the
Taiwan single-slope form) is kept only because ``invert`` imports it.

Used by ``magnitudes/ml/fit_ml.py``. To update: copy the five files from the
upstream tag unchanged and bump ``__upstream__``.
"""
__all__ = ["aggregation", "design", "invert", "model", "station_transport"]
__upstream__ = "taiwan-local-magnitude v3.1.0"
