"""pickerbench — phase-picker benchmark scoring utilities.

Self-contained module for scoring deep-learning phase pickers against
labelled event windows. Used to reproduce the RoSE and STEAD-test
benchmark numbers shipped under `results/`.

Modules:
    matching          — match_picks, evaluate_trace_detections, MCC pairs
    residual_stats    — exact MAE / MAD / std on matched-pick residuals
    trace_io          — Stream construction (SeisBench ZNE; RED-PAN ENZ)
    leaderboard       — aggregate per-model JSONs into the publication CSVs
"""
from .matching import (
    match_picks, evaluate_trace_detections,
    match_picks_cross_phase, aggregate_summary,
)
from .residual_stats import residual_statistics
from .trace_io import waveform_to_stream

__all__ = [
    "match_picks", "evaluate_trace_detections",
    "match_picks_cross_phase", "aggregate_summary",
    "residual_statistics", "waveform_to_stream",
]
__version__ = "1.0.0"
