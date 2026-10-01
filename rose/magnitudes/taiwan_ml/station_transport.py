# Vendored unchanged from taiwan-local-magnitude v3.1.0, src/taiwan_ml/station_transport.py
# (MIT, Copyright (c) 2026 Wu-Yu Liao). Provenance: rose/magnitudes/taiwan_ml/__init__.py
"""Training-only station-correction estimates for temporal transport tests.

This module deliberately keeps the released attenuation coefficients fixed.
It estimates only static station corrections from the training years and is
therefore a test of station-correction persistence, not an independent
validation of the attenuation relation or additive magnitude datum.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aggregation import station_location_ids


def balanced_observation_weights(
    public_ids: Iterable[str],
    station_keys: Iterable[str],
    is_borehole: Iterable[bool],
) -> np.ndarray:
    """Give equal event, installation-class, and station-location influence.

    The weights sum to one within each event.  Surface and borehole subsets
    receive equal weight when both occur, station locations receive equal
    weight within each subset, and channel prefixes at one location divide the
    station-location weight.
    """

    events = np.asarray([str(value) for value in public_ids], dtype=object)
    keys = np.asarray([str(value) for value in station_keys], dtype=object)
    borehole = np.asarray(list(is_borehole), dtype=bool)
    if not (events.shape == keys.shape == borehole.shape):
        raise ValueError("event, station-key, and borehole arrays must align")
    if not len(events):
        return np.array([], dtype=float)

    frame = pd.DataFrame(
        {
            "public_id": events,
            "installation": np.where(borehole, "borehole", "surface"),
            "station_location": station_location_ids(keys),
        }
    )
    location_group = ["public_id", "installation", "station_location"]
    network_group = ["public_id", "installation"]
    channel_multiplicity = frame.groupby(location_group, sort=False)[
        "public_id"
    ].transform("size")
    locations_per_network = frame.groupby(network_group, sort=False)[
        "station_location"
    ].transform("nunique")
    networks_per_event = frame.groupby("public_id", sort=False)[
        "installation"
    ].transform("nunique")
    weights = 1.0 / (
        channel_multiplicity.to_numpy(float)
        * locations_per_network.to_numpy(float)
        * networks_per_event.to_numpy(float)
    )
    event_sums = pd.Series(weights).groupby(frame["public_id"], sort=False).transform("sum")
    if not np.allclose(event_sums.to_numpy(float), 1.0, atol=1e-12):
        raise AssertionError("balanced observation weights do not sum to one per event")
    return weights


@dataclass(frozen=True)
class TransportTermFit:
    """One deterministic training-only station-correction fit."""

    terms: pd.Series
    observation_counts: pd.Series
    iterations: int
    median_absolute_last_change: float
    maximum_absolute_last_change: float
    converged: bool
    gauge_keys_present: int
    gauge_sum: float


def fit_transport_terms(
    public_ids: Iterable[str],
    raw_station_magnitudes: Iterable[float],
    term_keys: Iterable[str],
    observation_weights: Iterable[float],
    *,
    gauge_keys: Iterable[str],
    minimum_observations: int = 20,
    iterations: int = 16,
    minimum_iterations: int = 4,
    median_change_tolerance: float = 1.0e-4,
    maximum_change_tolerance: float = 1.0e-3,
) -> TransportTermFit:
    """Estimate static terms by alternating event means and median residuals.

    This robust, scalable estimator is used only for the temporal transport
    diagnostic.  It is not substituted for the released Huber-IRLS fit.
    """

    if minimum_observations < 1:
        raise ValueError("minimum_observations must be positive")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if not 1 <= minimum_iterations <= iterations:
        raise ValueError("minimum_iterations must be between one and iterations")
    if median_change_tolerance <= 0 or maximum_change_tolerance <= 0:
        raise ValueError("convergence tolerances must be positive")
    frame = pd.DataFrame(
        {
            "public_id": np.asarray([str(value) for value in public_ids], dtype=object),
            "raw": np.asarray(list(raw_station_magnitudes), dtype=float),
            "term_key": np.asarray([str(value) for value in term_keys], dtype=object),
            "weight": np.asarray(list(observation_weights), dtype=float),
        }
    )
    if len({len(frame[column]) for column in frame.columns}) != 1:
        raise ValueError("transport-fit arrays must align")
    finite = np.isfinite(frame["raw"]) & np.isfinite(frame["weight"]) & frame["weight"].gt(0)
    frame = frame.loc[finite].reset_index(drop=True)
    if frame.empty:
        raise ValueError("transport fit contains no finite observations")
    event_weight_sums = frame.groupby("public_id", sort=False)["weight"].sum()
    if not np.allclose(event_weight_sums.to_numpy(float), 1.0, atol=1e-10):
        raise ValueError("observation weights must sum to one within each event")

    counts = frame.groupby("term_key", sort=False).size().astype(int)
    supported = counts[counts.ge(minimum_observations)].index
    terms = pd.Series(0.0, index=supported, dtype=float)
    gauge = {str(value) for value in gauge_keys}
    median_change = float("nan")
    maximum_change = float("nan")
    converged = False
    gauge_present: list[str] = []

    completed_iterations = 0
    for completed_iterations in range(1, iterations + 1):
        previous = terms.copy()
        applied = frame["term_key"].map(previous).fillna(0.0).to_numpy(float)
        corrected = frame["raw"].to_numpy(float) - applied
        weighted = corrected * frame["weight"].to_numpy(float)
        event_center = pd.Series(weighted).groupby(
            frame["public_id"], sort=False
        ).transform("sum")
        residual = frame["raw"].to_numpy(float) - event_center.to_numpy(float)
        estimates = pd.Series(residual).groupby(frame["term_key"], sort=False).median()
        terms = estimates.reindex(supported).dropna().astype(float)
        gauge_present = sorted(gauge.intersection(terms.index.astype(str)))
        if not gauge_present:
            raise ValueError("no declared gauge key occurs in the training fit")
        terms -= float(terms.loc[gauge_present].mean())

        common = previous.index.intersection(terms.index)
        change = (terms.loc[common] - previous.loc[common]).abs()
        median_change = float(change.median())
        maximum_change = float(change.max())
        if (
            completed_iterations >= minimum_iterations
            and median_change <= median_change_tolerance
            and maximum_change <= maximum_change_tolerance
        ):
            converged = True
            break

    return TransportTermFit(
        terms=terms.sort_index(),
        observation_counts=counts.reindex(terms.index).astype(int),
        iterations=int(completed_iterations),
        median_absolute_last_change=median_change,
        maximum_absolute_last_change=maximum_change,
        converged=converged,
        gauge_keys_present=len(gauge_present),
        gauge_sum=float(terms.loc[gauge_present].sum()),
    )
