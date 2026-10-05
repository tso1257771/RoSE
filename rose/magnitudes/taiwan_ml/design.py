# Vendored unchanged from taiwan-local-magnitude commit 9d645e4, src/taiwan_ml/design.py
# (MIT, Copyright (c) 2026 Wu-Yu Liao). Provenance: rose/magnitudes/taiwan_ml/__init__.py
"""Sparse design matrix for the released local-magnitude inversion.

The only supported attenuation form is

    n*log10(R/Rref) + K*(R-Rref)
    + dK*(R-Rref)*max(0,h-h_break)/100.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp


def attenuation_columns(
    distance_km: np.ndarray,
    depth_km: np.ndarray,
    *,
    rref: float = 100.0,
    h_break: float = 40.0,
) -> tuple[np.ndarray, list[str]]:
    """Return the three positive attenuation columns of the released model."""

    distance = np.asarray(distance_km, dtype=float)
    depth = np.asarray(depth_km, dtype=float)
    if distance.shape != depth.shape:
        raise ValueError("distance_km and depth_km must have identical shapes")
    columns = np.column_stack(
        (
            np.log10(distance / float(rref)),
            distance - float(rref),
            (distance - float(rref))
            * np.maximum(0.0, depth - float(h_break))
            / 100.0,
        )
    )
    return columns, ["n", "K", "dK"]


def build_design(
    observations: pd.DataFrame,
    *,
    rref: float = 100.0,
    h_break: float = 40.0,
    n_cheb: int = 0,
    depth_term: bool = True,
    station_col: str = "station_key",
    center: bool = True,
) -> tuple[sp.csr_matrix, np.ndarray, dict]:
    """Assemble the released joint event/station/attenuation design."""

    if int(n_cheb) != 0:
        raise ValueError("the released fit supports only n_cheb=0")
    if not bool(depth_term):
        raise ValueError("the released fit requires the 40-km depth-hinge term")

    event_category = pd.Categorical(observations["public_id"])
    station_category = pd.Categorical(observations[station_col])
    n_events = len(event_category.categories)
    n_stations = len(station_category.categories)

    positive_attenuation, names = attenuation_columns(
        observations["R_km"].to_numpy(float),
        observations["depth_km"].to_numpy(float),
        rref=rref,
        h_break=h_break,
    )
    attenuation = -positive_attenuation
    attenuation_mean = (
        attenuation.mean(axis=0)
        if center
        else np.zeros(attenuation.shape[1], dtype=float)
    )
    attenuation = attenuation - attenuation_mean

    n_observations = len(observations)
    row_indices = np.repeat(np.arange(n_observations), 2)
    column_indices = np.empty(n_observations * 2, dtype=int)
    column_indices[0::2] = event_category.codes
    column_indices[1::2] = n_events + station_category.codes
    event_station = sp.coo_matrix(
        (
            np.ones(n_observations * 2),
            (row_indices, column_indices),
        ),
        shape=(n_observations, n_events + n_stations),
    ).tocsr()
    design = sp.hstack(
        [event_station, sp.csr_matrix(attenuation)]
    ).tocsr()
    data = observations["log10_A"].to_numpy(float)
    metadata = {
        "events": list(event_category.categories),
        "stations": list(station_category.categories),
        "atten_names": names,
        "atten_mean": attenuation_mean,
        "n_e": n_events,
        "n_s": n_stations,
        "rref": float(rref),
        "h_break": float(h_break),
        "n_cheb": 0,
        "depth_term": True,
    }
    return design, data, metadata
