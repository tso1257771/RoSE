# Vendored unchanged from taiwan-local-magnitude v3.1.0, src/taiwan_ml/model.py
# (MIT, Copyright (c) 2026 Wu-Yu Liao). Provenance: rose/magnitudes/taiwan_ml/__init__.py
"""Per-station ML computation and robust event aggregation.

ML_station = log10(A_mm) + (-log10 A0)(R,h) - S_j + C
ML_event   = median of station MLs with IQR trimming.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from .aggregation import station_location_ids
from .invert import neg_log_a0


def station_ml(obs: pd.DataFrame, coeffs: dict, station_terms: Mapping[str, float],
               anchor_c: float = 0.0, default_s: float = 0.0) -> np.ndarray:
    """Per-observation station ML estimate."""
    # -log10(A0): positive attenuation correction added to log10(A) to give ML
    atten = neg_log_a0(obs["R_km"].to_numpy(float), obs["depth_km"].to_numpy(float), coeffs)
    sj = obs["station_key"].map(station_terms).fillna(default_s).to_numpy(float)
    return obs["log10_A"].to_numpy(float) + atten - sj + anchor_c


def event_ml(obs: pd.DataFrame, coeffs: dict, station_terms: Mapping[str, float],
             anchor_c: float = 0.0, method: str = "median",
             min_stations: int = 3, trim_iqr: bool = True,
             default_s: float = 0.0) -> pd.DataFrame:
    """Aggregate station MLs to event ML. Returns public_id, ml, n_sta."""
    ml_vals = station_ml(obs, coeffs, station_terms, anchor_c, default_s)
    ml_df = pd.DataFrame(
        {
            "public_id": obs["public_id"].to_numpy(),
            "station_location": station_location_ids(
                obs["station_key"].astype(str)
            ),
            "ml": ml_vals,
        }
    ).dropna()
    ml_df = (
        ml_df.groupby(
            ["public_id", "station_location"],
            sort=False,
            as_index=False,
        )["ml"]
        .median()
    )
    rows = []
    for pid, grp in ml_df.groupby("public_id")["ml"]:
        vals = grp.to_numpy()
        if trim_iqr and len(vals) >= 5:
            q1, q3 = np.percentile(vals, [25, 75])
            iqr = q3 - q1
            vals = vals[(vals >= q1 - 1.5 * iqr) & (vals <= q3 + 1.5 * iqr)]
        if len(vals) < min_stations:
            continue
        rows.append((pid, float(np.median(vals) if method == "median" else np.mean(vals)), len(vals)))
    return pd.DataFrame(rows, columns=["public_id", "ml", "n_sta"])
