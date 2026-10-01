# Vendored unchanged from taiwan-local-magnitude v3.1.0, src/taiwan_ml/aggregation.py
# (MIT, Copyright (c) 2026 Wu-Yu Liao). Provenance: rose/magnitudes/taiwan_ml/__init__.py
"""Event aggregation shared by the released magnitude computations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd


def station_location_ids(
    station_keys: Iterable[str],
) -> np.ndarray:
    """Return ``NET.STA.LOC`` groups from channel-specific station keys.

    A station correction is specific to a channel prefix and epoch, whereas
    fitting weights and event aggregation group channel prefixes by
    station-location code. Mixed-network operations apply that grouping
    separately within the surface and borehole subsets because an FDSN
    location code does not uniquely determine sensor depth in this inventory.
    """

    identifiers: list[str] = []
    for raw_key in station_keys:
        key = str(raw_key)
        prefix = key.split("@", maxsplit=1)[0]
        parts = prefix.split(".")
        if len(parts) < 4 or any(not part for part in parts[:2]):
            raise ValueError(
                "station keys must have NET.STA.LOC.CHAN@EPOCH form"
            )
        identifiers.append(".".join(parts[:3]))
    return np.asarray(identifiers, dtype=object)


def network_scoped_group_ids(
    station_locations: Iterable[str],
    is_borehole: Iterable[bool],
) -> np.ndarray:
    """Return the released group identity for independent-site counting.

    The released aggregation treats a surface sensor and a downhole sensor as
    two independent groups even when the FDSN location code is shared, because
    a location code does not determine sensor depth in this inventory. Every
    place that counts groups or gates on a group minimum must use this one
    identity so the single-event and batch paths agree.
    """

    locations = [str(value) for value in station_locations]
    downhole = [bool(value) for value in is_borehole]
    if len(locations) != len(downhole):
        raise ValueError(
            "station-location identifiers and installation flags must align"
        )
    return np.asarray(
        [
            f"{'borehole' if flag else 'surface'}|{location}"
            for location, flag in zip(locations, downhole, strict=True)
        ],
        dtype=object,
    )


def _validated_station_locations(
    station_locations: Iterable[str],
    *,
    length: int,
) -> np.ndarray:
    locations = np.asarray(
        [str(value) for value in station_locations],
        dtype=object,
    )
    if locations.shape != (length,):
        raise ValueError(
            "station-location identifiers must align with observations"
        )
    if any(not value or value != value.strip() for value in locations):
        raise ValueError(
            "station-location identifiers must be non-blank and unpadded"
        )
    return locations


def _collapse_one_event_station_locations(
    station_magnitudes: np.ndarray,
    is_borehole: np.ndarray,
    station_locations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame(
        {
            "station_location": station_locations,
            "station_ml": station_magnitudes,
            "is_borehole": is_borehole,
        }
    )
    collapsed = (
        frame.groupby(["is_borehole", "station_location"], sort=False)
        .agg(
            station_ml=("station_ml", "median"),
            is_borehole=("is_borehole", "first"),
        )
        .reset_index(drop=True)
    )
    return (
        collapsed["station_ml"].to_numpy(float),
        collapsed["is_borehole"].to_numpy(bool),
    )


@dataclass(frozen=True)
class EventLocalMagnitude:
    """One event aggregate under the released mixed-network rule."""

    ml: float
    network_source: str
    surface_ml: float | None
    borehole_ml: float | None
    combined_ml: float | None
    surface_stations_used: int
    borehole_stations_used: int
    combined_stations_used: int


def _trimmed_median(
    values: np.ndarray,
    *,
    min_stations: int,
) -> tuple[float | None, int]:
    """Return a median after the fixed 1.5-IQR station filter."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size < min_stations:
        return None, 0
    retained = finite
    if finite.size >= 5:
        q1, q3 = np.quantile(finite, (0.25, 0.75))
        iqr = q3 - q1
        retained = finite[
            (finite >= q1 - 1.5 * iqr) & (finite <= q3 + 1.5 * iqr)
        ]
    if retained.size < min_stations:
        return None, int(retained.size)
    return float(np.median(retained)), int(retained.size)


def aggregate_event_ml(
    station_magnitudes: np.ndarray,
    is_borehole: np.ndarray,
    *,
    min_stations: int = 3,
    station_locations: Iterable[str] | None = None,
) -> EventLocalMagnitude:
    """Aggregate one event while balancing surface and borehole networks.

    Surface and borehole observations are filtered and aggregated separately.
    A station-location code can occur once in each network because Taiwan
    includes surface and downhole sensors under the same FDSN location code.
    Co-located channel prefixes within the same network are collapsed by
    median. Network medians receive equal weight when both independently meet
    ``min_stations``. The available network is used when only one qualifies;
    the union is used only when neither qualifies separately.
    """

    if not isinstance(min_stations, int | np.integer) or min_stations < 1:
        raise ValueError("min_stations must be a positive integer")
    magnitudes, borehole = np.broadcast_arrays(
        np.asarray(station_magnitudes, dtype=float),
        np.asarray(is_borehole, dtype=bool),
    )
    magnitudes = magnitudes.ravel()
    borehole = borehole.ravel()
    if station_locations is not None:
        locations = _validated_station_locations(
            station_locations,
            length=len(magnitudes),
        )
        magnitudes, borehole = _collapse_one_event_station_locations(
            magnitudes,
            borehole,
            locations,
        )
    surface_ml, surface_used = _trimmed_median(
        magnitudes[~borehole],
        min_stations=int(min_stations),
    )
    borehole_ml, borehole_used = _trimmed_median(
        magnitudes[borehole],
        min_stations=int(min_stations),
    )
    combined_ml, combined_used = _trimmed_median(
        magnitudes,
        min_stations=int(min_stations),
    )

    if surface_ml is not None and borehole_ml is not None:
        event_ml = 0.5 * (surface_ml + borehole_ml)
        source = "balanced_both"
    elif surface_ml is not None:
        event_ml = surface_ml
        source = "surface_only"
    elif borehole_ml is not None:
        event_ml = borehole_ml
        source = "borehole_only"
    elif combined_ml is not None:
        event_ml = combined_ml
        source = "union_fallback"
    else:
        event_ml = float("nan")
        source = "insufficient_stations"

    return EventLocalMagnitude(
        ml=float(event_ml),
        network_source=source,
        surface_ml=surface_ml,
        borehole_ml=borehole_ml,
        combined_ml=combined_ml,
        surface_stations_used=surface_used,
        borehole_stations_used=borehole_used,
        combined_stations_used=combined_used,
    )


def aggregate_event_table(
    public_ids: Iterable[str],
    station_magnitudes: Iterable[float],
    *,
    min_stations: int = 3,
    station_locations: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate a station-value table with the released median/IQR rule."""

    frame = pd.DataFrame(
        {
            "public_id": np.asarray(public_ids),
            "station_ml": np.asarray(station_magnitudes, dtype=float),
        }
    )
    if station_locations is not None:
        frame["station_location"] = _validated_station_locations(
            station_locations,
            length=len(frame),
        )
    frame = frame[
        frame["public_id"].notna() & np.isfinite(frame["station_ml"])
    ].copy()
    columns = [
        "public_id",
        "ml",
        "scatter_post_trim",
        "n_stations_used",
        "n_stations_raw",
        "scatter_untrimmed",
        "n_stations_trimmed",
    ]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    if station_locations is not None:
        frame = (
            frame.groupby(
                ["public_id", "station_location"],
                sort=False,
                as_index=False,
            )["station_ml"]
            .median()
        )

    grouped = frame.groupby("public_id", sort=False)["station_ml"]
    raw = grouped.agg(n_stations_raw="count", scatter_untrimmed="std")
    group_count = grouped.transform("count")
    quantiles = grouped.quantile([0.25, 0.75]).unstack()
    q1 = frame["public_id"].map(quantiles[0.25])
    q3 = frame["public_id"].map(quantiles[0.75])
    iqr = q3 - q1
    keep = (group_count < 5) | (
        (frame["station_ml"] >= q1 - 1.5 * iqr)
        & (frame["station_ml"] <= q3 + 1.5 * iqr)
    )
    retained = frame.loc[keep]
    result = (
        retained.groupby("public_id", sort=False)["station_ml"]
        .agg(ml="median", scatter_post_trim="std", n_stations_used="count")
        .join(raw, how="left")
        .reset_index()
    )
    result = result[
        (result["n_stations_raw"] >= min_stations)
        & (result["n_stations_used"] >= min_stations)
    ].copy()
    result["n_stations_trimmed"] = (
        result["n_stations_raw"] - result["n_stations_used"]
    )
    return result.reset_index(drop=True)


def aggregate_mixed_network_table(
    public_ids: Iterable[str],
    station_magnitudes: Iterable[float],
    is_borehole: Iterable[bool],
    *,
    min_stations: int = 3,
    station_locations: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Aggregate many events with the released mixed-network ``M_R`` rule.

    Surface and borehole observations are summarized separately and receive
    equal weight when both independently meet the station minimum. The union
    is used only when neither network qualifies by itself.
    """

    ids = np.asarray(public_ids)
    values = np.asarray(station_magnitudes, dtype=float)
    borehole = np.asarray(is_borehole, dtype=bool)
    if ids.ndim != 1 or values.ndim != 1 or borehole.ndim != 1:
        raise ValueError("mixed-network inputs must be one-dimensional")
    if len({len(ids), len(values), len(borehole)}) != 1:
        raise ValueError("mixed-network inputs must have equal length")
    locations = (
        None
        if station_locations is None
        else _validated_station_locations(
            station_locations,
            length=len(ids),
        )
    )
    network_locations = (
        None
        if locations is None
        else network_scoped_group_ids(locations, borehole)
    )

    parts: list[pd.DataFrame] = []
    for installation, mask in (
        ("surface", ~borehole),
        ("borehole", borehole),
        ("combined", np.ones(len(values), dtype=bool)),
    ):
        aggregated = aggregate_event_table(
            ids[mask],
            values[mask],
            min_stations=min_stations,
            station_locations=(
                network_locations[mask]
                if network_locations is not None
                else None
            ),
        )
        aggregated = aggregated.rename(
            columns={
                column: f"{installation}_{column}"
                for column in aggregated.columns
                if column != "public_id"
            }
        )
        parts.append(aggregated)
    result = parts[0]
    for part in parts[1:]:
        result = result.merge(
            part,
            on="public_id",
            how="outer",
            validate="one_to_one",
        )
    surface = result["surface_ml"]
    downhole = result["borehole_ml"]
    combined = result["combined_ml"]
    both = surface.notna() & downhole.notna()
    result["balanced_ml"] = surface.fillna(downhole).fillna(combined)
    result.loc[both, "balanced_ml"] = 0.5 * (
        surface.loc[both] + downhole.loc[both]
    )
    source = np.full(len(result), "missing", dtype=object)
    source[both.to_numpy()] = "balanced_both"
    source[(surface.notna() & downhole.isna()).to_numpy()] = "surface_only"
    source[(surface.isna() & downhole.notna()).to_numpy()] = "borehole_only"
    source[
        (surface.isna() & downhole.isna() & combined.notna()).to_numpy()
    ] = "union_fallback"
    result["network_source"] = source
    return result
