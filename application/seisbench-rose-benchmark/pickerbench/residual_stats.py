"""Exact residual statistics on matched-pick residuals."""
from __future__ import annotations

import math

import numpy as np


def residual_statistics(residuals: list[float]) -> dict:
    """Return the canonical pick-residual statistics.

        n        — sample count
        mae      — mean absolute error  = mean(|r|)
        median   — median signed residual
        std      — standard deviation around the mean
        mad      — median absolute deviation about the median
                   (robust scale; for normal MAD = 0.6745*std)
        rmse     — root mean square residual
        iqr      — interquartile range Q75 - Q25 (linear interpolation,
                   matches numpy.percentile / pandas.quantile defaults)
    """
    if not residuals:
        return {"n": 0, "mae": float("nan"), "median": float("nan"),
                "std": float("nan"), "mad": float("nan"),
                "rmse": float("nan"), "iqr": float("nan")}
    arr = np.asarray(residuals, dtype=np.float64)
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    return {
        "n": int(arr.size),
        "mae": float(np.mean(np.abs(arr))),
        "median": median,
        "std": float(np.sqrt(np.mean((arr - mean) ** 2))),
        "mad": float(np.median(np.abs(arr - median))),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "iqr": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
    }
