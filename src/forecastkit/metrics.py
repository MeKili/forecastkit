"""Point and interval forecast accuracy metrics."""

from __future__ import annotations

import numpy as np

from forecastkit.types import FloatArray


def mae(actual: FloatArray, forecast: FloatArray) -> float:
    return float(np.mean(np.abs(actual - forecast)))


def rmse(actual: FloatArray, forecast: FloatArray) -> float:
    return float(np.sqrt(np.mean((actual - forecast) ** 2)))


def smape(actual: FloatArray, forecast: FloatArray) -> float:
    """Symmetric MAPE in percent; terms where both values are zero count as zero error."""
    denom = (np.abs(actual) + np.abs(forecast)) / 2.0
    safe = np.where(denom == 0.0, 1.0, denom)
    terms = np.where(denom == 0.0, 0.0, np.abs(actual - forecast) / safe)
    return float(100.0 * np.mean(terms))


def mase(actual: FloatArray, forecast: FloatArray, train: FloatArray, season: int = 1) -> float:
    """Mean absolute scaled error against the in-sample seasonal-naive error."""
    if train.size <= season:
        raise ValueError("training series too short for the requested season")
    scale = float(np.mean(np.abs(train[season:] - train[:-season])))
    if scale == 0.0:
        raise ValueError("in-sample naive error is zero, MASE is undefined")
    return mae(actual, forecast) / scale


def pinball(actual: FloatArray, quantile_forecast: FloatArray, q: float) -> float:
    """Pinball (quantile) loss for quantile level ``q``."""
    if not 0.0 < q < 1.0:
        raise ValueError("q must be in (0, 1)")
    diff = actual - quantile_forecast
    return float(np.mean(np.maximum(q * diff, (q - 1.0) * diff)))


def coverage(actual: FloatArray, lower: FloatArray, upper: FloatArray) -> float:
    """Share of actual values that fall inside the interval."""
    return float(np.mean((actual >= lower) & (actual <= upper)))
