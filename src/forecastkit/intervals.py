"""Empirical (conformal-style) prediction intervals from backtest residuals, per horizon."""

from __future__ import annotations

import numpy as np

from forecastkit.backtest import BacktestResult
from forecastkit.types import FloatArray


def interval_offsets(result: BacktestResult, level: float = 0.8) -> tuple[FloatArray, FloatArray]:
    """Return (lower, upper) offsets to add to a point forecast, one per horizon step.

    Offsets are signed residual quantiles, so intervals can be asymmetric and widen with horizon
    when longer-range errors are larger.
    """
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    errors = result.errors
    if errors.shape[0] < 3:
        raise ValueError("need at least 3 backtest folds to estimate intervals")
    alpha = 1.0 - level
    lower: FloatArray = np.quantile(errors, alpha / 2.0, axis=0, method="lower")
    upper: FloatArray = np.quantile(errors, 1.0 - alpha / 2.0, axis=0, method="higher")
    return lower, upper


def with_intervals(
    point: FloatArray, result: BacktestResult, level: float = 0.8
) -> tuple[FloatArray, FloatArray]:
    """Add the residual-quantile offsets to a point forecast."""
    lower, upper = interval_offsets(result, level)
    if point.size != lower.size:
        raise ValueError("point forecast length must equal the backtest horizon")
    return point + lower, point + upper
