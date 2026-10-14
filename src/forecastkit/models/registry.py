"""Construct a forecaster by name."""

from __future__ import annotations

from forecastkit.models.base import Forecaster
from forecastkit.models.baselines import Drift, MovingAverage, Naive, SeasonalNaive
from forecastkit.models.holtwinters import HoltWinters
from forecastkit.models.ridge import RidgeLag

MODEL_NAMES = ("naive", "seasonal_naive", "drift", "moving_average", "holt_winters", "ridge_lag")


def build_model(name: str, season: int = 1) -> Forecaster:
    """Return a fresh, unfitted model. ``season`` is the seasonal period (1 = none)."""
    if name == "naive":
        return Naive()
    if name == "seasonal_naive":
        return SeasonalNaive(season)
    if name == "drift":
        return Drift()
    if name == "moving_average":
        return MovingAverage(window=max(season, 3))
    if name == "holt_winters":
        return HoltWinters(season)
    if name == "ridge_lag":
        return RidgeLag(lags=max(season, 4))
    raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODEL_NAMES)}")
