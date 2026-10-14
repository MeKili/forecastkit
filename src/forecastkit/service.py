"""High-level entry point: pick a model by backtest, fit on all data, return a forecast."""

from __future__ import annotations

from dataclasses import dataclass

from forecastkit.backtest import BacktestResult, compare_models, rolling_origin
from forecastkit.intervals import with_intervals
from forecastkit.models.registry import build_model
from forecastkit.types import FloatArray


@dataclass(frozen=True)
class ForecastOutput:
    model: str
    forecast: FloatArray
    lower: FloatArray
    upper: FloatArray
    backtest_mae: float
    backtest_smape: float
    folds: int


def forecast_series(
    y: FloatArray,
    horizon: int,
    season: int = 1,
    model: str = "auto",
    level: float = 0.8,
) -> ForecastOutput:
    """Forecast ``horizon`` steps. ``model="auto"`` chooses the lowest-MAE model by backtest."""
    initial = max(2 * season + 2, y.size // 2)
    best: BacktestResult
    if model == "auto":
        best = compare_models(y, horizon, season, initial=initial)[0]
    else:
        best = rolling_origin(
            y, lambda: build_model(model, season), horizon, initial, max(1, horizon // 2)
        )
    fitted = build_model(best.model, season)
    fitted.fit(y)
    point = fitted.predict(horizon)
    lower, upper = with_intervals(point, best, level)
    return ForecastOutput(
        model=best.model,
        forecast=point,
        lower=lower,
        upper=upper,
        backtest_mae=best.mae,
        backtest_smape=best.smape,
        folds=len(best.folds),
    )
