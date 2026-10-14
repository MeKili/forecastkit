"""Rolling-origin evaluation: refit at each origin, forecast ahead, compare with what happened."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from forecastkit.metrics import smape
from forecastkit.models.base import Forecaster
from forecastkit.models.registry import MODEL_NAMES, build_model
from forecastkit.types import FloatArray


@dataclass(frozen=True)
class Fold:
    origin: int
    actual: FloatArray
    forecast: FloatArray


@dataclass(frozen=True)
class BacktestResult:
    model: str
    folds: tuple[Fold, ...]

    @property
    def errors(self) -> FloatArray:
        """Signed errors (actual - forecast), shape (folds, horizon)."""
        return np.vstack([f.actual - f.forecast for f in self.folds])

    @property
    def mae(self) -> float:
        return float(np.mean(np.abs(self.errors)))

    @property
    def smape(self) -> float:
        actual = np.concatenate([f.actual for f in self.folds])
        forecast = np.concatenate([f.forecast for f in self.folds])
        return smape(actual, forecast)

    def mae_by_horizon(self) -> FloatArray:
        out: FloatArray = np.mean(np.abs(self.errors), axis=0)
        return out


def rolling_origin(
    y: FloatArray,
    make_model: Callable[[], Forecaster],
    horizon: int,
    initial: int,
    step: int = 1,
) -> BacktestResult:
    """Evaluate a model on expanding windows. Needs ``initial + horizon <= len(y)``."""
    if horizon < 1 or step < 1 or initial < 2:
        raise ValueError("horizon and step must be >= 1 and initial >= 2")
    if initial + horizon > y.size:
        raise ValueError("series too short for the requested initial window and horizon")
    folds: list[Fold] = []
    for origin in range(initial, y.size - horizon + 1, step):
        model = make_model()
        model.fit(y[:origin])
        folds.append(
            Fold(
                origin=origin,
                actual=y[origin : origin + horizon],
                forecast=model.predict(horizon),
            )
        )
    return BacktestResult(model=make_model().name, folds=tuple(folds))


def compare_models(
    y: FloatArray,
    horizon: int,
    season: int = 1,
    initial: int | None = None,
    step: int | None = None,
    names: tuple[str, ...] = MODEL_NAMES,
) -> list[BacktestResult]:
    """Backtest several models on the same folds and return them best-first by MAE."""
    start = initial if initial is not None else max(2 * season + 2, y.size // 2)
    stride = step if step is not None else max(1, horizon // 2)
    results: list[BacktestResult] = []
    for name in names:

        def factory(model_name: str = name) -> Forecaster:
            return build_model(model_name, season)

        try:
            results.append(rolling_origin(y, factory, horizon, start, stride))
        except ValueError:
            continue
    if not results:
        raise ValueError("no model could be backtested on this series")
    return sorted(results, key=lambda r: r.mae)
