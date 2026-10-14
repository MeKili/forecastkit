"""Additive Holt-Winters exponential smoothing with a small parameter grid search."""

from __future__ import annotations

import itertools

import numpy as np

from forecastkit.types import FloatArray

_ALPHAS = (0.1, 0.3, 0.5, 0.8)
_BETAS = (0.01, 0.1, 0.3)
_GAMMAS = (0.1, 0.3, 0.6)


class HoltWinters:
    """Level, trend and additive seasonality. With season 1 it reduces to Holt's linear method."""

    name = "holt_winters"

    def __init__(self, season: int = 1) -> None:
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season
        self._level = 0.0
        self._trend = 0.0
        self._seasonals: FloatArray = np.zeros(season, dtype=np.float64)
        self._n = 0
        self.params: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def _initial_state(self, y: FloatArray) -> tuple[float, float, FloatArray]:
        m = self.season
        if m == 1:
            slope = float(y[1] - y[0]) if y.size > 1 else 0.0
            return float(y[0]), slope, np.zeros(1, dtype=np.float64)
        first = y[:m]
        second = y[m : 2 * m]
        level = float(np.mean(first))
        trend = float((np.mean(second) - np.mean(first)) / m)
        seasonals: FloatArray = first - level
        return level, trend, seasonals

    def _run(
        self, y: FloatArray, alpha: float, beta: float, gamma: float
    ) -> tuple[float, float, float, FloatArray]:
        m = self.season
        level, trend, seasonals = self._initial_state(y)
        seasonals = seasonals.copy()
        sse = 0.0
        start = m if m > 1 else 1
        for t in range(start, y.size):
            s = float(seasonals[t % m]) if m > 1 else 0.0
            pred = level + trend + s
            err = float(y[t]) - pred
            sse += err * err
            new_level = alpha * (float(y[t]) - s) + (1 - alpha) * (level + trend)
            trend = beta * (new_level - level) + (1 - beta) * trend
            if m > 1:
                seasonals[t % m] = gamma * (float(y[t]) - new_level) + (1 - gamma) * s
            level = new_level
        return sse, level, trend, seasonals

    def fit(self, y: FloatArray) -> None:
        m = self.season
        if y.size < max(2 * m, 3):
            raise ValueError("series too short for Holt-Winters with this season")
        best_sse = float("inf")
        best_params = (0.0, 0.0, 0.0)
        gammas = _GAMMAS if m > 1 else (0.0,)
        for a, b, g in itertools.product(_ALPHAS, _BETAS, gammas):
            sse = self._run(y, a, b, g)[0]
            if sse < best_sse:
                best_sse = sse
                best_params = (a, b, g)
        self.params = best_params
        _, self._level, self._trend, self._seasonals = self._run(y, *best_params)
        self._n = int(y.size)

    def predict(self, horizon: int) -> FloatArray:
        if self._n == 0:
            raise RuntimeError("model is not fitted")
        m = self.season
        steps = np.arange(1, horizon + 1)
        out = self._level + steps * self._trend
        if m > 1:
            idx = (self._n - 1 + steps) % m
            out = out + self._seasonals[idx]
        return np.asarray(out, dtype=np.float64)
