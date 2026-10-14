"""Simple baselines that any serious model has to beat."""

from __future__ import annotations

import numpy as np

from forecastkit.types import FloatArray


class _Base:
    name = "base"

    def __init__(self) -> None:
        self._y: FloatArray | None = None

    def fit(self, y: FloatArray) -> None:
        self._y = y.copy()

    def _series(self) -> FloatArray:
        if self._y is None:
            raise RuntimeError("model is not fitted")
        return self._y


class Naive(_Base):
    """Repeat the last observation."""

    name = "naive"

    def predict(self, horizon: int) -> FloatArray:
        return np.full(horizon, self._series()[-1], dtype=np.float64)


class SeasonalNaive(_Base):
    """Repeat the last full season."""

    name = "seasonal_naive"

    def __init__(self, season: int) -> None:
        super().__init__()
        if season < 1:
            raise ValueError("season must be >= 1")
        self.season = season

    def predict(self, horizon: int) -> FloatArray:
        y = self._series()
        if y.size < self.season:
            raise ValueError("series shorter than one season")
        last = y[-self.season :]
        reps = int(np.ceil(horizon / self.season))
        out: FloatArray = np.tile(last, reps)[:horizon]
        return out


class Drift(_Base):
    """Extend the line between the first and last observation."""

    name = "drift"

    def predict(self, horizon: int) -> FloatArray:
        y = self._series()
        slope = 0.0 if y.size < 2 else (y[-1] - y[0]) / (y.size - 1)
        steps = np.arange(1, horizon + 1, dtype=np.float64)
        out: FloatArray = y[-1] + slope * steps
        return out


class MovingAverage(_Base):
    """Flat forecast at the mean of the last ``window`` points."""

    name = "moving_average"

    def __init__(self, window: int = 6) -> None:
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def predict(self, horizon: int) -> FloatArray:
        y = self._series()
        return np.full(horizon, float(np.mean(y[-self.window :])), dtype=np.float64)
