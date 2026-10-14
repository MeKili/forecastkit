"""Ridge regression on lagged values and a linear time index, forecast recursively."""

from __future__ import annotations

import numpy as np

from forecastkit.types import FloatArray


class RidgeLag:
    name = "ridge_lag"

    def __init__(self, lags: int = 12, alpha: float = 1.0) -> None:
        if lags < 1:
            raise ValueError("lags must be >= 1")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        self.lags = lags
        self.alpha = alpha
        self._coef: FloatArray | None = None
        self._mean: FloatArray | None = None
        self._std: FloatArray | None = None
        self._history: FloatArray | None = None

    def _features(self, window: FloatArray, t: int) -> FloatArray:
        out: FloatArray = np.concatenate([window[::-1], np.array([float(t)])])
        return out

    def fit(self, y: FloatArray) -> None:
        n_rows = y.size - self.lags
        if n_rows < self.lags + 2:
            raise ValueError("series too short for the requested number of lags")
        x = np.vstack([self._features(y[i : i + self.lags], i + self.lags) for i in range(n_rows)])
        target = y[self.lags :]
        self._mean = x.mean(axis=0)
        std = x.std(axis=0)
        self._std = np.where(std == 0.0, 1.0, std)
        z = (x - self._mean) / self._std
        design = np.hstack([np.ones((z.shape[0], 1)), z])
        penalty = self.alpha * np.eye(design.shape[1])
        penalty[0, 0] = 0.0
        self._coef = np.linalg.solve(design.T @ design + penalty, design.T @ target)
        self._history = y.copy()

    def predict(self, horizon: int) -> FloatArray:
        if self._coef is None or self._mean is None or self._std is None or self._history is None:
            raise RuntimeError("model is not fitted")
        history = [float(v) for v in self._history]
        out: list[float] = []
        for _ in range(horizon):
            window = np.asarray(history[-self.lags :], dtype=np.float64)
            feats = (self._features(window, len(history)) - self._mean) / self._std
            value = float(self._coef[0] + feats @ self._coef[1:])
            out.append(value)
            history.append(value)
        return np.asarray(out, dtype=np.float64)
