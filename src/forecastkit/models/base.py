"""The interface every forecaster implements."""

from __future__ import annotations

from typing import Protocol

from forecastkit.types import FloatArray


class Forecaster(Protocol):
    name: str

    def fit(self, y: FloatArray) -> None: ...

    def predict(self, horizon: int) -> FloatArray: ...
