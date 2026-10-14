"""Synthetic series for demos and tests, plus a small CSV loader."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from forecastkit.types import FloatArray, as_series


def make_synthetic(
    n: int,
    season: int = 12,
    trend: float = 0.05,
    amplitude: float = 5.0,
    noise: float = 1.0,
    level: float = 50.0,
    seed: int = 0,
) -> FloatArray:
    """Level + linear trend + sinusoidal seasonality + Gaussian noise."""
    if n <= 0:
        raise ValueError("n must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=np.float64)
    seasonal = amplitude * np.sin(2 * np.pi * t / season) if season > 1 else 0.0
    series: FloatArray = level + trend * t + seasonal + rng.normal(0.0, noise, size=n)
    return series


def load_csv(path: Path, column: str) -> FloatArray:
    """Read one numeric column from a CSV file."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column {column!r} not found in {path}")
        values = [float(row[column]) for row in reader if row[column] != ""]
    return as_series(values)
