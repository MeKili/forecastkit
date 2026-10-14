"""Shared type aliases."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def as_series(values: object) -> FloatArray:
    """Coerce input to a finite one-dimensional float array."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if arr.size == 0:
        raise ValueError("series is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError("series contains NaN or infinite values")
    return arr
