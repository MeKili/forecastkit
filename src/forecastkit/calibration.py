"""Prediction-interval calibration report: out-of-sample coverage by horizon step.

Checking whether an interval covers the actual value on the same folds used to fit its
offsets is circular: the offsets are quantiles of those folds' own errors, so coverage on
them is close to the nominal level by construction and says nothing about new data. This
module splits the backtest folds in time order, fits interval offsets on the earlier half
and reports realized coverage and width on the later, held-out half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecastkit.backtest import BacktestResult, rolling_origin
from forecastkit.intervals import interval_offsets
from forecastkit.metrics import coverage
from forecastkit.models.registry import build_model
from forecastkit.types import FloatArray


@dataclass(frozen=True)
class CalibrationReport:
    model: str
    level: float
    calibration_folds: int
    evaluation_folds: int
    coverage_by_horizon: FloatArray
    width_by_horizon: FloatArray

    @property
    def overall_coverage(self) -> float:
        return float(np.mean(self.coverage_by_horizon))

    @property
    def coverage_gap(self) -> float:
        """Empirical minus nominal coverage; negative means intervals are too narrow."""
        return self.overall_coverage - self.level


def calibration_report(
    y: FloatArray,
    model: str,
    horizon: int,
    season: int = 1,
    level: float = 0.8,
    initial: int | None = None,
    step: int | None = None,
) -> CalibrationReport:
    """Backtest ``model``, then measure how well its intervals calibrate out-of-sample.

    Needs at least 6 backtest folds: the earlier half fits the interval offsets, the later
    half is held out to measure realized coverage and width per horizon step.
    """
    start = initial if initial is not None else max(2 * season + 2, y.size // 2)
    stride = step if step is not None else max(1, horizon // 2)
    result = rolling_origin(y, lambda: build_model(model, season), horizon, start, stride)
    n_folds = len(result.folds)
    split = n_folds // 2
    if split < 3 or n_folds - split < 3:
        raise ValueError("need at least 6 backtest folds to split into calibration and evaluation")
    calib = BacktestResult(model=result.model, folds=result.folds[:split])
    evaln = BacktestResult(model=result.model, folds=result.folds[split:])
    lower_offset, upper_offset = interval_offsets(calib, level)
    actual = np.vstack([f.actual for f in evaln.folds])
    forecast = np.vstack([f.forecast for f in evaln.folds])
    lower = forecast + lower_offset
    upper = forecast + upper_offset
    coverage_by_horizon = np.array(
        [coverage(actual[:, h], lower[:, h], upper[:, h]) for h in range(horizon)]
    )
    width_by_horizon: FloatArray = np.mean(upper - lower, axis=0)
    return CalibrationReport(
        model=result.model,
        level=level,
        calibration_folds=len(calib.folds),
        evaluation_folds=len(evaln.folds),
        coverage_by_horizon=coverage_by_horizon,
        width_by_horizon=width_by_horizon,
    )


def format_calibration_report(report: CalibrationReport) -> str:
    """Render a calibration report as a plain-text table."""
    lines = [
        f"calibration report for {report.model} "
        f"(nominal level {report.level:.2f}, "
        f"{report.calibration_folds} calibration / {report.evaluation_folds} evaluation folds)",
        f"{'h':>3} {'coverage':>9} {'width':>10}",
    ]
    for h, (cov, width) in enumerate(
        zip(report.coverage_by_horizon, report.width_by_horizon, strict=True), start=1
    ):
        lines.append(f"{h:>3} {cov:>9.2f} {width:>10.3f}")
    lines.append(
        f"overall coverage {report.overall_coverage:.2f} "
        f"(nominal {report.level:.2f}, gap {report.coverage_gap:+.2f})"
    )
    return "\n".join(lines)
