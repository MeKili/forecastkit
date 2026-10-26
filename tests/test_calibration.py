import numpy as np
import pytest

from forecastkit.calibration import calibration_report, format_calibration_report
from forecastkit.data import make_synthetic


def test_calibration_report_shapes_and_fold_split() -> None:
    rng = np.random.default_rng(1)
    y = 10.0 + rng.normal(0, 1, size=200)
    report = calibration_report(y, "naive", horizon=3, initial=100, step=1)
    assert report.coverage_by_horizon.shape == (3,)
    assert report.width_by_horizon.shape == (3,)
    assert report.calibration_folds >= 3
    assert report.evaluation_folds >= 3
    assert report.calibration_folds <= report.evaluation_folds
    assert np.all(report.width_by_horizon >= 0.0)


def test_calibration_report_is_near_nominal_on_stationary_noise() -> None:
    rng = np.random.default_rng(5)
    y = 10.0 + rng.normal(0, 1, size=700)
    report = calibration_report(y, "naive", horizon=1, season=1, level=0.8, initial=100, step=1)
    assert 0.6 <= report.overall_coverage <= 0.95
    assert abs(report.coverage_gap) < 0.3


def test_calibration_report_rejects_too_few_folds() -> None:
    y = make_synthetic(40, season=1, seed=2)
    with pytest.raises(ValueError):
        calibration_report(y, "naive", horizon=2, initial=35, step=1)


def test_calibration_report_rejects_bad_level() -> None:
    y = make_synthetic(240, season=12, seed=3)
    with pytest.raises(ValueError):
        calibration_report(y, "naive", horizon=3, season=12, level=1.5)


def test_calibration_report_properties_match_their_arrays() -> None:
    y = make_synthetic(240, season=12, seed=4)
    report = calibration_report(y, "holt_winters", horizon=4, season=12, level=0.8)
    assert report.model == "holt_winters"
    assert report.overall_coverage == pytest.approx(float(np.mean(report.coverage_by_horizon)))
    assert report.coverage_gap == pytest.approx(report.overall_coverage - report.level)


def test_format_calibration_report_contains_expected_rows() -> None:
    y = make_synthetic(240, season=12, seed=6)
    report = calibration_report(y, "holt_winters", horizon=4, season=12, level=0.8)
    text = format_calibration_report(report)
    assert "calibration report for holt_winters" in text
    assert "overall coverage" in text
    assert len(text.splitlines()) == len(report.coverage_by_horizon) + 3
