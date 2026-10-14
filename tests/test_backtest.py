import numpy as np
import pytest

from forecastkit.backtest import compare_models, rolling_origin
from forecastkit.data import make_synthetic
from forecastkit.intervals import interval_offsets, with_intervals
from forecastkit.metrics import coverage
from forecastkit.models.baselines import Naive
from forecastkit.models.registry import build_model
from forecastkit.service import forecast_series


def test_rolling_origin_fold_count_and_shapes() -> None:
    y = np.arange(30, dtype=float)
    result = rolling_origin(y, Naive, horizon=3, initial=10, step=2)
    assert len(result.folds) == len(range(10, 28, 2))
    assert result.errors.shape == (len(result.folds), 3)


def test_rolling_origin_never_peeks_at_the_future() -> None:
    y = np.arange(20, dtype=float)
    result = rolling_origin(y, Naive, horizon=2, initial=10)
    for fold in result.folds:
        assert fold.forecast[0] == y[fold.origin - 1]


def test_rolling_origin_rejects_short_series() -> None:
    with pytest.raises(ValueError):
        rolling_origin(np.arange(5, dtype=float), Naive, horizon=3, initial=4)


def test_seasonal_models_beat_naive_on_seasonal_data() -> None:
    y = make_synthetic(240, season=12, noise=0.5, seed=3)
    results = compare_models(y, horizon=12, season=12)
    ranking = [r.model for r in results]
    assert ranking.index("holt_winters") < ranking.index("naive")
    assert ranking[0] != "naive"


def test_intervals_widen_or_hold_with_horizon_and_cover() -> None:
    y = make_synthetic(400, season=12, noise=1.0, seed=4)
    train, test = y[:360], y[360:372]
    result = rolling_origin(
        train, lambda: build_model("holt_winters", 12), horizon=12, initial=120, step=6
    )
    lower, upper = interval_offsets(result, level=0.8)
    assert np.all(upper >= lower)
    model = build_model("holt_winters", 12)
    model.fit(train)
    lo, hi = with_intervals(model.predict(12), result, level=0.8)
    assert coverage(test, lo, hi) >= 0.5


def test_empirical_coverage_is_near_nominal_on_stationary_noise() -> None:
    rng = np.random.default_rng(5)
    y = 10.0 + rng.normal(0, 1, size=600)
    result = rolling_origin(y[:500], Naive, horizon=1, initial=100, step=1)
    lo, hi = with_intervals(np.array([y[499]]), result, level=0.8)
    inside = [float(lo[0]) <= float(v) <= float(hi[0]) for v in y[500:] - 0.0]
    assert 0.6 <= float(np.mean(inside)) <= 0.95


def test_interval_validation() -> None:
    y = make_synthetic(60, season=1, seed=6)
    result = rolling_origin(y, Naive, horizon=2, initial=50, step=1)
    with pytest.raises(ValueError):
        interval_offsets(result, level=1.5)
    short = rolling_origin(y, Naive, horizon=2, initial=57, step=1)
    with pytest.raises(ValueError):
        interval_offsets(short)


def test_forecast_series_auto_returns_consistent_output() -> None:
    y = make_synthetic(180, season=12, noise=0.5, seed=7)
    out = forecast_series(y, horizon=6, season=12)
    assert out.forecast.shape == out.lower.shape == out.upper.shape == (6,)
    assert np.all(out.lower <= out.upper)
    assert out.folds >= 3
