import numpy as np
import pytest

from forecastkit.data import make_synthetic
from forecastkit.metrics import coverage, mae, mase, pinball, rmse, smape
from forecastkit.models.baselines import Drift, MovingAverage, Naive, SeasonalNaive
from forecastkit.models.holtwinters import HoltWinters
from forecastkit.models.registry import MODEL_NAMES, build_model
from forecastkit.models.ridge import RidgeLag
from forecastkit.types import as_series


def test_naive_repeats_last_value() -> None:
    m = Naive()
    m.fit(np.array([1.0, 2.0, 3.0]))
    assert m.predict(3).tolist() == [3.0, 3.0, 3.0]


def test_seasonal_naive_repeats_last_season() -> None:
    m = SeasonalNaive(3)
    m.fit(np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert m.predict(5).tolist() == [4.0, 5.0, 6.0, 4.0, 5.0]


def test_drift_extends_trend() -> None:
    m = Drift()
    m.fit(np.array([0.0, 1.0, 2.0, 3.0]))
    assert m.predict(2).tolist() == [4.0, 5.0]


def test_moving_average() -> None:
    m = MovingAverage(window=2)
    m.fit(np.array([1.0, 3.0, 5.0]))
    assert m.predict(2).tolist() == [4.0, 4.0]


def test_unfitted_model_raises() -> None:
    with pytest.raises(RuntimeError):
        Naive().predict(1)


def test_holt_winters_recovers_seasonal_pattern() -> None:
    y = make_synthetic(120, season=12, noise=0.2, seed=1)
    m = HoltWinters(season=12)
    m.fit(y[:108])
    pred = m.predict(12)
    assert mae(y[108:], pred) < 1.5


def test_holt_winters_requires_two_seasons() -> None:
    with pytest.raises(ValueError):
        HoltWinters(season=12).fit(np.arange(10, dtype=float))


def test_ridge_lag_tracks_seasonal_series() -> None:
    y = make_synthetic(160, season=12, noise=0.2, seed=2)
    m = RidgeLag(lags=12)
    m.fit(y[:148])
    assert mae(y[148:], m.predict(12)) < 2.0


def test_registry_builds_every_name_and_rejects_unknown() -> None:
    for name in MODEL_NAMES:
        assert build_model(name, season=4).name == name
    with pytest.raises(ValueError):
        build_model("nope")


def test_metrics_known_values() -> None:
    a = np.array([1.0, 2.0, 3.0])
    f = np.array([1.0, 3.0, 5.0])
    assert mae(a, f) == pytest.approx(1.0)
    assert rmse(a, f) == pytest.approx(np.sqrt(5 / 3))
    assert smape(np.array([0.0]), np.array([0.0])) == 0.0
    assert mase(a, f, np.array([1.0, 2.0, 3.0, 4.0])) == pytest.approx(1.0)
    assert pinball(a, f, 0.5) == pytest.approx(0.5)
    assert coverage(a, a - 0.1, a + 0.1) == 1.0


def test_as_series_validates() -> None:
    with pytest.raises(ValueError):
        as_series([])
    with pytest.raises(ValueError):
        as_series([1.0, float("nan")])
    with pytest.raises(ValueError):
        as_series([[1.0, 2.0]])
