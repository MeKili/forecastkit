import io
from contextlib import redirect_stdout
from pathlib import Path

from fastapi.testclient import TestClient

from forecastkit.api import app
from forecastkit.cli import main
from forecastkit.data import load_csv, make_synthetic

client = TestClient(app)


def test_health_and_models() -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert "holt_winters" in client.get("/models").json()["models"]


def test_forecast_endpoint_happy_path() -> None:
    values = [float(v) for v in make_synthetic(120, season=12, seed=8)]
    resp = client.post(
        "/forecast", json={"values": values, "horizon": 6, "season": 12, "level": 0.8}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["forecast"]) == len(body["lower"]) == len(body["upper"]) == 6
    assert all(lo <= hi for lo, hi in zip(body["lower"], body["upper"], strict=True))


def test_forecast_endpoint_rejects_bad_input() -> None:
    too_short = client.post("/forecast", json={"values": [1.0, 2.0], "horizon": 3})
    assert too_short.status_code == 422
    unknown = client.post("/forecast", json={"values": [float(i) for i in range(40)], "model": "x"})
    assert unknown.status_code == 422
    series_too_short = client.post(
        "/forecast", json={"values": [float(i) for i in range(10)], "horizon": 30}
    )
    assert series_too_short.status_code == 422


def test_cli_prints_leaderboard() -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = main(["--n", "120", "--season", "12", "--horizon", "6"])
    assert code == 0
    assert "holt_winters" in buffer.getvalue()


def test_load_csv(tmp_path: Path) -> None:
    path = tmp_path / "s.csv"
    path.write_text("date,value\n2026-01-01,1.5\n2026-01-02,2.5\n", encoding="utf-8")
    assert load_csv(path, "value").tolist() == [1.5, 2.5]
