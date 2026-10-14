"""FastAPI service: POST a series, get a forecast with prediction intervals."""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from forecastkit import __version__
from forecastkit.models.registry import MODEL_NAMES
from forecastkit.service import forecast_series
from forecastkit.types import as_series

MAX_POINTS = int(os.environ.get("FORECASTKIT_MAX_POINTS", "5000"))


class ForecastRequest(BaseModel):
    values: list[float] = Field(min_length=8)
    horizon: int = Field(default=12, ge=1, le=365)
    season: int = Field(default=1, ge=1, le=366)
    model: str = "auto"
    level: float = Field(default=0.8, gt=0.0, lt=1.0)


class ForecastResponse(BaseModel):
    model: str
    forecast: list[float]
    lower: list[float]
    upper: list[float]
    backtest_mae: float
    backtest_smape: float
    folds: int


app = FastAPI(title="forecastkit", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.get("/models")
def models() -> dict[str, list[str]]:
    return {"models": ["auto", *MODEL_NAMES]}


@app.post("/forecast", response_model=ForecastResponse)
def forecast(req: ForecastRequest) -> ForecastResponse:
    if len(req.values) > MAX_POINTS:
        raise HTTPException(status_code=413, detail=f"at most {MAX_POINTS} points are accepted")
    if req.model != "auto" and req.model not in MODEL_NAMES:
        raise HTTPException(status_code=422, detail=f"unknown model {req.model!r}")
    try:
        out = forecast_series(as_series(req.values), req.horizon, req.season, req.model, req.level)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return ForecastResponse(
        model=out.model,
        forecast=[float(v) for v in out.forecast],
        lower=[float(v) for v in out.lower],
        upper=[float(v) for v in out.upper],
        backtest_mae=out.backtest_mae,
        backtest_smape=out.backtest_smape,
        folds=out.folds,
    )
