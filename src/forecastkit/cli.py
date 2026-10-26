"""Command line: backtest all models on a synthetic or CSV series and print a leaderboard."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from forecastkit.backtest import compare_models
from forecastkit.calibration import calibration_report, format_calibration_report
from forecastkit.data import load_csv, make_synthetic


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forecastkit", description=__doc__)
    parser.add_argument("--csv", type=Path, help="CSV file with a numeric column")
    parser.add_argument("--column", default="value", help="column to read from the CSV")
    parser.add_argument("--n", type=int, default=240, help="length of the synthetic series")
    parser.add_argument("--season", type=int, default=12)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="also print an out-of-sample interval calibration report",
    )
    parser.add_argument(
        "--model",
        default="auto",
        help="model for --calibrate; 'auto' uses the leaderboard winner",
    )
    parser.add_argument("--level", type=float, default=0.8, help="nominal interval level")
    args = parser.parse_args(argv)

    y = (
        load_csv(args.csv, args.column)
        if args.csv
        else make_synthetic(args.n, season=args.season, seed=args.seed)
    )
    results = compare_models(y, args.horizon, args.season)
    print(f"{'model':<16} {'MAE':>8} {'sMAPE%':>8} {'folds':>6}")
    for r in results:
        print(f"{r.model:<16} {r.mae:>8.3f} {r.smape:>8.2f} {len(r.folds):>6}")

    if args.calibrate:
        model_name = args.model if args.model != "auto" else results[0].model
        print()
        try:
            report = calibration_report(y, model_name, args.horizon, args.season, args.level)
        except ValueError as exc:
            print(f"calibration report unavailable: {exc}")
        else:
            print(format_calibration_report(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
