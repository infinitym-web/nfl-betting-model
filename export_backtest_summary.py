"""
Backtest summary for the dashboard: FanDuel's real prices with NOAA forecasts,
2020-2026, UNDER bets only, flat 1-unit stakes (same staking as the paper log).

    python3 run_forecast_backtest.py   # writes output/forecast_bets_<day>.csv (model)
    python3 export_backtest_summary.py

Writes output/dashboard/backtest_summary.csv and backtest_curve.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_all
from run_odds_backtest import load_odds
from tool.odds import american_to_decimal

HERE = Path(__file__).parent
OUT = HERE / "output" / "dashboard"
DAYS = {"open": "Monday", "mid": "Thursday", "late": "Saturday", "close": "Sunday"}
RULE_WIND, RULE_RAIN = 15.0, 0.05


def flat(outcome, odds):
    return np.where(outcome == "win", odds.map(american_to_decimal) - 1, np.where(outcome == "push", 0.0, -1.0))


def main():
    rows, curves = [], []
    # Model (unders only) from the forecast backtest's saved bets
    for snap, day in DAYS.items():
        s = pd.read_csv(HERE / f"output/forecast_bets_{day.lower()}.csv")
        s = s[s.side == "under"].copy()
        s["units"] = flat(s.outcome, s.odds)
        s["strategy"] = "Model"
        s["bet_day"] = day
        curves.append(s[["gameday", "season", "strategy", "bet_day", "units"]])

    # Rule: under at FanDuel's price when the NOAA forecast says wind > 15 mph or rain
    feats, _, _ = load_all()
    fd = load_odds(feats)["fanduel"]
    fc = pd.read_csv(HERE / "data" / "weather_forecast_mos.csv")
    played = feats[feats.total.notna()].set_index("game_id")
    for snap, day in DAYS.items():
        x = fd[fd.snapshot == snap].set_index("game_id").join(
            played[["total", "is_indoor", "gameday", "season"]], how="inner")
        x = x[(x.is_indoor == 0)].dropna(subset=["point_under", "price_under"])
        x = x.join(fc[fc.snapshot == snap].set_index("game_id")[["w_wind", "w_precip"]], how="inner")
        x = x[(x.w_wind > RULE_WIND) | (x.w_precip > RULE_RAIN)]
        outcome = np.where(x.total < x.point_under, "win", np.where(x.total == x.point_under, "push", "loss"))
        x["units"] = flat(pd.Series(outcome, index=x.index), x.price_under)
        x["strategy"] = "Rule"
        x["bet_day"] = day
        curves.append(x.reset_index()[["gameday", "season", "strategy", "bet_day", "units"]])

    allb = pd.concat(curves, ignore_index=True)
    for (strategy, day), d in allb.groupby(["strategy", "bet_day"], sort=False):
        by_season = d.groupby("season").units.sum()
        rows.append(dict(strategy=strategy, bet_day=day, bets=len(d),
                         win_rate=round(float((d.units > 0).mean()), 4),
                         roi=round(float(d.units.mean()), 4),
                         units=round(float(d.units.sum()), 1),
                         seasons_won=f"{int((by_season > 0).sum())} of {len(by_season)}"))
    OUT.mkdir(parents=True, exist_ok=True)
    order = {d: i for i, d in enumerate(DAYS.values())}
    summ = pd.DataFrame(rows).sort_values(["strategy", "bet_day"], key=lambda c: c.map(order) if c.name == "bet_day" else c)
    summ.to_csv(OUT / "backtest_summary.csv", index=False)

    # Cumulative units over time for the headline strategies
    keep = [("Model", "Monday"), ("Model", "Thursday"), ("Rule", "Thursday"), ("Rule", "Saturday")]
    cur = []
    for strategy, day in keep:
        d = allb[(allb.strategy == strategy) & (allb.bet_day == day)].sort_values("gameday")
        d = d.groupby("gameday").units.sum().cumsum().reset_index()
        d["series"] = f"{strategy}, {day}"
        cur.append(d.rename(columns={"units": "cumulative_units", "gameday": "date"}))
    pd.concat(cur)[["series", "date", "cumulative_units"]].round(2).to_csv(OUT / "backtest_curve.csv", index=False)
    print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
