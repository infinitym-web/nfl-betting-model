"""
THE KEY TEST: over/under bets at real FanDuel prices, using the weather
FORECAST a bettor actually had at the time of the bet (not the weather that
happened).

    python3 fetch_forecasts.py        # first: historical forecasts
    python3 run_forecast_backtest.py

Bet time      FanDuel line     Weather used
  Monday      Mon 10am line    forecast issued ~6 days before kickoff
  Thursday    Thu 11am line    forecast issued ~3 days before
  Saturday    Sat 10am line    forecast issued ~1 day before
  Sunday      Sun 12:45pm      forecast issued ~1 day before

Each row is run twice on exactly the same games: with RECORDED weather (the
earlier, flattered version) and with the FORECAST. The gap between the two is
how much of the edge came from hindsight.
The model itself is still trained on recorded weather from earlier seasons -
that's realistic: you learn how weather matters from history, then bet on
the forecast.
"""
import numpy as np
import pandas as pd

from run_backtest import load_all, START
from run_odds_backtest import load_odds, with_line
from tool.backtest import find_total_bets, simulate_bankroll, summarize
from tool.features import TOTALS_FEATURES, environment_features
from tool.model import _fit, add_model_columns, predict_dev

LEAD = {"open": 6, "mid": 3, "late": 1, "close": 1}
NAMES = {"open": "Monday", "mid": "Thursday", "late": "Saturday", "close": "Sunday"}
SEASONS = range(2020, 2027)
EDGE = 0.03


def run(feats, fd, fc, snap, use_forecast):
    played = feats[feats.result.notna()].set_index("game_id")
    snap_fd = fd[fd.snapshot == snap].set_index("game_id")
    lead_fc = fc[fc.lead_days == LEAD[snap]].set_index("game_id")
    ok = feats.total.notna() & feats.total_line.notna()
    bets = []
    for season in SEASONS:
        test = played[played.season == season].join(snap_fd, how="inner")
        test = test.dropna(subset=["point_over", "price_over", "price_under"])
        # Same games in both versions: outdoor games need a forecast to be included
        outdoor = test.is_indoor == 0
        test = test[~outdoor | test.index.isin(lead_fc.index)]
        if test.empty:
            continue
        if use_forecast:
            for c in ("w_temp", "w_wind", "w_precip", "w_snow", "w_gust"):
                test.loc[test.is_indoor == 0, c] = lead_fc[c].reindex(test.index[test.is_indoor == 0])
            test = environment_features(test)
        test = test.reset_index()
        tr = feats[ok & (feats.season >= 2002) & (feats.season < season)]
        model, sd = _fit(add_model_columns(tr), TOTALS_FEATURES, "total", "total_line")
        x = with_line(test, total=test.point_over)
        x["pred_total"] = x.total_line + predict_dev(model, x[TOTALS_FEATURES])
        x["over_odds"], x["under_odds"], x["sd"] = x.price_over, x.price_under, sd
        b = find_total_bets(x, min_edge=EDGE)
        if b.empty:
            continue
        key = x.assign(game=x.away_team + " @ " + x.home_team)[["gameday", "game", "game_id", "wet", "windy", "is_indoor"]]
        bets.append(b.merge(key, on=["gameday", "game"], how="left"))
    if not bets:
        return None
    return simulate_bankroll(pd.concat(bets, ignore_index=True), start=START)


def describe(sim, pin_close):
    s = summarize(sim, start=START)
    line = sim.bet.str.split().str[1].astype(float)
    close = sim.game_id.map(pin_close.point_over)
    clv = pd.Series(np.where(sim.side == "under", line - close, close - line)).dropna()
    bad = (sim.is_indoor == 0) & ((sim.wet > 0) | (sim.windy > 0))
    bw = sim[bad]
    return dict(bets=s["bets"], win=s["win_rate"], roi=s["roi"], lo=s["roi_90pct_range"][0],
                hi=s["roi_90pct_range"][1], clv=clv.mean(), beat=(clv > 0).mean(),
                bad_bets=len(bw), bad_roi=bw.profit.sum() / bw.stake.sum() if len(bw) else np.nan)


def main():
    feats, _, _ = load_all()
    fc = pd.read_csv("data/weather_forecast.csv")
    odds = load_odds(feats)
    fd, pin = odds["fanduel"], odds["pinnacle"]
    pin_close = pin[pin.snapshot == "close"].set_index("game_id")
    print(f"Forecasts available for {fc.game_id.nunique():,} outdoor games\n")

    # How good were the forecasts? (bad-weather calls vs what happened)
    rec = feats.set_index("game_id")
    print("FORECAST ACCURACY vs recorded weather (outdoor games)")
    for d in (6, 3, 1):
        f = fc[fc.lead_days == d].set_index("game_id").join(rec[["w_temp", "w_wind", "w_precip"]], rsuffix="_rec").dropna()
        rain_f, rain_r = f.w_precip > 0.05, f.w_precip_rec > 0.05
        wind_f, wind_r = f.w_wind > 10, f.w_wind_rec > 10
        print(f"  {d} day(s) ahead: temp off by {np.abs(f.w_temp - f.w_temp_rec).mean():.1f}F, "
              f"wind off by {np.abs(f.w_wind - f.w_wind_rec).mean():.1f} mph, "
              f"rain called right {((rain_f == rain_r).mean()):.0%} "
              f"(caught {rain_f[rain_r].mean():.0%} of rainy games), "
              f"windy called right {(wind_f == wind_r).mean():.0%}")
    print()

    print(f"OVER/UNDER AT REAL FANDUEL PRICES (EV >= {EDGE:.0%}), RECORDED vs FORECAST WEATHER")
    print(f"  {'bet day':9s} {'weather':9s} {'bets':>5s} {'win%':>6s} {'ROI':>7s}  {'luck range':>17s} "
          f"{'CLV pts':>7s} {'beat close':>10s}  {'bad-wx bets':>11s} {'bad-wx ROI':>10s}")
    for snap in ("open", "mid", "late", "close"):
        for use_fc in (False, True):
            sim = run(feats, fd, fc, snap, use_fc)
            if sim is None:
                continue
            r = describe(sim, pin_close)
            print(f"  {NAMES[snap]:9s} {'forecast' if use_fc else 'recorded':9s} {r['bets']:5d} {r['win']:6.1%} "
                  f"{r['roi']:+7.1%}  [{r['lo']:+6.1%}, {r['hi']:+6.1%}] {r['clv']:+7.2f} {r['beat']:10.0%}  "
                  f"{r['bad_bets']:11d} {r['bad_roi']:+10.1%}")
            if use_fc:
                sim.to_csv(f"output/forecast_bets_{NAMES[snap].lower()}.csv", index=False)
        print()
    print("CLV pts = average points our line beat Pinnacle's closing line by (positive = good).")


if __name__ == "__main__":
    main()
