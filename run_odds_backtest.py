"""
Backtest against REAL FanDuel prices at different times of the week (2020-now).

    python3 fetch_odds.py          # first: download the odds history (paid API)
    python3 run_odds_backtest.py

Answers three questions:
  1. LINE MOVEMENT  Do totals move between Monday and Sunday, and do they move
                    toward the weather (down for rain/wind)? If the market
                    only adjusts late, betting earlier captures that move.
  2. PROFIT         The models' results betting FanDuel's line and price at
                    each snapshot (open/mid/late/close). Each snapshot only uses
                    the line available then - never the closing line.
  3. CLOSING LINE VALUE (CLV)  Did our bet's line beat Pinnacle's final line?
                    Beating the close is the most reliable early sign of a real
                    edge; it settles much faster than profit does.

CAVEAT: weather features are the RECORDED kickoff weather. At the Monday
snapshot a bettor only had a 6-day forecast, so "open" results are the most
flattered; Saturday ("late") is the most realistic test of the weather edge.
"""
import numpy as np
import pandas as pd

from run_backtest import load_all, START
from tool.backtest import find_bets, find_total_bets, simulate_bankroll, summarize
from tool.features import STEP2_FEATURES, TOTALS_FEATURES
from tool.model import _fit, add_model_columns, predict_dev

SNAPS = ["open", "mid", "late", "close"]
SEASONS = range(2020, 2027)


def load_odds(feats):
    o = pd.read_csv("data/odds_history.csv")
    o["gameday"] = (pd.to_datetime(o.commence_time).dt.tz_convert("America/New_York")
                    .dt.strftime("%Y-%m-%d"))
    key = feats[["game_id", "season", "home_team", "away_team", "gameday"]]
    o = o.merge(key, on=["season", "home_team", "away_team", "gameday"], how="inner")

    def side_table(book, market):
        x = o[(o.book == book) & (o.market == market)]
        x = x.sort_values("snapshot_time").groupby(["game_id", "snapshot", "side"]).last()
        return x[["point", "price"]].unstack("side")

    tables = {}
    for book in ("fanduel", "pinnacle"):
        t = side_table(book, "totals")
        t.columns = [f"{a}_{b}" for a, b in t.columns]
        s = side_table(book, "spreads")
        s.columns = [f"{a}_{b}" for a, b in s.columns]
        tables[book] = t.join(s, how="outer").reset_index()
    return tables


def with_line(df, total=None, spread=None):
    """Recompute every feature that depends on the market line."""
    df = df.copy()
    if total is not None:
        df["total_line"] = total
        df["wind_x_total"] = df.windy * df.total_line / 44
    if spread is not None:
        df["spread_line"] = spread
        df["wind_x_line"] = df.windy * df.spread_line / 7
        df["wet_x_line"] = df.wet * df.spread_line / 7
        df["div_game_x_line"] = df.div_game * df.spread_line / 7
    return add_model_columns(df)


def main():
    feats, _, _ = load_all()
    odds = load_odds(feats)
    fd, pin = odds["fanduel"], odds["pinnacle"]
    pin_close = pin[pin.snapshot == "close"].set_index("game_id")
    played = feats[feats.result.notna()].set_index("game_id")
    print(f"Matched FanDuel odds for {fd.game_id.nunique():,} games "
          f"({fd[fd.snapshot == 'open'].game_id.nunique():,} with Monday lines)\n")

    # ---------------- 1. line movement ----------------
    print("1. HOW FANDUEL TOTALS MOVE FROM MONDAY TO SUNDAY (points; negative = moved toward the under)")
    mv = fd.pivot_table(index="game_id", columns="snapshot", values="point_over")
    mv = mv.join(played[["is_indoor", "wet", "windy", "total", "season"]], how="inner").dropna(subset=["open", "close"])
    mv["move"] = mv["close"] - mv["open"]
    mv["env"] = np.select([mv.is_indoor == 1, (mv.wet > 0) | (mv.windy > 0)],
                          ["indoors", "outdoors, rain or wind>10mph"], "outdoors, normal")
    t = mv.groupby("env").agg(games=("move", "size"), avg_move=("move", "mean"),
                              open_vs_result=("open", lambda s: (mv.loc[s.index, "total"] - s).mean()),
                              close_vs_result=("close", lambda s: (mv.loc[s.index, "total"] - s).mean()))
    print(t.round(2).to_string())
    print("  (x_vs_result = actual points minus that line; closer to 0 = more accurate)\n")

    # ---------------- 2 & 3. models at each snapshot ----------------
    ok_tot = feats.total.notna() & feats.total_line.notna()
    ok_spr = feats.result.notna() & feats.spread_line.notna()
    results = []
    keep = {}
    for market in ("Over/under", "Spread"):
        for snap in SNAPS:
            snap_fd = fd[fd.snapshot == snap].set_index("game_id")
            all_bets = []
            for season in SEASONS:
                test = played[played.season == season].join(snap_fd, how="inner")
                if test.empty:
                    continue
                test = test.reset_index()
                if market == "Over/under":
                    test = test.dropna(subset=["point_over", "price_over", "price_under"])
                    tr = feats[ok_tot & (feats.season >= 2002) & (feats.season < season)]
                    model, sd = _fit(add_model_columns(tr), TOTALS_FEATURES, "total", "total_line")
                    x = with_line(test, total=test.point_over)
                    x["pred_total"] = x.total_line + predict_dev(model, x[TOTALS_FEATURES])
                    x["over_odds"], x["under_odds"], x["sd"] = x.price_over, x.price_under, sd
                    b = find_total_bets(x, min_edge=0.03)
                else:
                    test = test.dropna(subset=["point_home", "price_home", "price_away"])
                    tr = feats[ok_spr & (feats.season >= 2002) & (feats.season < season)]
                    model, sd = _fit(add_model_columns(tr), STEP2_FEATURES)
                    x = with_line(test, spread=-test.point_home)
                    x["pred_margin"] = x.spread_line + predict_dev(model, x[STEP2_FEATURES])
                    x["home_spread_odds"], x["away_spread_odds"], x["sd"] = x.price_home, x.price_away, sd
                    b = find_bets(x, sd=sd, min_edge=0.03)
                if b.empty:
                    continue
                b["game_id"] = b.merge(x[["gameday", "away_team", "home_team", "game_id"]]
                                       .assign(game=lambda d: d.away_team + " @ " + d.home_team),
                                       on=["gameday", "game"], how="left").game_id.values
                all_bets.append(b)
            if not all_bets:
                continue
            bets = pd.concat(all_bets, ignore_index=True)
            sim = simulate_bankroll(bets, start=START)
            s = summarize(sim, start=START)
            # CLV vs Pinnacle's closing line (points in our favour)
            if market == "Over/under":
                line = sim.bet.str.split().str[1].astype(float)
                close = sim.game_id.map(pin_close.point_over)
                clv = np.where(sim.side == "under", line - close, close - line)
            else:
                close_home = sim.game_id.map(pin_close.point_home)  # e.g. -3.5
                bet_line = sim.bet.str.split().str[-1].astype(float)  # our team's line
                close_side = np.where(sim.side == "home", close_home, -close_home)
                clv = bet_line - close_side
            clv = pd.Series(clv).dropna()
            results.append(dict(market=market, snapshot=snap, bets=s["bets"],
                                win=s["win_rate"], roi=s["roi"], lo=s["roi_90pct_range"][0],
                                hi=s["roi_90pct_range"][1], final=s["final_bankroll"],
                                clv=clv.mean() if len(clv) else np.nan,
                                beat=(clv > 0).mean() if len(clv) else np.nan,
                                worse=(clv < 0).mean() if len(clv) else np.nan))
            keep[(market, snap)] = sim
    print("2-3. MODEL RESULTS AT REAL FANDUEL PRICES, BY WHEN THE BET IS PLACED (EV >= 3%)")
    print(f"  {'market':10s} {'snapshot':8s} {'bets':>5s} {'win%':>6s} {'ROI':>7s}  {'luck range':>17s} "
          f"{'final $':>8s}  {'CLV pts':>7s} {'beat close':>10s} {'worse':>6s}")
    for r in results:
        print(f"  {r['market']:10s} {r['snapshot']:8s} {r['bets']:5d} {r['win']:6.1%} {r['roi']:+7.1%}  "
              f"[{r['lo']:+6.1%}, {r['hi']:+6.1%}] {r['final']:8,.0f}  {r['clv']:+7.2f} "
              f"{r['beat']:10.0%} {r['worse']:6.0%}")
    print("\n  CLV pts = average points our line beat Pinnacle's closing line by (positive = good).")
    print("  'open' uses Monday lines with RECORDED weather - a bettor only had a 6-day forecast then.")

    sim = keep.get(("Over/under", "late"))
    if sim is not None:
        sim = sim.join(played[["wet", "windy", "is_indoor"]], on="game_id")
        sim["env"] = np.where((sim.is_indoor == 0) & ((sim.wet > 0) | (sim.windy > 0)), "bad weather", "other")
        print("\nOver/under, Saturday bets, split by weather")
        print(sim.groupby("env").apply(lambda d: pd.Series({
            "bets": len(d), "win%": f"{(d.outcome == 'win').mean():.1%}",
            "ROI": f"{d.profit.sum() / d.stake.sum():+.1%}"})).to_string())
        sim.to_csv("output/fanduel_totals_saturday_bets.csv", index=False)


if __name__ == "__main__":
    main()
