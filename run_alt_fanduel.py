"""
Test the alternate-under strategy against FanDuel's REAL alternate-total prices
(2023-now, Saturday 10am snapshot).

    python3 fetch_odds.py && python3 fetch_alt_odds.py   # downloads first
    python3 run_alt_fanduel.py

Reports:
  * FanDuel's actual margin ("hold") on alternate totals, by distance from the main line
  * The strategy's results at those real prices, bad-weather games vs others
"""
import numpy as np
import pandas as pd

from run_backtest import load_all, START
from run_odds_backtest import load_odds, with_line
from tool.alt_lines import _pool
from tool.backtest import simulate_bankroll, summarize
from tool.features import TOTALS_FEATURES
from tool.model import _fit, add_model_columns, predict_dev
from tool.odds import american_to_decimal

EDGE = 0.03


def main():
    feats, _, _ = load_all()
    alt = pd.read_csv("data/fanduel_alt_totals.csv")
    fd = load_odds(feats)["fanduel"]
    late = fd[fd.snapshot == "late"].set_index("game_id")
    played = feats[feats.total.notna()].set_index("game_id")
    games = played.join(late[["point_over"]], how="inner").loc[lambda d: d.index.isin(alt.game_id)]
    print(f"{len(games)} outdoor games with FanDuel alternate totals\n")

    # FanDuel hold on alternate totals
    p = alt.pivot_table(index=["game_id", "point"], columns="side", values="price").dropna().reset_index()
    p = p.join(games[["point_over"]], on="game_id")
    p["dist"] = (p.point - p.point_over).abs()
    p["hold"] = 1 / p.over.map(american_to_decimal) + 1 / p.under.map(american_to_decimal) - 1
    p["bucket"] = pd.cut(p.dist, [-0.1, 0.5, 3, 6, 10, 30], labels=["main", "1-3", "4-6", "7-10", "10+"])
    print("FANDUEL'S MARGIN ON ALTERNATE TOTALS (my earlier stand-in assumed 4.5-20%)")
    print(p.groupby("bucket", observed=True).hold.agg(["size", "median"])
          .rename(columns={"size": "prices", "median": "median hold"})
          .to_string(formatters={"median hold": "{:.1%}".format}), "\n")

    # Strategy at real prices
    hist = feats[feats.total.notna() & feats.total_line.notna() & (feats.season >= 2002)]
    rows = []
    for season, test in games.groupby("season"):
        tr = hist[hist.season < season]
        model, sd = _fit(add_model_columns(tr), TOTALS_FEATURES, "total", "total_line")
        x = with_line(test.reset_index(), total=test.point_over.values)
        x["shift"] = predict_dev(model, x[TOTALS_FEATURES])
        tl, ta = tr.total_line.to_numpy(), tr.total.to_numpy()
        for g in x.itertuples():
            pool = _pool(tl, ta, g.total_line)
            best = None
            for a in alt[(alt.game_id == g.game_id) & (alt.side == "under")].itertuples():
                L = a.point
                p_over, p_push = (pool > L).mean(), (pool == L).mean()
                if min(p_over, 1 - p_over - p_push) < 0.04 or L > g.total_line - 4:
                    continue
                dens = ((pool > L - 1.5).mean() - (pool > L + 1.5).mean()) / 3
                q_under = 1 - np.clip(p_over + dens * g.shift, 0.01, 0.99 - p_push) - p_push
                dec = american_to_decimal(a.price)
                ev = q_under * (dec - 1) - (1 - q_under - p_push)
                if best is None or ev > best[0]:
                    best = (ev, L, a.price, dec, q_under, p_push)
            if best is None or best[0] < EDGE:
                continue
            ev, L, price, dec, q, pp = best
            pc = q / (1 - pp)
            kelly = max((pc * (dec - 1) - (1 - pc)) / (dec - 1), 0)
            outcome = "push" if g.total == L else ("win" if g.total < L else "loss")
            rows.append(dict(season=g.season, week=g.week, gameday=g.gameday, game_id=g.game_id,
                             bet=f"UNDER {L:g}", main=g.total_line, odds=price, ev=ev, kelly=kelly,
                             outcome=outcome,
                             env="bad weather" if (g.wet > 0 or g.windy > 0) else "normal"))
    bets = pd.DataFrame(rows)
    if bets.empty:
        print("No alternate-under bets qualified at FanDuel's real prices.")
        return
    sim = simulate_bankroll(bets, start=START)
    s = summarize(sim, start=START)
    lo, hi = s["roi_90pct_range"]
    print(f"ALTERNATE-UNDER STRATEGY AT FANDUEL'S REAL PRICES (EV >= {EDGE:.0%}, Saturday lines)")
    print(f"  bets {s['bets']}, win {s['win_rate']:.1%}, median odds {bets.odds.median():+.0f}, "
          f"ROI {s['roi']:+.1%} [luck range {lo:+.1%} to {hi:+.1%}], ${START:,.0f} -> ${s['final_bankroll']:,.0f}")
    print(sim.groupby("env").apply(lambda d: pd.Series({
        "bets": len(d), "win%": f"{(d.outcome == 'win').mean():.1%}",
        "ROI": f"{d.profit.sum() / d.stake.sum():+.1%}"})).to_string())
    sim.to_csv("output/fanduel_alt_under_bets.csv", index=False)


if __name__ == "__main__":
    main()
