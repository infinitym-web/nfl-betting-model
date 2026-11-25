"""
Do alternate lines beat main lines? Backtest 2010-2025.

    python3 run_alt_backtest.py

For both markets (spreads and over/unders) and several assumed book
margins ("hold"), it compares:
  main only    - the posted line, priced from history with the hold
  alt ladder   - every line from -10 to +10 points; take the single best bet per game
See tool/alt_lines.py for how prices are built (no real alt-price history exists here).
"""
from run_backtest import load_all, TEST_SEASONS, START
from tool.alt_lines import alt_ladder_bets
from tool.backtest import simulate_bankroll, summarize
from tool.features import STEP2_FEATURES, TOTALS_FEATURES
from tool.model import walk_forward

EDGE = 0.03
HOLDS = (0.045, 0.07, 0.10)


def main():
    feats, _, _ = load_all()
    totals, _ = walk_forward(feats, TOTALS_FEATURES, TEST_SEASONS,
                             target="total", line="total_line", out_col="pred_total")
    spreads, _ = walk_forward(feats, STEP2_FEATURES, TEST_SEASONS)
    markets = {
        "Over/under": dict(preds=totals, line_col="total_line", actual_col="total",
                           pred_col="pred_total", labels=("over", "under")),
        "Spread": dict(preds=spreads, line_col="spread_line", actual_col="result",
                       pred_col="pred_margin", labels=("home", "away")),
    }
    print(f"Bets need EV >= {EDGE:.0%}. Quarter Kelly, max 3% per bet, $1,000 start.\n")
    print(f"{'market':11s} {'hold':>5s}  {'lines':11s} {'bets':>5s} {'win%':>6s} {'avg odds':>9s} "
          f"{'ROI':>7s}  {'luck range':>17s} {'final $':>9s} {'worst dip':>9s}")
    keep = {}
    for mname, m in markets.items():
        for hold in HOLDS:
            for lname, offs in (("main only", [0]), ("alt ladder", range(-10, 11))):
                bets = alt_ladder_bets(seasons_all=feats, hold=hold, min_edge=EDGE, offsets=offs, **m)
                if bets.empty:
                    print(f"{mname:11s} {hold:5.1%}  {lname:11s}     0")
                    continue
                sim = simulate_bankroll(bets, start=START)
                s = summarize(sim, start=START)
                lo, hi = s["roi_90pct_range"]
                print(f"{mname:11s} {hold:5.1%}  {lname:11s} {s['bets']:5d} {s['win_rate']:6.1%} "
                      f"{bets.odds.median():+9.0f} {s['roi']:+7.1%}  [{lo:+6.1%}, {hi:+6.1%}] "
                      f"{s['final_bankroll']:9,.0f} {s['max_drawdown']:9.0%}")
                keep[(mname, hold, lname)] = sim
        print()

    sim = keep.get(("Over/under", 0.07, "alt ladder"))
    if sim is not None:
        print("Over/under alt ladder at 7% hold: which lines it picked")
        sim = sim.assign(dist=sim.offset.abs().clip(upper=10)
                         .map(lambda d: "main line" if d == 0 else ("1-3 pts away" if d <= 3 else
                                                                     ("4-6 pts away" if d <= 6 else "7-10 pts away"))))
        t = sim.groupby("dist").agg(bets=("profit", "size"), staked=("stake", "sum"), profit=("profit", "sum"),
                                    win_rate=("outcome", lambda o: (o == "win").mean()))
        t["roi"] = (t.profit / t.staked).map("{:+.1%}".format)
        t["win_rate"] = t.win_rate.map("{:.1%}".format)
        print(t[["bets", "win_rate", "roi"]].to_string(), "\n")
        print("By side:", sim.side.value_counts().to_dict())
        sim.to_csv("output/alt_totals_bets.csv", index=False)


if __name__ == "__main__":
    main()
