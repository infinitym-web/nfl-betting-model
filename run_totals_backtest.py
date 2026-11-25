"""
Backtest the OVER/UNDER model on 2010-2025.

    python3 run_totals_backtest.py
    python3 run_totals_backtest.py --edge 0.02

Models compared
  t0   simple: rolling points scored/allowed only (raw, NOT anchored to market)
  t1   full model WITHOUT environment/weather features
  t2   full model WITH environment/weather features
Same rules as the spread backtest: walk-forward (season S is predicted by a
model fitted only on seasons before S), predictions are "how far from the
posted total", and bet probabilities start from the market's own price.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from run_backtest import load_all, TEST_SEASONS, START
from tool.backtest import find_total_bets, simulate_bankroll, summarize
from tool.features import TOTALS_FEATURES, ENVIRONMENT_FEATURES
from tool.model import walk_forward, explain

HERE = Path(__file__).parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.03)
    args = ap.parse_args()

    feats, _, _ = load_all()
    test = feats[feats.season.isin(TEST_SEASONS) & feats.total.notna() & feats.total_line.notna()]
    market_mae = (test.total - test.total_line).abs().mean()

    models = {}
    t0 = test.copy()
    t0["pred_total"] = t0.pts_model
    t0["sd"] = float((t0.total - t0.pts_model).std())
    models["t0  points avg (raw)"] = t0
    no_env = [f for f in TOTALS_FEATURES if f not in ENVIRONMENT_FEATURES]
    kw = dict(target="total", line="total_line", out_col="pred_total")
    models["t1  full, no weather"], _ = walk_forward(feats, no_env, TEST_SEASONS, **kw)
    models["t2  full + weather"], last = walk_forward(feats, TOTALS_FEATURES, TEST_SEASONS, **kw)

    print(f"Test seasons {TEST_SEASONS.start}-{TEST_SEASONS.stop-1}, {len(test)} games. "
          f"Market total's average miss: {market_mae:.3f} pts\n")
    print("ACCURACY (average miss on combined points, lower is better)")
    for name, p in models.items():
        print(f"  {name:22s} {(p.total - p.pred_total).abs().mean():.3f} pts")

    print("\nBETTING RESULTS by edge threshold (ROI on money staked; 90% luck range in brackets)")
    print(f"  {'model':22s} {'edge':>5s} {'bets':>5s} {'win%':>6s} {'ROI':>7s}  {'luck range':>17s} {'final $':>9s}")
    sims = {}
    for name, p in models.items():
        for edge in (0.01, 0.02, 0.03, 0.05):
            bets = find_total_bets(p, min_edge=edge, anchor_to_market=not name.startswith("t0"))
            if bets.empty:
                print(f"  {name:22s} {edge:5.0%}     0")
                continue
            sim = simulate_bankroll(bets, start=START)
            s = summarize(sim, start=START)
            lo, hi = s["roi_90pct_range"]
            print(f"  {name:22s} {edge:5.0%} {s['bets']:5d} {s['win_rate']:6.1%} {s['roi']:+7.1%}  "
                  f"[{lo:+6.1%}, {hi:+6.1%}] {s['final_bankroll']:9,.0f}")
            if abs(edge - args.edge) < 1e-9:
                sims[name] = sim
        print()

    best = sims.get("t2  full + weather")
    if best is not None and not best.empty:
        print(f"FULL + WEATHER bets at edge >= {args.edge:.0%}, split by side")
        print(best.groupby("side").agg(bets=("profit", "size"),
                                       win_rate=("outcome", lambda o: (o == "win").mean()),
                                       roi=("profit", lambda x: x.sum()))
              .assign(roi=lambda d: d.roi / best.groupby("side").stake.sum())
              .to_string(formatters={"win_rate": "{:.1%}".format, "roi": "{:+.1%}".format}), "\n")

    print("WHAT THE FULL MODEL LEARNED (fit through 2024; points vs the posted total per 1 std-dev)")
    print("  positive = pushes toward the OVER")
    print(explain(last, TOTALS_FEATURES).to_string(), "\n")

    out = HERE / "output"
    out.mkdir(exist_ok=True)
    if best is not None:
        best.to_csv(out / "totals_backtest_bets.csv", index=False)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, sm in sims.items():
            d = pd.to_datetime(sm.gameday)
            ax.plot(np.r_[d.iloc[0], d], np.r_[START, sm.bankroll_after], label=name.strip())
        ax.axhline(START, color="gray", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"Over/under simulated bankroll, edge ≥ {args.edge:.0%}, ¼ Kelly")
        ax.set_ylabel("Bankroll ($, log scale)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "totals_bankroll.png", dpi=150)
        print("Saved output/totals_backtest_bets.csv and output/totals_bankroll.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
