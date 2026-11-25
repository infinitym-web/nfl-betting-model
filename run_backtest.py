"""
Backtest NFL spread models on 2010-2025 and compare them.

    python3 run_backtest.py               # default: bet when edge >= 3%
    python3 run_backtest.py --edge 0.02

Models compared
  v1  Elo only, raw (the first version - overconfident)
  s1  Step 1: Elo CALIBRATED against the market (learns how much to trust Elo)
  s2  Step 2: market + Elo + EPA + QB + schedule + home field + environment/weather
  s2- Step 2 without the environment/weather features (shows what weather adds)

All models except v1 are walk-forward: season S is predicted by a model
fitted only on seasons before S.
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from tool.backtest import find_bets, simulate_bankroll, summarize
from tool.elo import run_elo
from tool.features import (build_features, STEP1_FEATURES, STEP2_FEATURES,
                           ENVIRONMENT_FEATURES)
from tool.model import walk_forward, explain

HERE = Path(__file__).parent
DATA = HERE / "data"
ELO_TUNE_SEASONS = range(1999, 2010)
TEST_SEASONS = range(2010, 2026)
START = 1000.0


def load_games():
    g = pd.read_csv(DATA / "games.csv")
    g["gametime"] = g["gametime"].fillna("13:00")
    return g


def tune_elo(games):
    """Elo settings chosen on 1999-2009 by prediction accuracy (not profit)."""
    best = None
    for k, hfa, reg in itertools.product([15, 20, 25, 30], [30, 45, 60, 75], [0.25, 1/3, 0.5]):
        out = run_elo(games, k=k, home_adv=hfa, season_regress=reg)
        tr = out[out.season.isin(ELO_TUNE_SEASONS) & out.result.notna() & (out.week > 1)]
        mae = (tr.result - tr.pred_margin).abs().mean()
        if best is None or mae < best[0]:
            best = (mae, k, hfa, reg)
    return best


def load_all(extra_weather=None):
    games = load_games()
    _, k, hfa, reg = tune_elo(games)
    elo = run_elo(games, k=k, home_adv=hfa, season_regress=reg)
    tr = elo[elo.season.isin(ELO_TUNE_SEASONS) & elo.result.notna()]
    elo_sd = float((tr.result - tr.pred_margin).std())
    weather_file = DATA / "weather.csv"
    weather = pd.read_csv(weather_file) if weather_file.exists() else pd.DataFrame(columns=["game_id"])
    if extra_weather is not None and not extra_weather.empty:
        weather = pd.concat([weather, extra_weather], ignore_index=True)
    feats = build_features(games, pd.read_csv(DATA / "team_games.csv"),
                           pd.read_csv(DATA / "qb_games.csv"), weather,
                           elo.set_index("game_id").pred_margin)
    return feats, elo, elo_sd


def bet_summary(preds, edge, sd=13.5, anchor=True):
    bets = find_bets(preds, sd=sd, min_edge=edge, anchor_to_market=anchor)
    if bets.empty:
        return None, None
    sim = simulate_bankroll(bets, start=START)
    return summarize(sim, start=START), sim


def calibration_table(preds):
    """Does a predicted 55% cover actually cover ~55% of the time?"""
    from tool.odds import cover_probs
    rows = []
    for g in preds.itertuples():
        ph, pa, pp = cover_probs(g.pred_margin, g.spread_line, g.sd)
        if g.result == g.spread_line:
            continue
        # look at the side the model favours
        p, won = (ph / (1 - pp), g.result > g.spread_line) if ph >= pa else (pa / (1 - pp), g.result < g.spread_line)
        rows.append((p, won))
    c = pd.DataFrame(rows, columns=["p", "won"])
    c["bucket"] = pd.cut(c.p, [0.5, 0.52, 0.54, 0.56, 0.6, 1.0])
    t = c.groupby("bucket", observed=True).agg(games=("won", "size"), predicted=("p", "mean"), actual=("won", "mean"))
    t["predicted"] = t.predicted.map("{:.1%}".format)
    t["actual"] = t.actual.map("{:.1%}".format)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--edge", type=float, default=0.03)
    args = ap.parse_args()

    feats, elo, elo_sd = load_all()
    test_mask = feats.season.isin(TEST_SEASONS) & feats.result.notna() & feats.spread_line.notna()

    models = {}
    v1 = elo[elo.game_id.isin(feats[test_mask].game_id)].copy()
    v1["sd"] = elo_sd
    models["v1  Elo raw"] = v1
    models["s1  Elo calibrated"], _ = walk_forward(feats, STEP1_FEATURES, TEST_SEASONS)
    no_env = [f for f in STEP2_FEATURES if f not in ENVIRONMENT_FEATURES]
    models["s2- full, no weather"], _ = walk_forward(feats, no_env, TEST_SEASONS)
    models["s2  full model"], last_model = walk_forward(feats, STEP2_FEATURES, TEST_SEASONS)

    market_mae = (feats[test_mask].result - feats[test_mask].spread_line).abs().mean()
    print(f"Test seasons {TEST_SEASONS.start}-{TEST_SEASONS.stop-1}. "
          f"Betting market's average miss: {market_mae:.3f} pts\n")

    print("ACCURACY (average miss on the final margin, lower is better)")
    for name, p in models.items():
        print(f"  {name:24s} {(p.result - p.pred_margin).abs().mean():.3f} pts")

    print("\nBETTING RESULTS by edge threshold (ROI on money staked; 90% luck range in brackets)")
    print(f"  {'model':24s} {'edge':>5s} {'bets':>5s} {'win%':>6s} {'ROI':>7s}  {'luck range':>17s} {'final $':>9s}")
    sims = {}
    for name, p in models.items():
        for edge in (0.01, 0.02, 0.03, 0.05):
            s, sim = bet_summary(p, edge, anchor=not name.startswith("v1"))
            if s is None:
                print(f"  {name:24s} {edge:5.0%}     0")
                continue
            lo, hi = s["roi_90pct_range"]
            print(f"  {name:24s} {edge:5.0%} {s['bets']:5d} {s['win_rate']:6.1%} {s['roi']:+7.1%}  "
                  f"[{lo:+6.1%}, {hi:+6.1%}] {s['final_bankroll']:9,.0f}")
            if abs(edge - args.edge) < 1e-9:
                sims[name] = sim
        print()

    print("CALIBRATION of the full model (predicted vs actual cover rate)")
    print(calibration_table(models["s2  full model"]).to_string(), "\n")

    print("WHAT THE FULL MODEL LEARNED (fit through 2024; points of line error per 1 std-dev of feature)")
    print("  positive = pushes the prediction toward the HOME team beyond the line")
    print(explain(last_model, STEP2_FEATURES).to_string(), "\n")

    best = models["s2  full model"]
    s, sim = bet_summary(best, args.edge)
    out = HERE / "output"
    out.mkdir(exist_ok=True)
    if sim is not None:
        sim.to_csv(out / "backtest_bets.csv", index=False)
        per = sim.groupby("season").agg(bets=("profit", "size"), staked=("stake", "sum"),
                                        profit=("profit", "sum"), bankroll=("bankroll_after", "last"))
        per["roi"] = (per.profit / per.staked).map("{:+.1%}".format)
        per["bankroll"] = per.bankroll.map("${:,.0f}".format)
        print(f"FULL MODEL by season (edge >= {args.edge:.0%})")
        print(per[["bets", "roi", "bankroll"]].to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name, sm in sims.items():
            dates = pd.to_datetime(sm.gameday)
            ax.plot(np.r_[dates.iloc[0], dates], np.r_[START, sm.bankroll_after], label=name.strip())
        ax.axhline(START, color="gray", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"Simulated bankroll, bets with edge ≥ {args.edge:.0%}, ¼ Kelly")
        ax.set_ylabel("Bankroll ($, log scale)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / "bankroll.png", dpi=150)
        print("\nSaved output/backtest_bets.csv and output/bankroll.png")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
