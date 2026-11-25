"""
Backtesting: pick bets from past games, then simulate a bankroll betting them.
"""
import numpy as np
import pandas as pd

from .odds import cover_probs, expected_value, kelly_fraction, no_vig_probs


def find_bets(games: pd.DataFrame, sd: float, min_edge: float,
              anchor_to_market: bool = True) -> pd.DataFrame:
    """
    For each game with a spread and prices, decide whether either side
    is worth betting. A side qualifies when the model's expected value per
    $1 is at least `min_edge` (e.g. 0.03 = 3%).

    anchor_to_market=True (recommended): start from the market's own no-vig
    probability and only move it by how much the model's margin differs from
    the line. Books shade prices around key numbers (3, 7) for good reasons;
    a simple bell-curve model can't see that and would mistake every
    plus-money price for value. Anchoring removes that false signal.
    Returns one row per bet.
    """
    rows = []
    usable = games.dropna(subset=["spread_line", "home_spread_odds",
                                  "away_spread_odds", "pred_margin"])
    for g in usable.itertuples():
        game_sd = getattr(g, "sd", sd)  # models may supply their own per-game spread
        p_home, p_away, p_push = cover_probs(g.pred_margin, g.spread_line, game_sd)
        mkt_home, mkt_away = no_vig_probs(g.home_spread_odds, g.away_spread_odds)
        if anchor_to_market:
            b_home, b_away, _ = cover_probs(g.spread_line, g.spread_line, game_sd)
            shift = p_home / (p_home + p_away) - b_home / (b_home + b_away)
            q_home = min(max(mkt_home + shift, 0.01), 0.99)
            p_home, p_away = q_home * (1 - p_push), (1 - q_home) * (1 - p_push)
        for side, p, odds, mkt in (("home", p_home, g.home_spread_odds, mkt_home),
                                   ("away", p_away, g.away_spread_odds, mkt_away)):
            ev = expected_value(p, p_push, odds)
            if ev < min_edge:
                continue
            # Outcome (only known for completed games).
            outcome = None
            if not pd.isna(g.result):
                if g.result == g.spread_line:
                    outcome = "push"
                elif (g.result > g.spread_line) == (side == "home"):
                    outcome = "win"
                else:
                    outcome = "loss"
            team = g.home_team if side == "home" else g.away_team
            line = -g.spread_line if side == "home" else g.spread_line
            rows.append(dict(
                season=g.season, week=g.week, gameday=g.gameday,
                game=f"{g.away_team} @ {g.home_team}",
                bet=f"{team} {line:+g}", side=side, odds=odds,
                model_pred_margin=round(g.pred_margin, 1),
                model_prob=round(p / (1 - p_push), 3),
                market_prob=round(mkt, 3),
                ev=round(ev, 3),
                kelly=kelly_fraction(p, p_push, odds),
                outcome=outcome,
            ))
    return pd.DataFrame(rows)


def simulate_bankroll(bets: pd.DataFrame, start=1000.0, kelly_mult=0.25,
                      max_stake_pct=0.03, flat_stake_pct=None):
    """
    Walk through bets week by week. All bets in the same week are sized
    from the bankroll at the START of that week (they happen at the same time).

    kelly_mult     - fraction of the Kelly stake to bet (0.25 = "quarter Kelly")
    max_stake_pct  - never put more than this share of the bankroll on one bet
    flat_stake_pct - if set, ignore Kelly and bet this share every time
    """
    bets = bets[bets.outcome.notna()].copy()
    bankroll = start
    stakes, profits, after = [], [], []
    for _, week_bets in bets.groupby(["season", "week"], sort=True):
        week_start = bankroll
        for b in week_bets.itertuples():
            pct = flat_stake_pct if flat_stake_pct else min(b.kelly * kelly_mult, max_stake_pct)
            stake = week_start * pct
            if b.outcome == "win":
                profit = stake * (american_to_decimal_vec(b.odds) - 1)
            elif b.outcome == "loss":
                profit = -stake
            else:
                profit = 0.0
            bankroll += profit
            stakes.append(stake)
            profits.append(profit)
            after.append(bankroll)
    bets = bets.sort_values(["season", "week"], kind="stable")
    bets["stake"] = stakes
    bets["profit"] = profits
    bets["bankroll_after"] = after
    return bets


def american_to_decimal_vec(a):
    return 1 + a / 100 if a > 0 else 1 + 100 / abs(a)


def summarize(sim: pd.DataFrame, start=1000.0, n_boot=2000, seed=0):
    """Headline numbers plus a 90% bootstrap range for ROI."""
    if sim.empty:
        return {"bets": 0}
    wins = (sim.outcome == "win").sum()
    losses = (sim.outcome == "loss").sum()
    roi = sim.profit.sum() / sim.stake.sum()
    curve = np.concatenate([[start], sim.bankroll_after.values])
    peak = np.maximum.accumulate(curve)
    max_dd = ((peak - curve) / peak).max()

    # Bootstrap: resample bets with replacement to see how much ROI could
    # vary from luck alone.
    rng = np.random.default_rng(seed)
    profit, stake = sim.profit.values, sim.stake.values
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(profit), len(profit))
        boots.append(profit[i].sum() / stake[i].sum())
    lo, hi = np.percentile(boots, [5, 95])

    return {
        "bets": len(sim),
        "record": f"{wins}-{losses}-{len(sim) - wins - losses}",
        "win_rate": wins / max(wins + losses, 1),
        "roi": roi,
        "roi_90pct_range": (lo, hi),
        "final_bankroll": sim.bankroll_after.iloc[-1],
        "max_drawdown": max_dd,
    }


def find_total_bets(games: pd.DataFrame, min_edge: float, sd: float = 13.5,
                    anchor_to_market: bool = True, pred_col: str = "pred_total") -> pd.DataFrame:
    """
    Over/under version of find_bets. Same logic: the market's no-vig
    probability is the starting point, and the model's predicted total only
    nudges it by how far the prediction sits from the posted total.
    """
    rows = []
    usable = games.dropna(subset=["total_line", "over_odds", "under_odds", pred_col])
    for g in usable.itertuples():
        game_sd = getattr(g, "sd", sd)
        pred = getattr(g, pred_col)
        p_over, p_under, p_push = cover_probs(pred, g.total_line, game_sd)
        mkt_over, mkt_under = no_vig_probs(g.over_odds, g.under_odds)
        if anchor_to_market:
            b_over, b_under, _ = cover_probs(g.total_line, g.total_line, game_sd)
            shift = p_over / (p_over + p_under) - b_over / (b_over + b_under)
            q = min(max(mkt_over + shift, 0.01), 0.99)
            p_over, p_under = q * (1 - p_push), (1 - q) * (1 - p_push)
        for side, p, odds, mkt in (("over", p_over, g.over_odds, mkt_over),
                                   ("under", p_under, g.under_odds, mkt_under)):
            ev = expected_value(p, p_push, odds)
            if ev < min_edge:
                continue
            outcome = None
            if not pd.isna(g.total):
                if g.total == g.total_line:
                    outcome = "push"
                elif (g.total > g.total_line) == (side == "over"):
                    outcome = "win"
                else:
                    outcome = "loss"
            rows.append(dict(
                season=g.season, week=g.week, gameday=g.gameday,
                game=f"{g.away_team} @ {g.home_team}",
                bet=f"{side.upper()} {g.total_line:g}", side=side, odds=odds,
                model_pred_total=round(pred, 1),
                model_prob=round(p / (1 - p_push), 3), market_prob=round(mkt, 3),
                ev=round(ev, 3), kelly=kelly_fraction(p, p_push, odds), outcome=outcome,
            ))
    return pd.DataFrame(rows)
