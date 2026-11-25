"""
Alternate lines (e.g. Under 38.5 at +180 when the main total is 44.5).

We have no history of real alternate-line prices, so the backtest PRICES THEM
THE WAY A BOOK DOES and adds a margin ("hold") on top:

  1. Fair probability of every alternate line comes from the real historical
     distribution of results in games with a similar market line (training
     seasons only). This captures key numbers: NFL margins pile up on 3, 7, 10
     and totals on 37, 41, 44, 47, 51 - a bell curve misses that.
  2. The book's price = fair price with `hold` removed (4.5% is typical for
     main lines; alternate lines usually carry more).
  3. Our probability = the same historical distribution, nudged by the
     model's predicted shift (prediction minus the line) times the local
     density at that alternate line - exactly how the main-line model works.

So the backtest answers: "if our lean is right, does it pay more on an
alternate line than on the main line, after a realistic book margin?"
It cannot answer whether FanDuel's real alternate prices are softer or
harsher than this stand-in; that needs real historical alt prices.
"""
import numpy as np
import pandas as pd


def _pool(train_lines, train_actual, line, min_games=150):
    """Actual outcomes of training games whose market line was close to `line`."""
    for width in (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0):
        m = np.abs(train_lines - line) <= width
        if m.sum() >= min_games:
            return train_actual[m]
    return train_actual


def decimal_to_american(d):
    return (d - 1) * 100 if d >= 2 else -100 / (d - 1)


def alt_ladder_bets(preds, line_col, actual_col, pred_col, seasons_all,
                    hold=0.07, min_edge=0.03, offsets=range(-10, 11),
                    train_start=2002, labels=("over", "under")):
    """
    For each game, price every alternate line (main line + offset) on both
    sides, and keep the single best bet if its EV clears `min_edge`.
    `seasons_all` is the full feature table, used for walk-forward pools.
    Returns rows compatible with backtest.simulate_bankroll.
    """
    hist = seasons_all[seasons_all[actual_col].notna() & seasons_all[line_col].notna()
                       & (seasons_all.season >= train_start)]
    rows = []
    for season, games in preds.groupby("season"):
        tr = hist[hist.season < season]
        tl, ta = tr[line_col].to_numpy(), tr[actual_col].to_numpy()
        for g in games.itertuples():
            line = getattr(g, line_col)
            shift = getattr(g, pred_col) - line
            pool = _pool(tl, ta, line)
            best = None
            for k in offsets:
                L = line + k
                p_over = (pool > L).mean()
                p_push = (pool == L).mean()
                p_under = 1 - p_over - p_push
                if min(p_over, p_under) < 0.04:  # too far out to price sensibly
                    continue
                dens = ((pool > L - 1.5).mean() - (pool > L + 1.5).mean()) / 3
                q_over = np.clip(p_over + dens * shift, 0.01, 0.99 - p_push)
                q_under = 1 - q_over - p_push
                for side, fair, ours in ((labels[0], p_over, q_over), (labels[1], p_under, q_under)):
                    # book's decimal price: fair odds with the hold taken out
                    fair_cond = fair / (1 - p_push)
                    dec = 1 / (fair_cond * (1 + hold))
                    ev = ours * (dec - 1) - (1 - ours - p_push)
                    if best is None or ev > best[0]:
                        best = (ev, side, L, k, dec, ours, p_push)
            if best is None or best[0] < min_edge:
                continue
            ev, side, L, k, dec, ours, p_push = best
            b = dec - 1
            p_cond = ours / (1 - p_push)
            kelly = max((p_cond * b - (1 - p_cond)) / b, 0.0)
            actual = getattr(g, actual_col)
            if actual == L:
                outcome = "push"
            elif (actual > L) == (side == labels[0]):
                outcome = "win"
            else:
                outcome = "loss"
            rows.append(dict(season=g.season, week=g.week, gameday=g.gameday,
                             game=f"{g.away_team} @ {g.home_team}", side=side,
                             main_line=line, alt_line=L, offset=k,
                             odds=decimal_to_american(dec), ev=round(ev, 3),
                             kelly=kelly, outcome=outcome))
    return pd.DataFrame(rows)


def alt_targets(line, shift, hist_lines, hist_actual, min_edge=0.03, offsets=range(-10, 11)):
    """
    For one upcoming game: every alternate line where the model's lean is
    strongest, with the MINIMUM price at which a bet clears `min_edge`.
    You then compare that minimum to what FanDuel actually offers.
    """
    pool = _pool(hist_lines, hist_actual, line)
    out = []
    for k in offsets:
        L = line + k
        p_over, p_push = (pool > L).mean(), (pool == L).mean()
        p_under = 1 - p_over - p_push
        if min(p_over, p_under) < 0.04:
            continue
        dens = ((pool > L - 1.5).mean() - (pool > L + 1.5).mean()) / 3
        q_over = float(np.clip(p_over + dens * shift, 0.01, 0.99 - p_push))
        for side, q, fair in (("over", q_over, p_over), ("under", 1 - q_over - p_push, p_under)):
            if q <= fair:  # model doesn't lean this way here
                continue
            dec_min = (1 + min_edge - p_push) / q
            out.append(dict(side=side, alt_line=L, offset=k, model_prob=q,
                            min_odds=decimal_to_american(dec_min),
                            typical_odds=decimal_to_american(1 / (fair / (1 - p_push) * 1.07)),
                            edge_ratio=q / fair))
    return pd.DataFrame(out)
