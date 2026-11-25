"""
Helpers for converting betting odds and model predictions into probabilities.
"""
import math


def american_to_decimal(american: float) -> float:
    """-110 -> 1.909 (you get back $1.909 per $1 staked, stake included)."""
    if american > 0:
        return 1 + american / 100
    return 1 + 100 / abs(american)


def implied_prob(american: float) -> float:
    """Break-even win rate the price implies. -110 -> 52.4%."""
    return 1 / american_to_decimal(american)


def no_vig_probs(odds_a: float, odds_b: float):
    """Remove the bookmaker's cut so the two sides add up to 100%."""
    pa, pb = implied_prob(odds_a), implied_prob(odds_b)
    return pa / (pa + pb), pb / (pa + pb)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def cover_probs(pred_margin: float, spread_line: float, sd: float = 13.5):
    """
    Probability the HOME team covers, the away team covers, or it's a push.

    nflverse convention: spread_line is how many points the home team is
    favoured by (e.g. 3.0 means home -3). Home covers if result > spread_line.
    We model the actual margin as Normal(pred_margin, sd); NFL margins have a
    standard deviation of roughly 13-14 points around the expectation.
    """
    lo = _norm_cdf((spread_line - 0.5 - pred_margin) / sd)
    hi = _norm_cdf((spread_line + 0.5 - pred_margin) / sd)
    if spread_line == int(spread_line):  # whole-number line -> push possible
        return 1 - hi, lo, hi - lo
    mid = _norm_cdf((spread_line - pred_margin) / sd)
    return 1 - mid, mid, 0.0


def expected_value(p_win: float, p_push: float, american: float) -> float:
    """Expected profit per $1 staked (0.03 = +3%)."""
    profit = american_to_decimal(american) - 1
    p_lose = 1 - p_win - p_push
    return p_win * profit - p_lose


def kelly_fraction(p_win: float, p_push: float, american: float) -> float:
    """
    Kelly criterion: the fraction of bankroll that maximises long-run growth
    IF your probabilities are exactly right. They never are, so we bet a
    fraction of Kelly (see backtest.py).
    """
    b = american_to_decimal(american) - 1
    # Pushes return the stake, so ignore them: renormalise win/lose.
    p = p_win / (1 - p_push) if p_push < 1 else 0
    f = (p * b - (1 - p)) / b
    return max(f, 0.0)
