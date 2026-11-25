"""
Elo power ratings for NFL teams.

The idea (same family as chess ratings):
  * Every team starts at 1500.
  * Before each game we predict the point margin from the rating gap.
  * After the game, the winner takes rating points from the loser.
    Bigger upsets and bigger margins of victory move ratings more.
  * Between seasons every team is pulled part of the way back to 1500,
    because rosters change.

Crucially, a game's prediction is made BEFORE that game updates the ratings,
so the model can never "see the future". That's what makes it safe to backtest.
"""
import math
import pandas as pd

# Franchises that moved: treat them as the same team.
FRANCHISE = {"STL": "LA", "SD": "LAC", "OAK": "LV"}

ELO_POINTS_PER_POINT = 25.0  # 25 Elo points ~= 1 point on the scoreboard


def run_elo(games: pd.DataFrame, k=20.0, home_adv=48.0, season_regress=1/3):
    """
    Walk through every game in date order and return a copy of `games`
    with these new columns (all known before kickoff):
        elo_home, elo_away  - ratings going into the game
        pred_margin         - predicted home score minus away score
    """
    games = games.sort_values(["gameday", "gametime", "game_id"]).copy()
    ratings = {}
    last_season = None
    elo_home_list, elo_away_list, pred_list = [], [], []

    for g in games.itertuples():
        # New season: regress everyone toward the mean.
        if g.season != last_season:
            for t in ratings:
                ratings[t] = 1500 + (ratings[t] - 1500) * (1 - season_regress)
            last_season = g.season

        home = FRANCHISE.get(g.home_team, g.home_team)
        away = FRANCHISE.get(g.away_team, g.away_team)
        rh = ratings.get(home, 1500.0)
        ra = ratings.get(away, 1500.0)
        hfa = 0.0 if g.location == "Neutral" else home_adv

        diff = rh + hfa - ra
        elo_home_list.append(rh)
        elo_away_list.append(ra)
        pred_list.append(diff / ELO_POINTS_PER_POINT)

        # Only update ratings once the game has actually been played.
        if pd.isna(g.result):
            continue
        margin = g.result  # home score - away score
        expected_home_win = 1 / (1 + 10 ** (-diff / 400))
        actual = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)
        # Margin-of-victory multiplier (dampened when the favourite wins big).
        winner_diff = diff if margin > 0 else -diff
        mov_mult = math.log(abs(margin) + 1) * (2.2 / (winner_diff * 0.001 + 2.2))
        shift = k * mov_mult * (actual - expected_home_win)
        ratings[home] = rh + shift
        ratings[away] = ra - shift

    games["elo_home"] = elo_home_list
    games["elo_away"] = elo_away_list
    games["pred_margin"] = pred_list
    return games
