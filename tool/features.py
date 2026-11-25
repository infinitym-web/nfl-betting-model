"""
Turns raw data into one row per game of PRE-GAME facts ("features").

Golden rule: every number on a game's row must have been knowable before
kickoff. Each rolling rating is read BEFORE the game and updated AFTER it.

Feature groups
  market     spread_line (what the betting market thinks)
  strength   Elo, rolling EPA per play (offence & defence), success rate
  QB         rolling EPA per dropback of the listed starting QB; "new starter" flag
  schedule   rest days, short week, coming off a bye, travel distance, time zones
  home field rolling estimate of league-wide home advantage (it changes over time)
  environment dome / retractable / outdoors, temperature, wind, rain, snow,
             and how those interact with the line and with teams used to domes
"""
import math

import numpy as np
import pandas as pd

from .stadiums import COORDS

INDOOR = {"dome", "closed"}

# --- rolling team strength from play-by-play --------------------------------

GAME_DECAY = 0.92      # each older game counts 8% less
SEASON_DECAY = 0.55    # at a new season, last year's info counts ~half
PRIOR_PLAYS = 150      # shrink toward league average until a team has data


def _team_epa_ratings(games, team_games):
    """Pre-game rolling EPA/play and success rate for each team, offence and defence."""
    tg = team_games.merge(games[["game_id", "season", "gameday", "gametime"]], on="game_id")
    tg = tg.sort_values(["gameday", "gametime", "game_id"])
    by_game = {gid: grp for gid, grp in tg.groupby("game_id", sort=False)}

    state = {}  # team -> dict of weighted sums
    keys = ["o_plays", "o_epa", "o_succ", "o_db", "o_pepa", "d_plays", "d_epa", "d_succ", "d_db", "d_pepa"]

    def read(team):
        s = state.get(team)
        if s is None:
            return dict(off_epa=0.0, def_epa=0.0, off_succ=0.44, def_succ=0.44, off_pass=0.0, def_pass=0.0)
        return dict(
            off_epa=s["o_epa"] / (s["o_plays"] + PRIOR_PLAYS),
            def_epa=s["d_epa"] / (s["d_plays"] + PRIOR_PLAYS),
            off_succ=(s["o_succ"] + 0.44 * PRIOR_PLAYS) / (s["o_plays"] + PRIOR_PLAYS),
            def_succ=(s["d_succ"] + 0.44 * PRIOR_PLAYS) / (s["d_plays"] + PRIOR_PLAYS),
            off_pass=s["o_pepa"] / (s["o_db"] + PRIOR_PLAYS * 0.6),
            def_pass=s["d_pepa"] / (s["d_db"] + PRIOR_PLAYS * 0.6),
        )

    out = []
    last_season = None
    for g in games.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        if g.season != last_season:
            for s in state.values():
                for k in keys:
                    s[k] *= SEASON_DECAY
            last_season = g.season
        h, a = read(g.home_team), read(g.away_team)
        out.append({"game_id": g.game_id, **{f"home_{k}": v for k, v in h.items()},
                    **{f"away_{k}": v for k, v in a.items()}})
        grp = by_game.get(g.game_id)
        if grp is None:
            continue
        for r in grp.itertuples():
            for team, side in ((r.posteam, "o"), (r.defteam, "d")):
                s = state.setdefault(team, {k: 0.0 for k in keys})
                for k in [k for k in keys if k.startswith(side)]:
                    s[k] *= GAME_DECAY
                s[f"{side}_plays"] += r.plays
                s[f"{side}_epa"] += r.epa
                s[f"{side}_succ"] += r.success
                s[f"{side}_db"] += r.dropbacks
                s[f"{side}_pepa"] += r.pass_epa
    return pd.DataFrame(out)


# --- rolling QB quality -------------------------------------------------------

QB_GAME_DECAY = 0.96
QB_SEASON_DECAY = 0.8
QB_PRIOR_DB = 200
QB_PRIOR_EPA = -0.10   # an unknown QB is assumed to be below average


def _qb_ratings(games, qb_games):
    qg = qb_games.merge(games[["game_id", "gameday", "gametime"]], on="game_id")
    by_game = {gid: grp for gid, grp in qg.groupby("game_id", sort=False)}
    state = {}  # qb -> [weighted dropbacks, weighted epa, last season seen]
    prev_starter = {}

    def value(qb, season):
        if not isinstance(qb, str) or qb not in state:
            return QB_PRIOR_EPA
        db, epa, last = state[qb]
        if last != season:
            db, epa = db * QB_SEASON_DECAY, epa * QB_SEASON_DECAY
        return (epa + QB_PRIOR_EPA * QB_PRIOR_DB) / (db + QB_PRIOR_DB)

    out = []
    for g in games.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        row = {"game_id": g.game_id}
        for side, team, qb in (("home", g.home_team, g.home_qb_id), ("away", g.away_team, g.away_qb_id)):
            row[f"{side}_qb"] = value(qb, g.season)
            prev = prev_starter.get(team)
            row[f"{side}_new_qb"] = float(isinstance(qb, str) and prev is not None and qb != prev)
            if isinstance(qb, str) and not pd.isna(g.result):
                prev_starter[team] = qb
        out.append(row)
        grp = by_game.get(g.game_id)
        if grp is None:
            continue
        for r in grp.itertuples():
            db, epa, last = state.get(r.passer_player_id, [0.0, 0.0, g.season])
            if last != g.season:
                db, epa = db * QB_SEASON_DECAY, epa * QB_SEASON_DECAY
            state[r.passer_player_id] = [db * QB_GAME_DECAY + r.dropbacks,
                                         epa * QB_GAME_DECAY + r.qb_epa, g.season]
    return pd.DataFrame(out)


# --- home field, travel, climate ----------------------------------------------

def _haversine_km(a, b):
    (lat1, lon1), (lat2, lon2) = a, b
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


def _home_stadiums(games):
    """Each team's main home stadium per season (and whether it's indoor / warm)."""
    home = games[games.location == "Home"]
    main = (home.groupby(["season", "home_team"]).stadium_id
            .agg(lambda s: s.value_counts().index[0]).to_dict())
    roof = home.groupby("stadium_id").roof.agg(lambda r: r.dropna().mode().iat[0] if r.notna().any() else "outdoors")
    return main, roof.to_dict()


def _rolling_hfa(games):
    """League-wide home advantage estimated only from games already played:
    last 3 seasons plus this season so far, recent games weighted more."""
    g = games.sort_values(["gameday", "gametime", "game_id"])
    num = den = 0.0
    out, last_season = [], None
    for r in g.itertuples():
        if r.season != last_season:
            num, den = num * 0.6, den * 0.6
            last_season = r.season
        out.append((r.game_id, num / den if den > 20 else 2.5))
        if r.location == "Home" and not pd.isna(r.result):
            num, den = num * 0.997 + r.result, den * 0.997 + 1
    return dict(out)


# --- put it all together ------------------------------------------------------

def build_features(games, team_games, qb_games, weather, elo_pred):
    g = games.copy()
    g["pred_elo"] = g.game_id.map(elo_pred)

    g = g.merge(_team_epa_ratings(g, team_games), on="game_id", how="left")
    g = g.merge(_qb_ratings(g, qb_games), on="game_id", how="left")
    g = g.merge(weather, on="game_id", how="left")
    for c in ("w_temp", "w_wind", "w_gust", "w_precip", "w_snow"):
        if c not in g:
            g[c] = np.nan

    # Net strength differences (home minus away). Defence EPA is "allowed", so lower is better.
    g["epa_net"] = (g.home_off_epa - g.home_def_epa) - (g.away_off_epa - g.away_def_epa)
    g["pass_net"] = (g.home_off_pass - g.home_def_pass) - (g.away_off_pass - g.away_def_pass)
    g["succ_net"] = (g.home_off_succ - g.home_def_succ) - (g.away_off_succ - g.away_def_succ)
    g["qb_diff"] = g.home_qb - g.away_qb
    g["new_qb_diff"] = g.home_new_qb - g.away_new_qb

    # Schedule
    g["rest_diff"] = (g.home_rest - g.away_rest).clip(-7, 7)
    g["short_week_diff"] = (g.home_rest <= 5).astype(float) - (g.away_rest <= 5)
    g["bye_diff"] = (g.home_rest >= 12).astype(float) - (g.away_rest >= 12)

    main, roof_of = _home_stadiums(g)
    trav_h, trav_a, tz_a, cold_team_diff_base = [], [], [], []
    for r in g.itertuples():
        here = COORDS.get(r.stadium_id)
        hs = COORDS.get(main.get((r.season, r.home_team)))
        as_ = COORDS.get(main.get((r.season, r.away_team)))
        if here and hs and as_:
            trav_h.append(_haversine_km(hs, here) / 1000)
            trav_a.append(_haversine_km(as_, here) / 1000)
            tz_a.append((here[1] - as_[1]) / 15)  # rough time-zone hours moved east
        else:
            trav_h.append(0.0); trav_a.append(0.0); tz_a.append(0.0)

        def soft(team):  # team plays home games indoors or in a warm climate
            sid = main.get((r.season, team))
            if sid is None:
                return 0.0
            return float(roof_of.get(sid) in {"dome", "closed", "open"} or COORDS.get(sid, (90, 0))[0] < 34)
        cold_team_diff_base.append(soft(r.away_team) - soft(r.home_team))
    g["travel_diff"] = np.array(trav_a) - np.array(trav_h)
    g["tz_east_away"] = np.clip(tz_a, -4, 4)  # west-coast team flying east is positive
    g["neutral"] = (g.location == "Neutral").astype(float)
    g["hfa_est"] = g.game_id.map(_rolling_hfa(g)) * (1 - g.neutral)

    # Environment. Unknown future retractable-roof status counts as indoor.
    roof = g.roof.fillna("closed")
    g["is_indoor"] = roof.isin(INDOOR).astype(float)
    g["is_open_roof"] = (roof == "open").astype(float)
    outdoor = 1 - g.is_indoor
    g["temp"] = g.w_temp.fillna(g.temp).fillna(65).where(outdoor == 1, 70)
    g["wind"] = g.w_wind.fillna(g.wind).fillna(8).where(outdoor == 1, 0)
    g["precip"] = g.w_precip.fillna(0).where(outdoor == 1, 0).clip(0, 1)
    g["snow"] = g.w_snow.fillna(0).where(outdoor == 1, 0).clip(0, 3)
    g["cold"] = ((45 - g.temp).clip(lower=0) / 10) * outdoor
    g["hot"] = ((g.temp - 85).clip(lower=0) / 10) * outdoor
    g["windy"] = ((g.wind - 10).clip(lower=0) / 5) * outdoor
    g["wet"] = (g.precip > 0.05).astype(float) * outdoor
    # Interactions: bad weather tends to shrink margins (favourites cover less),
    # and dome/warm-weather teams may struggle in the cold.
    g["wind_x_line"] = g.windy * g.spread_line.fillna(0) / 7
    g["wet_x_line"] = g.wet * g.spread_line.fillna(0) / 7
    g["cold_x_softteam"] = g.cold * np.array(cold_team_diff_base)
    g["div_game_x_line"] = g.div_game * g.spread_line.fillna(0) / 7

    g = g.merge(_scoring_and_pace(g, team_games), on="game_id", how="left")
    # Totals features: combined ("sum") rather than difference
    g["off_epa_sum"] = g.home_off_epa + g.away_off_epa
    g["def_epa_sum"] = g.home_def_epa + g.away_def_epa
    g["pass_sum"] = g.home_off_pass + g.away_off_pass + g.home_def_pass + g.away_def_pass
    g["succ_sum"] = g.home_off_succ + g.away_off_succ + g.home_def_succ + g.away_def_succ
    g["qb_sum"] = g.home_qb + g.away_qb
    g["new_qb_sum"] = g.home_new_qb + g.away_new_qb
    g["pace_sum"] = g.home_pace + g.away_pace
    g["pts_model"] = (g.home_pf + g.away_pa) / 2 + (g.away_pf + g.home_pa) / 2
    g["short_week_any"] = ((g.home_rest <= 5) | (g.away_rest <= 5)).astype(float)
    g["playoff"] = (g.game_type != "REG").astype(float)
    g["late_season"] = ((g.game_type == "REG") & (g.week >= 15)).astype(float)
    g["gusty"] = ((g.w_gust.fillna(g.wind * 1.4) - 20).clip(lower=0) / 10) * outdoor
    g["wind_x_total"] = g.windy * g.total_line.fillna(44) / 44
    return g


def _scoring_and_pace(games, team_games):
    """Rolling points scored/allowed and offensive plays per game, pre-game,
    plus the league's recent scoring level relative to the closing totals."""
    plays = team_games.set_index(["game_id", "posteam"]).plays.to_dict()
    pts = {}  # team -> [w, pf, pa, plays_w, plays]
    lg_num = lg_den = 0.0
    out, last_season = [], None
    for r in games.sort_values(["gameday", "gametime", "game_id"]).itertuples():
        if r.season != last_season:
            for s in pts.values():
                for i in range(5):
                    s[i] *= SEASON_DECAY
            lg_num, lg_den = lg_num * 0.5, lg_den * 0.5
            last_season = r.season
        row = {"game_id": r.game_id,
               "league_vs_total": lg_num / lg_den if lg_den > 20 else 0.0}
        for side, team in (("home", r.home_team), ("away", r.away_team)):
            w, pf, pa, pw, pl = pts.get(team, [0, 0, 0, 0, 0])
            row[f"{side}_pf"] = (pf + 22 * 3) / (w + 3)
            row[f"{side}_pa"] = (pa + 22 * 3) / (w + 3)
            row[f"{side}_pace"] = (pl + 62 * 3) / (pw + 3)
        out.append(row)
        if pd.isna(r.result):
            continue
        if not pd.isna(r.total_line):
            lg_num, lg_den = lg_num * 0.995 + (r.total - r.total_line), lg_den * 0.995 + 1
        for team, pf, pa in ((r.home_team, r.home_score, r.away_score),
                             (r.away_team, r.away_score, r.home_score)):
            s = pts.setdefault(team, [0.0] * 5)
            s[0] = s[0] * GAME_DECAY + 1
            s[1] = s[1] * GAME_DECAY + pf
            s[2] = s[2] * GAME_DECAY + pa
            n = plays.get((r.game_id, team))
            if n is not None:
                s[3] = s[3] * GAME_DECAY + 1
                s[4] = s[4] * GAME_DECAY + n
    return pd.DataFrame(out)


# Which columns the models use
STEP1_FEATURES = ["elo_vs_line"]
STEP2_FEATURES = [
    "elo_vs_line", "line", "epa_net", "pass_net", "succ_net", "qb_diff", "new_qb_diff",
    "rest_diff", "short_week_diff", "bye_diff", "travel_diff", "tz_east_away",
    "hfa_est", "neutral",
    "is_indoor", "is_open_roof", "cold", "hot", "windy", "wet", "snow",
    "wind_x_line", "wet_x_line", "cold_x_softteam", "div_game_x_line",
]
ENVIRONMENT_FEATURES = ["is_indoor", "is_open_roof", "cold", "hot", "windy", "wet", "snow",
                        "wind_x_line", "wet_x_line", "cold_x_softteam",
                        "gusty", "wind_x_total"]

# Over/under model
TOTALS_FEATURES = [
    "total_line", "pts_model", "off_epa_sum", "def_epa_sum", "pass_sum", "succ_sum",
    "qb_sum", "new_qb_sum", "pace_sum", "league_vs_total",
    "short_week_any", "playoff", "late_season", "div_game", "neutral",
    "is_indoor", "is_open_roof", "cold", "hot", "windy", "gusty", "wet", "snow",
    "wind_x_total",
]
