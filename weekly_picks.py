"""
Weekly picks + paper-trading log.

    python3 build_data.py          # refresh games, scores, play-by-play, weather
    python3 weekly_picks.py        # make this week's picks, settle finished ones

Run it on Monday and Thursday mornings (and Saturday if you like). Each run:
  1. Pulls FanDuel's and Pinnacle's CURRENT totals and spreads (The Odds API, 2 credits)
  2. Pulls the latest NOAA airport forecast for every outdoor game
  3. Scores every game in the next 7 days two ways (UNDERS only: over bets
     lost in every backtest):
       model  the over/under model, under when EV >= 3% at FanDuel's price
       rule   UNDER when the forecast is wind over 15 mph or rain, outdoors
  4. Logs each pick with the price at that moment (first price kept for each
     game x system x bet day) to data/picks_log.csv
  5. Settles finished games: result, profit, and closing line value against
     the final market total
Outputs for the dashboard go to output/dashboard/.

Paper trading only: stakes are in units (1 unit = 1% of a $1,000 paper bankroll).
"""
import io
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_mos import STATION, _from_run, PUBLISH_DELAY
from fetch_odds import TEAM_ABBR, api_get
from run_backtest import load_all
from run_odds_backtest import with_line
from tool.features import TOTALS_FEATURES, environment_features
from tool.model import fit_latest, predict_dev
from tool.odds import american_to_decimal, cover_probs, expected_value, kelly_fraction, no_vig_probs

HERE = Path(__file__).parent
LOG = HERE / "data" / "picks_log.csv"
OUT = HERE / "output" / "dashboard"
ODDS_URL = ("https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"
            "?markets=totals,spreads&bookmakers=fanduel,pinnacle&oddsFormat=american")
MOS_URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station={st}&model={model}"
           "&sts={start}&ets={end}&format=csv")
EDGE = 0.03
RULE_WIND, RULE_RAIN = 15.0, 0.05
BANKROLL = 1000.0


def bet_day(now_et):
    """Which bet day this run counts as (matches the backtest's snapshots)."""
    return {0: "Monday", 1: "Monday", 2: "Thursday", 3: "Thursday",
            4: "Saturday", 5: "Saturday", 6: "Sunday"}[now_et.weekday()]


def live_odds(feats):
    data, left = api_get(ODDS_URL)
    rows = []
    for ev in data:
        home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
        if not home or not away:
            continue
        day = pd.Timestamp(ev["commence_time"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
        row = dict(home_team=home, away_team=away, gameday=day, commence_time=ev["commence_time"])
        for b in ev["bookmakers"]:
            for m in b["markets"]:
                for o in m["outcomes"]:
                    side = o["name"].lower() if m["key"] == "totals" else \
                        ("home" if TEAM_ABBR.get(o["name"]) == home else "away")
                    row[f"{b['key']}_{m['key']}_{side}_point"] = o.get("point")
                    row[f"{b['key']}_{m['key']}_{side}_price"] = o["price"]
        rows.append(row)
    odds = pd.DataFrame(rows)
    key = feats[["game_id", "home_team", "away_team", "gameday"]]
    return odds.merge(key, on=["home_team", "away_team", "gameday"], how="inner"), left


def live_forecasts(games, now):
    """Newest published NOAA MOS forecast for each upcoming outdoor game."""
    out, cache = {}, {}
    start = (now - pd.Timedelta(hours=36)).strftime("%Y-%m-%dT%H:%MZ")
    end = now.strftime("%Y-%m-%dT%H:%MZ")
    for g in games.itertuples():
        st = STATION.get(g.stadium_id)
        if st is None or g.is_indoor == 1:
            continue
        kick = pd.Timestamp(f"{g.gameday} {g.gametime}").tz_localize("America/New_York").tz_convert("UTC")
        k0, k1 = kick.floor("h"), kick.floor("h") + pd.Timedelta(hours=3)
        for model in ("GFS", "MEX"):
            if (st, model) not in cache:
                cache[(st, model)] = None
                for attempt in range(5):  # Iowa State limits request rate: space them out
                    try:
                        with urllib.request.urlopen(MOS_URL.format(st=st, model=model, start=start, end=end), timeout=60) as r:
                            d = pd.read_csv(io.BytesIO(r.read()), low_memory=False)
                        d["runtime"] = pd.to_datetime(d.runtime, utc=True)
                        d["ftime"] = pd.to_datetime(d.ftime, utc=True)
                        cache[(st, model)] = d
                        break
                    except Exception as e:
                        if attempt == 4:
                            print(f"  no {model} forecast for {st}: {e}")
                        time.sleep(5 * (attempt + 1))
                time.sleep(2)
            d = cache[(st, model)]
            if d is None or d.empty:
                continue
            runs = d[d.runtime + PUBLISH_DELAY <= now]
            if runs.empty:
                continue
            latest = runs[runs.runtime == runs.runtime.max()]
            if latest.ftime.max() < k1:
                continue
            w = _from_run(latest, model, k0, k1)
            if w["w_temp"] == w["w_temp"]:
                out[g.game_id] = dict(w, w_gust=np.nan, forecast=f"NOAA {model} MOS {latest.runtime.iloc[0]:%a %H:%M}Z")
                break
    return out


def settle(log, feats):
    if log.empty:
        return log
    res = feats.set_index("game_id")[["total", "total_line"]]
    for i, r in log[log.outcome.isna()].iterrows():
        if r.game_id not in res.index or pd.isna(res.at[r.game_id, "total"]):
            continue
        total, close = res.at[r.game_id, "total"], res.at[r.game_id, "total_line"]
        win = total < r.line if r.side == "under" else total > r.line
        push = total == r.line
        log.at[i, "final_total"] = total
        log.at[i, "outcome"] = "push" if push else ("win" if win else "loss")
        log.at[i, "profit_units"] = 0.0 if push else (r.units * (american_to_decimal(r.price) - 1) if win else -r.units)
        log.at[i, "close_line"] = close
        log.at[i, "clv_pts"] = (r.line - close) if r.side == "under" else (close - r.line)
    return log


def main():
    now = pd.Timestamp(datetime.now(timezone.utc))
    now_et = now.tz_convert("America/New_York")
    day = bet_day(now_et)
    feats, _, _ = load_all()
    odds, credits_left = live_odds(feats)
    kick = pd.to_datetime(odds.commence_time)
    odds = odds[(kick > now) & (kick <= now + pd.Timedelta(days=7))]  # this week only
    print(f"{len(odds)} upcoming games with FanDuel lines; {credits_left} Odds API credits left")

    up = feats.set_index("game_id").loc[odds.game_id].reset_index()
    up = up.merge(odds.drop(columns=["home_team", "away_team", "gameday"]), on="game_id")
    fc = live_forecasts(up, now)
    for gid, w in fc.items():
        for k, v in w.items():
            up.loc[up.game_id == gid, k] = v
    up["forecast"] = up.get("forecast", pd.Series(index=up.index, dtype=object))
    up = environment_features(up)

    model, sd = fit_latest(feats, TOTALS_FEATURES, target="total", line="total_line")
    line = up["fanduel_totals_over_point"]
    x = with_line(up, total=line)
    x["pred_total"] = x.total_line + predict_dev(model, x[TOTALS_FEATURES].fillna(0))

    board, picks = [], []
    for g in x.itertuples():
        if pd.isna(g.total_line) or pd.isna(getattr(g, "fanduel_totals_over_price", np.nan)):
            continue
        po, pu, pp = cover_probs(g.pred_total, g.total_line, sd)
        bo, bu, _ = cover_probs(g.total_line, g.total_line, sd)
        mo, _ = no_vig_probs(g.fanduel_totals_over_price, g.fanduel_totals_under_price)
        q = mo + po / (po + pu) - bo / (bo + bu)
        sides = {"over": (q * (1 - pp), g.fanduel_totals_over_price),
                 "under": ((1 - q) * (1 - pp), g.fanduel_totals_under_price)}
        evs = {s: expected_value(p, pp, price) for s, (p, price) in sides.items()}
        best = max(evs, key=evs.get)
        outdoor = g.is_indoor == 0
        has_fc = outdoor and isinstance(g.forecast, str)
        rule = bool(has_fc and (g.wind > RULE_WIND or g.precip > RULE_RAIN))
        env = "Indoors" if not outdoor else (
            f"{g.temp:.0f}°F, wind {g.wind:.0f} mph" + (", rain" if g.precip > RULE_RAIN else "")
            + (", snow" if g.snow > 0.1 else "") if has_fc else "Outdoors, no forecast yet")
        kick = pd.Timestamp(g.commence_time).tz_convert("America/New_York")
        row = dict(game_id=g.game_id, kickoff=kick.strftime("%a %b %d %I:%M %p ET"), kickoff_utc=g.commence_time,
                   game=f"{g.away_team} @ {g.home_team}", fanduel_total=g.total_line,
                   over_price=g.fanduel_totals_over_price, under_price=g.fanduel_totals_under_price,
                   pinnacle_total=getattr(g, "pinnacle_totals_over_point", np.nan),
                   model_total=round(g.pred_total, 1), lean=best, lean_ev=round(evs[best], 4),
                   environment=env, forecast_source=g.forecast if has_fc else "",
                   wind_mph=round(g.wind, 1) if outdoor else np.nan,
                   rain_in=round(g.precip, 2) if outdoor else np.nan,
                   model_pick=evs[best] >= EDGE, rule_pick=rule)
        board.append(row)
        # Over bets lost in every backtest, so both systems only bet unders.
        model_ok = evs["under"] >= EDGE
        row["model_pick"] = model_ok
        for system, ok, side in (("model", model_ok, "under"), ("rule", rule, "under")):
            if not ok:
                continue
            p, price = sides[side]
            k = kelly_fraction(p, pp, price)
            units = round(min(k * 0.25, 0.03) * 100, 2) if system == "model" else 1.0
            picks.append(dict(logged_at=now.strftime("%Y-%m-%d %H:%M UTC"), bet_day=day,
                              season=g.season, week=g.week, game_id=g.game_id, game=row["game"],
                              kickoff=row["kickoff"], system=system, side=side, line=g.total_line,
                              price=price, model_ev=round(evs[side], 4), units=max(units, 0.25),
                              environment=env, final_total=np.nan, outcome=None, profit_units=np.nan,
                              close_line=np.nan, clv_pts=np.nan))

    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=list(picks[0]) if picks else [])
    new = pd.DataFrame(picks)
    if not new.empty:
        seen = set(zip(log.get("game_id", []), log.get("system", []), log.get("bet_day", [])))
        new = new[[(a, b, c) not in seen for a, b, c in zip(new.game_id, new.system, new.bet_day)]]
        log = pd.concat([log, new], ignore_index=True)
    log = settle(log, feats)
    LOG.parent.mkdir(exist_ok=True)
    log.to_csv(LOG, index=False)

    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.DataFrame(board).sort_values("kickoff_utc")
    b.to_csv(OUT / "board.csv", index=False)
    log.to_csv(OUT / "picks_log.csv", index=False)
    (OUT / "run.json").write_text(json.dumps({"run_at": now.strftime("%Y-%m-%d %H:%M UTC"), "bet_day": day,
                                              "games": len(b), "credits_left": credits_left}))
    print(f"Bet day: {day}. {int(b.model_pick.sum())} model picks, {int(b.rule_pick.sum())} rule picks "
          f"({len(new)} newly logged). Log: {len(log)} picks, "
          f"{log.outcome.notna().sum()} settled.")
    show = b[b.model_pick | b.rule_pick][["kickoff", "game", "fanduel_total", "model_total", "lean", "lean_ev",
                                         "environment", "model_pick", "rule_pick"]]
    print(show.to_string(index=False) if len(show) else "No picks this run.")


if __name__ == "__main__":
    main()
