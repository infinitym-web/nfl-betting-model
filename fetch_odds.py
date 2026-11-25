"""
Download historical sportsbook odds from The Odds API (paid plan).

    python3 fetch_odds.py            # main lines: 4 snapshots per NFL week, 2020-now
    python3 fetch_odds.py --parse    # only rebuild data/odds_history.csv from saved files

The API key is NOT in this code: the environment's network settings add it
to every request to api.the-odds-api.com automatically (x-api-key header).
If an ODDS_API_KEY environment variable is set instead, it's used.

Snapshots each week (US Eastern time):
  open   Monday 10am     early-week lines
  mid    Thursday 11am
  late   Saturday 10am   weather forecasts are reliable by now
  close  Sunday 12:45pm  final line for 1pm games (most games)
Cost: 10 credits x 2 markets per snapshot (up to 10 bookmakers count as one).
Every response is saved in data/odds_raw/, so nothing is downloaded twice.
"""
import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

HERE = Path(__file__).parent
RAW = HERE / "data" / "odds_raw"
BOOKS = "fanduel,draftkings,pinnacle,betmgm,williamhill_us,betonlineag,lowvig,bovada"
URL = ("https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/odds"
       "?markets=spreads,totals&bookmakers=" + BOOKS + "&oddsFormat=american&date={date}")
MIN_CREDITS_LEFT = 3000

TEAM_ABBR = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF", "Carolina Panthers": "CAR", "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE", "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC", "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC",
    "Los Angeles Rams": "LA", "Miami Dolphins": "MIA", "Minnesota Vikings": "MIN",
    "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT",
    "San Francisco 49ers": "SF", "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
    "Washington Football Team": "WAS", "Washington Redskins": "WAS",
}


def api_get(url):
    req = urllib.request.Request(url, headers={"Accept-Encoding": "identity"})
    if os.environ.get("ODDS_API_KEY"):
        req.add_header("x-api-key", os.environ["ODDS_API_KEY"])
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r), int(r.headers.get("x-requests-remaining", -1))


def snapshot_times(games):
    """Four snapshot times (UTC) per NFL week, for weeks already played."""
    g = games[(games.season >= 2020) & games.result.notna()].copy()
    g["day"] = pd.to_datetime(g.gameday)
    out = []
    for (season, week), wk in g.groupby(["season", "week"]):
        sunday = wk.day[wk.day.dt.dayofweek == 6]
        sunday = sunday.min() if len(sunday) else wk.day.max()
        monday = sunday - pd.Timedelta(days=6)
        thursday = sunday - pd.Timedelta(days=3)
        saturday = sunday - pd.Timedelta(days=1)
        for label, day, hm in (("open", monday, "10:00"), ("mid", thursday, "11:00"),
                               ("late", saturday, "10:00"), ("close", sunday, "12:45")):
            t = pd.Timestamp(f"{day.date()} {hm}").tz_localize("America/New_York").tz_convert("UTC")
            out.append(dict(season=season, week=week, label=label, time=t))
    return pd.DataFrame(out)


def fetch(snaps):
    RAW.mkdir(parents=True, exist_ok=True)
    todo = [s for s in snaps.itertuples()
            if not (RAW / f"{s.season}_{s.week:02d}_{s.label}.json").exists()]
    print(f"{len(snaps)} snapshots planned, {len(todo)} still to download "
          f"(~{len(todo) * 20} credits)")
    for i, s in enumerate(todo, 1):
        date = s.time.strftime("%Y-%m-%dT%H:%M:%SZ")
        try:
            data, left = api_get(URL.format(date=date))
        except urllib.error.HTTPError as e:
            print(f"  {s.season} wk{s.week} {s.label}: HTTP {e.code} {e.read()[:200]!r}")
            if e.code in (401, 403):
                return
            time.sleep(5)
            continue
        (RAW / f"{s.season}_{s.week:02d}_{s.label}.json").write_text(json.dumps(data))
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done, {left} credits left", flush=True)
        if 0 <= left < MIN_CREDITS_LEFT:
            print(f"Stopping: only {left} credits left (keeping a reserve of {MIN_CREDITS_LEFT}).")
            return
        time.sleep(0.3)


def parse():
    """Flatten saved snapshots into one row per game x book x market x side."""
    rows = []
    for f in sorted(RAW.glob("*_*_*.json")):
        season, week, label = f.stem.split("_")
        snap = json.loads(f.read_text())
        for ev in snap.get("data", []):
            home, away = TEAM_ABBR.get(ev["home_team"]), TEAM_ABBR.get(ev["away_team"])
            if not home or not away:
                continue
            for b in ev.get("bookmakers", []):
                for m in b.get("markets", []):
                    for o in m.get("outcomes", []):
                        side = ("home" if TEAM_ABBR.get(o["name"]) == home else "away") \
                            if m["key"] == "spreads" else o["name"].lower()
                        rows.append(dict(season=int(season), week=int(week), snapshot=label,
                                         snapshot_time=snap["timestamp"], event_id=ev["id"],
                                         commence_time=ev["commence_time"], home_team=home,
                                         away_team=away, book=b["key"], market=m["key"],
                                         side=side, point=o.get("point"), price=o["price"]))
    df = pd.DataFrame(rows)
    # Keep only snapshots taken BEFORE kickoff (a snapshot can list games already started)
    df = df[pd.to_datetime(df.snapshot_time) < pd.to_datetime(df.commence_time)]
    out = HERE / "data" / "odds_history.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df):,} odds rows for {df.event_id.nunique():,} games to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()
    if not args.parse:
        games = pd.read_csv(HERE / "data" / "games.csv")
        fetch(snapshot_times(games))
    parse()


if __name__ == "__main__":
    main()
