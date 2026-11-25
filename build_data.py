"""
Builds the data files the model needs. Run it once, and again each week
to pick up new games.

    python3 build_data.py            # refresh everything (weather is cached)
    python3 build_data.py --skip-pbp # only refresh games + weather

Produces, in data/:
  games.csv        schedule, scores, betting lines, roof, starting QBs (nflverse)
  team_games.csv   per team per game: offensive plays, EPA, success rate
  qb_games.csv     per QB per game: dropbacks and EPA
  weather.csv      kickoff-window weather for every outdoor / open-roof game

WHAT IS EPA? "Expected Points Added": every play is scored by how much it
changed the offence's expected points on that drive. A 3rd-and-1 conversion
is worth more than a 3rd-and-10 short gain. Averaged per play it is one of
the best publicly available measures of how good an offence or defence is.
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from tool.stadiums import COORDS

DATA = Path(__file__).parent / "data"
RAW = DATA / "raw_pbp"
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"


def download(url, dest):
    with urllib.request.urlopen(url, timeout=120) as r:
        dest.write_bytes(r.read())


def build_pbp(seasons, current_season):
    RAW.mkdir(exist_ok=True)
    team_rows, qb_rows = [], []
    for y in seasons:
        f = RAW / f"pbp_{y}.parquet"
        if not f.exists() or y == current_season:  # current season changes weekly
            print(f"  downloading play-by-play {y}")
            download(PBP_URL.format(y), f)
        p = pd.read_parquet(f, columns=["game_id", "posteam", "defteam", "epa", "success",
                                        "play_type", "qb_dropback", "passer_player_id", "qb_epa"])
        p = p[p.play_type.isin(["pass", "run"]) & p.epa.notna() & p.posteam.notna()]
        p["is_pass"] = (p.qb_dropback == 1)
        p["pass_epa"] = p.epa.where(p.is_pass, 0.0)
        t = p.groupby(["game_id", "posteam", "defteam"]).agg(
            plays=("epa", "size"), epa=("epa", "sum"), success=("success", "sum"),
            dropbacks=("is_pass", "sum"), pass_epa=("pass_epa", "sum")).reset_index()
        team_rows.append(t)
        q = p[p.is_pass & p.passer_player_id.notna()].groupby(["game_id", "passer_player_id"]).agg(
            dropbacks=("qb_epa", "size"), qb_epa=("qb_epa", "sum")).reset_index()
        qb_rows.append(q)
    pd.concat(team_rows).to_csv(DATA / "team_games.csv", index=False)
    pd.concat(qb_rows).to_csv(DATA / "qb_games.csv", index=False)


WEATHER_URL = ("https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}"
               "&start_date={start}&end_date={end}"
               "&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m"
               "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
               "&timezone=America%2FNew_York")


def build_weather(games):
    """
    Actual recorded weather (ERA5 reanalysis via Open-Meteo) for the kickoff
    window (kickoff hour + next 3 hours) of every game not played under a
    closed roof. Kickoff times in nflverse are US Eastern, so we request hours
    in Eastern time too. Cached: only games not already in weather.csv are fetched.
    """
    out = DATA / "weather.csv"
    have = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["game_id"])
    need = games[games.result.notna() & ~games.roof.isin(["dome", "closed"])
                 & ~games.game_id.isin(have.game_id) & games.stadium_id.isin(COORDS)]
    if need.empty:
        print("  weather: up to date")
        return
    print(f"  weather: fetching {len(need)} games")
    groups = list(need.groupby(["stadium_id", "season"]))
    saved = 0
    for i, ((stadium, season), grp) in enumerate(groups, 1):
        lat, lon = COORDS[stadium]
        url = WEATHER_URL.format(lat=lat, lon=lon, start=grp.gameday.min(), end=grp.gameday.max())
        h = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    h = json.load(r)["hourly"]
                break
            except urllib.error.HTTPError as e:
                body = e.read().decode(errors="ignore")
                if "Daily API request limit" in body:
                    print(f"  weather: free daily limit reached after {saved} new games. "
                          f"Run build_data.py --skip-pbp again tomorrow; it resumes where it stopped.")
                    return
                time.sleep(10 * (attempt + 1))  # per-minute/hour limit: back off
            except Exception:  # network blip
                time.sleep(5 * (attempt + 1))
        if h is None:
            print(f"    skipped {stadium} {season} (will retry next run)")
            continue
        rows = []
        hourly = pd.DataFrame(h)
        hourly["time"] = pd.to_datetime(hourly["time"])
        for g in grp.itertuples():
            kick = pd.Timestamp(f"{g.gameday} {g.gametime}").floor("h")
            w = hourly[(hourly.time >= kick) & (hourly.time < kick + pd.Timedelta(hours=4))]
            if w.empty:
                continue
            rows.append(dict(game_id=g.game_id, w_temp=w.temperature_2m.mean(),
                             w_wind=w.wind_speed_10m.mean(), w_gust=w.wind_gusts_10m.max(),
                             w_precip=w.precipitation.sum(), w_snow=w.snowfall.sum()))
        # Save after every stadium-season so progress is never lost.
        new = pd.DataFrame(rows)
        new.to_csv(out, mode="a", header=not out.exists(), index=False)
        saved += len(new)
        if i % 25 == 0:
            print(f"    {i}/{len(groups)} stadium-seasons, {saved} games saved", flush=True)
        time.sleep(1.0)  # be polite to the free API
    print(f"  weather: saved {saved} new games")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-pbp", action="store_true")
    args = ap.parse_args()
    DATA.mkdir(exist_ok=True)
    print("Downloading schedule + lines")
    download(GAMES_URL, DATA / "games.csv")
    games = pd.read_csv(DATA / "games.csv")
    games["gametime"] = games.gametime.fillna("13:00")
    current = int(games[games.result.notna()].season.max())
    if not args.skip_pbp:
        print("Building play-by-play summaries")
        build_pbp(range(1999, current + 1), current)
    print("Building weather")
    build_weather(games)
    print("Done.")


if __name__ == "__main__":
    main()
