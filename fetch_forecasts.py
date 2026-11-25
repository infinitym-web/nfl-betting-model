"""
Download historical weather FORECASTS (what a bettor could have known), from
Open-Meteo's Previous Runs archive, for every outdoor / open-roof game since 2020.

    python3 fetch_forecasts.py          # resumes; waits out the free daily limit
    python3 fetch_forecasts.py --parse  # only rebuild data/weather_forecast.csv

For each game it saves the kickoff-window forecast as issued:
  6 days before  (matches a Monday bet)
  3 days before  (Thursday bet)
  1 day before   (Saturday / Sunday bet)
Models: 2024 onward uses Open-Meteo's best blend; before 2024 only Japan's JMA GSM
global model is archived (coarser, so those forecasts are a bit less accurate
than what was really available - a conservative test).
"""
import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from tool.stadiums import COORDS

HERE = Path(__file__).parent
RAW = HERE / "data" / "forecast_raw"
LEADS = (1, 3, 6)
VARS = ["temperature_2m", "precipitation", "wind_speed_10m"]
URL = ("https://previous-runs-api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
       "&hourly={hourly}&start_date={day}&end_date={day}&timezone=America%2FNew_York"
       "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch{model}")
MAX_RUNTIME_S = 105 * 60


def fetch(games):
    RAW.mkdir(parents=True, exist_ok=True)
    g = games[(games.season >= 2020) & games.result.notna() & ~games.roof.isin(["dome", "closed"])
              & games.stadium_id.isin(COORDS)]
    todo = [r for r in g.itertuples() if not (RAW / f"{r.game_id}.json").exists()]
    print(f"{len(g)} outdoor/open-roof games since 2020, {len(todo)} still to download", flush=True)
    hourly = ",".join(f"{v}_previous_day{d}" for v in VARS for d in LEADS)
    start, done = time.time(), 0
    i = 0
    while i < len(todo):
        if time.time() - start > MAX_RUNTIME_S:
            print(f"Time limit reached; {len(todo) - i} games left. Run again to resume.")
            return
        r = todo[i]
        lat, lon = COORDS[r.stadium_id]
        model = "" if r.season >= 2024 else "&models=jma_gsm"
        url = URL.format(lat=lat, lon=lon, hourly=hourly, day=r.gameday, model=model)
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.load(resp)
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="ignore")
            if "Daily" in body:
                print(f"  daily limit hit after {done} games; waiting 15 min...", flush=True)
                time.sleep(900)
            elif e.code == 429:
                time.sleep(60)
            else:
                print(f"  {r.game_id}: HTTP {e.code} {body[:120]}")
                i += 1
            continue
        except Exception:
            time.sleep(10)
            continue
        (RAW / f"{r.game_id}.json").write_text(json.dumps(data))
        done += 1
        i += 1
        if done % 100 == 0:
            print(f"  {done} downloaded, {len(todo) - i} left", flush=True)
        time.sleep(0.5)
    print(f"All forecasts downloaded ({done} this run).")


def parse(games):
    gt = games.set_index("game_id")
    rows = []
    for f in RAW.glob("*.json"):
        gid = f.stem
        h = pd.DataFrame(json.loads(f.read_text()).get("hourly", {}))
        if h.empty or gid not in gt.index:
            continue
        h["time"] = pd.to_datetime(h.time)
        kick = pd.Timestamp(f"{gt.at[gid, 'gameday']} {gt.at[gid, 'gametime'] or '13:00'}").floor("h")
        w = h[(h.time >= kick) & (h.time < kick + pd.Timedelta(hours=4))]
        for d in LEADS:
            t = w[f"temperature_2m_previous_day{d}"]
            if t.notna().sum() < 2:
                continue
            precip = w[f"precipitation_previous_day{d}"].sum()
            temp = t.mean()
            rows.append(dict(game_id=gid, lead_days=d, w_temp=temp,
                             w_wind=w[f"wind_speed_10m_previous_day{d}"].mean(),
                             w_precip=precip, w_snow=precip * 10 if temp <= 33 else 0.0,
                             w_gust=float("nan")))
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "weather_forecast.csv", index=False)
    print(f"Saved forecasts for {out.game_id.nunique()} games "
          f"({out.groupby('lead_days').size().to_dict()} by lead days)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()
    games = pd.read_csv(HERE / "data" / "games.csv")
    games["gametime"] = games.gametime.fillna("13:00")
    if not args.parse:
        fetch(games)
    parse(games)


if __name__ == "__main__":
    main()
