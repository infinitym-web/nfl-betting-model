"""
Download FanDuel's REAL historical alternate totals (available from May 2023).

    python3 fetch_alt_odds.py

To keep credit use low it only fetches the games the strategy would consider:
outdoor games with rain or wind over 10 mph, plus a random sample of other
outdoor games for comparison. One request per game at the Saturday 10am
snapshot, 10 credits each.
"""
import argparse
import json
import time
import urllib.error
from pathlib import Path

import pandas as pd

from fetch_odds import api_get, MIN_CREDITS_LEFT
from run_backtest import load_all

HERE = Path(__file__).parent
RAW = HERE / "data" / "alt_odds_raw"
URL = ("https://api.the-odds-api.com/v4/historical/sports/americanfootball_nfl/events/{eid}/odds"
       "?markets=alternate_totals&bookmakers=fanduel&oddsFormat=american&date={date}")
SAMPLE_NORMAL = 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pregame", action="store_true",
                    help="fetch 30 minutes before kickoff instead of Saturday 10am "
                         "(FanDuel's alternate lines are often missing from Saturday snapshots)")
    args = ap.parse_args()
    raw = RAW.parent / "alt_odds_raw_pregame" if args.pregame else RAW
    out_csv = HERE / "data" / ("fanduel_alt_totals_pregame.csv" if args.pregame else "fanduel_alt_totals.csv")
    feats, _, _ = load_all()
    odds = pd.read_csv(HERE / "data" / "odds_history.csv")
    late = odds[(odds.snapshot == "late") & (odds.book == "fanduel") & (odds.market == "totals")
                & (pd.to_datetime(odds.commence_time) >= "2023-05-04")]
    late = late.drop_duplicates("event_id")
    late["gameday"] = (pd.to_datetime(late.commence_time).dt.tz_convert("America/New_York")
                       .dt.strftime("%Y-%m-%d"))
    g = late.merge(feats[["season", "home_team", "away_team", "gameday", "game_id", "is_indoor", "wet", "windy"]],
                   on=["season", "home_team", "away_team", "gameday"])
    g = g[g.is_indoor == 0]
    bad = g[(g.wet > 0) | (g.windy > 0)]
    normal = g.drop(bad.index).sample(min(SAMPLE_NORMAL, len(g) - len(bad)), random_state=1)
    todo = pd.concat([bad, normal])
    raw.mkdir(parents=True, exist_ok=True)
    todo = todo[[not (raw / f"{gid}.json").exists() for gid in todo.game_id]]
    print(f"{len(bad)} bad-weather + {len(normal)} comparison games; {len(todo)} to download "
          f"(~{len(todo) * 10} credits)")
    for i, r in enumerate(todo.itertuples(), 1):
        try:
            when = (pd.Timestamp(r.commence_time) - pd.Timedelta(minutes=30)) if args.pregame \
                else pd.Timestamp(r.snapshot_time)
            data, left = api_get(URL.format(eid=r.event_id, date=when.strftime("%Y-%m-%dT%H:%M:%SZ")))
        except urllib.error.HTTPError as e:
            print(f"  {r.game_id}: HTTP {e.code}")
            continue
        (raw / f"{r.game_id}.json").write_text(json.dumps(data))
        if i % 25 == 0 or i == len(todo):
            print(f"  {i}/{len(todo)} done, {left} credits left", flush=True)
        if 0 <= left < MIN_CREDITS_LEFT:
            print(f"Stopping: only {left} credits left.")
            break
        time.sleep(0.3)

    rows = []
    for f in raw.glob("*.json"):
        d = json.loads(f.read_text()).get("data") or {}
        for b in d.get("bookmakers", []):
            for m in b.get("markets", []):
                for o in m.get("outcomes", []):
                    rows.append(dict(game_id=f.stem, side=o["name"].lower(), point=o["point"], price=o["price"]))
    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False)
    print(f"Saved {len(out):,} alternate-total prices for {out.game_id.nunique()} games")


if __name__ == "__main__":
    main()
