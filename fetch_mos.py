"""
Free historical weather FORECASTS: NOAA's official airport forecasts (MOS,
"Model Output Statistics"), archived by Iowa State's Mesonet.

    python3 fetch_mos.py            # download (cached) + build data/weather_forecast_mos.csv
    python3 fetch_mos.py --parse    # rebuild the CSV only

Two NOAA products are used:
  GFS MOS  (short range)  every 3 hours, out to ~3 days   -> Saturday & Sunday bets
  MEX MOS  (extended)     every 12 hours, out to ~8 days  -> Monday & Thursday bets
For each game and each bet time (Mon 10am, Thu 11am, Sat 10am, Sun 12:45pm ET),
we use the newest forecast published at least 4 hours BEFORE the bet, so
nothing used was unavailable at the time.

Conversions: wind knots -> mph; NOAA's rain-amount category -> an inch estimate
(category midpoint, scaled to the 4-hour game window); snow when NOAA's precip
type is snow or the temperature is at or below 33F.
"""
import argparse
import io
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_odds import snapshot_times

HERE = Path(__file__).parent
RAW = HERE / "data" / "mos_raw"
URL = ("https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py?station={st}&model={model}"
       "&sts={start}T00:00Z&ets={end}T00:00Z&format=csv")
PUBLISH_DELAY = pd.Timedelta(hours=4)  # MOS is published ~3-4h after its model run

# Nearest airport with NOAA MOS for each outdoor / retractable-roof stadium
STATION = {
    "BAL00": "KBWI", "BOS00": "KBOS", "BUF00": "KBUF", "CAR00": "KCLT", "CHI98": "KMDW",
    "CIN00": "KCVG", "CLE00": "KCLE", "DEN00": "KDEN", "GNB00": "KGRB", "JAX00": "KJAX",
    "KAN00": "KMCI", "MIA00": "KMIA", "NAS00": "KBNA", "NYC01": "KEWR", "OAK00": "KOAK",
    "PHI00": "KPHL", "PIT00": "KPIT", "SDG00": "KSAN", "SEA00": "KBFI", "SFO01": "KSJC",
    "TAM00": "KTPA", "WAS00": "KDCA", "LAX99": "KLAX", "LAX97": "KLAX", "DAL00": "KDFW",
    "PHO00": "KPHX", "IND00": "KIND", "HOU00": "KHOU", "ATL97": "KATL", "MIN98": "KMSP",
}
# Rain-amount category -> inches (category midpoint)
Q_INCHES = {0: 0.0, 1: 0.05, 2: 0.17, 3: 0.37, 4: 0.75, 5: 1.4, 6: 2.5}


def download(games):
    RAW.mkdir(parents=True, exist_ok=True)
    g = games[(games.season >= 2020) & games.result.notna() & ~games.roof.isin(["dome", "closed"])
              & games.stadium_id.isin(STATION)]
    jobs = sorted({(STATION[s], int(y)) for s, y in zip(g.stadium_id, g.season)})
    print(f"{len(jobs)} station-seasons x 2 forecast products", flush=True)
    for i, (st, season) in enumerate(jobs, 1):
        for model in ("GFS", "MEX"):
            f = RAW / f"{st}_{season}_{model}.csv"
            if f.exists():
                continue
            url = URL.format(st=st, model=model, start=f"{season}-08-25", end=f"{season + 1}-02-20")
            for attempt in range(4):
                try:
                    with urllib.request.urlopen(url, timeout=180) as r:
                        f.write_bytes(r.read())
                    break
                except Exception as e:
                    print(f"  retry {st} {season} {model}: {e}", flush=True)
                    time.sleep(10 * (attempt + 1))
            time.sleep(1)
        if i % 20 == 0:
            print(f"  {i}/{len(jobs)} station-seasons done", flush=True)


def _load(st, season, model):
    f = RAW / f"{st}_{season}_{model}.csv"
    if not f.exists() or f.stat().st_size < 200:
        return None
    d = pd.read_csv(f, low_memory=False)
    d["runtime"] = pd.to_datetime(d.runtime, utc=True)
    d["ftime"] = pd.to_datetime(d.ftime, utc=True)
    return d


def _from_run(run, model, kick, end):
    """Kickoff-window weather from one forecast run."""
    run = run.sort_values("ftime")
    t = run.set_index("ftime")
    window = pd.date_range(kick, end, freq="h")
    def interp(col):
        s = pd.to_numeric(t[col], errors="coerce").dropna()
        if len(s) < 2 or s.index.min() > kick or s.index.max() < end:
            return np.nan
        full = s.reindex(s.index.union(window)).interpolate(method="time")
        return float(full.reindex(window).mean())
    temp = interp("tmp")
    wind = interp("wsp") * 1.151  # knots -> mph
    qcol, hours = ("q06", 6) if model == "GFS" else ("q12", 12)
    q = pd.to_numeric(t[qcol], errors="coerce")
    # precipitation periods (ending at ftime) that overlap the game window
    over = q[(q.index > kick) & (q.index - pd.Timedelta(hours=hours) < end)].dropna()
    precip = float(np.mean([Q_INCHES.get(int(v), 0.0) for v in over]) * 4 / hours) if len(over) else 0.0
    typ = run.loc[(run.ftime >= kick) & (run.ftime <= end + pd.Timedelta(hours=hours)), "typ"].astype(str)
    snowy = typ.str.strip().eq("S").any() or (temp == temp and temp <= 33)
    return dict(w_temp=temp, w_wind=wind, w_precip=precip, w_snow=precip * 10 if snowy else 0.0)


def parse(games):
    g = games[(games.season >= 2020) & games.result.notna() & ~games.roof.isin(["dome", "closed"])
              & games.stadium_id.isin(STATION)].copy()
    g["kick"] = [pd.Timestamp(f"{d} {t}").tz_localize("America/New_York").tz_convert("UTC")
                 for d, t in zip(g.gameday, g.gametime)]
    snaps = snapshot_times(games).set_index(["season", "week", "label"]).time
    cache, rows = {}, []
    for r in g.itertuples():
        st = STATION[r.stadium_id]
        kick, end = r.kick.floor("h"), r.kick.floor("h") + pd.Timedelta(hours=3)
        for label in ("open", "mid", "late", "close"):
            bet_time = snaps.get((r.season, r.week, label))
            if bet_time is None or bet_time >= r.kick:
                continue
            for model in ("GFS", "MEX"):  # prefer the short-range product when it reaches the game
                key = (st, r.season, model)
                if key not in cache:
                    cache[key] = _load(st, r.season, model)
                d = cache[key]
                if d is None:
                    continue
                runs = d[d.runtime + PUBLISH_DELAY <= bet_time]
                if runs.empty:
                    continue
                latest = runs[runs.runtime == runs.runtime.max()]
                if latest.ftime.max() < end:
                    continue  # this run doesn't reach kickoff; try the extended product
                w = _from_run(latest, model, kick, end)
                if w["w_temp"] != w["w_temp"]:
                    continue
                rows.append(dict(game_id=r.game_id, snapshot=label, product=model,
                                 hours_ahead=round((r.kick - latest.runtime.iloc[0]).total_seconds() / 3600),
                                 w_gust=np.nan, **w))
                break
    out = pd.DataFrame(rows)
    out.to_csv(HERE / "data" / "weather_forecast_mos.csv", index=False)
    print(f"Saved {len(out):,} forecasts for {out.game_id.nunique():,} games")
    print(out.groupby("snapshot").agg(games=("game_id", "size"),
                                      product=("product", lambda p: p.value_counts().to_dict()),
                                      median_hours_ahead=("hours_ahead", "median")).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parse", action="store_true")
    args = ap.parse_args()
    games = pd.read_csv(HERE / "data" / "games.csv")
    games["gametime"] = games.gametime.fillna("13:00")
    if not args.parse:
        download(games)
    parse(games)


if __name__ == "__main__":
    main()
