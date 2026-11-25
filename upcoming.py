"""
The full model's view of upcoming games that already have a market line.

    python3 build_data.py --skip-pbp   # refresh lines first (pbp weekly is enough)
    python3 upcoming.py

For outdoor and open-roof stadiums it pulls the actual weather FORECAST for
kickoff (Open-Meteo, up to ~16 days ahead). Retractable roofs whose status
isn't known yet are treated as closed.

NOTE: in backtests this model does not beat closing lines (see README), so
this is a demonstration of the weekly workflow, NOT betting advice. Lines
come from nflverse's consensus data, not necessarily FanDuel's.
"""
import json
import urllib.request

import pandas as pd

from run_backtest import load_all, load_games
from tool.features import STEP2_FEATURES, TOTALS_FEATURES
from tool.model import fit_latest, add_model_columns, predict_dev
from tool.odds import cover_probs, no_vig_probs, expected_value
from tool.stadiums import COORDS

FORECAST_URL = ("https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                "&hourly=temperature_2m,precipitation,snowfall,wind_speed_10m,wind_gusts_10m"
                "&temperature_unit=fahrenheit&wind_speed_unit=mph&precipitation_unit=inch"
                "&timezone=America%2FNew_York&forecast_days=16")


def forecast_weather(games):
    up = games[games.result.isna() & games.spread_line.notna()
               & ~games.roof.fillna("closed").isin(["dome", "closed"])]
    rows = []
    for stadium, grp in up.groupby("stadium_id"):
        if stadium not in COORDS:
            continue
        lat, lon = COORDS[stadium]
        try:
            with urllib.request.urlopen(FORECAST_URL.format(lat=lat, lon=lon), timeout=30) as r:
                h = pd.DataFrame(json.load(r)["hourly"])
        except Exception as e:
            print(f"  (no forecast for {stadium}: {e})")
            continue
        h["time"] = pd.to_datetime(h.time)
        for g in grp.itertuples():
            kick = pd.Timestamp(f"{g.gameday} {g.gametime}").floor("h")
            w = h[(h.time >= kick) & (h.time < kick + pd.Timedelta(hours=4))]
            if w.empty:
                continue  # beyond forecast range
            rows.append(dict(game_id=g.game_id, w_temp=w.temperature_2m.mean(),
                             w_wind=w.wind_speed_10m.mean(), w_gust=w.wind_gusts_10m.max(),
                             w_precip=w.precipitation.sum(), w_snow=w.snowfall.sum()))
    return pd.DataFrame(rows)


def main():
    games = load_games()
    fc = forecast_weather(games)
    feats, _, _ = load_all(extra_weather=fc)
    model, sd = fit_latest(feats, STEP2_FEATURES)
    up = add_model_columns(feats[feats.result.isna() & feats.spread_line.notna()
                                 & feats.home_spread_odds.notna()])
    if up.empty:
        print("No upcoming games with lines. Run: python3 build_data.py --skip-pbp")
        return
    up = up.assign(pred_margin=up.spread_line + predict_dev(model, up[STEP2_FEATURES]))
    tmodel, tsd = fit_latest(feats, TOTALS_FEATURES, target="total", line="total_line")
    up = up.assign(pred_total=up.total_line + predict_dev(tmodel, up[TOTALS_FEATURES].fillna(0)))

    rows = []
    for g in up.itertuples():
        ph, pa, pp = cover_probs(g.pred_margin, g.spread_line, sd)
        bh, ba, _ = cover_probs(g.spread_line, g.spread_line, sd)
        mh, _ = no_vig_probs(g.home_spread_odds, g.away_spread_odds)
        q = mh + ph / (ph + pa) - bh / (bh + ba)
        ev_h = expected_value(q * (1 - pp), pp, g.home_spread_odds)
        ev_a = expected_value((1 - q) * (1 - pp), pp, g.away_spread_odds)
        # Over/under, same market-anchored logic
        po, pu, ppt = cover_probs(g.pred_total, g.total_line, tsd)
        bo, bu, _ = cover_probs(g.total_line, g.total_line, tsd)
        mo, _ = no_vig_probs(g.over_odds, g.under_odds) if not pd.isna(g.over_odds) else (0.5, 0.5)
        qo = mo + po / (po + pu) - bo / (bo + bu)
        ev_o = expected_value(qo * (1 - ppt), ppt, g.over_odds if not pd.isna(g.over_odds) else -110)
        ev_u = expected_value((1 - qo) * (1 - ppt), ppt, g.under_odds if not pd.isna(g.under_odds) else -110)
        if g.is_indoor:
            env = "indoors"
        elif pd.isna(g.w_temp):
            env = "outdoors (no forecast yet)"
        else:
            env = f"{g.w_temp:.0f}F wind {g.w_wind:.0f}mph" + (" rain" if g.w_precip > 0.05 else "") \
                  + (" snow" if g.w_snow > 0.1 else "")
        rows.append({
            "date": g.gameday, "game": f"{g.away_team} @ {g.home_team}",
            "line": f"{g.home_team} {-g.spread_line:+g}",
            "model": f"{g.home_team} {-g.pred_margin:+.1f}",
            "spread EV": f"{'home' if ev_h > ev_a else 'away'} {max(ev_h, ev_a):+.1%}",
            "total": f"{g.total_line:g}",
            "model total": f"{g.pred_total:.1f}",
            "total EV": f"{'over' if ev_o > ev_u else 'under'} {max(ev_o, ev_u):+.1%}",
            "environment": env,
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))
    print("\nEV = expected profit per $1 at the listed price. Spread EVs have not held up in")
    print("backtests; over/under EVs of +1-3% were roughly break-even to slightly positive.")


if __name__ == "__main__":
    main()
