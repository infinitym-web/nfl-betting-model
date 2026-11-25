"""
How does the game environment relate to scoring and to betting results?

    python3 analyze_weather.py

For each environment bucket it shows:
  points      average combined points scored
  vs_total    average (actual total - market total line): negative = went under
  under%      how often the game went UNDER the total
  home_cover% how often the home team covered the spread
If the market already prices weather in, vs_total and under% stay near 0 / 50%.
"""
import pandas as pd

from run_backtest import load_all

f, _, _ = load_all()
f = f[f.result.notna() & f.total_line.notna() & (f.season >= 2006)].copy()
f["vs_total"] = f.total - f.total_line
f["under"] = (f.total < f.total_line).astype(float)
f.loc[f.total == f.total_line, "under"] = float("nan")
f["home_cover"] = (f.result > f.spread_line).astype(float)
f.loc[f.result == f.spread_line, "home_cover"] = float("nan")

outdoor = f.is_indoor == 0
buckets = {
    "Indoors (dome / roof closed)": f.is_indoor == 1,
    "Outdoors, all": outdoor,
    "Outdoors, wind < 10 mph": outdoor & (f.wind < 10),
    "Outdoors, wind 10-15 mph": outdoor & f.wind.between(10, 15),
    "Outdoors, wind 15-20 mph": outdoor & (f.wind > 15) & (f.wind <= 20),
    "Outdoors, wind > 20 mph": outdoor & (f.wind > 20),
    "Outdoors, rain/snow": outdoor & ((f.precip > 0.05) | (f.snow > 0.1)),
    "Outdoors, below 32F": outdoor & (f.temp < 32),
    "Outdoors, above 85F": outdoor & (f.temp > 85),
    "Dome/warm team visiting, below 40F": outdoor & (f.temp < 40) & (f.cold_x_softteam > 0),
}
rows = []
for name, m in buckets.items():
    s = f[m]
    rows.append({"environment": name, "games": len(s),
                 "points": round(s.total.mean(), 1),
                 "vs_total": round(s.vs_total.mean(), 2),
                 "under%": f"{s.under.mean():.1%}",
                 "home_cover%": f"{s.home_cover.mean():.1%}"})
pd.set_option("display.width", 200)
print("Games 2006-present with a total line\n")
print(pd.DataFrame(rows).to_string(index=False))
print("\nNote: weather is the recorded kickoff weather, a stand-in for the forecast"
      " that was available before the game.")
