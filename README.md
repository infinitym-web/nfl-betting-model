# NFL Betting Model

Finds NFL bets worth more than they cost (point spreads and over/unders), tested honestly on 2010–2025.

## Setup

```
pip install -r requirements.txt
```

## How to run

```
python3 build_data.py                # download / refresh all data (first run: a few minutes)
python3 run_backtest.py              # spread models: backtest report
python3 run_totals_backtest.py       # over/under models: backtest report
python3 analyze_weather.py           # how weather relates to scoring and results
python3 run_alt_backtest.py          # alternate spreads / totals vs main lines
python3 fetch_mos.py                 # free historical NOAA airport forecasts (Iowa State Mesonet)
python3 run_forecast_backtest.py     # FanDuel over/unders with forecast vs recorded weather
python3 upcoming.py                  # this week's games: model vs market, forecasts, alt-under targets
```

Weekly refresh: `python3 build_data.py`. Historical weather comes mostly from Meteostat's bulk archive (a few minutes, no daily cap). Open-Meteo fills the remaining gaps, mainly the current season, and has a daily request limit. If it stops partway, run it again the next day and it picks up where it left off.

## Data

| File | What | Source |
|---|---|---|
| `games.csv` | schedule, scores, closing lines and prices, roof, starting QBs | nflverse |
| `team_games.csv` | each team's plays, EPA, success rate per game | nflverse play-by-play |
| `qb_games.csv` | each QB's dropbacks and EPA per game | nflverse play-by-play |
| `weather.csv` | temperature, wind, rain, snow at kickoff (outdoor and open-roof games) | Meteostat weather stations (93% of games); Open-Meteo for the rest |

**EPA (Expected Points Added)** scores every play by how much it changed the offense's expected points. Averaged per play, it's one of the best public measures of team quality.

## How the models work

1. **Pre-game features only.** Each rating is read before the game and updated after it. Features include:
   * Elo ratings and rolling EPA per play (offense and defense)
   * the listed starting QB's rolling EPA, and whether he's a new starter
   * rest days, short weeks, byes, travel distance, time zones
   * a rolling estimate of home-field advantage
   * dome, retractable roof or outdoors; temperature, wind, gusts, rain, snow; dome or warm-weather teams playing in the cold
   * for totals, also pace (plays per game) and rolling points scored and allowed
2. **Predict how wrong the line is, not the score.** With no information, the model says "the market is right." A penalized regression (ridge) lets a feature move the prediction only if it has helped in earlier seasons.
3. **Centered on the line.** Scores are lopsided: shootouts pull the average total above the line even though overs and unders each hit about half the time. The model ignores that average gap and keeps only game-specific signals.
4. **Probabilities start from the market's price.** Sportsbooks shade prices around 3 and 7 for good reason. The model starts from the market's no-vig probability and only nudges it.
5. **Walk-forward testing.** Each season is predicted by a model trained only on earlier seasons.
6. **Bet sizing.** ¼ Kelly, capped at 3% of the bankroll per bet. All bets in a week are sized from the bankroll at the start of that week.

## Results (2010–2025, closing lines)

**Spreads: no edge.**

| Model | Avg miss (pts) | Bets (3% edge) | ROI |
|---|---|---|---|
| Betting market | 10.054 | | |
| v1 Elo, raw | 10.392 | 3,208 | −1.0% |
| Step 1: Elo calibrated to market | 10.057 | 3 | n/a |
| Step 2: full model | 10.083 | 528 | −4.0% |

Elo, EPA, QB, rest, travel and weather add nothing beyond the closing spread; the closing line already reflects them. The full model only bet heavily in 2010–2013, when it had little history and was fooled by noise. After that it learned to trust the market and rarely bets.

**Over/unders: small, uncertain edge, mainly from weather.**

| Model | Avg miss (pts) | Bets (1% edge) | Win% | ROI | Luck range (90%) |
|---|---|---|---|---|---|
| Betting market | 10.471 | | | | |
| Points averages, raw | 10.721 | 3,511 | 49.9% | −2.6% | −6.4% to +1.0% |
| Full model, no weather | 10.468 | 633 | 51.7% | −0.4% | −7.4% to +6.8% |
| **Full model + weather** | **10.453** | **1,528** | **54.8%** | **+4.8%** | −0.7% to +10.5% |

* Adding weather turns break-even into a profit. The strongest effects are **rain** (rain or snow outdoors: under 57.5%) and **wind** (10–15 mph: under 57%; over 20 mph: under 60%).
* It made money in 11 of 16 seasons, with about the same ROI in 2010–17 (+4.6%) and 2018–25 (+4.9%). **2024 and 2025 were losing years.**
* The luck range still just includes zero, so this isn't proven yet.

**Alternate lines (`run_alt_backtest.py`): promising for over/unders in bad weather only.**

There's no history of real alternate-line prices, so each alternate line from −10 to +10 points is priced the way a book would: from the historical distribution of results in games with a similar line (which handles key numbers such as 3, 7, 41 and 44), minus an assumed book margin. The model's lean then nudges that probability, and the single best line per game is bet.

| Over/unders, 2010–2025 | Book margin | Bets | Win% | Typical odds | ROI |
|---|---|---|---|---|---|
| Main line only | 4.5% | 547 | 54.0% | −111 | +0.4% |
| Alternate ladder | 4.5% | 1,552 | 29.6% | +295 | +7.3% |
| Alternate ladder | 10% | 753 | 28.4% | +291 | +5.4% |
| Alternate ladder | 20% | 263 | n/a | n/a | +5.1% |

* The ladder mostly picked **unders 7–10 points below the main total**, at long odds.
* All of the profit came from **bad-weather games**: rain or wind over 10 mph made +11.0% on 633 bets; normal weather lost 9.4% on 461 bets.
* Spreads: alternate lines lose less than main lines but still lose (−0.1% to −3.1%).
* **Expect big swings.** Win rate is about 28%, and the bankroll fell as much as 45% from a peak.
* `upcoming.py` lists alternate unders for bad-weather games with the **minimum price** needed. Only bet if FanDuel offers at least that.
* Real FanDuel alternate prices could be better or worse than this stand-in. Validating it needs historical alternate-line odds (paid, from The Odds API).

## Results against real FanDuel prices (The Odds API, 2020–2026)

`fetch_odds.py` downloads four snapshots a week (Monday 10am, Thursday 11am, Saturday 10am and Sunday 12:45pm Eastern) of spreads and totals from 8 books, including FanDuel and Pinnacle. `run_odds_backtest.py` bets FanDuel's line and price at each snapshot and measures closing line value against Pinnacle's final line.

**Lines move toward the weather, but not enough.** In rain or wind games, FanDuel's total drops about 0.7 points from Monday to Sunday, yet the final line is still about 2 points too high on average.

| Over/under, bet placed | Bets | Win% | ROI | Luck range (90%) | Beat Pinnacle close |
|---|---|---|---|---|---|
| Monday* | 233 | 61.3% | +18.2% | +6.6% to +29.5% | 70% |
| Thursday* | 251 | 57.2% | +13.1% | +2.2% to +23.6% | 63% |
| **Saturday** | 238 | 52.5% | **+4.8%** | −6.4% to +15.4% | 53% |
| Sunday | 216 | 51.9% | +4.0% | −7.0% to +15.5% | 29% |

\*Monday and Thursday are inflated: they use the weather that actually happened, which nobody knew that early. Saturday is the most realistic test. At Saturday prices, bad-weather games made +6.0% and the rest lost 4.2%. Spreads lose at every snapshot.

**Alternate totals at FanDuel's real prices** (`fetch_alt_odds.py --pregame`, `run_alt_fanduel.py --pregame`, 2023–2026, 278 outdoor games):
* FanDuel's cut on alternate totals is about **6% at every distance from the main line**.
* The alternate-under strategy made **−2.6%** on 178 bets (median odds +560; luck range −32% to +31%). Bad-weather games made +2.2%; normal games lost.
* That's far weaker than the stand-in backtest (+11% in bad weather). The model mostly picked lines more than 10 points below the total, where its probability estimates are least reliable. **Alternate lines are not proven; main-line unders in bad weather remain the best-supported bet.**
* FanDuel's historical alternate lines are mostly missing from Saturday snapshots but present 30 minutes before kickoff.

## The key test: real FanDuel prices with real weather forecasts

`fetch_mos.py` downloads NOAA's official airport forecasts (MOS), archived free by Iowa State's Mesonet. For each bet time it uses the newest forecast published at least 4 hours before the bet: about 162 hours ahead for Monday bets, 86 for Thursday, 36 for Saturday and 8 for Sunday. `run_forecast_backtest.py` reruns the FanDuel over/under backtest on the same games with recorded weather and with the forecast.

**Forecast accuracy (outdoor games, 2020–2026):** temperature is off by about 7°F six days out and 2–3°F a day out. Rain forecasts catch about half of rainy games from Thursday on (almost none on Monday). Wind over 10 mph is caught about 80% of the time.

| Over/under, bet placed | Weather | Bets | Win% | ROI | Luck range (90%) | Winning seasons |
|---|---|---|---|---|---|---|
| Monday | recorded | 215 | 61.3% | +17.8% | +5.8% to +28.9% | |
| Monday | **forecast** | 281 | 58.8% | **+12.0%** | +1.7% to +21.8% | 7 of 7 |
| Thursday | recorded | 245 | 57.4% | +13.2% | +2.2% to +23.7% | |
| Thursday | **forecast** | 383 | 56.7% | **+10.8%** | +1.6% to +19.2% | 5 of 7 |
| Saturday | recorded | 225 | 52.4% | +5.6% | −5.3% to +16.8% | |
| Saturday | **forecast** | 274 | 57.9% | **+9.6%** | −0.5% to +19.8% | 5 of 7 |
| Sunday | **forecast** | 173 | 60.1% | +13.5% | +0.4% to +26.6% | 5 of 7 |

* **The edge survives realistic forecasts.** It's smaller on Monday and Thursday than with hindsight weather, but still positive.
* **It's almost all unders.** Over bets lose or break even on every day.
* **Early-week outdoor totals run high.** Betting every outdoor under at FanDuel's Monday price made +5.0% (1,039 bets) with no model and no weather filter. That fades to +1.0% on Thursday and −1.0% on Saturday.
* **A simple rule does as well as the model.** Under only when the forecast is wind over 15 mph or rain: Monday +12.6% (240 bets), Thursday +10.3% (353), Saturday +16.8% (238). Under when the forecast is wind over 10 mph or rain: +9.8%, +6.2%, +6.6%. These thresholds were picked partly after seeing earlier results, so treat them as a hypothesis to paper-trade.

## Caveats

1. **Recorded vs forecast weather.** The 2010–2025 backtests use recorded weather. The 2020–2026 FanDuel tests were redone with real forecasts (above), and the edge held up.
2. **Consensus prices before 2020.** The 2010–2019 backtests use consensus closing lines. FanDuel's real prices at four snapshots a week are tested from 2020 (see above).
3. **Small real-price samples.** About 230 FanDuel over/under bets per snapshot and 178 alternate bets is too few to separate a 5% edge from luck.
4. **Many variants tried.** The more versions and thresholds tested, the more likely one looks good by chance. Treat +4.8% as a lead, not a result.
5. **Weather approximations.** Readings come from the nearest weather station, often an airport, not the stadium. Snowfall is estimated from precipitation at or below 33°F. Gusts aren't recorded, so they're estimated from wind speed.

## Planned next steps

1. Fill the last weather gaps (mainly 2026) by rerunning `build_data.py`.
2. Backtest with historical weather **forecasts** instead of recorded weather.
3. Done: the FanDuel backtests were rerun with historical NOAA forecasts (see above).
4. Paper-trade the over/under model for the rest of 2026 before risking money.
