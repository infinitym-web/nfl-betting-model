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

## Caveats

1. **Recorded weather, not forecasts.** The backtest uses weather recorded at kickoff. A bettor only has the forecast, which can be wrong, especially for wind. This flatters the weather results; how much is unknown until the backtest runs on real historical forecasts.
2. **Closing lines only.** These are the hardest lines to beat. Earlier-week lines are softer, but testing them needs paid historical odds.
3. **Not FanDuel's exact prices.** The data is a market consensus.
4. **Many variants tried.** The more versions and thresholds tested, the more likely one looks good by chance. Treat +4.8% as a lead, not a result.
5. **Weather approximations.** Readings come from the nearest weather station, often an airport, not the stadium. Snowfall is estimated from precipitation at or below 33°F. Gusts aren't recorded, so they're estimated from wind speed.

## Planned next steps

1. Fill the last weather gaps (mainly 2026) by rerunning `build_data.py`.
2. Backtest with historical weather **forecasts** instead of recorded weather.
3. Get historical opening and mid-week lines (FanDuel via The Odds API) to test betting earlier in the week and track closing line value.
4. Paper-trade the over/under model for the rest of 2026 before risking money.
