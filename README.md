# College Football Elo Model

A small, readable predictive model for college football (FBS) built with pandas and
numpy. It downloads every FBS game since 2010 from ESPN's public scoreboard API,
fits an Elo rating system, measures itself on seasons it never saw, and predicts
upcoming games (win probability and point spread).

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh                      # opens the menu (or double-click run.command in Finder)
```

The menu shows the state of the data, the model and the prediction log, then offers the
everyday tasks as numbered options. Option 1, "Game-day update", is the Saturday routine:
refresh results and lines, settle finished games, lock in today's predictions, and print the
slate. `python app.py 4` runs a single option without the menu, which is handy for scripts.
Every option echoes the command line it runs, so the scripts below can also be used directly.

## Workflow

```bash
.venv/bin/python fetch_data.py                   # 1. download/refresh results -> data/games.csv
.venv/bin/python backtest.py                     # 2. tune parameters, report out-of-sample accuracy
.venv/bin/python predict.py "Ohio State" "Indiana"   # 3. predict a game (away team first, then home)
.venv/bin/python predict.py --date 2026-10-17    #    or every game on a date, next to the Vegas line
.venv/bin/python predict.py --rankings           #    current top 25
```

Re-run `fetch_data.py` each week; it re-downloads only the current season and reuses
cached files in `data/seasons/` for past seasons.

## Files

| File | Purpose |
| --- | --- |
| `app.py`, `run.sh` | Menu front end; starts with any Python and runs the scripts below with the `.venv` interpreter. |
| `fetch_data.py` | Pulls game results week by week from ESPN (no API key needed) and caches them as CSV. |
| `elo.py` | The model: ratings, home-field advantage, margin-of-victory updates, preseason regression, FCS pooling. |
| `backtest.py` | Grid-searches the parameters on 2012-2023, then reports accuracy, Brier score, log loss and calibration on the held-out 2024-2025 seasons. Saves the winning parameters to `data/model_params.json`. |
| `predict.py` | Replays all games to date with the saved parameters and prints predictions. |
| `track.py` | Logs predictions before kickoff, grades them after the games, and summarises the record (`data/predictions.csv`). |

## How the model works

1. Every FBS team starts at 1500. FCS opponents share one pooled "FCS" rating, so a win
   over an FCS team is worth about what it should be.
2. Before each game the home team gets a home-field bonus (`hfa` Elo points; none at a
   neutral site). The home win probability is `1 / (1 + 10 ** (-diff / 400))`.
3. After the game the winner takes `K * margin_multiplier * (result - expected)` points
   from the loser. The margin multiplier (FiveThirtyEight's NFL formula) rewards big wins
   but shrinks when a heavy favourite runs up the score.
4. Between seasons every rating is pulled part of the way back toward 1500 (`regress`),
   because rosters turn over. Teams promoted from FCS start at 1300.
5. Projected spread = Elo difference / `elo_per_point`, where `elo_per_point` is fit by
   least squares on the training seasons.

Parameters `k`, `hfa`, `regress` and `mov` (margin multiplier on/off) are chosen by grid
search on log loss over the training seasons. The 2024 and 2025 seasons are never used to
choose anything, so the numbers reported for them are a fair estimate of how the model does
on games it has not seen.

## Reading the backtest

* **Accuracy**: share of games where the team the model favoured actually won. Picking the
  home team every time gets about 62-64% (FBS teams almost always host FCS teams), so that is the bar to beat.
* **Brier score**: mean squared error of the probabilities (0 is perfect, 0.25 is a coin
  flip). Lower is better.
* **Log loss**: like Brier but punishes confident misses harder. Lower is better.
* **Calibration table**: groups games by predicted probability and shows how often the home
  team really won. A well-calibrated model's `predicted` and `actual` columns match.

## Results (run on 2026-10-17, 2026 season through week 7)

Grid search chose `k=40`, `hfa=55`, `regress=0.2`, margin multiplier on, and 19.85 Elo per
point of spread. Held-out seasons were never used to pick those values.

| Split | Games | Accuracy | Brier | Log loss | Always-home accuracy | Always-home Brier |
| --- | --- | --- | --- | --- | --- | --- |
| Train 2012-2023 | 10,221 | 74.3% | 0.170 | 0.507 | 61.9% | 0.236 |
| **Test 2024-2025 (never seen)** | 1,853 | 73.6% | 0.171 | 0.507 | 63.8% | 0.231 |
| Test, FBS vs FBS only | 1,606 | 70.2% | 0.192 | 0.563 | | |
| 2026 so far | 458 | 78.0% | 0.147 | 0.451 | 71.4% | 0.204 |

Test-season log loss matches training log loss, so the parameters are not overfit.
Vegas favourites win roughly 75% of FBS-vs-FBS games straight up, so the model sits a few
points behind the market, which is normal for a scores-only model.

**Calibration.** On the held-out seasons the model is already close to calibrated: the Brier
score decomposes into uncertainty 0.231 minus resolution 0.060 plus reliability (calibration
error) of only 0.002. Fitting a logistic recalibration on the training seasons (Platt scaling)
lowers the test log loss from 0.5069 to 0.5057, a 0.2% change. The remaining error is almost
all resolution, meaning the model lacks information, not calibration. The weak spot is home
underdogs: when the model gives the home team 30-40% it has been winning about 43%.

**Today's test game: Ohio State at Indiana (Oct 17, 2026).**

| Source | Favourite | Spread | Win probability |
| --- | --- | --- | --- |
| Elo model | Indiana (Elo 2023 vs 1939, +55 home) | Indiana -7.0 | Indiana 69% |
| Vegas (ESPN pregame line) | Ohio State | Ohio State -3 | roughly 60% |

The two disagree by ten points. Indiana finished the 2025 season at Elo 2124 after winning the
national title, carried 80% of that into 2026, and is 6-0, though two of those wins were by 6 and
3 points. Ohio State is 5-1 with a one-point loss at Texas. The market is pricing in things Elo
cannot see, chiefly roster turnover at Indiana and Ohio State's talent, so the gap says more
about the model's blind spots than about the game. Check the outcome with:

```bash
.venv/bin/python fetch_data.py && .venv/bin/python predict.py --date 2026-10-17
```

## Keeping a record

`track.py` keeps a running log of predictions and how they turned out in `data/predictions.csv`.

```bash
.venv/bin/python fetch_data.py          # refresh lines and results first
.venv/bin/python track.py record        # lock in today's predictions (or --date YYYY-MM-DD)
.venv/bin/python track.py settle        # after the games: fill in scores and grade every pick
.venv/bin/python track.py report        # the full history plus a summary
```

Rules that keep the record honest:

* A game is only recorded while it is still scheduled and kickoff is in the future, and a
  recorded prediction is never overwritten. The Vegas line is captured at the same moment.
* Each row stores the model parameters that produced it, so the record survives retuning.
* Grading uses the line that was recorded, not the closing line.

What gets graded for each game:

* **Straight-up pick**: did the team the model favoured win? The Vegas favourite is graded
  the same way for comparison.
* **Spread bet**: the model's projected margin is compared with the recorded Vegas spread,
  and the model "bets" whichever side it thinks covers. The edge is the gap between the two
  in points. The result is win, loss or push, with profit in units at the recorded spread
  price, usually about -110 (a winning one-unit bet returns about 0.91 units).
* **Moneyline bet**: one unit on the model's favourite at the recorded moneyline price,
  when ESPN published one.
* **Brier score** for the win probability, so calibration can be checked as the log grows.

The summary reports the straight-up record, the against-the-spread record and return on
investment for all bets and for bets with an edge of at least 3 points, and the moneyline
return. Expect the spread record to hover near 50%: that is what a market this efficient
does to a scores-only model, and the log is how you find out rather than guess.

## Limitations

* Elo only sees final scores. It knows nothing about injuries, quarterbacks, weather,
  travel or scheme, all of which the betting market prices in.
* Preseason ratings are just last year's ratings regressed toward the mean; recruiting
  rankings and returning production would sharpen the first few weeks of a season.
* ESPN only stores betting lines for the current week, so the Vegas comparison covers
  upcoming games, not history.

## Ideas for the next version

* Add a logistic regression on top of Elo with extra features (recent point differential,
  yards per play, rest days, travel distance).
* Blend in preseason priors from recruiting composites or returning production.
* Track the model's picks against closing lines each week to see whether any edge exists.
