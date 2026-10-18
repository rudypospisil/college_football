# College Football Models

Small, readable predictive models for college football (FBS) built with pandas and
numpy. The project downloads every FBS game since 2010 from ESPN's public scoreboard API,
fits a rating model (Elo today, more to come), measures it on seasons it never saw, predicts
upcoming games (win probability and point spread), logs its picks, and replays past seasons
against closing lines. Every model plugs into the same harness, so they are compared on
identical games.

## Quick start

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
./run.sh                      # opens the menu (or double-click run.command in Finder)
```

The menu shows the state of the data, the model and the prediction log, then offers the
everyday tasks as numbered options. Option 1, "Game-day update", is the Saturday routine:
refresh results and lines, settle finished games, lock in today's predictions, and print the
slate. Option 10, "What if I had bet?", replays a past week or season and shows what flat bets on
the model's picks would have returned. `python app.py 4` runs a single option without the menu,
which is handy for scripts.
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

## Weekly routine

Start the menu with `./run.sh` from the project folder. Times are US/Eastern.

| When | Menu | What it does |
| --- | --- | --- |
| Sunday evening | 2, then 6 (date = next Saturday, model `prior`) | Refresh lines (the openers post Sunday) and lock in next week's picks against the opening numbers. These are the picks the closing-line test is about. |
| Sunday evening | 4 (next Saturday, `prior`) | Read the slate. Games where the model and the opener differ by 3+ points are the only candidates for a real bet; 8+ points is the strong tier. |
| Through the week | 14 | Enter an injury adjustment when a starter is ruled out, before kickoff. |
| Saturday morning | 1 | Game-day update: refresh scores and lines, snapshot injuries, record picks for any model not yet logged (Sunday's picks stand), print the slate. |
| Saturday night or Sunday morning | 2, then 7, then 8 | Refresh, settle every finished game, read the record per model and the overlay section. |
| Sunday morning | `git add -A && git commit -m "week N" && git push` | Keep the log on GitHub. |
| Monthly | 13, then 12 | Refresh this season's CFBD data (efficiency, lines) and re-run the model comparison. |
| Season end | 9 for each model, then 12 | Re-tune parameters with the finished season added, compare again, and re-read the closing line value with `clv.py`. |

What to watch, in order: the Brier score column (is the market model still best, is anything
closing the gap), the "edge >= 3 points" line in the record summary for the prior model, and the
closing-line value of Sunday picks. If the market keeps moving toward the Sunday picks by a point
or more on average over four to six weeks, the edge is still alive. Betting, if ever, is 1% of a
dedicated bankroll per qualifying game, flat, never on game day.

## Files

| File | Purpose |
| --- | --- |
| `app.py`, `run.sh` | Menu front end; starts with any Python and runs the scripts below with the `.venv` interpreter. |
| `fetch_data.py` | Pulls game results week by week from ESPN (no API key needed) and caches them as CSV. |
| `common.py` | Shared paths, `load_games()`, atomic CSV writes and the per-model parameter files in `data/params/`. |
| `models/base.py` | The interface every model implements (`fit`, `predict`, `rating`, `params`) plus shared FBS classification, rankings and team lookup. |
| `models/elo.py` | The Elo model: ratings, home-field advantage, margin-of-victory updates, preseason regression, FCS pooling. |
| `models/margin.py` | The margin model: weighted ridge regression on point differentials, refit weekly, with last season's ratings as a prior. |
| `models/__init__.py` | The registry (`MODELS`); every script takes `--model <name>`. |
| `compare.py` | Every registered model side by side on the held-out seasons, with the Vegas closing line scored the same way. |
| `fetch_cfbd.py` | Downloads recruiting, returning production, per-game efficiency, lines and games from CollegeFootballData.com (free key in `.env`) and maps ESPN to CFBD team ids. |
| `models/prior.py` | The margin model with a preseason prior regressed on last season's rating, recruiting, four-class talent and returning production. |
| `models/efficiency.py` | The prior model fit to a blend of actual margins and efficiency margins from CFBD's predicted points added per play. Registered as `eff`; a documented null result. |
| `models/ensemble.py` | Weighted average of other models' margins (`ens`). Tiny gain: the members' errors are 94-99.9% correlated. |
| `models/market.py` | Market-shading model (`market`): the Vegas line as the baseline, adjusted by a fitted fraction of our rating's disagreement with it. |
| `clv.py` | Closing line value: picks a side against each game's opening line and measures whether the closing line moved toward it. The honest test of a betting edge. |
| `fetch_injuries.py` | Snapshots ESPN's college football injury feed into `data/injuries/injuries.csv` (no history exists, so it accumulates from the first run). |
| `models/overlay.py`, `adjust.py` | Injury overlay (`overlay`): the prior model plus point adjustments entered before kickoff; graded against the prior model on the adjusted games. |
| `backtest.py` | Grid-searches a model's parameters on 2012-2023, then reports accuracy, Brier score, log loss and calibration on the held-out 2024-2025 seasons. Saves the winning parameters to `data/params/<model>.json`. |
| `predict.py` | Replays all games to date with a model's saved parameters and prints predictions. |
| `track.py` | Logs each model's predictions before kickoff, grades them after the games, and summarises the record per model (`data/predictions.csv`). |
| `fetch_odds.py` | Back-fills closing lines, moneylines and spread prices for finished games from ESPN's per-game odds endpoint. |
| `simulate.py` | Replays past games with the model's pregame predictions and grades flat bets on its picks against the stored lines. |

## How the Elo model works

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

**Would betting on it have made money?** `simulate.py` replays finished games against the
closing line with a flat $100 on every spread pick.

| Period | Games | Model ATS record | Spread ROI | Vegas favourite ATS ROI |
| --- | --- | --- | --- | --- |
| Week 6, 2026 (Oct 6-10) | 58 | 38-20 (65.5%) | +25.2% (+$1,464) | -20.9% |
| 2026 through week 7 | 458 | 225-227-6 (49.8%) | -4.8% (-$2,197) | +2.2% |
| 2024-2025 (held out) | 1,799 | 908-888-3 (50.6%) | -3.4% (-$6,081) | -3.4% |

Over two full seasons the model's spread picks win 50.6% of the time, below the 52.4% needed to
break even at -110, and its return equals that of blindly betting the Vegas favourite, which is
just the house edge. Filtering to the model's largest disagreements with the market makes it
worse, not better. Week 6 of 2026 was one of the three best weeks out of 39, and most of its
profit came from underdog picks in a week when underdogs covered 59% of games. Weekly results
over 2024-2025 ranged from -$1,377 to +$1,523 with a standard deviation near $770, so one week
says nothing. Full output: `data/simulate_week6.txt`, `data/simulate_2026.txt`,
`data/simulate_2024_2025.txt`.

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
* Each row stores the model's name and the parameters that produced it, so the record survives
  retuning and several models can be compared on the same games.
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

## What if I had bet?

```bash
.venv/bin/python fetch_odds.py --seasons 2026             # one-time back-fill of closing lines
.venv/bin/python simulate.py --season 2026 --week 6 --details
.venv/bin/python simulate.py --season 2026                # season to date, week by week
```

`simulate.py` uses the prediction the model would have made before each game (its pregame
ratings, replayed in date order) and the closing line stored for that game, then grades a flat
stake on every pick. Lines for 2013-2025 come from CollegeFootballData's cached sportsbook lines
(`fetch_odds.py --from-cfbd`); ESPN's per-game odds are used only for the current season, because
its history stores prices in place of spreads for some seasons. Every season passes a sanity check
(favourites win 74-80% straight up, average miss 12-13 points). It grades: spread bets on the model's side, spread bets only when the edge is at least
3, 5 or 7 points, moneyline bets on the model's favourite, and moneyline bets only when the model
disagrees with the Vegas favourite. Blindly betting the Vegas favourite is shown alongside as the
house-edge baseline. Lines for past games come from ESPN's per-game odds endpoint, which keeps
the closing DraftKings line for the current season; `fetch_odds.py --from-cfbd` fills every
season since 2013 from CollegeFootballData's lines, so the replay covers 13 seasons.

Read the results with the sample size in mind. One week is about 60 bets, where a 6-game swing
is pure luck. The 52.4% cover rate needed to break even at -110 is the bar, and a scores-only
model should not be expected to clear it over a full season.

## How the margin model works

`models/margin.py` treats each team's rating as points above an average FBS team and fits all
ratings at once by weighted least squares on the season's point differentials:

1. Before every week, solve for the ratings that best explain the margins so far, where
   `margin = rating_home - rating_away + 3 (home field)`. Older games count less: a game
   180 days old has half the weight of one played yesterday.
2. A ridge penalty pulls each rating toward a prior, last season's final rating shrunk 20% toward
   average, with the strength of two games' worth of evidence. In week 1 the ratings are the
   prior; by midseason the games dominate.
3. Win probability is the normal distribution applied to the predicted margin, with a noise
   level of 16.9 points estimated from the training seasons.

The grid search chose no cap on margins, a 180-day half-life, a 3-point home field and a light
prior. Everything is walk-forward: a game's prediction uses only games played before the first
kickoff of its week, and a perturbation test confirms a game never influences its own prediction.

### Elo vs. margin vs. the market

`python compare.py` scores every registered model on the held-out 2024-2025 seasons; the Vegas
closing line is scored the same way on the 1,839 games that have one, with its own noise level
fitted on the training seasons so the probability scores are comparable.

| Model | Accuracy | Brier | Log loss | Avg. margin miss | ATS 2024-25 |
| --- | --- | --- | --- | --- | --- |
| Elo | 73.4% | 0.173 | 0.511 | 13.4 pts | 49.5% |
| Margin | 74.4% | 0.165 | 0.491 | 12.7 pts | 51.4% |
| Prior (margin + recruiting/returning prior) | 74.7% | 0.165 | 0.489 | 12.5 pts | 51.1% |
| Efficiency (prior + 25% PPA blend) | 74.8% | 0.165 | 0.490 | 12.6 pts | 50.7% |
| Ensemble (10% Elo, 80% prior, 10% efficiency) | 74.8% | 0.165 | 0.489 | 12.6 pts | 50.7% |
| Market (Vegas line shaded by the prior model) | 75.2% | 0.161 | 0.478 | 11.9 pts | 50.1% |
| Vegas closing line | 75.3% | 0.161 | 0.479 | 11.9 pts | 50.7% (favourites) |

Each step closes part of the gap to the market. The margin model takes 0.7 points off Elo's
average margin error; the prior model takes another 0.15 and helps most where it should, in weeks
1-4 of a season (Brier 0.136 vs. 0.137 for margin and 0.145 for Elo). Its preseason regression,
fit on 2011-2023 only, says a one-standard-deviation edge in four-class recruiting talent is worth
3.7 points and the same edge in returning production 1.9 points, with last year's rating carrying
a weight of 0.64.

Against the closing spread none of them clears the 52.4% break-even over 2024-2025, and the order
(margin 51.4%, prior 51.1%, Elo 49.5%) does not follow prediction quality, which is what an
efficient market looks like: better calibration does not create an edge against the closer.
Over all 13 seasons with lines (11,177 games, 2013-2025) the prior model goes 50.8% against the
spread for -2.9% ROI, and even its 7-point-plus disagreements only reach 52.2%. In week 6 of
2026, where Elo had its +25% week, the margin model went 28-30 and the prior model 30-28.

### The market model

`models/market.py` starts from the closing line and lets a regression decide how much our rating
adds: `margin = -0.30 + 1.01 x Vegas margin + 0.16 x (prior margin - Vegas margin)`, fitted on
2013-2023 games with a line. The market keeps essentially all the weight and our rating's
disagreement with it gets 16%. On the held-out seasons the result is indistinguishable from the
closing line itself (average miss 11.95 vs 11.94 points), which makes it the best predictor in
the project and, by construction, useless against the closer (50.1% ATS). Its value is as the
number to compare an opening line against: see the closing line value section.

**Efficiency numbers did not help.** `models/efficiency.py` fits the same ridge regression to a
blend of the actual margin and an efficiency margin built from CollegeFootballData's predicted
points added per play, which explains 74% of the variance in final margins across 12,000 games
(about 64 points per unit of PPA-per-play difference). The grid search settled on a 25% blend,
and at that weight the model matches the prior model to the third decimal on held-out Brier,
log loss and margin error; more weight makes it worse, and excluding garbage time from the PPA
changes nothing. Once a season of margins is averaged with a sensible prior, the extra signal in
play-by-play efficiency is offset by what it strips out: turnovers, special teams and finishing
drives are partly skill. The model stays registered as a fourth entrant in the weekly log.

Today's Ohio State at Indiana game shows the models' temperaments: Elo says Indiana by 7,
the margin model Indiana by 2, the prior model Ohio State by 2.6, the market model Ohio State by 3.3, and the market Ohio State by 3.
The efficiency model says Ohio State by 2.4. All six models' picks for today's 52 games are in the
prediction log, so option 8 will show the head-to-head record as soon as the games are settled.

**How the first live Saturday went (Oct 17, 2026, all 52 games).** Ohio State won 19-15 and covered
by a point. Across the slate:

| Model | Winners called | Against the spread | $100 a game, spread | Brier |
| --- | --- | --- | --- | --- |
| Elo | 35-17 | 23-27-2 | -$583 | 0.210 |
| Margin | 36-16 | 23-27-2 | -$589 | 0.208 |
| Prior | 36-16 | 20-30-2 | -$1,160 | 0.204 |
| Efficiency | 36-16 | 21-29-2 | -$970 | 0.202 |
| Ensemble | 36-16 | 21-29-2 | -$964 | 0.203 |
| Market | 39-13 | 23-27-2 | -$579 | 0.194 |
| Vegas favourite | 39-13 | 23-27-2 | -$609 | |

The order on prediction quality matched the two-season backtest exactly: market, then the
margin-regression family, then Elo. Against the spread every model lost on an underdog day
(favourites covered 23 of 52), which is what the replay said to expect. One Saturday decides
nothing; the log is where the evidence accumulates.

## CollegeFootballData.com data

`fetch_cfbd.py` downloads season-level datasets from CollegeFootballData.com into `data/cfbd/`:
every game with CFBD's team ids and its own pregame Elo, per-game predicted points added (an
efficiency measure, with and without garbage time), betting lines from several sportsbooks back to 2013, recruiting class
rankings, and returning production. It also builds `team_map.csv`, which links ESPN and CFBD
team ids by matching games on date and score (no name matching).

It needs a free API key from collegefootballdata.com. Put it in a file named `.env` next to the
scripts, as `CFBD_API_KEY=...`. That file is in `.gitignore` and must never be committed. The
free tier allows 1,000 calls a month, so the script makes one call per season per dataset and
never re-downloads a cached season unless you name it: `python fetch_cfbd.py --seasons 2026`
refreshes the current season (games, efficiency and lines change weekly). Weather data needs a
paid tier and is not used.

## Closing line value: the one test that matters for betting

`clv.py` takes each game's opening spread from CFBD's lines files, has the model pick a side
against that opener, and then asks what the closing line did. If the market moves toward the
model's side after it would have bet, the model knew something the market learned later. This
shows up in weeks, not seasons, and is far less noisy than win-loss records.

| 2024-2025 | Bets | Close moved toward us / away | Mean CLV | Record vs opener | Record vs closer |
| --- | --- | --- | --- | --- | --- |
| Prior model, every game | 1,795 | 50% / 32% | +0.83 pts | 51.8% | 49.3% |
| Prior, disagree with opener by 3+ pts | 797 | 55% / 26% | +1.45 pts | 54.1% | 50.3% |
| Prior, disagree by 8+ pts | 173 | 64% / 24% | +3.05 pts | 62.4% | |
| Market model, disagree with opener by 2+ pts | 41 | 76% / 17% | +5.46 pts | 74.4% | 51.2% |

Read together with the replay results this is a coherent picture: the model has real information
relative to the opening number, the market absorbs it during the week, and by kickoff there is
nothing left. An edge that exists only against openers is small in practice (books post openers
with low limits and move them fast), but it is the first positive betting signal in this project,
and it is the metric to watch as the tracker accumulates weeks.

## Ensemble

`models/ensemble.py` averages member models' margins; the default is 10% Elo, 80% prior, 10%
efficiency, chosen by training-season log loss among genuine mixes. Held-out Brier improves from
0.1636 to 0.1634, which is what averaging models whose errors correlate at 0.94 to 0.999 can do.
It stays registered because members with different information (the market model, a future
offense/defense model) would make it useful.

## Injuries

College football has no official injury report; the SEC, Big Ten, ACC and Big 12 publish
game-day availability reports, and ESPN aggregates reported injuries into a feed with no history.
`fetch_injuries.py` snapshots that feed into `data/injuries/injuries.csv` and the game-day menu
option runs it each week, so a record builds from the first run. The practical use is a
position-based adjustment to a model's margin (a starting quarterback out is worth several points,
a running back or lineman a fraction of one), first as a manual overlay, later estimated from the
accumulated snapshots.

## Injury overlay

The models know scores, efficiency and last season's roster. They do not know that a starting
quarterback is out this week. `adjust.py` lets you add that yourself, and the tracker grades
whether your reads helped:

```bash
.venv/bin/python adjust.py add --team "Ohio State" --points -1.5 \
    --reason "RB Ja'Kobi Jackson out (leg); OL shuffling" --source "game preview"
.venv/bin/python adjust.py list
.venv/bin/python adjust.py remove --row 0        # before kickoff only
```

Positive points favour the team. Each row (game, team, points, reason, source, time entered) goes
to `data/adjustments.csv`. The `overlay` model is the prior model plus the net adjustment for a
game, with the win probability recomputed, and it applies only to games with an adjustment, so its
rows in the prediction log are exactly the games you touched. Adding an adjustment locks in the
overlay's pick for that game immediately, which is why it must happen before kickoff; the report
(menu option 8) then shows the overlay and the prior model side by side on those games.

Rules of thumb for points, roughly what sportsbooks move lines by: a starting quarterback 4 to 7
depending on the backup, a top running back or receiver 1 to 2, a starting lineman or top pass
rusher 0.5 to 1 each, a kicker 0.5. Halve it for "questionable" and keep a team's total under
about 7. The weekly injury snapshots in `data/injuries/` will eventually let these values be
estimated from results instead of assumed. Remember that the market model already carries whatever
the line has priced, so an overlay mainly matters for the standalone models and for news you learn
before the opening line moves.

## Adding a model

Every model is a class in `models/` that subclasses `RatingModel` (see `models/base.py`) and
is listed in `MODELS` in `models/__init__.py`. It needs four things:

* `fit(games)`: replay the games in date order and return one row per finished game with the
  prediction made *before* that game (`p_home`, `pred_margin`) next to the result.
* `predict(home_id, away_id, neutral)`: probability and margin for an upcoming game.
* `rating(team_id)` and `params()`: a number per team for the rankings, and the constructor
  arguments so the model can be rebuilt from `data/params/<name>.json`.
* Optionally `GRID`, the parameter grid `backtest.py` searches, and `post_fit_params()` for
  anything fitted afterwards (Elo fits its points-per-rating scale this way).

Once registered, `--model <name>` works everywhere: `backtest.py` tunes and evaluates it on the
same held-out seasons as Elo, `track.py record` logs its picks alongside Elo's each week, and
`simulate.py` replays it against the same closing lines. The game-day menu option records
every registered model, so a new idea is judged on identical games from its first Saturday,
and `compare.py` (menu option 12) puts all of them in one table.

## Limitations

* Elo only sees final scores. It knows nothing about injuries, quarterbacks, weather,
  travel or scheme, all of which the betting market prices in.
* Preseason ratings are just last year's ratings regressed toward the mean; recruiting
  rankings and returning production would sharpen the first few weeks of a season.
* The weekly scoreboard only carries lines for the current week; past seasons' lines come from
  CollegeFootballData (`fetch_odds.py --from-cfbd`). ESPN's own odds history is unreliable
  before 2024.

## Ideas for the next version

* Add a logistic regression on top of Elo with extra features (recent point differential,
  yards per play, rest days, travel distance).
* Blend in preseason priors from recruiting composites or returning production.
* Track the model's picks against closing lines each week to see whether any edge exists.
