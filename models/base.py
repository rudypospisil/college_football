"""
The interface every rating model implements.

A model is a class that can replay history and predict games. backtest.py, predict.py,
track.py, simulate.py and the menu only ever call the methods below, so a new model that
implements them gets the full harness for free: tuning, out-of-sample evaluation, the
prediction log and the betting replay, all on the same games and lines as every other model.

Required columns in the DataFrame returned by fit():
    game_id, season, week, date, home_team, away_team, home_is_fbs, away_is_fbs,
    p_home        probability the home team wins
    pred_margin   predicted home-team margin (positive = home favoured)
    home_won      1.0 / 0.0 (0.5 for a tie)
    margin        actual home-team margin
    odds_spread   the stored Vegas spread, if any (home perspective)
Models may add their own columns (Elo adds elo_home, elo_away, elo_diff).

predict() returns a dict with: home, away, rating_home, rating_away, p_home, p_away, pred_margin.
"""
import pandas as pd

FCS_ID = "FCS"
FCS_NAME = "FCS (pooled)"


class RatingModel:
    name = "base"        # key used by the registry, --model flags and parameter files
    GRID = {}            # parameter grid searched by backtest.py, e.g. {"k": [20, 30]}

    def __init__(self, fbs_min_games=6):
        self.fbs_min_games = fbs_min_games
        self.names = {FCS_ID: FCS_NAME}   # team_id -> display name
        self.fbs_by_season = {}           # season -> set of FBS team ids
        self.current_season = None

    # ------------------------------------------------------------ to implement
    def params(self):
        """All constructor parameters, so a model can be rebuilt from a JSON file."""
        raise NotImplementedError

    def fit(self, games):
        """Replay `games` in date order. Record the pregame prediction for every finished
        game *before* learning from it, and return those predictions as a DataFrame."""
        raise NotImplementedError

    def predict(self, home_id, away_id, neutral=False):
        """Predict an upcoming game from the current state of the ratings."""
        raise NotImplementedError

    def rating(self, team_id):
        """A single number per team, used for the rankings table."""
        raise NotImplementedError

    def post_fit_params(self, pred):
        """Parameters fitted after the grid search from the training predictions, e.g. a
        rating-to-points scale. Return {} if the model has none."""
        return {}

    def applies_to(self, game_id):
        """Whether track.py should log this model's pick for a game (the injury overlay says
        yes only for games with an adjustment)."""
        return True

    # ------------------------------------------------------------------ shared
    def classify_fbs(self, games):
        """Decide which teams are FBS in each season from the schedule itself.

        FBS teams play 12+ games against this pool; FCS teams only show up when they
        visit an FBS opponent, usually 1-2 times (occasionally 3-4). A team counts as FBS
        if it played >= fbs_min_games games, or >= 3 games and was FBS the season before
        (that second rule keeps the shortened 2020 season from dropping anyone)."""
        previous = set()
        for season, grp in games.groupby("season"):
            counts = pd.concat([grp["home_id"], grp["away_id"]]).value_counts()
            fbs = set(counts[counts >= self.fbs_min_games].index)
            fbs |= {t for t in counts[counts >= 3].index if t in previous}
            self.fbs_by_season[season] = fbs
            previous = fbs

    def entity(self, team_id, season):
        """Map a team to its rating key: its own id for FBS teams, the FCS pool otherwise."""
        return team_id if team_id in self.fbs_by_season.get(season, set()) else FCS_ID

    def rankings(self, n=25):
        fbs = self.fbs_by_season.get(self.current_season, set())
        ranked = sorted(((self.rating(t), self.names.get(t, t)) for t in fbs if t in self.names), reverse=True)
        return pd.DataFrame([{"rank": i + 1, "team": name, "rating": round(r, 1)}
                             for i, (r, name) in enumerate(ranked[:n])])

    def find_team(self, query):
        """Resolve a user-typed name ('Ohio State') to an ESPN team id among current FBS teams."""
        q = query.strip().lower()
        fbs = self.fbs_by_season.get(self.current_season, set())
        candidates = [(tid, self.names[tid]) for tid in fbs if tid in self.names]
        exact = [c for c in candidates if c[1].lower() == q]
        if exact:
            return exact[0][0]
        prefix = [c for c in candidates if c[1].lower().startswith(q)]
        if prefix:                      # 'Ohio' -> Ohio Bobcats, not Ohio State Buckeyes
            return min(prefix, key=lambda c: len(c[1]))[0]
        contains = [c for c in candidates if q in c[1].lower()]
        if contains:
            return min(contains, key=lambda c: len(c[1]))[0]
        raise KeyError(f"No FBS team matching '{query}'")
