"""
A basic Elo rating model for college football.

How it works
------------
* Every FBS team has a rating; the average team is ~1500.
* Before a game, the home team gets a home-field bonus (`hfa` Elo points, 0 at neutral sites).
* Win probability for the home team = 1 / (1 + 10 ** (-(rating_diff) / 400)).
* After a game, the winner takes points from the loser:
      delta = K * margin_multiplier * (actual_result - expected_result)
  The margin multiplier (same form FiveThirtyEight uses for the NFL) rewards bigger wins,
  but shrinks the reward when a heavy favourite runs up the score.
* Between seasons each rating is pulled `regress` of the way back toward the mean,
  because rosters turn over.
* FCS opponents are pooled into one "FCS" entity so a win over an FCS team is worth
  roughly what it should be (not much).
* Point spread estimate = rating difference / `elo_per_point` (fit from history, ~25).
"""
import math
from pathlib import Path

import pandas as pd

MEAN_RATING = 1500.0
FCS_ID = "FCS"
FCS_NAME = "FCS (pooled)"


def win_probability(elo_diff):
    """Probability that the side with +elo_diff wins."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


class EloModel:
    def __init__(self, k=25.0, hfa=60.0, regress=0.33, mov=True, elo_per_point=25.0,
                 fbs_min_games=6, new_team_rating=1300.0, fcs_rating=1200.0):
        self.k = k
        self.hfa = hfa
        self.regress = regress
        self.mov = mov
        self.elo_per_point = elo_per_point
        self.fbs_min_games = fbs_min_games
        self.new_team_rating = new_team_rating
        self.fcs_rating = fcs_rating

        self.ratings = {FCS_ID: fcs_rating}   # team_id -> rating
        self.names = {FCS_ID: FCS_NAME}       # team_id -> display name
        self.fbs_by_season = {}               # season -> set of FBS team ids
        self.first_season = None
        self.current_season = None

    # ------------------------------------------------------------------ helpers
    def params(self):
        return {"k": self.k, "hfa": self.hfa, "regress": self.regress, "mov": self.mov,
                "elo_per_point": self.elo_per_point, "fbs_min_games": self.fbs_min_games,
                "new_team_rating": self.new_team_rating, "fcs_rating": self.fcs_rating}

    def _classify_fbs(self, games):
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

    def _entity(self, team_id, season):
        """Map a team to its rating key (its own id for FBS teams, the FCS pool otherwise)."""
        return team_id if team_id in self.fbs_by_season.get(season, set()) else FCS_ID

    def _start_season(self, season):
        """Regress every FBS rating toward the mean at the start of a new season."""
        if self.current_season is not None and season != self.current_season:
            for team_id, r in self.ratings.items():
                if team_id != FCS_ID:
                    self.ratings[team_id] = MEAN_RATING + (1 - self.regress) * (r - MEAN_RATING)
        if self.first_season is None:
            self.first_season = season
        self.current_season = season
        # Teams promoted from FCS start below average.
        init = MEAN_RATING if season == self.first_season else self.new_team_rating
        for team_id in self.fbs_by_season.get(season, set()):
            self.ratings.setdefault(team_id, init)

    def rating(self, team_id):
        return self.ratings.get(team_id, self.new_team_rating)

    def pregame_diff(self, home_id, away_id, neutral=False):
        """Home-minus-away rating difference including home-field advantage."""
        return self.rating(home_id) - self.rating(away_id) + (0.0 if neutral else self.hfa)

    def margin_multiplier(self, margin, favourite_diff):
        """FiveThirtyEight-style margin-of-victory multiplier.
        `favourite_diff` is the pregame Elo edge of the *winner* (negative for an upset)."""
        if not self.mov:
            return 1.0
        return math.log(abs(margin) + 1) * (2.2 / (favourite_diff * 0.001 + 2.2))

    # --------------------------------------------------------------------- fit
    def fit(self, games):
        """Replay games in date order, recording the pregame prediction for every completed
        game and updating ratings afterwards. Returns a DataFrame of predictions."""
        games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
        self._classify_fbs(games)
        rows = []
        for g in games.itertuples(index=False):
            if g.season != self.current_season:
                self._start_season(g.season)
            self.names[g.home_id] = g.home_team
            self.names[g.away_id] = g.away_team
            if not g.completed or pd.isna(g.home_score) or pd.isna(g.away_score):
                continue
            home = self._entity(g.home_id, g.season)
            away = self._entity(g.away_id, g.season)
            if home == away:          # FCS vs FCS never happens here, but be safe
                continue
            diff = self.pregame_diff(home, away, g.neutral_site)
            p_home = win_probability(diff)
            margin = int(g.home_score) - int(g.away_score)
            home_won = 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0)

            rows.append({
                "game_id": g.game_id, "season": g.season, "week": g.week, "date": g.date,
                "home_team": g.home_team, "away_team": g.away_team,
                "home_is_fbs": home != FCS_ID, "away_is_fbs": away != FCS_ID,
                "elo_home": self.ratings[home], "elo_away": self.ratings[away], "elo_diff": diff,
                "p_home": p_home, "pred_margin": diff / self.elo_per_point,
                "home_won": home_won, "margin": margin,
                "odds_spread": g.odds_spread,
            })

            favourite_diff = diff if margin >= 0 else -diff
            delta = self.k * self.margin_multiplier(margin, favourite_diff) * (home_won - p_home)
            self.ratings[home] += delta
            self.ratings[away] -= delta
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- predict
    def predict(self, home_id, away_id, neutral=False):
        season = self.current_season
        home = self._entity(home_id, season)
        away = self._entity(away_id, season)
        diff = self.pregame_diff(home, away, neutral)
        p_home = win_probability(diff)
        return {
            "home": self.names.get(home_id, home_id), "away": self.names.get(away_id, away_id),
            "elo_home": self.ratings[home], "elo_away": self.ratings[away],
            "elo_diff": diff, "p_home": p_home, "p_away": 1 - p_home,
            "pred_margin": diff / self.elo_per_point,   # positive = home favoured
        }

    def rankings(self, n=25):
        fbs = self.fbs_by_season.get(self.current_season, set())
        ranked = sorted(((self.ratings[t], self.names.get(t, t)) for t in fbs if t in self.ratings), reverse=True)
        return pd.DataFrame([{"rank": i + 1, "team": name, "elo": round(r, 1)}
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


def load_games(path=None):
    """Load data/games.csv (next to this file) or the CSV at `path`."""
    path = path or Path(__file__).parent / "data" / "games.csv"
    games = pd.read_csv(path, dtype={"game_id": str, "home_id": str, "away_id": str})
    games["date"] = pd.to_datetime(games["date"], utc=True)
    return games
