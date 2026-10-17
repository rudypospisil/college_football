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
* Point spread estimate = rating difference / `elo_per_point` (fit from history, ~20).
"""
import math

import pandas as pd

from .base import FCS_ID, RatingModel

MEAN_RATING = 1500.0


def win_probability(elo_diff):
    """Probability that the side with +elo_diff wins."""
    return 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))


class EloModel(RatingModel):
    name = "elo"
    GRID = {
        "k": [20, 30, 40, 50, 60],
        "hfa": [40, 55, 70, 85],
        "regress": [0.1, 0.2, 0.33, 0.5],
        "mov": [True, False],
    }

    def __init__(self, k=25.0, hfa=60.0, regress=0.33, mov=True, elo_per_point=25.0,
                 fbs_min_games=6, new_team_rating=1300.0, fcs_rating=1200.0):
        super().__init__(fbs_min_games=fbs_min_games)
        self.k = k
        self.hfa = hfa
        self.regress = regress
        self.mov = mov
        self.elo_per_point = elo_per_point
        self.new_team_rating = new_team_rating
        self.fcs_rating = fcs_rating

        self.ratings = {FCS_ID: fcs_rating}   # team_id -> rating
        self.first_season = None

    # ------------------------------------------------------------------ helpers
    def params(self):
        return {"k": self.k, "hfa": self.hfa, "regress": self.regress, "mov": self.mov,
                "elo_per_point": self.elo_per_point, "fbs_min_games": self.fbs_min_games,
                "new_team_rating": self.new_team_rating, "fcs_rating": self.fcs_rating}

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

    def post_fit_params(self, pred):
        """Least-squares fit of actual margin vs Elo difference (through the origin)."""
        d, m = pred["elo_diff"].to_numpy(), pred["margin"].to_numpy()
        slope = (d * m).sum() / (d * d).sum()      # points per Elo point
        return {"elo_per_point": round(1.0 / slope, 2)}

    # --------------------------------------------------------------------- fit
    def fit(self, games):
        """Replay games in date order, recording the pregame prediction for every completed
        game and updating ratings afterwards. Returns a DataFrame of predictions."""
        games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
        self.classify_fbs(games)
        rows = []
        for g in games.itertuples(index=False):
            if g.season != self.current_season:
                self._start_season(g.season)
            self.names[g.home_id] = g.home_team
            self.names[g.away_id] = g.away_team
            if not g.completed or pd.isna(g.home_score) or pd.isna(g.away_score):
                continue
            home = self.entity(g.home_id, g.season)
            away = self.entity(g.away_id, g.season)
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
        home = self.entity(home_id, season)
        away = self.entity(away_id, season)
        diff = self.pregame_diff(home, away, neutral)
        p_home = win_probability(diff)
        return {
            "home": self.names.get(home_id, home_id), "away": self.names.get(away_id, away_id),
            "rating_home": self.ratings[home], "rating_away": self.ratings[away],
            "elo_diff": diff, "p_home": p_home, "p_away": 1 - p_home,
            "pred_margin": diff / self.elo_per_point,   # positive = home favoured
        }
