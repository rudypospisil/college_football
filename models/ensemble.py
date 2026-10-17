"""
An ensemble: the weighted average of other registered models' predicted margins.

Averaging models whose errors are not identical usually buys a small, reliable improvement. Here
the members' errors are 94-99.9% correlated, so the gain is tiny (about 0.0004 Brier on the
held-out seasons); the default weights (10% Elo, 80% prior, 10% efficiency) were chosen by
training-season log loss among genuine mixes. Members are fitted with their own tuned parameters (data/params/<member>.json) on the same games,
their predicted margins are averaged, and the win probability is the normal distribution applied
to that average with a noise level fitted on the training seasons. Elo ratings are converted to
points through its elo_per_point scale so every member speaks in points.
"""
import numpy as np
import pandas as pd

from common import load_params
from .base import FCS_ID, RatingModel
from .margin import normal_cdf

ELO_MEAN = 1500.0


class EnsembleModel(RatingModel):
    name = "ens"
    GRID = {}

    def __init__(self, members=("elo", "prior", "eff"), weights=(0.1, 0.8, 0.1), sigma=16.6, fbs_min_games=6):
        super().__init__(fbs_min_games=fbs_min_games)
        self.members = list(members)
        self.weights = list(weights) if weights else [1.0 / len(self.members)] * len(self.members)
        self.sigma = sigma
        self.models = {}

    def params(self):
        return {"members": self.members, "weights": self.weights, "sigma": self.sigma, "fbs_min_games": self.fbs_min_games}

    def _build(self):
        from . import MODELS                          # late import: the registry imports this module
        return {name: MODELS[name](**load_params(name)) for name in self.members}

    def _points(self, model, team_id):
        """A member's rating in points above average (Elo converted through its scale)."""
        if model.name == "elo":
            return (model.rating(team_id) - ELO_MEAN) / model.elo_per_point
        return model.rating(team_id)

    def post_fit_params(self, pred):
        return {"sigma": round(float((pred["margin"] - pred["pred_margin"]).std()), 2)}

    def fit(self, games):
        self.models = self._build()
        frame = None
        for name, model in self.models.items():
            p = model.fit(games)
            if frame is None:
                frame = p.drop(columns=[c for c in p.columns if c.startswith(("elo_", "rating_"))]).rename(columns={"pred_margin": f"m_{name}"})
            else:
                frame = frame.merge(p[["game_id", "pred_margin"]].rename(columns={"pred_margin": f"m_{name}"}), on="game_id", how="inner")
        first = next(iter(self.models.values()))
        self.names, self.fbs_by_season, self.current_season = first.names, first.fbs_by_season, first.current_season
        cols = [f"m_{n}" for n in self.members]
        frame["pred_margin"] = frame[cols].to_numpy() @ np.array(self.weights)
        frame["p_home"] = [normal_cdf(m / self.sigma) for m in frame["pred_margin"]]
        return frame.drop(columns=cols)

    def rating(self, team_id):
        return float(sum(w * self._points(m, team_id) for w, m in zip(self.weights, self.models.values())))

    def predict(self, home_id, away_id, neutral=False):
        preds = [m.predict(home_id, away_id, neutral) for m in self.models.values()]
        margin = float(sum(w * p["pred_margin"] for w, p in zip(self.weights, preds)))
        p_home = normal_cdf(margin / self.sigma)
        return {
            "home": preds[0]["home"], "away": preds[0]["away"],
            "rating_home": self.rating(home_id), "rating_away": self.rating(away_id),
            "p_home": p_home, "p_away": 1 - p_home, "pred_margin": margin,
        }
