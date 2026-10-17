"""
A market-shading model: start from the Vegas line and adjust it with our own rating.

The closing line is the sharpest single predictor of a college football margin, so instead of
competing with it this model treats it as the baseline and asks how much our rating adds:

    predicted margin = b0 + b1 * vegas margin + b2 * (our margin - vegas margin)

where "vegas margin" is the home team's margin implied by the spread and "our margin" comes from
the base model (the prior model by default). The coefficients are fitted by least squares on the
training seasons that have a stored line (through `train_through`); b2 measures how much the
market under-weights whatever our rating knows. Games without a line fall back to the base model.

For upcoming games the model uses the line stored in data/games.csv at prediction time, so a
pick logged on Saturday morning is shaded from Saturday morning's line. clv.py evaluates the
same idea against opening lines: did the market move toward our side after we would have bet?
"""
import numpy as np

from common import load_params
from .base import RatingModel
from .margin import normal_cdf


class MarketModel(RatingModel):
    name = "market"
    GRID = {}

    def __init__(self, base="prior", train_through=2023, coef=None, sigma=16.0, fbs_min_games=6):
        super().__init__(fbs_min_games=fbs_min_games)
        self.base_name = base
        self.train_through = train_through
        self.coef = list(coef) if coef else None     # [b0, b1, b2]; None -> fitted in fit()
        self.sigma = sigma
        self.base = None
        self.lines = {}                               # (home_id, away_id) -> stored spread for upcoming games
        self.fit_info = None

    def params(self):
        return {"base": self.base_name, "train_through": self.train_through, "coef": self.coef,
                "sigma": self.sigma, "fbs_min_games": self.fbs_min_games}

    def post_fit_params(self, pred):
        return {"sigma": round(float((pred["margin"] - pred["pred_margin"]).std()), 2),
                "coef": [round(float(c), 4) for c in self.coef]}

    def shade(self, vegas_margin, model_margin):
        b0, b1, b2 = self.coef
        return b0 + b1 * vegas_margin + b2 * (model_margin - vegas_margin)

    def fit(self, games):
        from . import MODELS
        self.base = MODELS[self.base_name](**load_params(self.base_name))
        pred = self.base.fit(games).copy()
        self.names, self.fbs_by_season, self.current_season = self.base.names, self.base.fbs_by_season, self.base.current_season

        has_line = pred["odds_spread"].notna() & (pred["odds_spread"] != 0)
        vegas = -pred["odds_spread"].to_numpy(dtype=float)          # home team's margin implied by the spread
        model = pred["pred_margin"].to_numpy(dtype=float)
        if self.coef is None:
            train = has_line & (pred["season"] <= self.train_through) & (pred["season"] >= 2012)
            X = np.column_stack([np.ones(train.sum()), vegas[train], (model - vegas)[train]])
            y = pred.loc[train, "margin"].to_numpy(dtype=float)
            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            self.coef = [float(c) for c in coef]
            self.fit_info = {"n": int(train.sum()), "seasons": sorted(pred.loc[train, "season"].unique().tolist())}
        shaded = self.shade(vegas, model)
        pred["base_margin"] = model
        pred["pred_margin"] = np.where(has_line, shaded, model)
        pred["p_home"] = [normal_cdf(m / self.sigma) for m in pred["pred_margin"]]

        upcoming = games[(~games["completed"]) & games["odds_spread"].notna()]
        self.lines = dict(zip(zip(upcoming["home_id"], upcoming["away_id"]), upcoming["odds_spread"]))
        return pred

    def rating(self, team_id):
        return self.base.rating(team_id)

    def predict(self, home_id, away_id, neutral=False):
        p = self.base.predict(home_id, away_id, neutral)
        spread = self.lines.get((home_id, away_id))
        margin = self.shade(-spread, p["pred_margin"]) if spread is not None and spread != 0 else p["pred_margin"]
        p_home = normal_cdf(margin / self.sigma)
        return {**p, "pred_margin": margin, "p_home": p_home, "p_away": 1 - p_home, "vegas_spread": spread}
