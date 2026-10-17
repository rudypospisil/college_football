"""
The margin model with a smarter preseason prior built from CollegeFootballData.com data.

The plain margin model starts every season from last year's rating shrunk toward average.
That misses the two things that change a roster most: recruiting and who came back. Here the
preseason prior for each team is a regression

    prior = b0 + b1 * last season's rating + b2 * recruiting class (z) + b3 * four-class talent (z)
                 + b4 * returning production (z)

whose coefficients are fitted by least squares on training seasons only (through
`prior_train_through`), using each season's final ratings as the target. From that point on the
model is exactly the margin model: weekly weighted ridge fits that start from this prior and let
the season's games take over as they accumulate.

Needs data/cfbd/recruiting_<year>.csv, returning_<year>.csv, games_<year>.csv and team_map.csv
(run fetch_cfbd.py). Seasons or teams without data fall back to the plain prior.
"""
import numpy as np
import pandas as pd

from common import CFBD_DIR
from .base import FCS_ID
from .margin import MarginModel


def zscore(series):
    s = series.astype(float)
    return (s - s.mean()) / (s.std() if s.std() > 0 else 1.0)


def load_features():
    """Per (season, espn_id): recruiting z-score, four-class talent z-score, returning-production z-score."""
    team_map = pd.read_csv(CFBD_DIR / "team_map.csv", dtype={"espn_id": str})
    cfbd_to_espn = dict(zip(team_map["cfbd_id"], team_map["espn_id"]))
    # CFBD recruiting/returning files are keyed by school name; games files give name -> cfbd id.
    name_to_cfbd = {}
    for path in sorted(CFBD_DIR.glob("games_*.csv")):
        g = pd.read_csv(path, usecols=["home_id", "home_team", "away_id", "away_team"])
        name_to_cfbd.update(dict(zip(g["home_team"], g["home_id"])))
        name_to_cfbd.update(dict(zip(g["away_team"], g["away_id"])))

    def espn_id(school):
        return cfbd_to_espn.get(name_to_cfbd.get(school))

    rec_frames = []
    for path in sorted(CFBD_DIR.glob("recruiting_*.csv")):
        r = pd.read_csv(path)
        r["espn_id"] = r["team"].map(espn_id)
        r = r.dropna(subset=["espn_id"])
        r["recruit_z"] = zscore(r["points"])
        rec_frames.append(r[["year", "espn_id", "recruit_z"]].rename(columns={"year": "season"}))
    rec = pd.concat(rec_frames, ignore_index=True)
    # talent: mean recruiting z over this class and the three before it (the roster's four classes)
    rec = rec.sort_values(["espn_id", "season"])
    rec["talent_z"] = rec.groupby("espn_id")["recruit_z"].transform(lambda s: s.rolling(4, min_periods=1).mean())
    rec["talent_z"] = rec.groupby("season")["talent_z"].transform(zscore)

    ret_frames = []
    for path in sorted(CFBD_DIR.glob("returning_*.csv")):
        r = pd.read_csv(path)
        r["espn_id"] = r["team"].map(espn_id)
        r = r.dropna(subset=["espn_id", "percent_ppa"])
        r["returning_z"] = zscore(r["percent_ppa"])
        ret_frames.append(r[["season", "espn_id", "returning_z"]])
    ret = pd.concat(ret_frames, ignore_index=True)

    feats = rec.merge(ret, on=["season", "espn_id"], how="outer")
    return feats.set_index(["season", "espn_id"]).sort_index()


class PriorMarginModel(MarginModel):
    name = "prior"
    GRID = {
        "lam": [2.0, 4.0, 8.0],
        "half_life": [120.0, 180.0, 365.0],
        "hfa": [2.5, 3.0],
        "cap": [28.0, 0.0],
    }
    FEATURES = ["last_rating", "recruit_z", "talent_z", "returning_z"]

    def __init__(self, lam=2.0, half_life=180.0, regress=0.2, hfa=3.0, cap=0.0, sigma=16.9,
                 prior_train_through=2023, fbs_min_games=6, new_team_prior=-12.0, fcs_prior=-25.0):
        super().__init__(lam=lam, half_life=half_life, regress=regress, hfa=hfa, cap=cap, sigma=sigma,
                         fbs_min_games=fbs_min_games, new_team_prior=new_team_prior, fcs_prior=fcs_prior)
        self.prior_train_through = prior_train_through
        self.features = None
        self.coef = None        # fitted prior regression coefficients (intercept first)

    def params(self):
        return {**super().params(), "prior_train_through": self.prior_train_through}

    # ------------------------------------------------------------ prior regression
    def _feature_row(self, season, team_id, last):
        f = self.features.loc[(season, team_id)] if (season, team_id) in self.features.index else None
        return [
            last.get(team_id, np.nan),
            float(f["recruit_z"]) if f is not None and pd.notna(f["recruit_z"]) else 0.0,
            float(f["talent_z"]) if f is not None and pd.notna(f["talent_z"]) else 0.0,
            float(f["returning_z"]) if f is not None and pd.notna(f["returning_z"]) else 0.0,
        ]

    def _fit_prior_regression(self, base_final):
        """Least squares: final rating in season S ~ features known before season S.
        Only seasons through prior_train_through are used, so held-out seasons stay clean."""
        X, y = [], []
        for season in sorted(base_final):
            if season <= min(base_final) or season > self.prior_train_through:
                continue
            last = base_final[season - 1]
            for team_id, rating in base_final[season].items():
                if team_id == FCS_ID or team_id not in last:
                    continue
                X.append([1.0] + self._feature_row(season, team_id, last))
                y.append(rating)
        X, y = np.array(X), np.array(y)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - (X * coef).sum(axis=1)      # elementwise: avoids a spurious BLAS warning on macOS
        self.coef = coef
        self.prior_fit_info = {"n": len(y), "rmse": float(np.sqrt((resid ** 2).mean())),
                               "coef": dict(zip(["intercept"] + self.FEATURES, np.round(coef, 3)))}
        return coef

    def _prior(self, season, entities):
        """Regression prior where features exist; the plain shrink-toward-zero prior otherwise."""
        base = super()._prior(season, entities)
        if self.coef is None:
            return base
        last = self.final_ratings.get(season - 1, {})
        prior = base.copy()
        for i, e in enumerate(entities):
            if e == FCS_ID or e not in last:
                continue                        # FCS pool and promoted teams keep the plain prior
            prior[i] = float(np.dot(self.coef, [1.0] + self._feature_row(season, e, last)))
        return prior

    # ----------------------------------------------------------------------- fit
    def fit(self, games):
        self.features = load_features()
        # Pass 1: plain margin model gives each season's final ratings (the regression targets).
        base = MarginModel(lam=self.lam, half_life=self.half_life, regress=self.regress, hfa=self.hfa,
                           cap=self.cap, sigma=self.sigma, fbs_min_games=self.fbs_min_games,
                           new_team_prior=self.new_team_prior, fcs_prior=self.fcs_prior)
        base.fit(games)
        self._fit_prior_regression(base.final_ratings)
        # Pass 2: the real walk-forward run with regression priors.
        self.final_ratings = {}
        return super().fit(games)
