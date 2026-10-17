"""
The prior model fit to efficiency instead of (only) the scoreboard.

Final scores are noisy: a fumble at the goal line or a blocked punt swings the margin by two
touchdowns without saying much about either team. CollegeFootballData's predicted points added
(PPA) per play measures how efficiently each offense actually moved the ball, and across 12,000
games it explains 74% of the variance in the final margin (correlation 0.86). This model fits the
weekly ridge regression to a blend:

    target = (1 - w) * actual margin + w * efficiency margin
    efficiency margin = ppa_scale * (home offense PPA per play - away offense PPA per play)

`ppa_scale` (about 64 points per unit of PPA-per-play difference, i.e. roughly a game's worth
of plays) is fitted on the training seasons. Games without PPA for both teams (mostly FBS vs
FCS) fall back to the actual margin. Everything else, the recruiting/returning prior, the time
decay and the walk-forward weekly fits, is inherited from the prior model.

Needs data/cfbd/ppa_<year>.csv and games_<year>.csv (fetch_cfbd.py).
"""
import numpy as np
import pandas as pd

from common import CFBD_DIR
from .prior import PriorMarginModel


def load_ppa_by_game(games, source="ppa"):
    """Per game: offense PPA per play for the home and away team, keyed by ESPN game id
    (CFBD uses the same ids). `source` is "ppa" or "ppa_clean" (garbage time excluded)."""
    frames = [pd.read_csv(p, dtype={"game_id": str}, usecols=["game_id", "team", "offense_overall"])
              for p in sorted(CFBD_DIR.glob(f"{source}_[0-9]*.csv"))]
    ppa = pd.concat(frames, ignore_index=True)
    names = {}
    for path in sorted(CFBD_DIR.glob("games_*.csv")):
        g = pd.read_csv(path, usecols=["home_id", "home_team", "away_id", "away_team"])
        names.update(dict(zip(g["home_team"], g["home_id"].astype(str))))
        names.update(dict(zip(g["away_team"], g["away_id"].astype(str))))
    ppa["espn_id"] = ppa["team"].map(names)
    ppa = ppa.dropna(subset=["espn_id", "offense_overall"]).drop_duplicates(["game_id", "espn_id"])
    m = ppa.merge(games[["game_id", "home_id", "away_id"]], on="game_id")
    home = m[m["espn_id"] == m["home_id"]].set_index("game_id")["offense_overall"]
    away = m[m["espn_id"] == m["away_id"]].set_index("game_id")["offense_overall"]
    return home, away


class EfficiencyModel(PriorMarginModel):
    name = "eff"
    GRID = {
        "w": [0.25, 0.5, 0.75, 1.0],
        "lam": [2.0, 4.0, 8.0],
        "half_life": [180.0, 365.0],
        "hfa": [2.5, 3.0],
    }

    def __init__(self, w=0.5, ppa_scale=None, ppa_source="ppa", lam=4.0, half_life=365.0, regress=0.2, hfa=2.5,
                 cap=0.0, sigma=16.6, prior_train_through=2023, fbs_min_games=6, new_team_prior=-12.0, fcs_prior=-25.0):
        super().__init__(lam=lam, half_life=half_life, regress=regress, hfa=hfa, cap=cap, sigma=sigma,
                         prior_train_through=prior_train_through, fbs_min_games=fbs_min_games,
                         new_team_prior=new_team_prior, fcs_prior=fcs_prior)
        self.w = w
        self.ppa_source = ppa_source
        self.ppa_scale = ppa_scale       # None -> fitted on the training seasons in fit()
        self.ppa_coverage = None

    def params(self):
        return {**super().params(), "w": self.w, "ppa_scale": self.ppa_scale, "ppa_source": self.ppa_source}

    def target(self, done):
        margin = done["margin"].astype(float).to_numpy()
        if "ppa_diff" not in done.columns:
            return margin
        eff = self.ppa_scale * done["ppa_diff"].to_numpy(dtype=float)
        return np.where(np.isnan(eff), margin, (1.0 - self.w) * margin + self.w * eff)

    def fit(self, games):
        games = games.copy()
        home, away = load_ppa_by_game(games, self.ppa_source)
        games["ppa_diff"] = games["game_id"].map(home) - games["game_id"].map(away)
        done = games[games["completed"] & games["home_score"].notna()]
        self.ppa_coverage = float(done["ppa_diff"].notna().mean())
        if self.ppa_scale is None:
            train = done[(done["season"] <= self.prior_train_through) & done["ppa_diff"].notna()]
            d = train["ppa_diff"].to_numpy(dtype=float)
            m = (train["home_score"] - train["away_score"]).to_numpy(dtype=float)
            self.ppa_scale = round(float((d * m).sum() / (d * d).sum()), 2)
        return super().fit(games)
