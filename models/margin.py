"""
A margin-based rating model: weighted ridge regression on point differentials.

How it works
------------
* Each team has a rating in points above an average FBS team. For a game,
      margin = rating_home - rating_away + hfa (0 at a neutral site) + noise.
* Before every week, the ratings are re-fit by least squares on the season's games so far,
  with two adjustments that make the early season sane:
    - time decay: a game `half_life` days old counts half as much as one played yesterday;
    - a prior: each rating is pulled toward last season's final rating, shrunk `regress` of
      the way to average, with strength `lam` (measured in equivalent games). With no games
      played yet the ratings simply equal the prior.
* Margins are capped at +/- `cap` points so running up the score does not count extra.
* Win probability = Phi(predicted margin / sigma), the normal distribution; `sigma` is the
  standard deviation of actual minus predicted margin, estimated from the training seasons.
* FCS opponents are pooled into one entity, as in the Elo model.

Everything is walk-forward: the prediction for a game only uses games played before the first
kickoff of that game's week.
"""
import math

import numpy as np
import pandas as pd

from .base import FCS_ID, RatingModel


def normal_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class MarginModel(RatingModel):
    name = "margin"
    GRID = {
        "lam": [1.0, 2.0, 4.0, 8.0],
        "half_life": [90.0, 180.0, 365.0],
        "regress": [0.2, 0.4],
        "hfa": [2.0, 2.5, 3.0],
        "cap": [28.0, 0.0],          # 0 = no cap
    }

    def __init__(self, lam=10.0, half_life=120.0, regress=0.4, hfa=2.5, cap=28.0, sigma=17.0,
                 fbs_min_games=6, new_team_prior=-12.0, fcs_prior=-25.0):
        super().__init__(fbs_min_games=fbs_min_games)
        self.lam = lam
        self.half_life = half_life
        self.regress = regress
        self.hfa = hfa
        self.cap = cap
        self.sigma = sigma
        self.new_team_prior = new_team_prior
        self.fcs_prior = fcs_prior

        self.ratings = {FCS_ID: fcs_prior}   # latest fit: team_id -> points above average
        self.final_ratings = {}              # season -> {team_id: rating} after the last game

    def params(self):
        return {"lam": self.lam, "half_life": self.half_life, "regress": self.regress, "hfa": self.hfa,
                "cap": self.cap, "sigma": self.sigma, "fbs_min_games": self.fbs_min_games,
                "new_team_prior": self.new_team_prior, "fcs_prior": self.fcs_prior}

    def rating(self, team_id):
        return self.ratings.get(team_id, self.new_team_prior)

    def target(self, done):
        """What the ratings are fit to for each finished game: the actual margin here.
        Subclasses can swap in something less noisy (see models/efficiency.py)."""
        return done["margin"].astype(float).to_numpy()

    def post_fit_params(self, pred):
        """The noise level: standard deviation of the margin residuals on the training seasons."""
        resid = pred["margin"] - pred["pred_margin"]
        return {"sigma": round(float(resid.std()), 2)}

    # ------------------------------------------------------------------ fitting
    def _prior(self, season, entities):
        """Prior rating for each entity this season: last season's final rating shrunk toward 0."""
        last = self.final_ratings.get(season - 1, {})
        prior = np.zeros(len(entities))
        for i, e in enumerate(entities):
            if e == FCS_ID:
                prior[i] = last.get(e, self.fcs_prior)
            elif e in last:
                prior[i] = (1 - self.regress) * last[e]
            elif self.final_ratings:          # promoted from FCS (or first appearance after 2010)
                prior[i] = self.new_team_prior
        return prior

    def _solve(self, home_idx, away_idx, y, weights, prior):
        """Weighted ridge regression with a prior mean: (X'WX + lam*I) r = X'Wy + lam*prior."""
        n = len(prior)
        a = np.zeros((n, n))
        b = np.zeros(n)
        for h, aw, yy, w in zip(home_idx, away_idx, y, weights):
            a[h, h] += w
            a[aw, aw] += w
            a[h, aw] -= w
            a[aw, h] -= w
            b[h] += w * yy
            b[aw] -= w * yy
        a[np.diag_indices(n)] += self.lam
        return np.linalg.solve(a, b + self.lam * prior)

    def fit(self, games):
        games = games.sort_values(["date", "game_id"]).reset_index(drop=True)
        self.classify_fbs(games)
        for g in games.itertuples(index=False):
            self.names[g.home_id] = g.home_team
            self.names[g.away_id] = g.away_team

        rows = []
        for season, season_games in games.groupby("season", sort=True):
            self.current_season = season
            entities = sorted(self.fbs_by_season.get(season, set()) | {FCS_ID})
            index = {e: i for i, e in enumerate(entities)}
            prior = self._prior(season, entities)

            done = season_games[season_games["completed"] & season_games["home_score"].notna()].copy()
            done["h"] = [index[self.entity(t, season)] for t in done["home_id"]]
            done["a"] = [index[self.entity(t, season)] for t in done["away_id"]]
            done["margin"] = (done["home_score"] - done["away_score"]).astype(int)
            done["t"] = done["date"].dt.tz_convert("UTC").dt.tz_localize(None)   # naive UTC -> datetime64 arrays
            y_all = self.target(done)
            if self.cap:
                y_all = np.clip(y_all, -self.cap, self.cap)
            y_all = y_all - self.hfa * (~done["neutral_site"].astype(bool)).to_numpy()
            dates = done["t"].to_numpy()
            h_all, a_all = done["h"].to_numpy(), done["a"].to_numpy()

            # One fit per week, using only games that finished before that week's first kickoff.
            for _, week_games in done.groupby(["season_type", "week"], sort=False):
                start = week_games["t"].min().to_datetime64()
                before = dates < start
                if before.any():
                    age_days = (start - dates[before]) / np.timedelta64(1, "D")
                    weights = 0.5 ** (age_days / self.half_life)
                    r = self._solve(h_all[before], a_all[before], y_all[before], weights, prior)
                else:
                    r = prior.copy()
                for g in week_games.itertuples(index=False):
                    if g.h == g.a:
                        continue
                    diff = r[g.h] - r[g.a] + (0.0 if g.neutral_site else self.hfa)
                    margin = int(g.margin)
                    rows.append({
                        "game_id": g.game_id, "season": season, "week": g.week, "date": g.date,
                        "home_team": g.home_team, "away_team": g.away_team,
                        "home_is_fbs": entities[g.h] != FCS_ID, "away_is_fbs": entities[g.a] != FCS_ID,
                        "rating_home": r[g.h], "rating_away": r[g.a],
                        "p_home": normal_cdf(diff / self.sigma), "pred_margin": diff,
                        "home_won": 1.0 if margin > 0 else (0.5 if margin == 0 else 0.0), "margin": margin,
                        "odds_spread": g.odds_spread,
                    })

            # Season-final fit on everything played: next season's prior and the live ratings.
            if len(done):
                now = max(dates.max(), season_games["date"].max().tz_convert("UTC").tz_localize(None).to_datetime64())
                age_days = (now - dates) / np.timedelta64(1, "D")
                r = self._solve(h_all, a_all, y_all, 0.5 ** (age_days / self.half_life), prior)
            else:
                r = prior.copy()
            self.final_ratings[season] = dict(zip(entities, r))
            self.ratings = dict(zip(entities, r))
        return pd.DataFrame(rows)

    # ----------------------------------------------------------------- predict
    def predict(self, home_id, away_id, neutral=False):
        season = self.current_season
        home, away = self.entity(home_id, season), self.entity(away_id, season)
        diff = self.rating(home) - self.rating(away) + (0.0 if neutral else self.hfa)
        p_home = normal_cdf(diff / self.sigma)
        return {
            "home": self.names.get(home_id, home_id), "away": self.names.get(away_id, away_id),
            "rating_home": self.rating(home), "rating_away": self.rating(away),
            "p_home": p_home, "p_away": 1 - p_home, "pred_margin": diff,
        }
