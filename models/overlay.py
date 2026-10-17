"""
Injury / news overlay: a base model's margin plus point adjustments you enter before kickoff.

The models only know final scores, efficiency and rosters as of last season. When you know a
starting quarterback is out or an offensive line is patched together, enter it with adjust.py:

    python adjust.py add --team "Ohio State" --points -1.5 --reason "RB Ja'Kobi Jackson out; OL shuffling"

Rows live in data/adjustments.csv (game, team, points, reason, source, time entered). Positive
points favour the team. This model adds the net adjustment (home points minus away points) to
the base model's margin and recomputes the win probability; everything else is the base model.
It only applies to games with an adjustment, so its rows in the prediction log are exactly the
games you touched, and track.py grades them next to the base model's picks on the same games.

Rules of thumb for point values (roughly what sportsbooks move lines by): starting quarterback
4-7 depending on the backup, top running back or receiver 1-2, starting lineman or top pass
rusher 0.5-1 each, kicker 0.5; halve it for "questionable"; keep a team's total under about 7.
"""
import pandas as pd

from common import DATA_DIR, load_params
from .base import RatingModel
from .margin import normal_cdf

ADJUSTMENTS_CSV = DATA_DIR / "adjustments.csv"
ADJ_COLUMNS = ["game_id", "date", "team_id", "team", "points", "reason", "source", "entered_at"]


def load_adjustments():
    if ADJUSTMENTS_CSV.exists():
        return pd.read_csv(ADJUSTMENTS_CSV, dtype={"game_id": str, "team_id": str})
    return pd.DataFrame(columns=ADJ_COLUMNS)


def save_adjustments(df):
    ADJUSTMENTS_CSV.parent.mkdir(exist_ok=True)
    df[ADJ_COLUMNS].to_csv(ADJUSTMENTS_CSV, index=False)


def adjustments_by_game(games):
    """{game_id: net points for the home team} from every adjustment row."""
    adj = load_adjustments()
    if adj.empty:
        return {}
    sides = games.set_index("game_id")[["home_id", "away_id"]]
    out = {}
    for row in adj.itertuples(index=False):
        if row.game_id not in sides.index:
            continue
        home_id, away_id = sides.at[row.game_id, "home_id"], sides.at[row.game_id, "away_id"]
        sign = 1.0 if row.team_id == home_id else -1.0 if row.team_id == away_id else 0.0
        out[row.game_id] = out.get(row.game_id, 0.0) + sign * float(row.points)
    return out


class OverlayModel(RatingModel):
    name = "overlay"
    GRID = {}

    def __init__(self, base="prior", fbs_min_games=6):
        super().__init__(fbs_min_games=fbs_min_games)
        self.base_name = base
        self.base = None
        self.sigma = 16.6
        self.adjust = {}
        self.upcoming = {}

    def params(self):
        return {"base": self.base_name, "fbs_min_games": self.fbs_min_games}

    def fit(self, games):
        from . import MODELS
        self.base = MODELS[self.base_name](**load_params(self.base_name))
        pred = self.base.fit(games).copy()
        self.names, self.fbs_by_season, self.current_season = self.base.names, self.base.fbs_by_season, self.base.current_season
        self.sigma = float(getattr(self.base, "sigma", 16.6))
        self.adjust = adjustments_by_game(games)
        upcoming = games[~games["completed"]]
        self.upcoming = dict(zip(zip(upcoming["home_id"], upcoming["away_id"]), upcoming["game_id"]))
        pred["adjustment"] = pred["game_id"].map(self.adjust).fillna(0.0)
        pred["pred_margin"] = pred["pred_margin"] + pred["adjustment"]
        pred["p_home"] = [normal_cdf(m / self.sigma) for m in pred["pred_margin"]]
        return pred

    def applies_to(self, game_id):
        return bool(self.adjust.get(game_id))

    def rating(self, team_id):
        return self.base.rating(team_id)

    def predict(self, home_id, away_id, neutral=False):
        p = self.base.predict(home_id, away_id, neutral)
        adj = self.adjust.get(self.upcoming.get((home_id, away_id)), 0.0)
        margin = p["pred_margin"] + adj
        p_home = normal_cdf(margin / self.sigma)
        return {**p, "pred_margin": margin, "p_home": p_home, "p_away": 1 - p_home, "adjustment": adj}
