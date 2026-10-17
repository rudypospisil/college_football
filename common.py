"""
Shared paths, data loading and parameter files. Everything here is model-agnostic.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SEASON_DIR = DATA_DIR / "seasons"     # one CSV of games per season (raw cache)
GAMES_CSV = DATA_DIR / "games.csv"    # all seasons combined
PARAMS_DIR = DATA_DIR / "params"      # tuned parameters, one JSON per model
LOG_PATH = DATA_DIR / "predictions.csv"
LOCAL_TZ = "US/Eastern"               # college football schedules are published in Eastern time


def current_season(today=None):
    """College football seasons are labelled by the year they start (Aug/Sep)."""
    today = today or dt.date.today()
    return today.year if today.month >= 8 else today.year - 1


def load_games(path=None):
    """Load data/games.csv (or the CSV at `path`) with ids as strings and UTC timestamps."""
    games = pd.read_csv(path or GAMES_CSV, dtype={"game_id": str, "home_id": str, "away_id": str})
    games["date"] = pd.to_datetime(games["date"], utc=True)
    return games


def write_csv(df, path):
    """Write via a temporary file so readers never see a half-written CSV."""
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def rebuild_games():
    """Combine data/seasons/*.csv into data/games.csv (sorted by date)."""
    frames = [pd.read_csv(p, dtype={"game_id": str, "home_id": str, "away_id": str})
              for p in sorted(SEASON_DIR.glob("*.csv"))]
    games = pd.concat(frames, ignore_index=True).sort_values(["date", "game_id"]).reset_index(drop=True)
    write_csv(games, GAMES_CSV)
    return games


def params_path(model_name):
    return PARAMS_DIR / f"{model_name}.json"


def load_params(model_name):
    """Tuned parameters saved by backtest.py, or {} to use the model's defaults."""
    path = params_path(model_name)
    return json.loads(path.read_text()) if path.exists() else {}


def save_params(model_name, params):
    PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    params_path(model_name).write_text(json.dumps(params, indent=2))
    return params_path(model_name)
