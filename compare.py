"""
Compare every registered model on the same games.

Usage:
    python compare.py                  # held-out 2024-2025 plus the current season
    python compare.py --seasons 2026   # any seasons

Each model is built with its tuned parameters (data/params/<model>.json, defaults if absent),
replayed over history, and scored on the chosen seasons: accuracy, Brier score, log loss and
the average miss on the final margin. The Vegas closing line is scored the same way on the
games that have one, so the market sits in the table as the benchmark.
"""
import argparse

import numpy as np
import pandas as pd

from backtest import TEST_SEASONS, metrics
from common import current_season, load_games, load_params
from models import MODELS


def score(frame, label):
    m = metrics(frame)
    return {"model": label, "games": m["games"], "accuracy": m["accuracy"], "brier": m["brier"],
            "log_loss": m["log_loss"], "margin_mae": (frame["margin"] - frame["pred_margin"]).abs().mean()}


def vegas_sigma(pred, through=2023):
    """Noise level for the closing line, fitted like the models': std of margin minus the line's
    margin on training-season games with a line."""
    t = pred[(pred["season"] >= 2012) & (pred["season"] <= through)].dropna(subset=["odds_spread"])
    return float((t["margin"] + t["odds_spread"]).std())


def vegas_rows(frame, sigma):
    """Score the closing spread as if it were a model: favourite = pick, margin = -spread,
    win probability from a normal with the fitted sigma."""
    v = frame.dropna(subset=["odds_spread"]).copy()
    v["pred_margin"] = -v["odds_spread"]
    v["p_home"] = 0.5 * (1 + np.vectorize(__import__("math").erf)(v["pred_margin"] / (sigma * np.sqrt(2))))
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="*", default=TEST_SEASONS + [current_season()])
    args = parser.parse_args()
    pd.set_option("display.width", 160)

    games = load_games()
    preds = {}
    for name, cls in MODELS.items():
        if name == "overlay":            # identical to its base model on past games
            continue
        params = load_params(name)
        preds[name] = cls(**params).fit(games)
        print(f"{name:8s} params: {params or 'defaults'}")
    sigma_v = vegas_sigma(next(iter(preds.values())))
    print(f"Vegas closing line scored with sigma {sigma_v:.2f} (fitted on training seasons)")

    for season in args.seasons:
        rows, with_line = [], None
        for name, pred in preds.items():
            f = pred[pred["season"] == season]
            if f.empty:
                continue
            rows.append(score(f, name))
            lined = f.dropna(subset=["odds_spread"])
            rows.append(score(lined, f"{name} (games with a line)"))
            with_line = lined if with_line is None else with_line
        if with_line is not None and len(with_line):
            rows.append(score(vegas_rows(with_line, sigma_v), "Vegas closing line"))
        print(f"\nSeason {season}:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # pooled over the held-out seasons, when they are all in the selection
    held = [s for s in TEST_SEASONS if s in args.seasons]
    if len(held) == len(TEST_SEASONS):
        rows = []
        for name, pred in preds.items():
            f = pred[pred["season"].isin(held)]
            rows.append(score(f, name))
        f0 = next(iter(preds.values()))
        lined = f0[f0["season"].isin(held)].dropna(subset=["odds_spread"])
        for name, pred in preds.items():
            fl = pred[pred["season"].isin(held)].dropna(subset=["odds_spread"])
            rows.append(score(fl, f"{name} (games with a line)"))
        rows.append(score(vegas_rows(lined, sigma_v), "Vegas closing line"))
        print(f"\nHeld-out seasons {held[0]}-{held[-1]} pooled:")
        print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
