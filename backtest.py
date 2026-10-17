"""
Tune a model's parameters on past seasons and measure how well it predicts seasons it has
never seen.

Usage:
    python backtest.py                    # Elo: grid search + out-of-sample report
    python backtest.py --model elo        # any registered model (see models/__init__.py)
    python backtest.py --quick            # skip the grid search, use the model's defaults

Writes the chosen parameters to data/params/<model>.json (used by predict.py and friends).
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd

from common import load_games, save_params
from models import MODELS, add_model_argument

BURN_IN_SEASONS = 2          # the first seasons only exist to warm up the ratings
TEST_SEASONS = [2024, 2025]  # held out: never used to choose parameters


def metrics(pred):
    """Accuracy, Brier score and log loss for a frame of pregame predictions."""
    p = pred["p_home"].clip(1e-6, 1 - 1e-6)
    y = pred["home_won"]
    picked_right = ((p >= 0.5) == (y >= 0.5)).mean()
    brier = ((p - y) ** 2).mean()
    logloss = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    return {"games": len(pred), "accuracy": picked_right, "brier": brier, "log_loss": logloss}


def run(model_cls, games, **params):
    model = model_cls(**params)
    pred = model.fit(games)
    return model, pred


def grid_search(model_cls, games, train_seasons):
    grid = model_cls.GRID
    results = []
    combos = list(itertools.product(*grid.values()))
    print(f"Grid search over {len(combos)} parameter combinations on seasons "
          f"{train_seasons[0]}-{train_seasons[-1]} ...")
    for values in combos:
        params = dict(zip(grid.keys(), values))
        _, pred = run(model_cls, games, **params)
        scored = pred[pred["season"].isin(train_seasons)]
        results.append({**params, **metrics(scored)})
    results = pd.DataFrame(results).sort_values("log_loss").reset_index(drop=True)
    return results


def calibration_table(pred, bins=(0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    cut = pd.cut(pred["p_home"], bins=bins, include_lowest=True)
    table = pred.groupby(cut, observed=True).agg(games=("home_won", "size"),
                                                predicted=("p_home", "mean"),
                                                actual=("home_won", "mean"))
    return table.round(3)


def vegas_comparison(pred):
    """Where a spread is stored, compare the model's pick and margin to Vegas's."""
    v = pred.dropna(subset=["odds_spread"]).copy()
    if v.empty:
        return None
    # ESPN's `spread` is from the home team's perspective (negative = home favoured).
    v["vegas_home_fav"] = v["odds_spread"] < 0
    v["model_home_fav"] = v["p_home"] >= 0.5
    v["home_actually_won"] = v["home_won"] >= 0.5
    v = v[v["odds_spread"] != 0]
    return {
        "games_with_line": len(v),
        "vegas_accuracy": (v["vegas_home_fav"] == v["home_actually_won"]).mean(),
        "model_accuracy": (v["model_home_fav"] == v["home_actually_won"]).mean(),
        "agreement": (v["vegas_home_fav"] == v["model_home_fav"]).mean(),
        "model_spread_mae": (v["pred_margin"] - v["margin"]).abs().mean(),
        "vegas_spread_mae": (-v["odds_spread"] - v["margin"]).abs().mean(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_argument(parser)
    parser.add_argument("--quick", action="store_true", help="skip grid search, use the model's defaults")
    args = parser.parse_args()
    pd.set_option("display.width", 160)
    model_cls = MODELS[args.model]

    games = load_games()
    seasons = sorted(games["season"].unique())
    train_seasons = [s for s in seasons[BURN_IN_SEASONS:] if s < min(TEST_SEASONS)]
    print(f"Model: {args.model}. Loaded {len(games)} games, {int(games['completed'].sum())} completed, "
          f"seasons {seasons[0]}-{seasons[-1]}")

    if args.quick or not model_cls.GRID:
        best = {k: v for k, v in model_cls().params().items() if k in model_cls.GRID}
    else:
        results = grid_search(model_cls, games, train_seasons)
        print("\nTop 10 parameter sets (training seasons, lower log loss is better):")
        print(results.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        best = results.iloc[0][list(model_cls.GRID)].to_dict()
        best = {k: (v.item() if hasattr(v, "item") else v) for k, v in best.items()}   # plain Python types for JSON

    # Anything the model fits after the grid search (e.g. Elo's rating-to-points scale) is
    # fitted on the training seasons only, then the model is re-run with it.
    model, pred = run(model_cls, games, **best)
    best.update(model.post_fit_params(pred[pred["season"].isin(train_seasons)]))
    model, pred = run(model_cls, games, **best)
    print(f"\nChosen parameters: {json.dumps(best)}")
    if "elo_per_point" in best:
        print(f"(1 point of spread ~ {best['elo_per_point']} Elo points)")

    # ---- out-of-sample report
    train = pred[pred["season"].isin(train_seasons)]
    test = pred[pred["season"].isin(TEST_SEASONS)]
    current = pred[pred["season"] > max(TEST_SEASONS)]

    def row(label, frame):
        m = metrics(frame)
        home_rate = frame["home_won"].mean()
        return {"split": label, **m,
                "always_home_acc": max(home_rate, 1 - home_rate),
                "always_home_brier": ((home_rate - frame["home_won"]) ** 2).mean()}

    report = pd.DataFrame([row(f"train {train_seasons[0]}-{train_seasons[-1]}", train),
                           row(f"TEST {TEST_SEASONS[0]}-{TEST_SEASONS[-1]} (out of sample)", test)]
                          + ([row(f"current season {int(current['season'].iloc[0])} so far", current)] if len(current) else []))
    print("\nModel vs. naive baseline (always pick the home team / constant home-win rate):")
    print(report.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nOut-of-sample results by season:")
    by_season = test.groupby("season").apply(lambda f: pd.Series(metrics(f)), include_groups=False)
    print(by_season.to_string(float_format=lambda x: f"{x:.4f}"))

    fbs_only = test[test["home_is_fbs"] & test["away_is_fbs"]]
    print(f"\nFBS-vs-FBS games only (test seasons): {json.dumps({k: round(v, 4) for k, v in metrics(fbs_only).items()})}")

    print("\nCalibration on test seasons (predicted home-win probability vs. what happened):")
    print(calibration_table(test).to_string())

    vegas = vegas_comparison(test)
    if vegas:
        print("\nModel vs. Vegas on test-season games with a stored point spread:")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in vegas.items()}, indent=2))

    print(f"\nCurrent top 25 ({args.model}):")
    print(model.rankings(25).to_string(index=False))

    path = save_params(args.model, best)
    print(f"\nSaved parameters to {path}")


if __name__ == "__main__":
    main()
