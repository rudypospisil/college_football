"""
Tune the Elo model's parameters on past seasons and measure how well it predicts
seasons it has never seen.

Usage:
    python backtest.py              # grid search + out-of-sample report
    python backtest.py --quick      # skip the grid search, use default parameters

Writes the chosen parameters to data/model_params.json (used by predict.py).
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from elo import EloModel, load_games

PARAMS_PATH = Path(__file__).parent / "data" / "model_params.json"
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


def fit_elo_per_point(pred):
    """Least-squares fit of actual margin vs Elo difference (through the origin)."""
    d, m = pred["elo_diff"].to_numpy(), pred["margin"].to_numpy()
    slope = (d * m).sum() / (d * d).sum()      # points per Elo point
    return 1.0 / slope


def run(games, **params):
    model = EloModel(**params)
    pred = model.fit(games)
    return model, pred


def grid_search(games, train_seasons):
    grid = {
        "k": [20, 30, 40, 50, 60],
        "hfa": [40, 55, 70, 85],
        "regress": [0.1, 0.2, 0.33, 0.5],
        "mov": [True, False],
    }
    results = []
    combos = list(itertools.product(*grid.values()))
    print(f"Grid search over {len(combos)} parameter combinations on seasons "
          f"{train_seasons[0]}-{train_seasons[-1]} ...")
    for values in combos:
        params = dict(zip(grid.keys(), values))
        _, pred = run(games, **params)
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
    """Where ESPN stored a closing/pregame spread, compare the model's pick to Vegas's pick."""
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
    parser.add_argument("--quick", action="store_true", help="skip grid search, use defaults")
    args = parser.parse_args()
    pd.set_option("display.width", 160)

    games = load_games()
    seasons = sorted(games["season"].unique())
    train_seasons = [s for s in seasons[BURN_IN_SEASONS:] if s < min(TEST_SEASONS)]
    print(f"Loaded {len(games)} games, {int(games['completed'].sum())} completed, seasons {seasons[0]}-{seasons[-1]}")

    if args.quick:
        best = {"k": 25, "hfa": 60, "regress": 0.33, "mov": True}
    else:
        results = grid_search(games, train_seasons)
        print("\nTop 10 parameter sets (training seasons, lower log loss is better):")
        print(results.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        best = results.iloc[0][["k", "hfa", "regress", "mov"]].to_dict()
        best = {"k": float(best["k"]), "hfa": float(best["hfa"]), "regress": float(best["regress"]), "mov": bool(best["mov"])}

    # Fit the spread conversion on the training seasons only, then re-run with it.
    _, pred = run(games, **best)
    best["elo_per_point"] = round(fit_elo_per_point(pred[pred["season"].isin(train_seasons)]), 2)
    model, pred = run(games, **best)
    print(f"\nChosen parameters: {json.dumps(best)}")
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
        print("\nModel vs. Vegas on test-season games where ESPN stored a point spread:")
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in vegas.items()}, indent=2))

    print("\nCurrent top 25 by Elo:")
    print(model.rankings(25).to_string(index=False))

    PARAMS_PATH.parent.mkdir(exist_ok=True)
    PARAMS_PATH.write_text(json.dumps(best, indent=2))
    print(f"\nSaved parameters to {PARAMS_PATH}")


if __name__ == "__main__":
    main()
