"""
Predict college football games.

Usage:
    python predict.py "Ohio State" "Indiana"            # away team first, then home team
    python predict.py "Ohio State" "Indiana" --neutral  # neutral-site game
    python predict.py --date 2026-10-17                 # every game on a date, with Vegas lines
    python predict.py --rankings                        # current top 25
    python predict.py --model elo ...                   # choose a registered model (default: elo)

Parameters come from data/params/<model>.json (written by backtest.py); defaults are used if missing.
"""
import argparse
import datetime as dt
import json

import pandas as pd

from common import LOCAL_TZ, load_games
from models import DEFAULT_MODEL, add_model_argument, fitted_model


def load_model(name=DEFAULT_MODEL):
    """The named model, tuned and replayed on everything in data/games.csv."""
    games = load_games()
    return fitted_model(name, games), games


def describe(pred, vegas=None):
    fav = pred["home"] if pred["p_home"] >= 0.5 else pred["away"]
    p_fav = max(pred["p_home"], pred["p_away"])
    spread = abs(pred["pred_margin"])
    lines = [
        f"{pred['away']} (rating {pred['rating_away']:.0f}) at {pred['home']} (rating {pred['rating_home']:.0f})",
        f"  Model: {fav} {p_fav:.1%} to win, projected spread {fav} -{spread:.1f}",
    ]
    if vegas:
        lines.append(f"  Vegas: {vegas}")
    return "\n".join(lines)


def predict_date(model, games, date):
    day = games[(games["date"].dt.tz_convert(LOCAL_TZ).dt.date == date)]
    if day.empty:
        print(f"No games found on {date} in data/games.csv (run fetch_data.py to refresh).")
        return
    rows = []
    for g in day.sort_values("date").itertuples(index=False):
        p = model.predict(g.home_id, g.away_id, neutral=g.neutral_site)
        rows.append({
            "away": g.away_team, "home": g.home_team,
            "p_home": round(p["p_home"], 3),
            "model_spread(home)": round(-p["pred_margin"], 1),
            "vegas_spread(home)": g.odds_spread,
            "vegas_line": g.odds_details,
            "status": g.status,
            "score": f"{g.away_score:.0f}-{g.home_score:.0f}" if g.completed else "",
        })
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("away", nargs="?", help="away team, e.g. 'Ohio State'")
    parser.add_argument("home", nargs="?", help="home team, e.g. 'Indiana'")
    parser.add_argument("--neutral", action="store_true", help="neutral site (no home-field advantage)")
    parser.add_argument("--date", help="predict all games on a date (YYYY-MM-DD, US/Eastern)")
    parser.add_argument("--rankings", action="store_true", help="print the current top 25")
    add_model_argument(parser)
    args = parser.parse_args()

    model, games = load_model(args.model)
    print(f"Model: {model.name}, parameters: {json.dumps(model.params())}\n")

    if args.rankings:
        print(model.rankings(25).to_string(index=False))
    if args.date:
        predict_date(model, games, dt.date.fromisoformat(args.date))
    if args.away and args.home:
        home_id, away_id = model.find_team(args.home), model.find_team(args.away)
        pred = model.predict(home_id, away_id, neutral=args.neutral)
        match = games[(games["home_id"] == home_id) & (games["away_id"] == away_id) & (~games["completed"])]
        vegas = match["odds_details"].dropna().iloc[0] if not match.empty and match["odds_details"].notna().any() else None
        print(describe(pred, vegas))
    elif not (args.rankings or args.date):
        parser.print_help()


if __name__ == "__main__":
    main()
