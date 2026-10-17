"""
What if you had bet on the model's picks?

Replays finished games that have a betting line, using the prediction the model would have
made before each game (its pregame Elo ratings), and grades a flat stake on every pick.

Usage:
    python simulate.py --season 2026 --week 6            # one week
    python simulate.py --from 2026-10-09 --to 2026-10-11 # a date range (US/Eastern)
    python simulate.py --season 2026                     # season to date, with a week-by-week table
    python simulate.py --season 2024 2025                # the held-out seasons (after fetch_odds.py)
Options:
    --stake 100     dollars per bet (default 100)
    --edge 3        minimum edge in points for the "confident spread bets" strategy
    --details       print every game

Lines come from data/games.csv; run fetch_odds.py first to back-fill lines for past games.
"""
import argparse
import datetime as dt
import json
from pathlib import Path

import pandas as pd

from elo import EloModel, load_games
from track import DEFAULT_SPREAD_ODDS, LOCAL_TZ, american_profit, grade, isnum, picks

PARAMS_PATH = Path(__file__).parent / "data" / "model_params.json"


def pregame_predictions(games):
    """Replay history and return one row per finished game with the model's pregame view
    plus the stored betting line."""
    params = json.loads(PARAMS_PATH.read_text()) if PARAMS_PATH.exists() else {}
    pred = EloModel(**params).fit(games)
    cols = ["game_id", "home_id", "away_id", "neutral_site", "odds_details", "home_moneyline",
            "away_moneyline", "home_spread_odds", "away_spread_odds", "home_score", "away_score"]
    return pred.merge(games[cols], on="game_id", how="left")


def graded_rows(frame):
    rows = []
    for g in frame.itertuples(index=False):
        chosen = picks(g.home_team, g.away_team, g.p_home, g.pred_margin, g.odds_spread)
        row = {
            "game_id": g.game_id, "season": g.season, "week": g.week, "date": g.date,
            "away_team": g.away_team, "home_team": g.home_team,
            "p_home": g.p_home, "model_home_margin": round(g.pred_margin, 1),
            "vegas_home_spread": g.odds_spread, "vegas_line": g.odds_details,
            "home_moneyline": g.home_moneyline, "away_moneyline": g.away_moneyline,
            "home_spread_odds": g.home_spread_odds, "away_spread_odds": g.away_spread_odds,
            **chosen,
        }
        row.update(grade(pd.Series(row), g.home_score, g.away_score))
        rows.append(row)
    return pd.DataFrame(rows)


def vegas_favourite_results(df):
    """Baseline: bet the Vegas favourite every game, against the spread and on the moneyline."""
    out = []
    for r in df.itertuples(index=False):
        if not isinstance(r.vegas_pick, str):
            continue
        fav_home = r.vegas_pick == r.home_team
        cover = r.actual_home_margin + r.vegas_home_spread
        if cover == 0:
            ats = 0.0
        else:
            price = r.home_spread_odds if fav_home else r.away_spread_odds
            ats = american_profit(price if isnum(price) else DEFAULT_SPREAD_ODDS) if (cover > 0) == fav_home else -1.0
        ml_price = r.home_moneyline if fav_home else r.away_moneyline
        ml = None
        if isnum(ml_price):
            ml = american_profit(ml_price) if r.vegas_pick == r.winner else -1.0
        out.append({"ats": ats, "ml": ml})
    return pd.DataFrame(out)


def line(label, results, stake):
    """Format one betting strategy: record, dollars won or lost, return on investment."""
    results = [r for r in results if r is not None and not pd.isna(r)]
    if not results:
        return f"{label:<42} no bets"
    wins = sum(1 for r in results if r > 0)
    losses = sum(1 for r in results if r < 0)
    pushes = len(results) - wins - losses
    profit = sum(results) * stake
    staked = len(results) * stake
    rate = wins / (wins + losses) if wins + losses else float("nan")
    rec = f"{wins}-{losses}" + (f"-{pushes}" if pushes else "")
    return f"{label:<42} {rec:>9} ({rate:5.1%})  {profit:+9,.0f} $ on {staked:>8,.0f} $ staked  ROI {profit / staked:+6.1%}"


def report(df, stake, edge, details):
    n = len(df)
    print(f"{n} finished games with a betting line, {df['date'].min().tz_convert(LOCAL_TZ):%a %b %d} to "
          f"{df['date'].max().tz_convert(LOCAL_TZ):%a %b %d, %Y}. Flat stake ${stake:,.0f} per bet.\n")

    su = df["pick_correct"].astype(bool)
    vg = df.dropna(subset=["vegas_pick_correct"])["vegas_pick_correct"].astype(bool)
    print(f"Straight up: model picked the winner in {su.sum()} of {n} ({su.mean():.1%}); "
          f"the Vegas favourite won {vg.sum()} of {len(vg)} ({vg.mean():.1%}).\n")

    print("Betting the model's picks:")
    print(line("Spread, every game", df["spread_profit"].tolist(), stake))
    for e in sorted({edge, 5.0, 7.0}):
        sub = df[df["spread_edge"].astype(float) >= e]
        print(line(f"Spread, only when edge >= {e:g} points", sub["spread_profit"].tolist(), stake))
    print(line("Moneyline, every game", df["moneyline_profit"].tolist(), stake))
    upsets = df[df["model_pick"] != df["vegas_pick"]]
    print(line("Moneyline, only when model disagrees w/ Vegas", upsets["moneyline_profit"].tolist(), stake))

    base = vegas_favourite_results(df)
    if len(base):
        print("\nFor comparison, blindly betting the Vegas favourite:")
        print(line("Spread, every game", base["ats"].tolist(), stake))
        print(line("Moneyline, every game", base["ml"].tolist(), stake))

    if df["week"].nunique() > 1:
        print("\nWeek by week (spread bets on every game):")
        rows = []
        for (season, week), grp in df.groupby(["season", "week"]):
            res = grp["spread_profit"].dropna()
            rows.append({"season": season, "week": week, "bets": len(res),
                         "record": f"{(res > 0).sum()}-{(res < 0).sum()}" + (f"-{(res == 0).sum()}" if (res == 0).any() else ""),
                         "profit $": round(res.sum() * stake)})
        print(pd.DataFrame(rows).to_string(index=False))

    if details:
        show = df.sort_values("date").copy()
        show["date"] = show["date"].dt.tz_convert(LOCAL_TZ).dt.strftime("%a %m-%d")
        show["matchup"] = show["away_team"] + " at " + show["home_team"]
        show["score"] = show["away_score"].astype(int).astype(str) + "-" + show["home_score"].astype(int).astype(str)
        show["model pick"] = show["model_pick"] + " " + (show["model_pick_prob"] * 100).round().astype(int).astype(str) + "%"
        show["spread bet"] = show["spread_pick"].fillna("-") + " (" + show["spread_edge"].astype(float).round(1).astype(str) + ")"
        show["ATS"] = show["spread_result"].fillna("").str.upper()
        show["ATS $"] = (show["spread_profit"].astype(float) * stake).round(0)
        show["ML"] = show["moneyline_result"].fillna("").str.upper()
        show["ML $"] = (show["moneyline_profit"].astype(float) * stake).round(0)
        pd.set_option("display.width", 250)
        print("\nGame by game:")
        print(show[["date", "matchup", "score", "model pick", "vegas_line", "spread bet", "ATS", "ATS $", "ML", "ML $"]]
              .rename(columns={"vegas_line": "vegas"}).to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, nargs="*")
    parser.add_argument("--week", type=int, nargs="*")
    parser.add_argument("--from", dest="date_from", help="YYYY-MM-DD (US/Eastern)")
    parser.add_argument("--to", dest="date_to", help="YYYY-MM-DD (US/Eastern)")
    parser.add_argument("--stake", type=float, default=100.0)
    parser.add_argument("--edge", type=float, default=3.0)
    parser.add_argument("--details", action="store_true")
    args = parser.parse_args()

    games = load_games()
    pred = pregame_predictions(games)
    sel = pred.copy()
    if args.season:
        sel = sel[sel["season"].isin(args.season)]
    if args.week:
        sel = sel[sel["week"].isin(args.week)]
    local = sel["date"].dt.tz_convert(LOCAL_TZ).dt.date
    if args.date_from:
        sel = sel[local >= dt.date.fromisoformat(args.date_from)]
        local = sel["date"].dt.tz_convert(LOCAL_TZ).dt.date
    if args.date_to:
        sel = sel[local <= dt.date.fromisoformat(args.date_to)]

    no_line = sel["odds_spread"].isna().sum()
    sel = sel.dropna(subset=["odds_spread"])
    if sel.empty:
        print("No finished games with a betting line in that selection. Run fetch_odds.py to back-fill lines.")
        return
    if no_line:
        print(f"({no_line} finished games in the selection have no stored line and are skipped; fetch_odds.py can fill them.)\n")
    report(graded_rows(sel), args.stake, args.edge, args.details)


if __name__ == "__main__":
    main()
