"""
Enter, list and remove injury / news adjustments for the overlay model.

Usage:
    python adjust.py add --team "Ohio State" --points -1.5 --reason "RB Ja'Kobi Jackson out (leg); OL shuffling"
    python adjust.py add --team Indiana --points 1 --reason "..." --source "The Athletic" --date 2026-10-17
    python adjust.py list [--date 2026-10-17]
    python adjust.py remove --row 2            # row number from `list`

Positive points favour the team. An adjustment can only be entered before the game kicks off,
and adding one immediately locks in the overlay model's pick for that game in the prediction
log, so the overlay is graded on exactly the games you adjusted. To change an adjustment before
kickoff, remove it (which also removes the logged pick) and add it again.
"""
import argparse
import datetime as dt

import pandas as pd

from common import LOCAL_TZ
from models.overlay import ADJ_COLUMNS, load_adjustments, save_adjustments
from predict import load_model
from track import load_log, record, save_log


def game_for(games, team_id, date):
    day = games[games["date"].dt.tz_convert(LOCAL_TZ).dt.date == date]
    g = day[(day["home_id"] == team_id) | (day["away_id"] == team_id)]
    return g.iloc[0] if len(g) else None


def add(args):
    model, games = load_model("overlay")
    team_id = model.find_team(args.team)
    team = model.names[team_id]
    date = dt.date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz=LOCAL_TZ).date()
    g = game_for(games, team_id, date)
    if g is None:
        raise SystemExit(f"{team} has no game on {date} in data/games.csv.")
    now = pd.Timestamp.now(tz="UTC")
    if g["completed"] or g["status"] != "Scheduled" or g["date"] <= now:
        raise SystemExit(f"{g['away_team']} at {g['home_team']} has already started; adjustments must be entered before kickoff.")
    log = load_log()
    if ((log["model"] == "overlay") & (log["game_id"] == g["game_id"])).any():
        raise SystemExit("The overlay pick for this game is already locked in. Use `adjust.py remove` first if it must change.")

    adj = load_adjustments()
    row = {"game_id": g["game_id"], "date": date.isoformat(), "team_id": team_id, "team": team,
           "points": float(args.points), "reason": args.reason, "source": args.source or "",
           "entered_at": now.isoformat(timespec="minutes")}
    save_adjustments(pd.concat([adj, pd.DataFrame([row])], ignore_index=True))
    print(f"Adjustment saved: {team} {float(args.points):+.1f} points for {g['away_team']} at {g['home_team']} "
          f"({g['date'].tz_convert(LOCAL_TZ):%a %b %d %H:%M} ET). Reason: {args.reason}\n")
    record(date, "overlay")          # lock in the overlay model's pick(s) for the adjusted game(s)


def list_(args):
    adj = load_adjustments()
    if args.date:
        adj = adj[adj["date"] == args.date]
    if adj.empty:
        print("No adjustments" + (f" on {args.date}" if args.date else "") + ".")
        return
    show = adj.reset_index().rename(columns={"index": "row"})
    pd.set_option("display.width", 200)
    print(show[["row", "date", "team", "points", "reason", "source", "entered_at"]].to_string(index=False))


def remove(args):
    adj = load_adjustments()
    if args.row not in adj.index:
        raise SystemExit(f"No adjustment row {args.row}. Run `adjust.py list`.")
    row = adj.loc[args.row]
    _, games = load_model("overlay")
    g = games[games["game_id"] == row["game_id"]].iloc[0]
    if g["completed"] or g["date"] <= pd.Timestamp.now(tz="UTC"):
        raise SystemExit("That game has started; the record stays as it was entered.")
    save_adjustments(adj.drop(index=args.row))
    log = load_log()
    mask = (log["model"] == "overlay") & (log["game_id"] == row["game_id"])
    if mask.any():
        save_log(log[~mask])
    print(f"Removed adjustment for {row['team']} ({row['points']:+.1f}) and {int(mask.sum())} logged overlay pick(s).")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("add", help="enter an adjustment and lock in the overlay pick")
    a.add_argument("--team", required=True)
    a.add_argument("--points", type=float, required=True, help="positive favours the team")
    a.add_argument("--reason", required=True)
    a.add_argument("--source", default="")
    a.add_argument("--date", help="YYYY-MM-DD in US/Eastern (default: today)")
    l = sub.add_parser("list", help="show adjustments")
    l.add_argument("--date")
    r = sub.add_parser("remove", help="remove an adjustment (before kickoff)")
    r.add_argument("--row", type=int, required=True)
    args = parser.parse_args()
    {"add": add, "list": list_, "remove": remove}[args.command](args)


if __name__ == "__main__":
    main()
