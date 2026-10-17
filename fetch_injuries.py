"""
Snapshot ESPN's college football injury feed.

College football has no official injury report. ESPN's site API carries a current list of
reported injuries and game-day availability per team, sourced from beat reporters and the
conference availability reports. It has no history, so this script appends today's snapshot to
data/injuries/injuries.csv; run it each game week (the game-day menu option does) and a usable
record accumulates: which players were out, for which game, at which position.

Usage:
    python fetch_injuries.py
"""
import datetime as dt

import pandas as pd
import requests

from common import DATA_DIR, write_csv

URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/injuries"
OUT = DATA_DIR / "injuries" / "injuries.csv"


def fetch():
    data = requests.get(URL, timeout=30, headers={"User-Agent": "cfb-elo-model/1.0"}).json()
    today = dt.date.today().isoformat()
    rows = []
    for team in data.get("injuries", []):
        for inj in team.get("injuries", []):
            athlete = inj.get("athlete") or {}
            details = inj.get("details") or {}
            rows.append({
                "snapshot": today,
                "team_id": team.get("id"), "team": team.get("displayName"),
                "player": athlete.get("displayName"), "position": (athlete.get("position") or {}).get("abbreviation"),
                "status": inj.get("status"), "type": details.get("type"), "detail": details.get("detail"),
                "return_date": details.get("returnDate"), "date": inj.get("date"),
                "comment": inj.get("shortComment"),
            })
    return pd.DataFrame(rows)


def main():
    new = fetch()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        old = pd.read_csv(OUT, dtype=str)
        combined = pd.concat([old, new.astype(str)], ignore_index=True).drop_duplicates(["snapshot", "team_id", "player", "status"])
    else:
        combined = new
    write_csv(combined, OUT)
    print(f"{len(new)} injury entries for {new['team'].nunique() if len(new) else 0} teams today; "
          f"{len(combined)} rows in {OUT}")
    if len(new):
        print(new[["team", "player", "position", "status", "type", "comment"]].head(12).to_string(index=False))


if __name__ == "__main__":
    main()
