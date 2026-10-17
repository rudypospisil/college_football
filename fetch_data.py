"""
Download college football (FBS) game results from ESPN's public scoreboard API
and cache them as CSV files.

Usage:
    python fetch_data.py                 # fetch 2010..current season (skips cached seasons)
    python fetch_data.py --seasons 2026  # refresh just one season
    python fetch_data.py --refresh       # re-download everything

Output:
    data/seasons/<year>.csv   one file per season
    data/games.csv            all seasons combined, sorted by date
"""
import argparse
import time

import pandas as pd
import requests

from common import GAMES_CSV, SEASON_DIR, current_season, rebuild_games, write_csv

BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/football/college-football/scoreboard"
FBS_GROUP = 80           # ESPN group id for FBS
FIRST_SEASON = 2010

COLUMNS = [
    "game_id", "season", "season_type", "week", "date", "neutral_site", "conference_game",
    "home_id", "home_team", "home_score", "away_id", "away_team", "away_score",
    "completed", "status", "odds_details", "odds_spread", "over_under",
    "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds",
]

session = requests.Session()
session.headers["User-Agent"] = "cfb-elo-model/1.0 (personal research project)"


def get_json(params, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(BASE_URL, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))
            print(f"  retry {attempt + 1} after error: {exc}")


def parse_event(event):
    comp = event["competitions"][0]
    teams = {c["homeAway"]: c for c in comp["competitors"]}
    home, away = teams["home"], teams["away"]
    status = event["status"]["type"]
    odds = (comp.get("odds") or [{}])[0]

    def score(team):
        s = team.get("score")
        return int(s) if s not in (None, "") else None

    def price(market, side):
        """American odds (e.g. '+136' -> 136, '-162' -> -162) for one side of a betting
        market, preferring the latest ('close') price over the opening one."""
        block = (odds.get(market) or {}).get(side) or {}
        raw = (block.get("close") or {}).get("odds") or (block.get("open") or {}).get("odds")
        if raw in (None, "", "OFF"):
            return None
        if str(raw).upper() in ("EVEN", "EV"):
            return 100
        try:
            return int(str(raw).replace("+", ""))
        except ValueError:
            return None

    return {
        "game_id": event["id"],
        "season": event["season"]["year"],
        "season_type": event["season"]["type"],   # 2 = regular season, 3 = postseason
        "week": event.get("week", {}).get("number"),
        "date": event["date"],
        "neutral_site": bool(comp.get("neutralSite", False)),
        "conference_game": bool(comp.get("conferenceCompetition", False)),
        "home_id": home["team"]["id"],
        "home_team": home["team"]["displayName"],
        "home_score": score(home),
        "away_id": away["team"]["id"],
        "away_team": away["team"]["displayName"],
        "away_score": score(away),
        "completed": bool(status.get("completed", False)),
        "status": status.get("description"),
        "odds_details": odds.get("details"),
        "odds_spread": odds.get("spread"),
        "over_under": odds.get("overUnder"),
        "home_moneyline": price("moneyline", "home"),
        "away_moneyline": price("moneyline", "away"),
        "home_spread_odds": price("pointSpread", "home"),
        "away_spread_odds": price("pointSpread", "away"),
    }


def fetch_week(season, season_type, week):
    data = get_json({
        "dates": season, "seasontype": season_type, "week": week,
        "groups": FBS_GROUP, "limit": 400,
    })
    rows = []
    for event in data.get("events", []):
        try:
            rows.append(parse_event(event))
        except (KeyError, IndexError, TypeError) as exc:
            # Occasionally ESPN returns a stub event with no competition data; skip it.
            print(f"  skipping event {event.get('id')} ({event.get('name', '?')}): missing {exc}")
    return data, rows


def fetch_season(season, pause=0.25):
    print(f"Season {season}")
    # Week 1 of the regular season also tells us which weeks exist this season.
    data, rows = fetch_week(season, 2, 1)
    weeks = {2: ["1"], 3: ["1"]}
    for cal in data.get("leagues", [{}])[0].get("calendar", []):
        stype = int(cal.get("value", 0))
        if stype in (2, 3):
            weeks[stype] = [e["value"] for e in cal.get("entries", [])] or ["1"]
    # ESPN uses '999' as a catch-all postseason bucket; query a few numbered weeks instead.
    weeks[3] = [w for w in weeks[3] if w != "999"] or ["1"]
    weeks[3] = sorted(set(weeks[3] + ["1", "2", "3", "4", "5"]), key=int)

    for season_type in (2, 3):
        for week in weeks[season_type]:
            if season_type == 2 and week == "1":
                continue  # already fetched
            time.sleep(pause)
            _, more = fetch_week(season, season_type, week)
            rows.extend(more)
            print(f"  type {season_type} week {week:>3}: {len(more):3d} games")

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.drop_duplicates("game_id").sort_values(["date", "game_id"]).reset_index(drop=True)
    print(f"  -> {len(df)} games, {int(df['completed'].sum())} completed")
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="*", help="specific seasons to fetch")
    parser.add_argument("--refresh", action="store_true", help="re-download seasons already cached")
    args = parser.parse_args()

    this_season = current_season()
    seasons = args.seasons or list(range(FIRST_SEASON, this_season + 1))
    SEASON_DIR.mkdir(parents=True, exist_ok=True)

    for season in seasons:
        path = SEASON_DIR / f"{season}.csv"
        # Always refresh the current season (new results every week); reuse cached past seasons.
        if path.exists() and not args.refresh and season != this_season:
            print(f"Season {season}: cached ({path})")
            continue
        write_csv(fetch_season(season), path)

    games = rebuild_games()
    print(f"\nWrote {len(games)} games across {games['season'].nunique()} seasons to {GAMES_CSV}")


if __name__ == "__main__":
    main()
