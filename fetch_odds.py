"""
Back-fill betting lines for finished games from ESPN's per-game odds endpoint.

The weekly scoreboard only carries odds for the current week, so past games in
data/games.csv have none. This fills them in (closing spread, moneylines and spread prices)
one request per game, which is what simulate.py needs to grade hypothetical bets.

Usage:
    python fetch_odds.py --seasons 2026              # every finished 2026 game without a line
    python fetch_odds.py --seasons 2026 --weeks 6
    python fetch_odds.py --seasons 2024 2025         # several minutes per season
    python fetch_odds.py --seasons 2026 --refresh    # replace lines already stored

Progress is saved every 50 games, so it is safe to interrupt and rerun.
"""
import argparse
import time

import pandas as pd
import requests

from fetch_data import SEASON_DIR, current_season, rebuild_games, write_csv

ODDS_URL = ("https://sports.core.api.espn.com/v2/sports/football/leagues/college-football/"
            "events/{id}/competitions/{id}/odds")
# Prefer the book the weekly scoreboard uses (DraftKings), then ESPN BET, then anything else.
PROVIDER_PRIORITY = {"100": 0, "58": 1, "1004": 2}
ODDS_COLUMNS = ["odds_details", "odds_spread", "over_under",
                "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds"]

session = requests.Session()
session.headers["User-Agent"] = "cfb-elo-model/1.0 (personal research project)"


def american(block):
    """ESPN wraps prices as {'american': '-112', ...}; return the number, or None."""
    raw = (block or {}).get("american") if isinstance(block, dict) else block
    if raw in (None, "", "OFF"):
        return None
    if str(raw).upper() in ("EVEN", "EV"):
        return 100.0
    try:
        return float(str(raw).replace("+", ""))
    except ValueError:
        return None


def side_values(side):
    """Closing (else current, else top-level) spread line, spread price and moneyline for one team."""
    for key in ("close", "current"):
        block = side.get(key) or {}
        if block:
            vals = (american(block.get("pointSpread")), american(block.get("spread")), american(block.get("moneyLine")))
            if any(v is not None for v in vals):
                return vals
    return None, side.get("spreadOdds"), side.get("moneyLine")


def choose_provider(items):
    usable = [i for i in items if "live" not in (i.get("provider", {}).get("name") or "").lower()
              and (i.get("spread") is not None or i.get("details"))]
    usable.sort(key=lambda i: PROVIDER_PRIORITY.get(str(i.get("provider", {}).get("id")), 9))
    return usable[0] if usable else None


def parse_odds(item):
    home_line, home_price, home_ml = side_values(item.get("homeTeamOdds") or {})
    _, away_price, away_ml = side_values(item.get("awayTeamOdds") or {})
    spread = home_line if home_line is not None else item.get("spread")   # home perspective
    return {
        "odds_details": item.get("details"), "odds_spread": spread, "over_under": item.get("overUnder"),
        "home_moneyline": home_ml, "away_moneyline": away_ml,
        "home_spread_odds": home_price, "away_spread_odds": away_price,
    }


def fetch_game_odds(game_id, retries=3):
    for attempt in range(retries):
        try:
            resp = session.get(ODDS_URL.format(id=game_id), timeout=20)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return choose_provider(resp.json().get("items", []))
        except (requests.RequestException, ValueError) as exc:
            if attempt == retries - 1:
                print(f"  giving up on {game_id}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))


def backfill_season(season, weeks=None, refresh=False, pause=0.15):
    path = SEASON_DIR / f"{season}.csv"
    if not path.exists():
        print(f"Season {season}: no data (run fetch_data.py first)")
        return
    df = pd.read_csv(path, dtype={"game_id": str, "home_id": str, "away_id": str})
    if "odds_checked" not in df.columns:
        df["odds_checked"] = False
    df["odds_checked"] = df["odds_checked"].fillna(False).astype(bool)

    todo = df["completed"] & (df["odds_spread"].isna() if not refresh else True)
    if not refresh:
        todo &= ~df["odds_checked"]
    if weeks:
        todo &= df["week"].isin(weeks)
    idx = df.index[todo].tolist()
    print(f"Season {season}: {len(idx)} finished games to look up")

    filled = 0
    for n, i in enumerate(idx, 1):
        item = fetch_game_odds(df.at[i, "game_id"])
        df.at[i, "odds_checked"] = True
        if item:
            for col, val in parse_odds(item).items():
                df.at[i, col] = val
            filled += 1
        if n % 25 == 0 or n == len(idx):
            print(f"  {n}/{len(idx)} looked up, {filled} lines found")
        if n % 50 == 0:
            write_csv(df, path)
        time.sleep(pause)
    write_csv(df, path)
    return filled


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="*", default=[current_season()])
    parser.add_argument("--weeks", type=int, nargs="*")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    for season in args.seasons:
        backfill_season(season, args.weeks, args.refresh)
    games = rebuild_games()
    with_lines = games["odds_spread"].notna() & games["completed"]
    print(f"\ngames.csv rebuilt: {int(with_lines.sum())} finished games now have a line "
          f"({', '.join(f'{int(s)}: {int(c)}' for s, c in games[with_lines].groupby('season').size().items())})")


if __name__ == "__main__":
    main()
