"""
Download season-level datasets from CollegeFootballData.com (CFBD) and cache them as CSV.

Needs a free API key (register at collegefootballdata.com) in the CFBD_API_KEY environment
variable or in a .env file next to this script:  CFBD_API_KEY=...   (.env is git-ignored).
The free tier allows 1,000 calls a month, so everything here is one call per season per
dataset and cached files are never re-downloaded unless asked.

Usage:
    python fetch_cfbd.py                       # all datasets, 2010..current season
    python fetch_cfbd.py --seasons 2026        # one season (re-downloads it: current data changes weekly)
    python fetch_cfbd.py --datasets games ppa  # a subset
    python fetch_cfbd.py --map                 # only rebuild the ESPN <-> CFBD team id map

Output in data/cfbd/:
    games_<year>.csv      every game with CFBD team ids, CFBD's own pregame Elo, attendance, venue
    ppa_<year>.csv        per team per game: predicted points added (offense/defense, by down)
    lines_<year>.csv      betting lines, one row per game per sportsbook (2013+)
    recruiting_<year>.csv team recruiting class rank and composite points
    returning_<year>.csv  returning production (share of last season's PPA coming back), 2014+
    team_map.csv          espn_id <-> cfbd_id, derived by matching games on date and score
"""
import argparse
import re
import time

import pandas as pd
import requests

from common import CFBD_DIR, GAMES_CSV, cfbd_api_key, current_season, write_csv

BASE_URL = "https://api.collegefootballdata.com"
FIRST_SEASON = 2010
DATASETS = {
    #  name        endpoint              params                      first season
    "games":      ("/games",             {"seasonType": "both"},     2010),
    "ppa":        ("/ppa/games",         {"seasonType": "both"},     2010),
    "lines":      ("/lines",             {"seasonType": "both"},     2013),
    "recruiting": ("/recruiting/teams",  {},                         2010),
    "returning":  ("/player/returning",  {},                         2014),
}


def snake(name):
    """camelCase -> snake_case, keeping acronyms together: percentPPA -> percent_ppa."""
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).lower()


def flatten(record, prefix=""):
    """Flatten nested dicts (ppa offense/defense) into prefixed columns; leave lists as JSON-ish text."""
    out = {}
    for k, v in record.items():
        key = f"{prefix}{snake(k)}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "_"))
        elif isinstance(v, list):
            out[key] = str(v) if v else None
        else:
            out[key] = v
    return out


class Client:
    def __init__(self):
        key = cfbd_api_key()
        if not key:
            raise SystemExit("No CFBD API key. Put CFBD_API_KEY=... in .env or the environment.")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Accept": "application/json"})
        self.remaining = None

    def get(self, path, **params):
        for attempt in range(3):
            resp = self.session.get(BASE_URL + path, params=params, timeout=60)
            if resp.status_code == 429:
                time.sleep(10 * (attempt + 1))
                continue
            if resp.status_code == 401:
                raise SystemExit(f"{path}: {resp.json().get('message', 'unauthorized')}")
            resp.raise_for_status()
            self.remaining = resp.headers.get("X-CallLimit-Remaining")
            return resp.json()
        raise RuntimeError(f"{path}: rate limited three times")


def fetch_dataset(client, name, season, refresh=False):
    path, params, first = DATASETS[name]
    if season < first:
        return None
    out = CFBD_DIR / f"{name}_{season}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out)
    records = client.get(path, year=season, **params)
    rows = []
    for rec in records:
        if name == "lines":                      # one row per game per sportsbook
            base = flatten({k: v for k, v in rec.items() if k != "lines"})
            for line in rec.get("lines") or []:
                rows.append({**base, **flatten(line)})
        else:
            rows.append(flatten(rec))
    df = pd.DataFrame(rows)
    write_csv(df, out)
    print(f"  {name:<10} {season}: {len(df):5d} rows  (calls left this month: {client.remaining})")
    return df


def build_team_map():
    """Match CFBD games to ESPN games on (UTC date, home points, away points) and vote on the
    team-id pairing. Needs data/games.csv and data/cfbd/games_<year>.csv."""
    espn = pd.read_csv(GAMES_CSV, dtype={"game_id": str, "home_id": str, "away_id": str})
    espn = espn[espn["completed"]].copy()
    espn["day"] = pd.to_datetime(espn["date"], utc=True).dt.strftime("%Y-%m-%d")
    frames = [pd.read_csv(p) for p in sorted(CFBD_DIR.glob("games_*.csv"))]
    cfbd = pd.concat(frames, ignore_index=True)
    cfbd = cfbd[cfbd["completed"] & cfbd["home_points"].notna()].copy()
    cfbd["day"] = pd.to_datetime(cfbd["start_date"], utc=True).dt.strftime("%Y-%m-%d")

    keys = ["day", "home_points", "away_points"]
    e = espn.rename(columns={"home_score": "home_points", "away_score": "away_points"})
    e = e.drop_duplicates(keys, keep=False)          # ambiguous (same day, same score) pairs are useless
    c = cfbd.drop_duplicates(keys, keep=False)
    m = c.merge(e[keys + ["home_id", "away_id", "home_team", "away_team"]], on=keys, how="inner")
    votes = pd.concat([
        m[["home_id_x", "home_id_y", "home_team_x", "home_team_y"]].set_axis(["cfbd_id", "espn_id", "cfbd_team", "espn_team"], axis=1),
        m[["away_id_x", "away_id_y", "away_team_x", "away_team_y"]].set_axis(["cfbd_id", "espn_id", "cfbd_team", "espn_team"], axis=1),
    ])
    tally = votes.groupby(["cfbd_id", "espn_id", "cfbd_team", "espn_team"]).size().reset_index(name="matches")
    best = tally.sort_values("matches", ascending=False).drop_duplicates("cfbd_id")
    best = best[best["matches"] >= 3]
    conflicts = best[best.duplicated("espn_id", keep=False)]
    if len(conflicts):
        print("  warning: ESPN ids claimed by more than one CFBD team:\n", conflicts.to_string(index=False))
    best = best.sort_values("espn_team")
    best["cfbd_id"] = best["cfbd_id"].astype(int)
    write_csv(best[["espn_id", "cfbd_id", "espn_team", "cfbd_team", "matches"]], CFBD_DIR / "team_map.csv")
    fbs_cfbd = set(cfbd.loc[cfbd["home_classification"].eq("fbs"), "home_id"]) | set(cfbd.loc[cfbd["away_classification"].eq("fbs"), "away_id"])
    print(f"  team_map.csv: {len(best)} teams mapped; {len(fbs_cfbd - set(best['cfbd_id']))} CFBD FBS ids unmapped")
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="*", help="seasons to (re)download; default 2010..current, cached ones skipped")
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS), choices=list(DATASETS))
    parser.add_argument("--map", action="store_true", help="only rebuild team_map.csv")
    args = parser.parse_args()
    CFBD_DIR.mkdir(parents=True, exist_ok=True)

    if not args.map:
        client = Client()
        explicit = bool(args.seasons)
        seasons = args.seasons or list(range(FIRST_SEASON, current_season() + 1))
        for season in seasons:
            for name in args.datasets:
                fetch_dataset(client, name, season, refresh=explicit)
                time.sleep(0.2)
        print(f"CFBD calls left this month: {client.remaining}")
    build_team_map()


if __name__ == "__main__":
    main()
