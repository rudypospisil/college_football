"""
Keep a historical record of each model's predictions and how they turned out.

Usage:
    python track.py record                 # lock in predictions for today's scheduled games
    python track.py record --date 2026-10-18 --model elo
    python track.py settle                 # after the games: fill in scores and grade every pick
    python track.py report                 # print the whole log plus a summary (per model)

The log is data/predictions.csv. A game is only recorded while it is still scheduled and
kickoff is in the future, a recorded prediction is never overwritten, and the Vegas line and
prices are captured at the same moment so the grading is fair. Several models can record
picks on the same games; each row carries the model's name and parameters.

Each settled game is graded three ways:
  * straight-up pick   - did the model's favourite win? (Vegas's favourite graded alongside)
  * spread bet         - the model takes whichever side its projected margin says covers the
                         recorded spread; win / loss / push, profit in units at the recorded
                         spread price (default -110)
  * moneyline bet      - one unit on the model's favourite at the recorded moneyline price
"""
import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from common import GAMES_CSV, LOCAL_TZ, LOG_PATH, load_games
from models import DEFAULT_MODEL, add_model_argument
from predict import load_model

DEFAULT_SPREAD_ODDS = -110      # used when ESPN did not publish a spread price
EDGE_THRESHOLD = 3.0            # "confident" spread bets: model and Vegas differ by >= 3 points

COLUMNS = [
    # recorded before kickoff
    "game_id", "model", "season", "week", "kickoff", "recorded_at", "away_team", "home_team", "neutral_site",
    "rating_away", "rating_home", "p_home", "model_home_margin", "adjustment", "model_pick", "model_pick_prob",
    "vegas_home_spread", "vegas_line", "vegas_pick",
    "home_moneyline", "away_moneyline", "home_spread_odds", "away_spread_odds",
    "spread_pick", "spread_edge", "model_params",
    # filled in by settle
    "settled", "home_score", "away_score", "actual_home_margin", "winner",
    "pick_correct", "vegas_pick_correct", "brier",
    "spread_result", "spread_profit", "moneyline_odds", "moneyline_result", "moneyline_profit",
]
LEGACY_COLUMNS = {"elo_away": "rating_away", "elo_home": "rating_home"}   # log files written before models/


# ----------------------------------------------------------------------------- helpers
def american_profit(odds):
    """Profit on a winning 1-unit bet at American odds: +136 -> 1.36, -162 -> 0.617."""
    if odds is None or pd.isna(odds):
        return None
    odds = float(odds)
    return odds / 100.0 if odds > 0 else 100.0 / abs(odds)


def isnum(x):
    return x is not None and not pd.isna(x)


def load_log():
    if LOG_PATH.exists():
        log = pd.read_csv(LOG_PATH, dtype={"game_id": str, "model_params": str}).rename(columns=LEGACY_COLUMNS)
        if "model" not in log.columns:
            log["model"] = DEFAULT_MODEL
        log["model"] = log["model"].fillna(DEFAULT_MODEL)
        for col in COLUMNS:
            if col not in log.columns:
                log[col] = np.nan
        return log[COLUMNS]
    return pd.DataFrame(columns=COLUMNS)


def save_log(log):
    LOG_PATH.parent.mkdir(exist_ok=True)
    log.sort_values(["kickoff", "game_id", "model"]).to_csv(LOG_PATH, index=False)


def local_date(series_utc):
    return series_utc.dt.tz_convert(LOCAL_TZ).dt.date


# ------------------------------------------------------------------------------ record
def picks(home_team, away_team, p_home, pred_margin, spread):
    """Turn a prediction plus the Vegas spread (home perspective) into the picks we grade:
    the model's favourite, Vegas's favourite, and the side the model takes against the spread."""
    out = {
        "model_pick": home_team if p_home >= 0.5 else away_team,
        "model_pick_prob": round(max(p_home, 1 - p_home), 4),
        "vegas_pick": None, "spread_pick": None, "spread_edge": None,
    }
    if spread is None or pd.isna(spread):
        return out
    if spread != 0:
        out["vegas_pick"] = home_team if spread < 0 else away_team
    value = pred_margin + spread          # >0: the model thinks the home side beats the spread
    out["spread_edge"] = round(abs(value), 1)
    if value > 0:
        out["spread_pick"] = home_team
    elif value < 0:
        out["spread_pick"] = away_team
    return out


def build_row(model, g, now):
    pred = model.predict(g.home_id, g.away_id, neutral=g.neutral_site)
    spread = g.odds_spread if isnum(g.odds_spread) else None
    chosen = picks(g.home_team, g.away_team, pred["p_home"], pred["pred_margin"], spread)

    return {
        "game_id": g.game_id, "model": model.name, "season": g.season, "week": g.week,
        "kickoff": g.date.isoformat(), "recorded_at": now.isoformat(timespec="minutes"),
        "away_team": g.away_team, "home_team": g.home_team, "neutral_site": g.neutral_site,
        "rating_away": round(pred["rating_away"], 1), "rating_home": round(pred["rating_home"], 1),
        "p_home": round(pred["p_home"], 4), "model_home_margin": round(pred["pred_margin"], 1),
        "adjustment": round(pred["adjustment"], 1) if pred.get("adjustment") else None,
        "model_pick": chosen["model_pick"], "model_pick_prob": chosen["model_pick_prob"],
        "vegas_home_spread": spread, "vegas_line": g.odds_details if isinstance(g.odds_details, str) else None,
        "vegas_pick": chosen["vegas_pick"],
        "home_moneyline": g.home_moneyline if isnum(g.home_moneyline) else None,
        "away_moneyline": g.away_moneyline if isnum(g.away_moneyline) else None,
        "home_spread_odds": g.home_spread_odds if isnum(g.home_spread_odds) else None,
        "away_spread_odds": g.away_spread_odds if isnum(g.away_spread_odds) else None,
        "spread_pick": chosen["spread_pick"], "spread_edge": chosen["spread_edge"],
        "model_params": json.dumps(model.params(), sort_keys=True),
        "settled": False,
    }


def record(date, model_name=DEFAULT_MODEL, now=None):
    now = now or pd.Timestamp.now(tz="UTC")
    age_hours = (dt.datetime.now().timestamp() - GAMES_CSV.stat().st_mtime) / 3600
    if age_hours > 6:
        print(f"Warning: data/games.csv is {age_hours:.0f} hours old; run fetch_data.py for fresh lines.\n")

    model, games = load_model(model_name)
    log = load_log()
    log, n_settled = settle_frame(log, games)
    if n_settled:
        print(f"Settled {n_settled} earlier game(s) first.\n")

    day = games[local_date(games["date"]) == date]
    future = day[(day["date"] > now) & (~day["completed"]) & (day["status"] == "Scheduled")]
    skipped_started = len(day) - len(future)
    future = future[[model.applies_to(gid) for gid in future["game_id"]]]   # the overlay: adjusted games only
    not_applicable = len(day) - skipped_started - len(future)
    already = future["game_id"].isin(log.loc[log["model"] == model.name, "game_id"])
    rows = [build_row(model, g, now) for g in future[~already].sort_values("date").itertuples(index=False)]

    if rows:
        new = pd.DataFrame(rows, columns=COLUMNS)
        # Object dtype for the merge: the log mixes numbers, text and blanks, and it is
        # re-read from CSV on every run anyway.
        log = new if log.empty else pd.concat([log.astype(object), new.astype(object)], ignore_index=True)
        save_log(log)
    print(f"{date} [{model.name}]: {len(day)} games on the schedule, {len(rows)} new predictions recorded, "
          f"{int(already.sum())} already in the log, {skipped_started} skipped (started or finished)"
          + (f", {not_applicable} without an adjustment." if not_applicable else "."))
    if rows:
        show = pd.DataFrame(rows)
        show["kick (ET)"] = pd.to_datetime(show["kickoff"]).dt.tz_convert(LOCAL_TZ).dt.strftime("%H:%M")
        show["matchup"] = show["away_team"] + " at " + show["home_team"]
        show["model pick"] = show["model_pick"] + " " + (show["model_pick_prob"] * 100).round().astype(int).astype(str) + "%"
        show["spread bet"] = show.apply(
            lambda r: f"{r['spread_pick']} (edge {r['spread_edge']:.1f})" if isinstance(r["spread_pick"], str) else "no line", axis=1)
        pd.set_option("display.width", 200)
        print(show[["kick (ET)", "matchup", "model pick", "model_home_margin", "vegas_line", "spread bet"]]
              .rename(columns={"model_home_margin": "model margin (home)", "vegas_line": "vegas"}).to_string(index=False))
    return log


# ------------------------------------------------------------------------------ settle
def grade(row, home_score, away_score):
    """Fill in the outcome columns for one recorded prediction."""
    margin = int(home_score) - int(away_score)
    home_won = margin > 0
    winner = row["home_team"] if home_won else row["away_team"]
    out = {
        "settled": True, "home_score": int(home_score), "away_score": int(away_score),
        "actual_home_margin": margin, "winner": winner,
        "pick_correct": row["model_pick"] == winner,
        "vegas_pick_correct": (row["vegas_pick"] == winner) if isinstance(row["vegas_pick"], str) else None,
        "brier": round((float(row["p_home"]) - (1.0 if home_won else 0.0)) ** 2, 4),
        "spread_result": None, "spread_profit": None,
        "moneyline_odds": None, "moneyline_result": None, "moneyline_profit": None,
    }

    if isinstance(row["spread_pick"], str) and isnum(row["vegas_home_spread"]):
        cover = margin + float(row["vegas_home_spread"])      # >0 home covers, <0 away covers
        if cover == 0:
            out["spread_result"], out["spread_profit"] = "push", 0.0
        else:
            bet_home = row["spread_pick"] == row["home_team"]
            won = (cover > 0) == bet_home
            price = row["home_spread_odds"] if bet_home else row["away_spread_odds"]
            price = price if isnum(price) else DEFAULT_SPREAD_ODDS
            out["spread_result"] = "win" if won else "loss"
            out["spread_profit"] = round(american_profit(price), 4) if won else -1.0

    ml = row["home_moneyline"] if row["model_pick"] == row["home_team"] else row["away_moneyline"]
    if isnum(ml):
        out["moneyline_odds"] = int(ml)
        out["moneyline_result"] = "win" if out["pick_correct"] else "loss"
        out["moneyline_profit"] = round(american_profit(ml), 4) if out["pick_correct"] else -1.0
    return out


def settle_frame(log, games):
    """Grade every unsettled row whose game has finished. Returns (log, number settled)."""
    if log.empty:
        return log, 0
    pending = log[log["settled"] != True]
    finished = games[games["completed"] & games["home_score"].notna()].set_index("game_id")
    outcome_cols = COLUMNS[COLUMNS.index("settled"):]
    for col in outcome_cols:                 # blank columns load as float NaN; allow text/bools
        if log[col].dtype != object:
            log[col] = log[col].astype(object)
    n = 0
    for idx, row in pending.iterrows():
        if row["game_id"] in finished.index:
            g = finished.loc[row["game_id"]]
            for col, val in grade(row, g["home_score"], g["away_score"]).items():
                log.at[idx, col] = val
            n += 1
    if n:
        save_log(log)
    return log, n


def settle():
    games = load_games()
    log = load_log()
    log, n = settle_frame(log, games)
    pending = int((log["settled"] != True).sum())
    print(f"Settled {n} game(s); {pending} still pending.")
    if pending:
        stale = log[(log["settled"] != True) & (pd.to_datetime(log["kickoff"]) < pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=5))]
        if len(stale):
            print(f"  {len(stale)} pending game(s) kicked off more than 5 hours ago: run fetch_data.py, then settle again.")
    summary(log)


# ------------------------------------------------------------------------------ report
def bet_line(frame, profit_col, result_col):
    bets = frame.dropna(subset=[result_col])
    wins = (bets[result_col] == "win").sum()
    losses = (bets[result_col] == "loss").sum()
    pushes = (bets[result_col] == "push").sum()
    profit = bets[profit_col].astype(float).sum()
    rate = wins / (wins + losses) if wins + losses else float("nan")
    roi = profit / len(bets) if len(bets) else float("nan")
    return f"{wins}-{losses}" + (f"-{pushes}" if pushes else "") + f" ({rate:.1%}), {profit:+.2f} units on {len(bets)} bets, ROI {roi:+.1%}"


def summary_block(log):
    done = log[log["settled"] == True]
    print(f"Record: {len(log)} predictions logged, {len(done)} settled, {len(log) - len(done)} pending.")
    if done.empty:
        return
    su = done["pick_correct"].astype(bool)
    print(f"Straight up, model:  {su.sum()}-{(~su).sum()} ({su.mean():.1%}), Brier {done['brier'].astype(float).mean():.3f}")
    vg = done.dropna(subset=["vegas_pick_correct"])
    if len(vg):
        vc = vg["vegas_pick_correct"].astype(bool)
        print(f"Straight up, Vegas:  {vc.sum()}-{(~vc).sum()} ({vc.mean():.1%}) on the {len(vg)} games with a line")
    if done["spread_result"].notna().any():
        print(f"Against the spread:  {bet_line(done, 'spread_profit', 'spread_result')}")
        strong = done[done["spread_edge"].astype(float) >= EDGE_THRESHOLD]
        if len(strong):
            print(f"  edge >= {EDGE_THRESHOLD:.0f} points: {bet_line(strong, 'spread_profit', 'spread_result')}")
    if done["moneyline_result"].notna().any():
        print(f"Moneyline bets:      {bet_line(done, 'moneyline_profit', 'moneyline_result')}")


def summary(log):
    """One summary block per model (a single block when only one model has recorded picks)."""
    models = sorted(log["model"].dropna().unique()) if len(log) else []
    if len(models) <= 1:
        print()
        summary_block(log)
        return
    for name in models:
        print(f"\n--- {name} ---")
        summary_block(log[log["model"] == name])


def overlay_report(log):
    """The injury overlay against its base model on exactly the games that were adjusted."""
    ov = log[(log["model"] == "overlay") & log["adjustment"].astype(float).fillna(0).ne(0)]
    if ov.empty:
        return
    base_name = json.loads(ov.iloc[0]["model_params"]).get("base", "prior")
    base = log[(log["model"] == base_name) & log["game_id"].isin(ov["game_id"])]
    m = ov.merge(base, on="game_id", suffixes=("_ov", "_base"))
    if m.empty:
        return
    print(f"\nInjury overlay vs {base_name} on the {len(m)} adjusted game(s):")
    show = pd.DataFrame({
        "date (ET)": pd.to_datetime(m["kickoff_ov"]).dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d"),
        "matchup": m["away_team_ov"] + " at " + m["home_team_ov"],
        "adj (home)": m["adjustment_ov"].astype(float).round(1),
        f"{base_name} margin": m["model_home_margin_base"].astype(float).round(1),
        "overlay margin": m["model_home_margin_ov"].astype(float).round(1),
        "vegas": m["vegas_line_ov"],
        "score": m.apply(lambda r: f"{int(r['away_score_ov'])}-{int(r['home_score_ov'])}" if r["settled_ov"] == True else "pending", axis=1),
        f"{base_name} ATS": m["spread_result_base"].fillna("").str.upper(),
        "overlay ATS": m["spread_result_ov"].fillna("").str.upper(),
    })
    print(show.to_string(index=False))
    done = m[m["settled_ov"] == True]
    if len(done):
        for label, suf in ((base_name, "_base"), ("overlay", "_ov")):
            su = done[f"pick_correct{suf}"].astype(bool)
            ats = done[f"spread_result{suf}"]
            print(f"  {label:8s} straight up {su.sum()}-{(~su).sum()}, against the spread {(ats == 'win').sum()}-{(ats == 'loss').sum()}"
                  + (f"-{(ats == 'push').sum()}" if (ats == 'push').any() else "")
                  + f", Brier {done[f'brier{suf}'].astype(float).mean():.3f}")


def report():
    log = load_log()
    if log.empty:
        print("No predictions recorded yet. Run: python track.py record")
        return
    show = log.copy()
    show["date (ET)"] = pd.to_datetime(show["kickoff"]).dt.tz_convert(LOCAL_TZ).dt.strftime("%Y-%m-%d")
    show["matchup"] = show["away_team"] + " at " + show["home_team"]
    show["model pick"] = show["model_pick"] + " " + (show["model_pick_prob"].astype(float) * 100).round().astype(int).astype(str) + "%"
    show["spread bet"] = show.apply(
        lambda r: f"{r['spread_pick']} (edge {float(r['spread_edge']):.1f})" if isinstance(r["spread_pick"], str) else "no line", axis=1)
    show["score"] = show.apply(
        lambda r: f"{int(r['away_score'])}-{int(r['home_score'])}" if r["settled"] == True else "pending", axis=1)
    show["pick"] = show["pick_correct"].map({True: "WIN", False: "LOSS"}).fillna("")
    show["ATS"] = show["spread_result"].fillna("").str.upper()
    show["ML"] = show["moneyline_result"].fillna("").str.upper()
    pd.set_option("display.width", 220)
    cols = ["date (ET)", "matchup", "model pick", "vegas_line", "spread bet", "score", "pick", "ATS", "ML"]
    if show["model"].nunique() > 1:
        cols.insert(1, "model")
    print(show.sort_values(["kickoff", "game_id", "model"])[cols].rename(columns={"vegas_line": "vegas"}).to_string(index=False))
    summary(log)
    overlay_report(log)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    rec = sub.add_parser("record", help="log predictions for scheduled games")
    rec.add_argument("--date", help="YYYY-MM-DD in US/Eastern (default: today)")
    add_model_argument(rec)
    sub.add_parser("settle", help="grade logged games that have finished")
    sub.add_parser("report", help="print the log and summary")
    args = parser.parse_args()

    if args.command == "record":
        date = dt.date.fromisoformat(args.date) if args.date else pd.Timestamp.now(tz=LOCAL_TZ).date()
        record(date, args.model)
    elif args.command == "settle":
        settle()
    elif args.command == "report":
        report()


if __name__ == "__main__":
    main()
