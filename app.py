"""
Menu-driven front end for the college football Elo model.

    python app.py          # interactive menu
    python app.py 4        # run one menu option and exit (handy for scripts)

Uses only the standard library itself, then runs the project's scripts with the
virtual-environment interpreter in .venv, so it can be started with any Python.
"""
import csv
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_PY = ROOT / ".venv" / "bin" / "python"
PY = str(VENV_PY) if VENV_PY.exists() else sys.executable
DATA = ROOT / "data"
GAMES = DATA / "games.csv"
PARAMS = DATA / "model_params.json"
LOG = DATA / "predictions.csv"


# ------------------------------------------------------------------ small helpers
def ask(prompt, default=None):
    """input() with a default; EOF (piped input ran out, Ctrl-D) quits cleanly."""
    suffix = f" [{default}]" if default else ""
    try:
        answer = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        print()
        raise SystemExit(0)
    return answer or (default or "")


def pause():
    try:
        input("\nPress Enter to return to the menu...")
    except EOFError:
        raise SystemExit(0)


def age_text(path):
    seconds = dt.datetime.now().timestamp() - path.stat().st_mtime
    if seconds < 3600:
        return f"{seconds / 60:.0f} min ago"
    if seconds < 86400:
        return f"{seconds / 3600:.0f} h ago"
    return f"{seconds / 86400:.0f} d ago"


def run(script, *args):
    """Run one of the project scripts, echoing the equivalent command line first."""
    shown = " ".join(f'"{a}"' if " " in a else a for a in args)
    print(f"\n> python {script} {shown}".rstrip())
    print("-" * 72, flush=True)
    result = subprocess.run([PY, str(ROOT / script), *args], cwd=ROOT)
    if result.returncode != 0:
        print(f"\n(command exited with code {result.returncode})")
    return result.returncode == 0


def today():
    return dt.date.today().isoformat()


def need_data():
    if GAMES.exists():
        return True
    print("\nNo data yet. Run 'Refresh data' first.")
    return False


# ------------------------------------------------------------------ status header
def status_lines():
    lines = []
    if GAMES.exists():
        n, last = 0, ""
        with GAMES.open(newline="") as f:
            for row in csv.DictReader(f):
                n += 1
                if row["completed"] == "True" and row["date"] > last:
                    last = row["date"]
        if last:   # ESPN timestamps are UTC; show the local date
            last = dt.datetime.strptime(last[:16], "%Y-%m-%dT%H:%M").replace(tzinfo=dt.timezone.utc).astimezone().date().isoformat()
        lines.append(f"Data : {n:,} games, last result {last}, refreshed {age_text(GAMES)}")
    else:
        lines.append("Data : none yet (run 'Refresh data')")
    if PARAMS.exists():
        p = json.loads(PARAMS.read_text())
        lines.append(f"Model: K={p.get('k'):g}  home field={p.get('hfa'):g}  regress={p.get('regress'):g}  "
                     f"{p.get('elo_per_point')} Elo per point")
    else:
        lines.append("Model: default parameters (run 'Backtest' to tune)")
    if LOG.exists():
        with LOG.open(newline="") as f:
            rows = list(csv.DictReader(f))
        settled = sum(r["settled"] == "True" for r in rows)
        lines.append(f"Log  : {len(rows)} predictions, {settled} settled, {len(rows) - settled} pending")
    else:
        lines.append("Log  : no predictions recorded yet")
    return lines


# ------------------------------------------------------------------ menu actions
def game_day():
    """The usual Saturday-morning routine in one go."""
    date = ask("Date (YYYY-MM-DD)", today())
    ok = run("fetch_data.py")
    ok = ok and run("track.py", "record", "--date", date)
    ok and run("predict.py", "--date", date)


def refresh():
    everything = ask("Re-download every season instead of just the current one? (y/N)", "n").lower().startswith("y")
    run("fetch_data.py", *(["--refresh"] if everything else []))


def matchup():
    if not need_data():
        return
    away = ask("Away team (e.g. Ohio State)")
    home = ask("Home team (e.g. Indiana)")
    if not away or not home:
        print("Need both teams.")
        return
    neutral = ask("Neutral site? (y/N)", "n").lower().startswith("y")
    run("predict.py", away, home, *(["--neutral"] if neutral else []))


def slate():
    if need_data():
        run("predict.py", "--date", ask("Date (YYYY-MM-DD)", today()))


def rankings():
    if need_data():
        run("predict.py", "--rankings")


def record():
    if need_data():
        run("track.py", "record", "--date", ask("Date (YYYY-MM-DD)", today()))


def settle():
    if need_data():
        run("track.py", "settle")


def report():
    run("track.py", "report")


def backtest():
    if need_data():
        print("Grid search over 160 parameter sets, about 10 seconds. Saves data/model_params.json.")
        run("backtest.py")


MENU = [
    ("1", "Game-day update", "refresh data, record today's picks (settling old ones), show the slate", game_day),
    ("2", "Refresh data", "download this season's results and betting lines from ESPN", refresh),
    ("3", "Predict a matchup", "enter two teams", matchup),
    ("4", "Slate for a date", "every game that day: model pick and spread next to Vegas", slate),
    ("5", "Top 25", "current Elo rankings", rankings),
    ("6", "Record predictions", "lock in picks for a date before kickoff", record),
    ("7", "Settle results", "grade finished games in the prediction log", settle),
    ("8", "Prediction record", "full history of picks, outcomes and bet results", report),
    ("9", "Backtest", "re-tune parameters and report out-of-sample accuracy", backtest),
]
ACTIONS = {key: fn for key, _, _, fn in MENU}


def show_menu():
    print("\n" + "=" * 72)
    print("  College Football Elo Model")
    print("=" * 72)
    for line in status_lines():
        print(f"  {line}")
    print()
    for key, title, desc, _ in MENU:
        print(f"  {key}  {title:<20} {desc}")
    print(f"  {'q':<3}{'Quit':<20}")
    print()


def main():
    os.chdir(ROOT)
    if len(sys.argv) > 1:                       # one-shot: python app.py 4
        choice = sys.argv[1]
        if choice not in ACTIONS:
            print(f"Unknown option '{choice}'. Options: {', '.join(ACTIONS)}")
            return 1
        ACTIONS[choice]()
        return 0
    while True:
        show_menu()
        choice = ask("Choose an option").lower()
        if choice in ("q", "quit", "exit", "0"):
            print("Bye.")
            return 0
        if choice not in ACTIONS:
            print(f"'{choice}' is not an option.")
            continue
        try:
            ACTIONS[choice]()
        except KeyboardInterrupt:
            print("\n(cancelled)")
        pause()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nBye.")
