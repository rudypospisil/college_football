"""
Menu-driven front end for the college football models.

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
PARAMS_DIR = DATA / "params"
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


def current_season():
    now = dt.date.today()
    return now.year if now.month >= 8 else now.year - 1


def need_data():
    if GAMES.exists():
        return True
    print("\nNo data yet. Run 'Refresh data' first.")
    return False


_MODELS = None


def model_names():
    """Registered models, asked of the models package (which needs the venv interpreter)."""
    global _MODELS
    if _MODELS is None:
        try:
            out = subprocess.run([PY, "-c", "from models import MODELS, DEFAULT_MODEL; "
                                  "print(DEFAULT_MODEL); print(' '.join(sorted(MODELS)))"],
                                 cwd=ROOT, capture_output=True, text=True, timeout=60).stdout.split("\n")
            default, names = out[0].strip(), out[1].split()
            _MODELS = [default] + [n for n in names if n != default]
        except (subprocess.SubprocessError, IndexError, OSError):
            _MODELS = ["elo"]
    return _MODELS


def choose_model():
    """Ask which model to use, but only when more than one is registered."""
    names = model_names()
    if len(names) == 1:
        return names[0]
    choice = ask(f"Model ({', '.join(names)})", names[0])
    return choice if choice in names else names[0]


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
    tuned = sorted(p.stem for p in PARAMS_DIR.glob("*.json")) if PARAMS_DIR.exists() else []
    if tuned:
        parts = []
        for name in tuned:
            p = json.loads((PARAMS_DIR / f"{name}.json").read_text())
            parts.append(f"{name} (" + ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else f"{k}={v}"
                                                   for k, v in p.items()) + ")")
        lines.append("Model: " + "; ".join(parts))
    else:
        lines.append("Model: default parameters (run 'Backtest' to tune)")
    if LOG.exists():
        with LOG.open(newline="") as f:
            rows = list(csv.DictReader(f))
        settled = sum(r["settled"] == "True" for r in rows)
        models = sorted({r.get("model") or "elo" for r in rows})
        lines.append(f"Log  : {len(rows)} predictions, {settled} settled, {len(rows) - settled} pending"
                     + (f" ({', '.join(models)})" if len(models) > 1 else ""))
    else:
        lines.append("Log  : no predictions recorded yet")
    return lines


# ------------------------------------------------------------------ menu actions
def game_day():
    """The usual Saturday-morning routine in one go: every registered model records its picks."""
    date = ask("Date (YYYY-MM-DD)", today())
    ok = run("fetch_data.py")
    ok = ok and run("fetch_odds.py", "--seasons", str(current_season()))
    for name in model_names():
        ok = ok and run("track.py", "record", "--date", date, "--model", name)
    ok and run("predict.py", "--date", date, "--model", model_names()[0])


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
    run("predict.py", away, home, "--model", choose_model(), *(["--neutral"] if neutral else []))


def slate():
    if need_data():
        run("predict.py", "--date", ask("Date (YYYY-MM-DD)", today()), "--model", choose_model())


def rankings():
    if need_data():
        run("predict.py", "--rankings", "--model", choose_model())


def record():
    if need_data():
        run("track.py", "record", "--date", ask("Date (YYYY-MM-DD)", today()), "--model", choose_model())


def settle():
    if need_data():
        run("track.py", "settle")


def report():
    run("track.py", "report")


def backtest():
    if need_data():
        model = choose_model()
        print(f"Grid search for '{model}', usually under a minute. Saves data/params/{model}.json.")
        run("backtest.py", "--model", model)


def what_if():
    """Replay finished games with lines and grade flat bets on a model's picks."""
    if not need_data():
        return
    model = choose_model()
    season = ask("Season", str(current_season()))
    week = ask("Week number (blank = whole season)")
    stake = ask("Stake per bet in dollars", "100")
    args = ["--model", model, "--season", season, "--stake", stake]
    if week:
        args += ["--week", week, "--details"]
    run("simulate.py", *args)


def backfill_lines():
    """Fetch closing lines for finished games that do not have one yet."""
    if not need_data():
        return
    season = ask("Season to back-fill", str(current_season()))
    print("One request per game; a full season takes a few minutes. Safe to interrupt and rerun.")
    run("fetch_odds.py", "--seasons", season)


MENU = [
    ("1", "Game-day update", "refresh data and lines, record today's picks (every model), show the slate", game_day),
    ("2", "Refresh data", "download this season's results and betting lines from ESPN", refresh),
    ("3", "Predict a matchup", "enter two teams", matchup),
    ("4", "Slate for a date", "every game that day: model pick and spread next to Vegas", slate),
    ("5", "Top 25", "current rankings", rankings),
    ("6", "Record predictions", "lock in a model's picks for a date before kickoff", record),
    ("7", "Settle results", "grade finished games in the prediction log", settle),
    ("8", "Prediction record", "full history of picks, outcomes and bet results, per model", report),
    ("9", "Backtest", "re-tune a model's parameters and report out-of-sample accuracy", backtest),
    ("10", "What if I had bet?", "grade flat bets on a model's picks for a past week or season", what_if),
    ("11", "Back-fill lines", "fetch closing lines for finished games (needed by option 10)", backfill_lines),
]
ACTIONS = {key: fn for key, _, _, fn in MENU}


def show_menu():
    print("\n" + "=" * 72)
    print("  College Football Models")
    print("=" * 72)
    for line in status_lines():
        print(f"  {line}")
    print()
    for key, title, desc, _ in MENU:
        print(f"  {key:>2}  {title:<20} {desc}")
    print(f"  {'q':>2}  {'Quit':<20}")
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
