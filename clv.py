"""
Closing line value: did the market move toward our side after we would have bet?

Beating the closing line is the fastest honest test of a betting edge. A bettor who takes a side
at the opening number and then sees the closing number move in that direction has beaten the
market, whether or not the bet wins that week; over a season, positive closing line value is
what a real edge looks like and win-loss records are mostly noise.

For every finished game with an opening and a closing spread from the same sportsbook (CFBD's
lines files), this script takes the model's pregame margin, picks the side it favours against
the OPENING line, and measures how far the CLOSING line moved toward (+) or away from (-) that
side. It also grades the pick against the opener and against the closer.

Usage:
    python clv.py --model prior --season 2024 2025
    python clv.py --model market --season 2024 2025 --min-edge 2
"""
import argparse

import numpy as np
import pandas as pd

from common import CFBD_DIR, load_games, load_params
from models import add_model_argument, get_model

PROVIDER_PRIORITY = ["DraftKings", "ESPN Bet", "Bovada", "consensus", "Caesars"]


def load_lines(seasons):
    frames = []
    for season in seasons:
        path = CFBD_DIR / f"lines_{season}.csv"
        if path.exists():
            frames.append(pd.read_csv(path, dtype={"id": str}))
    lines = pd.concat(frames, ignore_index=True).dropna(subset=["spread", "spread_open"])
    lines["rank"] = lines["provider"].map({p: i for i, p in enumerate(PROVIDER_PRIORITY)}).fillna(99)
    lines = lines.sort_values(["id", "rank"]).drop_duplicates("id")
    return lines[["id", "provider", "spread_open", "spread"]].rename(columns={"id": "game_id"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", type=int, nargs="*", default=[2024, 2025])
    parser.add_argument("--min-edge", type=float, default=0.0, help="only bet when |our margin - opener| >= this many points")
    add_model_argument(parser)
    args = parser.parse_args()

    games = load_games()
    model = get_model(args.model, **load_params(args.model))
    pred = model.fit(games)
    pred = pred[pred["season"].isin(args.season)]
    lines = load_lines(args.season)
    df = pred.merge(lines, on="game_id", how="inner")

    # Our margin as it would have been with only the opening line known.
    open_vegas = -df["spread_open"].to_numpy(dtype=float)       # home margin implied by the opener
    close_vegas = -df["spread"].to_numpy(dtype=float)
    if args.model == "market":
        ours = model.shade(open_vegas, df["base_margin"].to_numpy(dtype=float))
    else:
        ours = df["pred_margin"].to_numpy(dtype=float)
    edge = ours - open_vegas                                    # >0: we like the home side vs the opener
    side = np.sign(edge)                                         # +1 home, -1 away
    bet = (np.abs(edge) >= args.min_edge) & (side != 0)
    df = df.assign(ours=ours, open_vegas=open_vegas, close_vegas=close_vegas, edge=edge, side=side)[bet].copy()

    df["clv"] = (df["close_vegas"] - df["open_vegas"]) * df["side"]        # points the market moved toward us
    actual = df["margin"].to_numpy(dtype=float)
    df["cover_open"] = np.sign(actual - df["open_vegas"]) * df["side"]      # +1 win, -1 loss, 0 push vs opener
    df["cover_close"] = np.sign(actual - df["close_vegas"]) * df["side"]

    n = len(df)
    print(f"Model: {args.model}. {n} bets vs the opening line, seasons {args.season}, min edge {args.min_edge:g} "
          f"(books: {df['provider'].value_counts().to_dict()})\n")
    print(f"Closing line value: mean {df['clv'].mean():+.2f} points per bet; the close moved toward us in "
          f"{(df['clv'] > 0).mean():.1%} of bets, away in {(df['clv'] < 0).mean():.1%}, unchanged in {(df['clv'] == 0).mean():.1%}.")
    for label, col in (("vs opener", "cover_open"), ("vs closer", "cover_close")):
        w, l, p = (df[col] > 0).sum(), (df[col] < 0).sum(), (df[col] == 0).sum()
        print(f"Record {label}: {w}-{l}-{p} ({w / (w + l):.1%}); break-even at -110 is 52.4%.")
    print("\nBy size of our disagreement with the opener:")
    bins = pd.cut(df["edge"].abs(), [0, 1, 2, 3, 5, 8, 100], include_lowest=True)
    table = df.groupby(bins, observed=True).agg(bets=("clv", "size"), mean_clv=("clv", "mean"),
                                                 clv_positive=("clv", lambda s: (s > 0).mean()),
                                                 cover_rate_vs_opener=("cover_open", lambda s: (s > 0).sum() / max((s != 0).sum(), 1)))
    print(table.round(3).to_string())
    print(f"\nSanity check: the opener itself moved toward the eventual winner's side in "
          f"{(np.sign(df['close_vegas'] - df['open_vegas']) == np.sign(actual - df['open_vegas'])).mean():.1%} of games.")


if __name__ == "__main__":
    main()
