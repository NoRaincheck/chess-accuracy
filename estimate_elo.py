#!/usr/bin/env python3
"""
Estimate chess player ELO from game moves using maia3.

Usage:
    python estimate_elo.py game.pgn              # 2D sweep (white & black ELO)
    python estimate_elo.py game.pgn --1d         # 1D sweep (same ELO for both)
"""

import argparse
import json
import io
import math
from pathlib import Path

import chess
import chess.pgn
import numpy as np
from tqdm import tqdm

CONFIG_PATH = Path(__file__).parent / "estimate_elo.json"
DATA_DIR = Path(__file__).parent / "data"

DEFAULT_SCAN = {"elo_lo": 300, "elo_hi": 3500}
FIDELITY = 50

MIN_ELO = 300
MAX_ELO = 3000


# ── Iterative refinement: 1D sweep → repeated 2D narrowing ───────────────────
def _batch_estimate_2d(pgn_text, scan, model_name):
    """Estimate ELO with iterative 2D refinement until grid step < FIDELITY."""
    from chess_accuracy.batch_inference import (
        load_inference_engine,
        estimate_elo_batch,
        estimate_elo_2d,
    )

    elo_lo, elo_hi = scan["elo_lo"], scan["elo_hi"]
    elo_lo = max(elo_lo, MIN_ELO)
    elo_hi = min(elo_hi, MAX_ELO)

    print(f"Loading maia3 ONNX model ({model_name})...")
    inf_engine = load_inference_engine(model_name)

    # Stage 1: 1D sweep to get initial estimate
    elo_values = np.arange(elo_lo, elo_hi + 1, FIDELITY, dtype=np.float32)
    n_grid = len(elo_values)
    print(f"Stage 1: 1D sweep ({n_grid} values, step={FIDELITY})...")
    best_elo, best_rate, _ = estimate_elo_batch(
        pgn_text, elo_values, inf_engine, model_name=model_name
    )
    print(f"  -> 1D estimate: {best_elo:.0f} (rate={best_rate:.4f})")

    n_evals = n_grid
    best_w, best_b, best_rate = best_elo, best_elo, best_rate

    # Stage 2: iterative 2D refinement
    margin = (elo_hi - elo_lo) // 2  # start with full range
    n_axis = int(
        math.ceil(math.sqrt(n_grid))
    )  # values per axis, total ≈ n_grid per round
    round_num = 0

    while True:
        step = (margin * 2) / max(n_axis - 1, 1)
        if step < FIDELITY:
            break

        round_num += 1
        w_lo = max(elo_lo, best_w - margin)
        w_hi = min(elo_hi, best_w + margin)
        b_lo = max(elo_lo, best_b - margin)
        b_hi = min(elo_hi, best_b + margin)

        white_elos = np.linspace(w_lo, w_hi, n_axis, dtype=np.float32)
        black_elos = np.linspace(b_lo, b_hi, n_axis, dtype=np.float32)

        print(
            f"Round {round_num}: 2D refinement "
            f"({n_axis}x{n_axis}, margin=±{margin:.0f}, step={step:.0f})..."
        )

        best_w, best_b, best_rate, _ = estimate_elo_2d(
            pgn_text,
            white_elos,
            black_elos,
            inf_engine,
            model_name=model_name,
        )
        n_evals += n_axis * n_axis

        tqdm.write(f"  -> best: W={best_w:.0f}, B={best_b:.0f} (rate={best_rate:.4f})")

        margin = int(margin / 2)

    tqdm.write(f"Final: W={best_w:.0f}, B={best_b:.0f} (rate={best_rate:.4f})")

    return best_w, best_b, best_rate, n_evals


def estimate(
    pgn_path,
    n_sample=0,
    model_name="maia3-5m",
):
    """Estimate ELO for a game."""
    pgn_text = pgn_path.read_text()
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    white_name = game.headers.get("White", "?")
    black_name = game.headers.get("Black", "?")
    white_elo_hdr = game.headers.get("WhiteElo", "?")
    black_elo_hdr = game.headers.get("BlackElo", "?")

    total_moves = sum(1 for _ in game.mainline_moves())
    est_white, est_black, peak_rate, n_evals = _batch_estimate_2d(
        pgn_text,
        {"elo_lo": MIN_ELO, "elo_hi": MAX_ELO},
        model_name,
    )

    print()
    print(f"Game: {white_name} vs {black_name}")
    print(f"WhiteElo: {white_elo_hdr}, BlackElo: {black_elo_hdr}")
    print()
    print(
        f"Estimated:  W {est_white:6.0f}   B {est_black:6.0f}  (rate {peak_rate * 100:.1f}%)"
    )
    print(f"PGN ref:    W {white_elo_hdr:>6s}   B {black_elo_hdr:>6s}")
    print()

    return {
        "white": white_name,
        "black": black_name,
        "white_elo_hdr": white_elo_hdr,
        "black_elo_hdr": black_elo_hdr,
        "est_white_elo": round(est_white, 1),
        "est_black_elo": round(est_black, 1),
        "peak_rate": round(peak_rate, 4),
        "n_evaluations": n_evals,
        "sampled": n_sample > 0,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate chess player ELO from game moves using maia3"
    )
    parser.add_argument("pgn", nargs="?", help="PGN file to estimate")
    parser.add_argument(
        "--calibrate", action="store_true", help="Calibrate against data/ directory"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Sample N positions heuristically",
    )
    parser.add_argument(
        "--model",
        default="maia3-5m",
        help="Maia3 model: maia3-5m, maia3-23m, maia3-79m",
    )
    parser.add_argument(
        "--1d",
        dest="mode_1d",
        action="store_true",
        help="1D sweep (same ELO for both players)",
    )
    args = parser.parse_args()

    pgn_path = Path(args.pgn) if args.pgn else Path("example2.pgn")
    result = estimate(
        pgn_path,
        n_sample=args.sample,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
