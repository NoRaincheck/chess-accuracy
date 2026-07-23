#!/usr/bin/env python3
"""
Estimate chess player ELO from game moves using maia3.

Usage:
    python estimate_elo.py game.pgn          # estimate ELO (uses calibration if available)
    python estimate_elo.py --calibrate       # calibrate against data/ directory
"""

import argparse
import json
import io
import math
import os
import random
import subprocess
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
from tqdm import tqdm

# Force maia3-uci to use local HuggingFace cache only (no network checks)
_MAIA_ENV = {**os.environ, "HF_HUB_OFFLINE": "1"}


CONFIG_PATH = Path(__file__).parent / "estimate_elo.json"
DATA_DIR = Path(__file__).parent / "data"

DEFAULT_SCAN = {"elo_lo": 300, "elo_hi": 3500}
BASE_MODEL = "maia3-5m"  # one of "maia3-79m, maia3-23m, maia3-5m"
FIDELITY = 50

# Cache for ELO evaluations: maps (pgn_hash, elo) -> match_rate
_eval_cache: dict[tuple[int, int], float] = {}


def _pgn_hash(pgn_text: str) -> int:
    return hash(pgn_text)


def _select_sample_indices(total_moves, n_sample):
    """Select move indices to evaluate, preferring middlegame positions.

    Heuristics:
      - Skip first ~8 moves (opening book, memorized, not discriminating)
      - Skip last ~5 moves (game often decided, resignation/mate)
      - Weight remaining moves by a bell curve centered on the middlegame
      - Sample n_sample moves weighted by that distribution
    """
    if n_sample <= 0 or n_sample >= total_moves:
        return list(range(total_moves))

    skip_open = min(8, total_moves // 6)
    skip_end = min(5, total_moves // 8)
    lo, hi = skip_open, total_moves - skip_end
    if hi - lo <= n_sample:
        return list(range(lo, hi))

    # Bell-curve weight centered at the middlegame (roughly 40% into the game)
    center = lo + (hi - lo) * 0.4
    spread = (hi - lo) / 3.0
    candidates = list(range(lo, hi))
    weights = [math.exp(-0.5 * ((i - center) / spread) ** 2) for i in candidates]

    # Weighted random sample without replacement
    chosen = []
    remaining = list(range(len(candidates)))
    for _ in range(n_sample):
        w_sum = sum(weights[i] for i in remaining)
        r = random.random() * w_sum
        cumul = 0.0
        for idx in remaining:
            cumul += weights[idx]
            if cumul >= r:
                chosen.append(candidates[idx])
                remaining.remove(idx)
                break

    return sorted(chosen)


def get_maia3_stats(pgn_text, elo, sample_indices=None):
    """Query maia3 at a given ELO, return match rate. Cached by (pgn, elo).

    If sample_indices is provided, only evaluate those moves.
    """
    key = (_pgn_hash(pgn_text), elo)
    if key in _eval_cache:
        return _eval_cache[key]

    game = chess.pgn.read_game(io.StringIO(pgn_text))
    board = game.board()

    eng = chess.engine.SimpleEngine.popen_uci(
        [
            "maia3-uci",
            "--model",
            "maia3-79m",
            "--elo",
            str(elo),
        ],
        env=_MAIA_ENV,
        stderr=subprocess.DEVNULL,
    )

    moves = list(game.mainline_moves())
    sample_set = set(sample_indices) if sample_indices is not None else None

    top1 = 0
    n = 0
    for i, move in enumerate(moves):
        if sample_set is not None and i not in sample_set:
            board.push(move)
            continue
        info = eng.analyse(board, chess.engine.Limit(nodes=1))
        pv = info.get("pv", [])
        if pv and pv[0] == move:
            top1 += 1
        board.push(move)
        n += 1

    eng.quit()
    rate = top1 / n if n > 0 else 0.0
    _eval_cache[key] = rate
    return rate


def ternary_search_elo(pgn_text, elo_lo, elo_hi, fidelity=FIDELITY, sample_indices=None):
    """Ternary search for ELO with highest match rate.

    Narrows [lo, hi] by evaluating at two interior points each iteration,
    discarding the third that cannot contain the peak. Stops when the
    interval is <= fidelity wide.
    """
    lo, hi = elo_lo, elo_hi
    total_evals = 0

    while hi - lo > fidelity:
        m1 = lo + (hi - lo) // 3
        m2 = hi - (hi - lo) // 3

        if m1 == m2:
            break

        tqdm.write(f"Ternary search: [{lo}, {hi}] (evaluating {m1}, {m2})")
        r1 = get_maia3_stats(pgn_text, m1, sample_indices)
        r2 = get_maia3_stats(pgn_text, m2, sample_indices)
        total_evals += 2

        if r1 < r2:
            lo = m1
        else:
            hi = m2

    # Evaluate the final interval to find the best point
    best_elo = lo
    best_rate = get_maia3_stats(pgn_text, lo, sample_indices)
    total_evals += 1

    for elo in range(lo + 1, hi + 1):
        rate = get_maia3_stats(pgn_text, elo, sample_indices)
        total_evals += 1
        if rate > best_rate:
            best_rate = rate
            best_elo = elo

    tqdm.write(
        f"Search complete: best ELO = {best_elo} (rate = {best_rate:.4f}), {total_evals} evaluations"
    )
    return float(best_elo), best_rate


def load_config():
    """Load calibration config."""
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return None


def apply_correction(raw_elo, config):
    """Apply calibration correction to raw estimate via piecewise linear interpolation."""
    if config is None or "correction" not in config:
        return raw_elo
    correction = config["correction"]
    raw_elos = [c["raw_elo"] for c in correction]
    actual_elos = [c["actual_elo"] for c in correction]
    return float(np.interp(raw_elo, raw_elos, actual_elos))


def estimate(pgn_path, n_sample=0):
    """Estimate ELO for a game.

    If n_sample > 0, only evaluate that many moves (heuristically selected)
    instead of the full game.
    """
    config = load_config()

    pgn_text = pgn_path.read_text()
    game = chess.pgn.read_game(io.StringIO(pgn_text))

    white_name = game.headers.get("White", "?")
    black_name = game.headers.get("Black", "?")
    white_elo_hdr = game.headers.get("WhiteElo", "?")
    black_elo_hdr = game.headers.get("BlackElo", "?")

    # Build sample indices if requested
    total_moves = sum(1 for _ in game.mainline_moves())
    sample_indices = None
    if n_sample > 0:
        sample_indices = _select_sample_indices(total_moves, n_sample)
        print(f"Sampling {len(sample_indices)} of {total_moves} moves (heuristic selection)")

    # Use calibrated scan params if available
    scan = config["scan"] if config else DEFAULT_SCAN

    raw_elo, peak_rate = ternary_search_elo(
        pgn_text, scan["elo_lo"], scan["elo_hi"], sample_indices=sample_indices
    )
    corrected_elo = apply_correction(raw_elo, config)

    print()
    print(f"Game: {white_name} vs {black_name}")
    print(f"WhiteElo: {white_elo_hdr}, BlackElo: {black_elo_hdr}")
    print()

    if config:
        print(
            f"Raw estimate:        {raw_elo:6.0f}  (peak rate {peak_rate * 100:.1f}%)"
        )
        print(f"Calibrated estimate: {corrected_elo:6.0f}")
    else:
        print(
            f"Maia3 estimate:      {raw_elo:6.0f}  (peak rate {peak_rate * 100:.1f}%)"
        )

    print(f"PGN reference:       W {white_elo_hdr:>6s}   B {black_elo_hdr:>6s}")

    print()
    if n_sample > 0:
        print("Method: maia3 is queried at each ELO level. A heuristic sample")
        print("of middlegame positions is used for faster estimation.")
    else:
        print("Method: maia3 is queried at each ELO level. For every position")
        print("in the game, we check whether the human's top-1 move matches")
        print("the engine's top-1 move.")
    print("A ternary search narrows the ELO range to the peak match rate")
    print("(fidelity ±50 ELO).")
    if config:
        print("Calibration correction applied from estimate_elo.json.")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate chess player ELO from game moves using maia3"
    )
    parser.add_argument("pgn", nargs="?", help="PGN file to estimate")
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Calibrate against data/ directory and save to estimate_elo.json",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Sample N positions heuristically (middlegame-weighted) instead of evaluating all moves",
    )
    args = parser.parse_args()

    pgn_path = Path(args.pgn) if args.pgn else Path("example2.pgn")
    estimate(pgn_path, n_sample=args.sample)


if __name__ == "__main__":
    main()
