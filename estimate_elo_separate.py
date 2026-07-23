#!/usr/bin/env python3
"""
Estimate chess player ELO from game moves using maia3.

Stage 1: 1D sweep assuming both players have the same ELO.
Stage 2: Two separate 1D sweeps (one per color), opponent fixed at 1500.

Usage:
    python estimate_elo_separate.py game.pgn
"""

import argparse
import io
import math
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch
from tqdm import tqdm

FIDELITY = 50

MIN_ELO = 300
MAX_ELO = 3000

OPPONENT_ELO = 1500


def _build_single_color_tensors(positions, elo_values, cfg, color_is_white, n_sample=0):
    """Build batch tensors for one color's positions only.

    For each position of the target color, evaluates all candidate ELO values
    with opponent fixed at 1500.

    Returns dict with tokens, self_elos, oppo_elos, human_moves, legal_masks,
    n_positions, n_elos.
    """
    from chess_accuracy.maia3.dataset import get_historical_tokens, tokenize_board
    from chess_accuracy.maia3.utils import get_all_possible_moves
    from chess_accuracy.pgn_parser import _select_sample_indices, move_to_index
    from chess_accuracy.maia3.dataset import get_legal_moves_mask

    ALL_MOVES = get_all_possible_moves()
    ALL_MOVES_DICT = {m: i for i, m in enumerate(ALL_MOVES)}

    n_elo = len(elo_values)

    # Filter positions to target color
    color_positions = [pos for pos in positions if pos["is_white_turn"] == color_is_white]

    if not color_positions:
        return {
            "tokens": torch.empty(0),
            "self_elos": torch.empty(0),
            "oppo_elos": torch.empty(0),
            "human_moves": np.array([], dtype=np.int64),
            "legal_masks": torch.empty(0, 4352),
            "n_positions": 0,
            "n_elos": n_elo,
        }

    # Sample from the filtered positions
    if n_sample > 0 and n_sample < len(color_positions):
        sample_indices = _select_sample_indices(len(color_positions), n_sample)
    else:
        sample_indices = list(range(len(color_positions)))

    n_pos = len(sample_indices)

    # We need to replay the full game to build correct history tokens
    # But only keep tokens for positions of the target color
    board = chess.Board()
    history: deque[torch.Tensor] = deque(maxlen=cfg.history)

    # Build a set of (original game indices) for target color positions
    target_indices = set()
    for i, pos in enumerate(positions):
        if pos["is_white_turn"] == color_is_white:
            target_indices.add(i)

    all_tokens = []
    all_human_moves = []
    all_legal_masks = []
    sampled_count = 0
    target_seen = 0

    for pos_idx, pos in enumerate(positions):
        token = tokenize_board(board)
        history.append(token)

        if pos_idx in target_indices:
            if target_seen in sample_indices:
                hist_tokens = get_historical_tokens(
                    history, cfg,
                    base=300.0, inc=0.0,
                    clk_left_before=pos["clk_left_before"],
                    clk_ponder=pos["clk_ponder"],
                )
                all_tokens.append(hist_tokens)
                all_human_moves.append(move_to_index(pos["move"], board))
                all_legal_masks.append(get_legal_moves_mask(board, ALL_MOVES_DICT))
                sampled_count += 1
            target_seen += 1

        board.push(pos["move"])

    if sampled_count == 0:
        return {
            "tokens": torch.empty(0),
            "self_elos": torch.empty(0),
            "oppo_elos": torch.empty(0),
            "human_moves": np.array([], dtype=np.int64),
            "legal_masks": torch.empty(0, 4352),
            "n_positions": 0,
            "n_elos": n_elo,
        }

    tokens_n = torch.stack(all_tokens, dim=0)
    human_moves = np.array(all_human_moves, dtype=np.int64)
    legal_masks = torch.stack(all_legal_masks, dim=0)

    if not cfg.include_time_info:
        tokens_n = tokens_n[:, :, :12 * cfg.history]

    tokens_batch = tokens_n.repeat_interleave(n_elo, dim=0)

    elo_t = torch.tensor(elo_values, dtype=torch.float32)
    self_elos = elo_t.repeat(sampled_count)
    oppo_elos = torch.full_like(self_elos, OPPONENT_ELO)

    return {
        "tokens": tokens_batch,
        "self_elos": self_elos,
        "oppo_elos": oppo_elos,
        "human_moves": human_moves,
        "legal_masks": legal_masks,
        "n_positions": sampled_count,
        "n_elos": n_elo,
    }


def _batch_estimate_single_color(pgn_text, elo_values, inf_engine, cfg, color_is_white, model_name):
    """Run 1D ELO sweep for a single color, opponent fixed at 1500."""
    from chess_accuracy.pgn_parser import parse_pgn_to_positions

    positions = parse_pgn_to_positions(pgn_text)
    if not positions:
        return 0.0, 0.0, np.zeros(len(elo_values))

    batch = _build_single_color_tensors(positions, elo_values, cfg, color_is_white)

    n_pos = batch["n_positions"]
    n_elo = batch["n_elos"]
    if n_pos == 0:
        return 0.0, 0.0, np.zeros(n_elo)

    tokens = batch["tokens"]
    self_elos = batch["self_elos"]
    oppo_elos = batch["oppo_elos"]
    human_moves = batch["human_moves"]
    legal_masks = batch["legal_masks"]

    logits_move, _, _ = inf_engine.predict(tokens, self_elos, oppo_elos)
    logits_move = logits_move.reshape(n_pos, n_elo, -1)

    legal_masks_np = legal_masks.numpy()
    legal_masks_expanded = legal_masks_np[:, np.newaxis, :]
    logits_masked = np.where(legal_masks_expanded, logits_move, -np.inf)

    all_rates = np.zeros(n_elo, dtype=np.float64)
    for elo_idx in range(n_elo):
        top1_moves = np.argmax(logits_masked[:, elo_idx, :], axis=1)
        matches = (top1_moves == human_moves).sum()
        all_rates[elo_idx] = matches / n_pos if n_pos > 0 else 0.0

    best_idx = np.argmax(all_rates)
    best_elo = float(elo_values[best_idx])
    best_rate = float(all_rates[best_idx])

    return best_elo, best_rate, all_rates


def _batch_estimate_separate(pgn_text, scan, model_name):
    """Estimate ELO with Stage 1 1D sweep + Stage 2 separate-color refinement."""
    from chess_accuracy.batch_inference import load_inference_engine, estimate_elo_batch
    from chess_accuracy.maia3.model_registry import resolve_model_spec
    from types import SimpleNamespace

    elo_lo, elo_hi = scan["elo_lo"], scan["elo_hi"]
    elo_lo = max(elo_lo, MIN_ELO)
    elo_hi = min(elo_hi, MAX_ELO)

    print(f"Loading maia3 ONNX model ({model_name})...")
    inf_engine = load_inference_engine(model_name)

    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    # Stage 1: 1D sweep (same ELO for both)
    elo_values = np.arange(elo_lo, elo_hi + 1, FIDELITY, dtype=np.float32)
    n_grid = len(elo_values)
    print(f"Stage 1: 1D sweep ({n_grid} values, step={FIDELITY})...")
    best_elo, best_rate, _ = estimate_elo_batch(
        pgn_text, elo_values, inf_engine, model_name=model_name
    )
    print(f"  -> 1D estimate: {best_elo:.0f} (rate={best_rate:.4f})")

    n_evals = n_grid

    # Stage 2: separate 1D sweeps for each color (opponent=1500)
    margin = (elo_hi - elo_lo) // 2
    n_axis = int(math.ceil(math.sqrt(n_grid)))
    round_num = 0
    best_w, best_b = best_elo, best_elo

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
            f"Round {round_num}: separate 1D refinement "
            f"({n_axis} values each, margin=±{margin:.0f}, step={step:.0f})..."
        )

        w_elo, w_rate, _ = _batch_estimate_single_color(
            pgn_text, white_elos, inf_engine, cfg, color_is_white=True, model_name=model_name
        )
        b_elo, b_rate, _ = _batch_estimate_single_color(
            pgn_text, black_elos, inf_engine, cfg, color_is_white=False, model_name=model_name
        )

        best_w, best_b = w_elo, b_elo
        best_rate = (w_rate + b_rate) / 2
        n_evals += n_axis * 2

        tqdm.write(
            f"  -> W={best_w:.0f} (rate={w_rate:.4f}), "
            f"B={best_b:.0f} (rate={b_rate:.4f})"
        )

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
    est_white, est_black, peak_rate, n_evals = _batch_estimate_separate(
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
        description="Estimate chess player ELO from game moves using maia3 (separate per-color)"
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
    args = parser.parse_args()

    pgn_path = Path(args.pgn) if args.pgn else Path("example2.pgn")
    result = estimate(
        pgn_path,
        n_sample=args.sample,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()
