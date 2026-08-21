"""
Estimate chess player ELO from game moves using maia3.

Default scorer ("loglik") — likelihood fingerprints with uncertainty:
  Stage A: sparse joint (white, black) anchor grid over the full range;
           the smooth log-likelihood surface is bicubically interpolated
           to localize per-color modes at a fraction of the cost of a
           dense sweep.
  Stage B: joint fine grid around the modes -> posterior mean / std / 95% CI
           per color, with a Gaussian population prior. All reported
           statistics come from real model evaluations.

Legacy scorer (--scorer legacy) — original top-1+MRR argmax sweeps:
Stage 1: 1D sweep assuming both players have the same ELO.
Stage 2: Two separate 1D sweeps (one per color), opponent fixed at the ELO in stage 1.

Usage:
    python estimate_elo.py game.pgn
"""

import argparse
import io
import json
import math
from collections import deque
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

FIDELITY = 50

MIN_ELO = 300
MAX_ELO = 3000

# Log-likelihood scorer configuration
ANCHOR_STEP = 550  # stage-A full-range joint anchor grid spacing (localization)
DENSE_STEP = 50  # interpolated surface resolution (mode finding only, no model evals)
FINE_MARGIN = 300  # stage-B window half-width around per-color modes
FINE_STEP = 100  # stage-B grid spacing (point estimates refined by parabola fit)
PRIOR_MEAN = 1500.0  # Gaussian population prior over player ELO
PRIOR_STD = 350.0
DEFAULT_MAX_POSITIONS = 80  # deterministic cap on evaluated positions (0 = all)


def roundup(x):
    return math.ceil(x / 100.0) * 100


def rounddown(x):
    return math.floor(x / 100.0) * 100


def _build_single_color_tensors(positions, elo_values, cfg, color_is_white, n_sample=0, opponent_elo=1500):
    """Build batch tensors for one color's positions only.

    For each position of the target color, evaluates all candidate ELO values
    with opponent fixed at 1500.

    Returns dict with tokens, self_elos, oppo_elos, human_moves, legal_masks,
    n_positions, n_elos.
    """
    from chess_accuracy.maia3.dataset import get_historical_tokens, get_legal_moves_mask, tokenize_board
    from chess_accuracy.pgn_parser import ALL_MOVES_DICT, _select_sample_indices, move_to_index

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
                    history,
                    cfg,
                    base=300.0,
                    inc=0.0,
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
        tokens_n = tokens_n[:, :, : 12 * cfg.history]

    tokens_batch = tokens_n.repeat_interleave(n_elo, dim=0)

    elo_t = torch.tensor(elo_values, dtype=torch.float32)
    self_elos = elo_t.repeat(sampled_count)
    oppo_elos = torch.full_like(self_elos, opponent_elo)

    return {
        "tokens": tokens_batch,
        "self_elos": self_elos,
        "oppo_elos": oppo_elos,
        "human_moves": human_moves,
        "legal_masks": legal_masks,
        "n_positions": sampled_count,
        "n_elos": n_elo,
    }


def _batch_estimate_single_color(pgn_text, elo_values, inf_engine, cfg, color_is_white, model_name, opponent_elo=1500):
    """Run 1D ELO sweep for a single color, opponent fixed at 1500."""
    from chess_accuracy.batch_inference import _compute_score
    from chess_accuracy.pgn_parser import parse_pgn_to_positions

    positions = parse_pgn_to_positions(pgn_text)
    if not positions:
        return 0.0, 0.0, np.zeros(len(elo_values))

    batch = _build_single_color_tensors(positions, elo_values, cfg, color_is_white, opponent_elo=opponent_elo)

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

    all_rates = _compute_score(logits_masked, human_moves, n_pos, alpha=0.6)

    best_idx = np.argmax(all_rates)
    best_elo = float(elo_values[best_idx])
    best_rate = float(all_rates[best_idx])

    return best_elo, best_rate, all_rates


def _batch_estimate_separate(pgn_text, scan, model_name, quiet=False):
    """Estimate ELO with Stage 1 1D sweep + Stage 2 separate-color refinement."""
    from types import SimpleNamespace

    from chess_accuracy.batch_inference import estimate_elo_batch, load_inference_engine
    from chess_accuracy.maia3.model_registry import resolve_model_spec

    elo_lo, elo_hi = scan["elo_lo"], scan["elo_hi"]
    elo_lo = max(elo_lo, MIN_ELO)
    elo_hi = min(elo_hi, MAX_ELO)

    def _log(msg):
        if not quiet:
            print(msg)

    _log(f"Loading maia3 ONNX model ({model_name})...")
    inf_engine = load_inference_engine(model_name)

    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    # Stage 1: 1D sweep (same ELO for both)
    elo_values = np.arange(elo_lo, elo_hi + 1, 200, dtype=np.float32)
    n_grid = len(elo_values)
    _log(f"Stage 1: 1D sweep ({n_grid} values, step={FIDELITY})...")
    best_elo, best_rate, _ = estimate_elo_batch(pgn_text, elo_values, inf_engine, model_name=model_name)
    _log(f"  -> 1D estimate: {best_elo:.0f} (rate={best_rate:.4f})")

    n_evals = n_grid

    # Stage 2: separate 1D sweeps for each color (opponent = other color's estimate)
    # Start with a tight margin around the Stage 1 estimate to avoid wandering
    # to ELO extremes where the model's predictions are flat/undifferentiated.
    margin = min(400, (elo_hi - elo_lo) // 2)
    n_axis = math.ceil(math.sqrt(n_grid))
    round_num = 0
    best_w, best_b = best_elo, best_elo
    # if opponent_elo is < 1500 then round up to nearest 100, else round down to nearest 100
    opponent_elo = roundup(best_elo + 50) if best_elo < 1500 else rounddown(best_elo - 50)

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

        _log(
            f"Round {round_num}: separate 1D refinement "
            f"({n_axis} values each, margin=±{margin:.0f}, step={step:.0f})..."
        )

        w_elo, w_rate, _ = _batch_estimate_single_color(
            pgn_text,
            white_elos,
            inf_engine,
            cfg,
            color_is_white=True,
            model_name=model_name,
            opponent_elo=opponent_elo,
        )
        b_elo, b_rate, _ = _batch_estimate_single_color(
            pgn_text,
            black_elos,
            inf_engine,
            cfg,
            color_is_white=False,
            model_name=model_name,
            opponent_elo=opponent_elo,
        )

        best_w, best_b = w_elo, b_elo
        best_rate = (w_rate + b_rate) / 2
        n_evals += n_axis * 2

        _log(f"  -> W={best_w:.0f} (rate={w_rate:.4f}), B={best_b:.0f} (rate={b_rate:.4f})")

        margin = int(margin / 2)

    _log(f"Final: W={best_w:.0f}, B={best_b:.0f} (rate={best_rate:.4f})")

    return best_w, best_b, best_rate, n_evals


def _make_grid(lo: float, hi: float, step: float) -> np.ndarray:
    """ELO grid from lo to hi (inclusive, hi snapped into the grid)."""
    n = math.ceil((hi - lo) / step)
    grid = lo + step * np.arange(n + 1, dtype=np.float64)
    return np.clip(grid, lo, hi).astype(np.float32)


def _window_grid(center: float, margin: float, step: float, lo: float, hi: float) -> np.ndarray:
    """Fine ELO grid centered on `center`, clamped to [lo, hi]."""
    return _make_grid(max(lo, center - margin), min(hi, center + margin), step)


def _logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    """Log-sum-exp along an axis, stable against under/overflow."""
    m = a.max(axis=axis, keepdims=True)
    out = m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)


def _vertex_refine(values: np.ndarray, log_curve: np.ndarray) -> float:
    """Sub-grid point estimate: parabola vertex through the 3 points around the mode."""
    k = int(np.argmax(log_curve))
    if 0 < k < len(values) - 1:
        y0, y1, y2 = log_curve[k - 1], log_curve[k], log_curve[k + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom < 0:
            delta = 0.5 * (y0 - y2) / denom * (values[1] - values[0])
            return float(values[k] + delta)
    return float(values[k])


def _estimate_posterior(
    pgn_text: str,
    model_name: str,
    quiet: bool = False,
    max_positions: int = DEFAULT_MAX_POSITIONS,
) -> dict:
    """Estimate per-color ELO posteriors via log-likelihood fingerprints."""
    from types import SimpleNamespace

    from chess_accuracy.batch_inference import (
        _marginal_stats,
        bicubic_upsample_surface,
        joint_posterior_2d,
        load_inference_engine,
        loglik_2d_grid,
    )
    from chess_accuracy.maia3.model_registry import resolve_model_spec
    from chess_accuracy.pgn_parser import cap_indices, informative_indices, parse_pgn_to_positions

    def _log(msg):
        if not quiet:
            print(msg)

    _log(f"Loading maia3 ONNX model ({model_name})...")
    inf_engine = load_inference_engine(model_name)

    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    positions = parse_pgn_to_positions(pgn_text)
    indices = cap_indices(informative_indices(positions), max_positions)
    n_pos = len(indices)

    if n_pos == 0:
        raise ValueError("No informative positions found in PGN")

    # Stage A: sparse joint anchor grid over the full range -> interpolated
    # surface -> per-color modes. The likelihood surface is very smooth in
    # (w, b), so a coarse anchor grid plus bicubic interpolation localizes
    # the modes far more cheaply than a dense sweep. The diagonal of the
    # anchor surface doubles as the shared-ELO likelihood curve.
    anchor_grid = _make_grid(MIN_ELO, MAX_ELO, ANCHOR_STEP)
    _log(f"Stage A: joint anchor grid ({len(anchor_grid)}x{len(anchor_grid)}, step={ANCHOR_STEP:.0f})...")
    surface_a, _match_a, _ = loglik_2d_grid(positions, anchor_grid, anchor_grid, cfg, inf_engine, indices=indices)
    n_evals = len(anchor_grid) ** 2

    dense_grid = _make_grid(MIN_ELO, MAX_ELO, DENSE_STEP)
    dense = bicubic_upsample_surface(surface_a, (len(dense_grid), len(dense_grid)))
    joint_dense = joint_posterior_2d(dense, dense_grid, dense_grid, PRIOR_MEAN, PRIOR_STD)
    w_marg_dense = joint_dense.sum(axis=1)
    b_marg_dense = joint_dense.sum(axis=0)
    center = 0.5 * float((w_marg_dense * dense_grid).sum() + (b_marg_dense * dense_grid).sum())
    w_mode = float(dense_grid[int(np.argmax(w_marg_dense))])
    b_mode = float(dense_grid[int(np.argmax(b_marg_dense))])
    _log(f"  -> game center: {center:.0f}, modes: W={w_mode:.0f}, B={b_mode:.0f}")

    # Stage B: joint fine grid around the modes -> final posteriors.
    # If a marginal mode lands on the window edge, the true mode may lie just
    # outside; slide the window there once and re-evaluate.
    for attempt in range(2):
        w_grid = _window_grid(w_mode, FINE_MARGIN, FINE_STEP, MIN_ELO, MAX_ELO)
        b_grid = _window_grid(b_mode, FINE_MARGIN, FINE_STEP, MIN_ELO, MAX_ELO)
        _log(
            f"Stage B{' (re-centered)' if attempt else ''}: joint fine grid "
            f"({len(w_grid)}x{len(b_grid)}, ±{FINE_MARGIN:.0f} @ step={FINE_STEP:.0f})..."
        )
        surface2, match2, _ = loglik_2d_grid(positions, w_grid, b_grid, cfg, inf_engine, indices=indices)
        n_evals += len(w_grid) * len(b_grid)

        joint_b = joint_posterior_2d(surface2, w_grid, b_grid, PRIOR_MEAN, PRIOR_STD)
        w_post = joint_b.sum(axis=1)
        b_post = joint_b.sum(axis=0)

        wi_best = int(np.argmax(w_post))
        bi_best = int(np.argmax(b_post))
        edge_hit = wi_best in (0, len(w_grid) - 1) or bi_best in (0, len(b_grid) - 1)
        if attempt == 0 and edge_hit:
            w_mode = float(w_grid[wi_best])
            b_mode = float(b_grid[bi_best])
            continue
        break

    _mean_w, std_w, ci_w = _marginal_stats(w_post, w_grid.astype(np.float64))
    _mean_b, std_b, ci_b = _marginal_stats(b_post, b_grid.astype(np.float64))

    # Point estimates: parabola-refined marginal modes (sub-grid precision).
    from chess_accuracy.batch_inference import elo_log_prior

    log_post2 = (
        surface2
        + elo_log_prior(w_grid, PRIOR_MEAN, PRIOR_STD)[:, np.newaxis]
        + elo_log_prior(b_grid, PRIOR_MEAN, PRIOR_STD)[np.newaxis, :]
    )
    est_w = _vertex_refine(w_grid, _logsumexp(log_post2, axis=1))
    est_b = _vertex_refine(b_grid, _logsumexp(log_post2, axis=0))

    # Familiar "match rate" at the grid cell nearest the posterior mean.
    wi = int(np.argmin(np.abs(w_grid - est_w)))
    bi = int(np.argmin(np.abs(b_grid - est_b)))
    peak_rate = float(match2[wi, bi] / n_pos) if n_pos else 0.0

    _log(f"Final: W={est_w:.0f} ± {std_w:.0f}, B={est_b:.0f} ± {std_b:.0f} (rate={peak_rate:.4f})")

    return {
        "est_white_elo": round(est_w, 1),
        "est_black_elo": round(est_b, 1),
        "white_std": round(std_w, 1),
        "black_std": round(std_b, 1),
        "white_ci95": [round(ci_w[0], 1), round(ci_w[1], 1)],
        "black_ci95": [round(ci_b[0], 1), round(ci_b[1], 1)],
        "peak_rate": round(peak_rate, 4),
        "n_evaluations": n_evals,
        "n_positions": n_pos,
        "stage1_center": round(center, 1),
    }


def estimate(
    pgn_path,
    n_sample=0,
    model_name="maia3-5m",
    quiet=False,
    scorer="loglik",
    max_positions=DEFAULT_MAX_POSITIONS,
):
    """Estimate ELO for a game."""
    pgn_text = pgn_path.read_text()
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None

    white_name = game.headers.get("White", "?")
    black_name = game.headers.get("Black", "?")
    white_elo_hdr = game.headers.get("WhiteElo", "?")
    black_elo_hdr = game.headers.get("BlackElo", "?")

    n_moves = sum(1 for _ in game.mainline_moves())

    if scorer == "legacy":
        est_white, est_black, peak_rate, n_evals = _batch_estimate_separate(
            pgn_text,
            {"elo_lo": MIN_ELO, "elo_hi": MAX_ELO},
            model_name,
            quiet=quiet,
        )
        result = {
            "white": white_name,
            "black": black_name,
            "white_elo_hdr": white_elo_hdr,
            "black_elo_hdr": black_elo_hdr,
            "est_white_elo": round(est_white, 1),
            "est_black_elo": round(est_black, 1),
            "peak_rate": round(peak_rate, 4),
            "n_evaluations": n_evals,
            "n_moves": n_moves,
            "sampled": n_sample > 0,
        }
    else:
        post = _estimate_posterior(pgn_text, model_name, quiet=quiet, max_positions=max_positions)
        result = {
            "white": white_name,
            "black": black_name,
            "white_elo_hdr": white_elo_hdr,
            "black_elo_hdr": black_elo_hdr,
            "n_moves": n_moves,
            "sampled": n_sample > 0,
            **post,
        }

    if not quiet:
        print()
        print(f"Game: {white_name} vs {black_name}")
        print(f"WhiteElo: {white_elo_hdr}, BlackElo: {black_elo_hdr}")
        print()
        if scorer == "legacy":
            print(f"Estimated:  W {result['est_white_elo']:6.0f}   B {result['est_black_elo']:6.0f}  (rate {result['peak_rate'] * 100:.1f}%)")
        else:
            print(
                f"Estimated:  W {result['est_white_elo']:6.0f} ± {result['white_std']:.0f}   "
                f"B {result['est_black_elo']:6.0f} ± {result['black_std']:.0f}  "
                f"(rate {result['peak_rate'] * 100:.1f}%)"
            )
            print(
                f"95% CI:     W [{result['white_ci95'][0]:.0f}, {result['white_ci95'][1]:.0f}]   "
                f"B [{result['black_ci95'][0]:.0f}, {result['black_ci95'][1]:.0f}]"
            )
        print(f"PGN ref:    W {white_elo_hdr:>6s}   B {black_elo_hdr:>6s}")
        print()

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Estimate chess player ELO from game moves using maia3 (separate per-color)"
    )
    parser.add_argument("pgn", nargs="?", help="PGN file to estimate")
    parser.add_argument("--calibrate", action="store_true", help="Calibrate against data/ directory")
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Sample N positions heuristically",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=DEFAULT_MAX_POSITIONS,
        metavar="N",
        help=f"Cap on evaluated positions, evenly spaced (default {DEFAULT_MAX_POSITIONS}; 0 = all)",
    )
    parser.add_argument(
        "--model",
        default="maia3-5m",
        help="Maia3 model: maia3-5m, maia3-23m, maia3-79m",
    )
    parser.add_argument("--json", action="store_true", help="Output result as NDJSON to stdout")
    parser.add_argument(
        "--scorer",
        choices=["loglik", "legacy"],
        default="loglik",
        help="loglik: likelihood fingerprint posterior (default); legacy: top-1+MRR argmax sweeps",
    )
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose progress output")
    args = parser.parse_args()

    pgn_path = Path(args.pgn) if args.pgn else Path("example2.pgn")
    result = estimate(
        pgn_path,
        n_sample=args.sample,
        model_name=args.model,
        quiet=args.quiet,
        scorer=args.scorer,
        max_positions=args.max_positions,
    )

    if args.json:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
