"""Batch ELO inference using ONNX runtime for maia3.

Provides fast ELO estimation by running a single forward pass over
N positions × M ELO values, instead of M separate inference calls.

Two scoring families are provided:
- Rank scores (`_compute_score`): blended top-1 accuracy + MRR (legacy).
- Log-likelihood fingerprints (`position_loglik`, `loglik_1d_sweep`,
  `loglik_2d_grid`): per-position log P(human move | model at elo e).
  The summed curve L(e) is smooth in e, so a posterior
  P(e | game) ∝ exp(L(e)) · prior(e) gives sub-grid estimates and
  credible intervals.
"""

import math

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import torch
from numpy import ndarray

from .maia3.model_registry import resolve_model_spec
from .pgn_parser import (
    build_batch_tensors,
    build_batch_tensors_2d,
    parse_pgn_to_positions,
)


class BatchMaia3Inference:
    """ONNX runtime inference engine for batched maia3 predictions."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            onnx_path,
            providers=["CPUExecutionProvider"],
        )
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def predict(
        self, tokens: torch.Tensor, self_elos: torch.Tensor, oppo_elos: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run ONNX inference.

        Returns (logits_move, logits_value, logits_ponder) as numpy arrays.
        """
        ort_inputs = {
            "tokens": tokens.numpy() if isinstance(tokens, torch.Tensor) else tokens,
            "self_elos": self_elos.numpy() if isinstance(self_elos, torch.Tensor) else self_elos,
            "oppo_elos": oppo_elos.numpy() if isinstance(oppo_elos, torch.Tensor) else oppo_elos,
        }
        return cast(tuple[ndarray, ndarray, ndarray], self.session.run(None, ort_inputs))


def load_inference_engine(model_name: str = "maia3-5m") -> BatchMaia3Inference:
    """Load a maia3 ONNX model, auto-downloading if needed."""
    # Check for pre-exported ONNX file
    onnx_dir = Path(__file__).parent / "maia3" / "onnx"
    onnx_path = onnx_dir / f"{model_name}.onnx"

    if not onnx_path.exists():
        print(f"ONNX model not found at {onnx_path}, exporting...")
        from scripts.export_onnx import export

        export(model_name, output_dir=str(onnx_dir))

    return BatchMaia3Inference(str(onnx_path))


def _compute_score(logits_masked, human_moves, n_pos, alpha=0.6):
    """Compute blended top-1 accuracy + MRR score.

    score = alpha * top1_accuracy + (1 - alpha) * mean_reciprocal_rank

    Args:
        logits_masked: (N, M, V) masked logits for legal moves
        human_moves: (N,) human move indices
        n_pos: number of positions
        alpha: weight on top-1 vs MRR (0 = pure MRR, 1 = pure top-1)

    Returns:
        (M,) score per ELO value
    """
    if n_pos == 0:
        return np.zeros(logits_masked.shape[1], dtype=np.float64)

    n_elo = logits_masked.shape[1]
    pos_idx = np.arange(n_pos)[:, None]  # (N, 1)
    elo_idx = np.arange(n_elo)[None, :]  # (1, M)

    # Logit of the human move for each (position, ELO): (N, M)
    human_logits = logits_masked[pos_idx, elo_idx, human_moves[:, None]]

    # Top-1 accuracy: (M,)
    top1_moves = logits_masked.argmax(axis=2)  # (N, M)
    top1_acc = (top1_moves == human_moves[:, None]).mean(axis=0)

    # Mean reciprocal rank: rank of human move among legal moves
    rank = (logits_masked >= human_logits[:, :, None]).sum(axis=2) + 1  # (N, M)
    mrr = (1.0 / rank).mean(axis=0)

    return alpha * top1_acc + (1.0 - alpha) * mrr


def _log_softmax_masked(logits_masked: np.ndarray) -> np.ndarray:
    """Log-softmax over the last axis, treating -inf entries as illegal."""
    shifted = logits_masked - logits_masked.max(axis=-1, keepdims=True)
    log_z = np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
    return shifted - log_z


def position_loglik(logits_masked: np.ndarray, human_moves: np.ndarray, n_pos: int) -> np.ndarray:
    """Per-position log-likelihood of the human move.

    Args:
        logits_masked: (N, M, V) logits with illegal moves set to -inf.
        human_moves: (N,) human move indices.
        n_pos: number of positions (N).

    Returns:
        (N, M) matrix of log P(human_move | elo).
    """
    if n_pos == 0:
        return np.zeros((0, logits_masked.shape[1]), dtype=np.float64)

    logp = _log_softmax_masked(logits_masked)
    n_elos = logits_masked.shape[1]
    pos_idx = np.arange(n_pos)[:, None]
    elo_idx = np.arange(n_elos)[None, :]
    return logp[pos_idx, elo_idx, human_moves[:, None]]


def elo_log_prior(elo_values: np.ndarray, mean: float = 1500.0, std: float = 350.0) -> np.ndarray:
    """Unnormalized log of a truncated-Gaussian population prior over ELO."""
    z = (np.asarray(elo_values, dtype=np.float64) - mean) / std
    return -0.5 * z * z


def _marginal_stats(posterior: np.ndarray, values: np.ndarray) -> tuple[float, float, tuple[float, float]]:
    """Posterior mean, std, and central 95% credible interval over a 1D grid."""
    mean = float((posterior * values).sum())
    var = float((posterior * (values - mean) ** 2).sum())
    std = math.sqrt(max(var, 0.0))
    cdf = np.cumsum(posterior)
    lo = float(np.interp(0.025, cdf, values))
    hi = float(np.interp(0.975, cdf, values))
    return mean, std, (lo, hi)


def curve_posterior(
    curve: np.ndarray,
    elo_values: np.ndarray,
    prior_mean: float = 1500.0,
    prior_std: float = 350.0,
) -> dict:
    """Turn a 1D log-likelihood curve into a normalized posterior summary.

    P(e | game) ∝ exp(curve(e)) · prior(e)

    Returns dict with keys: posterior, mean, std, ci95.
    """
    log_post = curve.astype(np.float64) + elo_log_prior(elo_values, prior_mean, prior_std)
    log_post -= log_post.max()
    posterior = np.exp(log_post)
    posterior /= posterior.sum()
    mean, std, ci95 = _marginal_stats(posterior, elo_values.astype(np.float64))
    return {"posterior": posterior, "mean": mean, "std": std, "ci95": ci95}


def loglik_1d_sweep(
    positions: list[dict],
    elo_values: np.ndarray,
    cfg,
    inference_engine: "BatchMaia3Inference",
    indices: list[int] | None = None,
) -> dict:
    """Sweep candidate ELOs with self = opponent = e, scoring by log-likelihood.

    Returns dict with:
        - ll: (N, M) per-position log-likelihoods
        - curve: (M,) summed log-likelihood per ELO
        - top1_counts: (M,) number of positions where the model's top-1
          matches the human move (for reporting a familiar "match rate")
        - n_pos: number of positions evaluated
    """
    batch = build_batch_tensors(positions, elo_values, cfg, n_sample=0, indices=indices)
    n_pos = batch["n_positions"]
    n_elos = batch["n_elos"]
    if n_pos == 0:
        zeros_m = np.zeros(n_elos, dtype=np.float64)
        return {"ll": np.zeros((0, n_elos)), "curve": zeros_m, "top1_counts": zeros_m, "n_pos": 0}

    logits_move, _, _ = inference_engine.predict(batch["tokens"], batch["self_elos"], batch["oppo_elos"])
    logits = logits_move.reshape(n_pos, n_elos, -1)
    masked = np.where(batch["legal_masks"].numpy()[:, np.newaxis, :], logits, -np.inf)

    ll = position_loglik(masked, batch["human_moves"], n_pos)
    top1 = masked.argmax(axis=2) == batch["human_moves"][:, None]
    return {
        "ll": ll,
        "curve": ll.sum(axis=0),
        "top1_counts": top1.sum(axis=0).astype(np.float64),
        "n_pos": n_pos,
    }


def loglik_2d_grid(
    positions: list[dict],
    white_elo_values: np.ndarray,
    black_elo_values: np.ndarray,
    cfg,
    inference_engine: "BatchMaia3Inference",
    indices: list[int] | None = None,
    chunk_combos: int = 64,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Joint (white, black) log-likelihood surface, computed in chunks.

    White-turn positions score self=white_elo vs oppo=black_elo and vice
    versa. Chunking bounds peak memory to roughly chunk_combos grid cells
    worth of tokens/logits at a time.

    Returns (surface, match_counts, n_pos):
        - surface: (W, B) summed log-likelihood per grid cell
        - match_counts: (W, B) top-1 match counts per grid cell
        - n_pos: number of positions evaluated
    """
    n_w = len(white_elo_values)
    n_b = len(black_elo_values)
    surface = np.zeros((n_w, n_b), dtype=np.float64)
    match_counts = np.zeros((n_w, n_b), dtype=np.float64)
    n_pos_total = 0

    w_block = max(1, chunk_combos // max(n_b, 1))
    for w_start in range(0, n_w, w_block):
        w_chunk = white_elo_values[w_start : w_start + w_block]
        batch = build_batch_tensors_2d(positions, w_chunk, black_elo_values, cfg, n_sample=0, indices=indices)
        n_pos = batch["n_positions"]
        n_pos_total = n_pos
        if n_pos == 0:
            continue

        logits_move, _, _ = inference_engine.predict(batch["tokens"], batch["self_elos"], batch["oppo_elos"])
        logits = logits_move.reshape(n_pos, len(w_chunk) * n_b, -1)
        masked = np.where(batch["legal_masks"].numpy()[:, np.newaxis, :], logits, -np.inf)

        ll = position_loglik(masked, batch["human_moves"], n_pos)
        top1 = masked.argmax(axis=2) == batch["human_moves"][:, None]

        surface[w_start : w_start + len(w_chunk), :] = ll.sum(axis=0).reshape(len(w_chunk), n_b)
        match_counts[w_start : w_start + len(w_chunk), :] = (
            top1.sum(axis=0).astype(np.float64).reshape(len(w_chunk), n_b)
        )

    return surface, match_counts, n_pos_total


def bicubic_upsample_surface(surface: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bicubically upsample a (W, B) log-likelihood surface onto a denser grid.

    Both the source and target grids are assumed to be uniform and to span the
    same ELO range, so corners map to corners (align_corners=True). The
    likelihood surface is smooth in (w, b), which makes bicubic reconstruction
    from a sparse anchor grid accurate to well under a grid cell.

    Used only for mode *localization*; reported statistics always come from
    real model evaluations.
    """
    t = torch.from_numpy(np.ascontiguousarray(surface, dtype=np.float32))[None, None]
    out = torch.nn.functional.interpolate(t, size=target_shape, mode="bicubic", align_corners=True)
    return out[0, 0].numpy().astype(np.float64)


def joint_posterior_2d(
    surface: np.ndarray,
    white_elo_values: np.ndarray,
    black_elo_values: np.ndarray,
    prior_mean: float = 1500.0,
    prior_std: float = 350.0,
) -> np.ndarray:
    """Normalize a 2D log-likelihood surface into P(w, b | game) ∝ exp(L)·prior(w)·prior(b)."""
    log_post = (
        surface.astype(np.float64)
        + elo_log_prior(white_elo_values, prior_mean, prior_std)[:, np.newaxis]
        + elo_log_prior(black_elo_values, prior_mean, prior_std)[np.newaxis, :]
    )
    log_post -= log_post.max()
    posterior = np.exp(log_post)
    posterior /= posterior.sum()
    return posterior


def estimate_elo_batch(
    pgn_text: str,
    elo_values: np.ndarray,
    inference_engine: BatchMaia3Inference,
    model_name: str = "maia3-5m",
    n_sample: int = 0,
    alpha: float = 0.6,
) -> tuple[float, float, np.ndarray]:
    """Estimate ELO for a PGN game using batch inference over a range of ELO values.

    Args:
        pgn_text: PGN text of the game
        elo_values: array of ELO values to evaluate (shape (M,))
        inference_engine: loaded BatchMaia3Inference
        model_name: model alias for config lookup
        n_sample: number of positions to sample (0 = all)
        alpha: blend weight for top-1 vs MRR (0.6 default)

    Returns:
        (best_elo, best_rate, all_rates)
        - best_elo: ELO with highest score
        - best_rate: the peak score
        - all_rates: array of scores, one per ELO value (shape (M,))
    """
    # Get model config
    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    # Parse PGN and build batch tensors
    positions = parse_pgn_to_positions(pgn_text)
    if not positions:
        return 0.0, 0.0, np.zeros(len(elo_values))

    batch = build_batch_tensors(positions, elo_values, cfg, n_sample=n_sample)

    tokens = batch["tokens"]
    self_elos = batch["self_elos"]
    oppo_elos = batch["oppo_elos"]
    human_moves = batch["human_moves"]
    legal_masks = batch["legal_masks"]  # (N, 4352)
    n_pos = batch["n_positions"]
    n_elo = batch["n_elos"]

    # Run single forward pass
    logits_move, _logits_value, _logits_ponder = inference_engine.predict(tokens, self_elos, oppo_elos)

    # logits_move: (N*M, 4352) -> reshape to (N, M, 4352)
    logits_move = logits_move.reshape(n_pos, n_elo, -1)

    # Apply legal move masking: set illegal move logits to -inf
    legal_masks_np = legal_masks.numpy()  # (N, 4352)
    # Broadcast: (N, 1, 4352) to mask all ELOs at once
    legal_masks_expanded = legal_masks_np[:, np.newaxis, :]  # (N, 1, 4352)
    logits_masked = np.where(legal_masks_expanded, logits_move, -np.inf)

    # Vectorized score: top-1 + MRR ensemble
    all_rates = _compute_score(logits_masked, human_moves, n_pos, alpha=alpha)

    best_idx = np.argmax(all_rates)
    best_elo = float(elo_values[best_idx])
    best_rate = float(all_rates[best_idx])

    return best_elo, best_rate, all_rates


def estimate_elo_2d(
    pgn_text: str,
    white_elo_values: np.ndarray,
    black_elo_values: np.ndarray,
    inference_engine: BatchMaia3Inference,
    model_name: str = "maia3-5m",
    n_sample: int = 0,
) -> tuple[float, float, float, np.ndarray]:
    """Estimate both white and black ELO using 2D grid search.

    Args:
        pgn_text: PGN text of the game
        white_elo_values: array of white ELO values to try (shape (W,))
        black_elo_values: array of black ELO values to try (shape (B,))
        inference_engine: loaded BatchMaia3Inference
        model_name: model alias for config lookup
        n_sample: number of positions to sample (0 = all)

    Returns:
        (best_white_elo, best_black_elo, best_rate, rate_grid)
        - best_white_elo: white ELO with highest match rate
        - best_black_elo: black ELO with highest match rate
        - best_rate: the peak match rate
        - rate_grid: (W, B) array of match rates
    """
    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    positions = parse_pgn_to_positions(pgn_text)
    if not positions:
        return 0.0, 0.0, 0.0, np.zeros((len(white_elo_values), len(black_elo_values)))

    batch = build_batch_tensors_2d(positions, white_elo_values, black_elo_values, cfg, n_sample=n_sample)

    tokens = batch["tokens"]
    self_elos = batch["self_elos"]
    oppo_elos = batch["oppo_elos"]
    human_moves = batch["human_moves"]
    legal_masks = batch["legal_masks"]
    n_pos = batch["n_positions"]
    n_w = batch["n_white_elo"]
    n_b = batch["n_black_elo"]

    # Run single forward pass
    logits_move, _, _ = inference_engine.predict(tokens, self_elos, oppo_elos)

    # Reshape: (N, W*B, 4352)
    logits_move = logits_move.reshape(n_pos, n_w * n_b, -1)

    # Apply legal move masking
    legal_masks_np = legal_masks.numpy()  # (N, 4352)
    legal_masks_expanded = legal_masks_np[:, np.newaxis, :]  # (N, 1, 4352)
    logits_masked = np.where(legal_masks_expanded, logits_move, -np.inf)

    # Compute match rate for each (W, B) pair
    # top1 for each position × each grid cell: (N, W*B)
    top1_moves = np.argmax(logits_masked, axis=2)  # (N, W*B)
    matches = top1_moves == human_moves[:, np.newaxis]  # (N, W*B)

    # Sum matches per grid cell, then reshape to (W, B)
    match_counts = matches.sum(axis=0)  # (W*B,)
    rate_grid = (match_counts / n_pos).reshape(n_w, n_b) if n_pos > 0 else np.zeros((n_w, n_b))

    # Find best
    best_idx = np.unravel_index(np.argmax(rate_grid), rate_grid.shape)
    best_white_elo = float(white_elo_values[best_idx[0]])
    best_black_elo = float(black_elo_values[best_idx[1]])
    best_rate = float(rate_grid[best_idx])

    return best_white_elo, best_black_elo, best_rate, rate_grid


def _eval_2d_grid(
    pgn_text: str,
    white_elo_values: np.ndarray,
    black_elo_values: np.ndarray,
    positions,
    cfg,
    inference_engine: BatchMaia3Inference,
    n_sample: int = 0,
    alpha: float = 0.6,
) -> np.ndarray:
    """Evaluate a 2D ELO grid, returning rate_grid (W, B)."""
    batch = build_batch_tensors_2d(positions, white_elo_values, black_elo_values, cfg, n_sample=n_sample)
    tokens = batch["tokens"]
    self_elos = batch["self_elos"]
    oppo_elos = batch["oppo_elos"]
    human_moves = batch["human_moves"]
    legal_masks = batch["legal_masks"]
    n_pos = batch["n_positions"]
    n_w = batch["n_white_elo"]
    n_b = batch["n_black_elo"]

    logits_move, _, _ = inference_engine.predict(tokens, self_elos, oppo_elos)
    logits_move = logits_move.reshape(n_pos, n_w * n_b, -1)

    legal_masks_np = legal_masks.numpy()[:, np.newaxis, :]
    logits_masked = np.where(legal_masks_np, logits_move, -np.inf)

    all_rates = _compute_score(logits_masked, human_moves, n_pos, alpha=alpha)
    return all_rates.reshape(n_w, n_b)


def estimate_elo_2d_halving(
    pgn_text: str,
    inference_engine: BatchMaia3Inference,
    model_name: str = "maia3-5m",
    elo_lo: float = 300,
    elo_hi: float = 3500,
    n_sample: int = 0,
    n_rounds: int = 3,
    keep_fraction: float = 0.25,
    initial_step: float = 400,
    min_step: float = 50,
    verbose: bool = True,
) -> tuple[float, float, float]:
    """Estimate white & black ELO via successive halving.

    Round 0: coarse grid over full range (step=initial_step)
    Round 1..n_rounds: keep top keep_fraction of candidates, refine around them
                       with step halved each round, until step < min_step.

    Returns (best_white_elo, best_black_elo, best_rate).
    """
    spec = resolve_model_spec(model_name)
    cfg = SimpleNamespace(**spec.config)

    positions = parse_pgn_to_positions(pgn_text)
    if not positions:
        return 0.0, 0.0, 0.0

    step = initial_step
    best_w = (elo_lo + elo_hi) / 2
    best_b = (elo_lo + elo_hi) / 2
    best_rate = 0.0

    for rd in range(n_rounds + 1):
        # Build grid around current best
        half_range = step * 2  # look ±2 steps around best
        w_lo = max(elo_lo, best_w - half_range)
        w_hi = min(elo_hi, best_w + half_range)
        b_lo = max(elo_lo, best_b - half_range)
        b_hi = min(elo_hi, best_b + half_range)

        white_elos = np.arange(w_lo, w_hi + 0.1, step, dtype=np.float32)
        black_elos = np.arange(b_lo, b_hi + 0.1, step, dtype=np.float32)

        n_combos = len(white_elos) * len(black_elos)
        if verbose:
            print(
                f"  Round {rd}: grid {len(white_elos)}x{len(black_elos)}="
                f"{n_combos} (step={step:.0f}, range W=[{w_lo:.0f}-{w_hi:.0f}] "
                f"B=[{b_lo:.0f}-{b_hi:.0f}])"
            )

        rate_grid = _eval_2d_grid(
            pgn_text,
            white_elos,
            black_elos,
            positions,
            cfg,
            inference_engine,
            n_sample=n_sample,
        )

        # Find best in this grid
        best_idx = np.unravel_index(np.argmax(rate_grid), rate_grid.shape)
        cand_w = float(white_elos[best_idx[0]])
        cand_b = float(black_elos[best_idx[1]])
        cand_rate = float(rate_grid[best_idx])

        if cand_rate >= best_rate:
            best_w, best_b, best_rate = cand_w, cand_b, cand_rate

        if verbose:
            print(f"    -> best: W={best_w:.0f} B={best_b:.0f} rate={best_rate:.4f}")

        # Halve step for next round
        step = max(step / 2, min_step)

    return best_w, best_b, best_rate
