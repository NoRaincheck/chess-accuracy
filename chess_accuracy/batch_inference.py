"""Batch ELO inference using ONNX runtime for maia3.

Provides fast ELO estimation by running a single forward pass over
N positions × M ELO values, instead of M separate inference calls.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from .maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
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
            "self_elos": self_elos.numpy()
            if isinstance(self_elos, torch.Tensor)
            else self_elos,
            "oppo_elos": oppo_elos.numpy()
            if isinstance(oppo_elos, torch.Tensor)
            else oppo_elos,
        }
        return self.session.run(None, ort_inputs)


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


def estimate_elo_batch(
    pgn_text: str,
    elo_values: np.ndarray,
    inference_engine: BatchMaia3Inference,
    model_name: str = "maia3-5m",
    n_sample: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Estimate ELO for a PGN game using batch inference over a range of ELO values.

    Args:
        pgn_text: PGN text of the game
        elo_values: array of ELO values to evaluate (shape (M,))
        inference_engine: loaded BatchMaia3Inference
        model_name: model alias for config lookup
        n_sample: number of positions to sample (0 = all)

    Returns:
        (best_elo, best_rate, all_rates)
        - best_elo: ELO with highest match rate
        - best_rate: the peak match rate
        - all_rates: array of match rates, one per ELO value (shape (M,))
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
    logits_move, logits_value, logits_ponder = inference_engine.predict(
        tokens, self_elos, oppo_elos
    )

    # logits_move: (N*M, 4352) -> reshape to (N, M, 4352)
    logits_move = logits_move.reshape(n_pos, n_elo, -1)

    # Apply legal move masking: set illegal move logits to -inf
    legal_masks_np = legal_masks.numpy()  # (N, 4352)
    # Broadcast: (N, 1, 4352) to mask all ELOs at once
    legal_masks_expanded = legal_masks_np[:, np.newaxis, :]  # (N, 1, 4352)
    logits_masked = np.where(legal_masks_expanded, logits_move, -np.inf)

    # For each ELO, compute match rate: top-1 legal prediction vs human's actual move
    all_rates = np.zeros(n_elo, dtype=np.float64)
    for elo_idx in range(n_elo):
        top1_moves = np.argmax(logits_masked[:, elo_idx, :], axis=1)  # (N,)
        matches = (top1_moves == human_moves).sum()
        all_rates[elo_idx] = matches / n_pos if n_pos > 0 else 0.0

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

    batch = build_batch_tensors_2d(
        positions, white_elo_values, black_elo_values, cfg, n_sample=0
    )

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
    rate_grid = (
        (match_counts / n_pos).reshape(n_w, n_b) if n_pos > 0 else np.zeros((n_w, n_b))
    )

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
) -> np.ndarray:
    """Evaluate a 2D ELO grid, returning rate_grid (W, B)."""
    batch = build_batch_tensors_2d(
        positions, white_elo_values, black_elo_values, cfg, n_sample=n_sample
    )
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

    top1_moves = np.argmax(logits_masked, axis=2)  # (N, W*B)
    matches = top1_moves == human_moves[:, np.newaxis]  # (N, W*B)
    match_counts = matches.sum(axis=0)  # (W*B,)
    rate_grid = (
        (match_counts / n_pos).reshape(n_w, n_b) if n_pos > 0 else np.zeros((n_w, n_b))
    )
    return rate_grid


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
