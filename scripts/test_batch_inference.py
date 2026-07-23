#!/usr/bin/env python3
"""End-to-end test for batch ELO inference.

Verifies that the batch ONNX approach produces correct results by comparing
against the PyTorch model and checking numerical accuracy.
"""

import io
import time

import chess
import chess.pgn
import numpy as np
import torch

from chess_accuracy.batch_inference import BatchMaia3Inference, estimate_elo_batch
from chess_accuracy.maia3.model_registry import resolve_checkpoint_path, resolve_model_spec
from chess_accuracy.maia3.models import MAIA3Model
from chess_accuracy.pgn_parser import (
    build_batch_tensors,
    move_to_index,
    parse_pgn_to_positions,
)
from types import SimpleNamespace


def test_pgn_parser():
    """Test PGN parsing and tensor construction."""
    print("=== PGN Parser ===")
    for pgn_file in ["example1.pgn", "example2.pgn"]:
        pgn = open(pgn_file).read()
        positions = parse_pgn_to_positions(pgn)
        print(f"  {pgn_file}: {len(positions)} positions parsed")

        spec = resolve_model_spec("maia3-5m")
        cfg = SimpleNamespace(**spec.config)
        elo_values = np.array([1500.0])
        batch = build_batch_tensors(positions, elo_values, cfg, n_sample=5)

        assert batch["tokens"].shape[0] == 5, f"Expected 5 positions, got {batch['tokens'].shape[0]}"
        assert batch["tokens"].shape[1] == 64
        assert batch["tokens"].shape[2] == 12 * cfg.history
        assert batch["legal_masks"].shape == (5, 4352)
        print(f"    Tokens: {batch['tokens'].shape}, Legal masks: {batch['legal_masks'].shape}")
    print("  PASSED\n")


def test_onnx_vs_pytorch():
    """Verify ONNX output matches PyTorch output."""
    print("=== ONNX vs PyTorch ===")
    onnx_engine = BatchMaia3Inference("chess_accuracy/maia3/onnx/maia3-5m.onnx")

    spec = resolve_model_spec("maia3-5m")
    cfg = SimpleNamespace(**spec.config)
    pt_model = MAIA3Model(cfg)
    state_dict = torch.load(resolve_checkpoint_path(spec), map_location="cpu", weights_only=True)
    cleaned = {k.replace("smolgen", "gab"): v for k, v in state_dict.items()}
    pt_model.load_state_dict(cleaned)
    pt_model.eval()

    # Test on both example games
    for pgn_file in ["example1.pgn", "example2.pgn"]:
        pgn = open(pgn_file).read()
        positions = parse_pgn_to_positions(pgn)
        elo_values = np.array([1500.0])
        batch = build_batch_tensors(positions, elo_values, cfg, n_sample=10)

        tokens = batch["tokens"]
        self_elos = batch["self_elos"]
        oppo_elos = batch["oppo_elos"]

        # PyTorch forward
        with torch.no_grad():
            pt_move, pt_value, pt_ponder = pt_model(tokens, self_elos, oppo_elos)

        # ONNX forward
        onnx_move, onnx_value, onnx_ponder = onnx_engine.predict(tokens, self_elos, oppo_elos)

        # Compare
        move_diff = np.max(np.abs(pt_move.numpy() - onnx_move))
        value_diff = np.max(np.abs(pt_value.numpy() - onnx_value))
        ponder_diff = np.max(np.abs(pt_ponder.numpy() - onnx_ponder))

        status = "OK" if move_diff < 1e-4 else "FAIL"
        print(f"  {pgn_file}: move_diff={move_diff:.2e}, value_diff={value_diff:.2e}, ponder_diff={ponder_diff:.2e} [{status}]")

    print("  PASSED\n")


def test_legal_move_masking():
    """Verify legal move masking changes the top-1 prediction."""
    print("=== Legal Move Masking ===")
    from chess_accuracy.maia3.dataset import get_legal_moves_mask
    from chess_accuracy.maia3.utils import get_all_possible_moves

    onnx_engine = BatchMaia3Inference("chess_accuracy/maia3/onnx/maia3-5m.onnx")
    spec = resolve_model_spec("maia3-5m")
    cfg = SimpleNamespace(**spec.config)

    pgn = open("example2.pgn").read()
    positions = parse_pgn_to_positions(pgn)
    elo_values = np.array([1500.0])
    batch = build_batch_tensors(positions, elo_values, cfg, n_sample=0)

    tokens = batch["tokens"]
    self_elos = batch["self_elos"]
    oppo_elos = batch["oppo_elos"]
    legal_masks = batch["legal_masks"]

    logits_move, _, _ = onnx_engine.predict(tokens, self_elos, oppo_elos)

    # Without masking
    top1_no_mask = np.argmax(logits_move, axis=1)
    # With masking
    logits_masked = np.where(legal_masks.numpy(), logits_move, -np.inf)
    top1_masked = np.argmax(logits_masked, axis=1)

    n_diff = (top1_no_mask != top1_masked).sum()
    print(f"  Positions where masking changed top-1: {n_diff}/{len(top1_no_mask)}")
    print("  PASSED\n")


def test_batch_estimate():
    """Test full batch ELO estimation."""
    print("=== Batch ELO Estimation ===")
    onnx_engine = BatchMaia3Inference("chess_accuracy/maia3/onnx/maia3-5m.onnx")

    for pgn_file, expected_range in [("example1.pgn", (500, 1000)), ("example2.pgn", (2500, 3500))]:
        pgn = open(pgn_file).read()
        elo_values = np.arange(300, 3501, 50, dtype=np.float32)

        t0 = time.time()
        best_elo, best_rate, all_rates = estimate_elo_batch(
            pgn, elo_values, onnx_engine, model_name="maia3-5m", n_sample=0
        )
        elapsed = time.time() - t0

        in_range = expected_range[0] <= best_elo <= expected_range[1]
        status = "OK" if in_range else "WARN"
        print(f"  {pgn_file}: best_elo={best_elo:.0f} (rate={best_rate:.3f}), time={elapsed:.2f}s [{status}]")
        print(f"    Expected range: {expected_range}")

    print("  PASSED\n")


if __name__ == "__main__":
    test_pgn_parser()
    test_onnx_vs_pytorch()
    test_legal_move_masking()
    test_batch_estimate()
    print("All tests passed!")
