#!/usr/bin/env python3
"""Export a maia3 PyTorch checkpoint to ONNX format.

Usage:
    python scripts/export_onnx.py --model maia3-5m
    python scripts/export_onnx.py --model maia3-79m --output-dir chess_accuracy/maia3/onnx
"""

import argparse
from pathlib import Path
from types import SimpleNamespace

import torch

from chess_accuracy.maia3.model_registry import (
    resolve_checkpoint_path,
    resolve_model_spec,
)


def export(model_name: str = "maia3-5m", opset: int = 14, output_dir: str | None = None):
    """Export a maia3 model to ONNX."""
    from chess_accuracy.maia3.models import MAIA3Model

    spec = resolve_model_spec(model_name)
    print(f"Model: {spec.display_name} ({spec.name})")
    print(f"Config: dim_vit={spec.config['dim_vit']}, heads={spec.config['num_heads']}, blocks={spec.config['num_blocks']}")

    # Resolve checkpoint
    checkpoint_path = resolve_checkpoint_path(spec)
    print(f"Checkpoint: {checkpoint_path}")

    # Load model
    cfg = SimpleNamespace(**spec.config)
    model = MAIA3Model(cfg)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Handle legacy key naming (smolgen -> gab)
    cleaned = {}
    for k, v in state_dict.items():
        cleaned[k.replace("smolgen", "gab")] = v
    model.load_state_dict(cleaned)
    model.eval()
    print(f"Loaded checkpoint, {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Input dimensions
    time_info_dims = 4 if cfg.include_time_info else 1
    D_in = 12 * cfg.history + time_info_dims - 1
    print(f"Input: tokens(1, 64, {D_in}), self_elos(1,), oppo_elos(1,)")

    # Dummy inputs
    dummy_tokens = torch.zeros(1, 64, D_in, dtype=torch.float32)
    dummy_self_elo = torch.tensor([1500.0], dtype=torch.float32)
    dummy_oppo_elo = torch.tensor([1500.0], dtype=torch.float32)

    # Verify forward pass works
    with torch.no_grad():
        logits_move, logits_value, logits_ponder = model(dummy_tokens, dummy_self_elo, dummy_oppo_elo)
    print(f"Forward pass OK: logits_move={logits_move.shape}, logits_value={logits_value.shape}, logits_ponder={logits_ponder.shape}")

    # Export to ONNX
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "chess_accuracy" / "maia3" / "onnx"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = output_dir / f"{model_name}.onnx"
    print(f"Exporting to {onnx_path} (opset {opset})...")

    # Use legacy TorchScript tracer (dynamo-based exporter has issues with MHA view ops)
    torch.onnx.export(
        model,
        (dummy_tokens, dummy_self_elo, dummy_oppo_elo),
        str(onnx_path),
        opset_version=opset,
        input_names=["tokens", "self_elos", "oppo_elos"],
        output_names=["logits_move", "logits_value", "logits_ponder"],
        dynamic_axes={
            "tokens": {0: "batch"},
            "self_elos": {0: "batch"},
            "oppo_elos": {0: "batch"},
            "logits_move": {0: "batch"},
            "logits_value": {0: "batch"},
            "logits_ponder": {0: "batch"},
        },
        dynamo=False,
    )
    print(f"Exported to {onnx_path}")

    # Verify with onnx checker
    try:
        import onnx
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified OK")
    except ImportError:
        print("onnx package not installed, skipping verification")
    except Exception as e:
        print(f"ONNX verification failed: {e}")

    # Verify numerical correctness with onnxruntime
    try:
        import numpy as np
        import onnxruntime as ort

        session = ort.InferenceSession(str(onnx_path))
        ort_inputs = {
            "tokens": dummy_tokens.numpy(),
            "self_elos": dummy_self_elo.numpy(),
            "oppo_elos": dummy_oppo_elo.numpy(),
        }
        ort_logits_move, ort_logits_value, ort_logits_ponder = session.run(None, ort_inputs)

        # Compare
        pt_move = logits_move.detach().numpy()
        pt_value = logits_value.detach().numpy()
        pt_ponder = logits_ponder.detach().numpy()

        move_diff = np.max(np.abs(pt_move - ort_move)) if (ort_move := ort_logits_move) is not None else 0
        value_diff = np.max(np.abs(pt_value - ort_logits_value))
        ponder_diff = np.max(np.abs(pt_ponder - ort_logits_ponder))

        print(f"ONNX vs PyTorch max diff: move={move_diff:.2e}, value={value_diff:.2e}, ponder={ponder_diff:.2e}")
        if move_diff < 1e-4 and value_diff < 1e-4 and ponder_diff < 1e-4:
            print("Numerical correctness verified OK")
        else:
            print("WARNING: Numerical differences exceed tolerance")
    except ImportError:
        print("onnxruntime not installed, skipping numerical verification")

    return str(onnx_path)


def main():
    parser = argparse.ArgumentParser(description="Export maia3 model to ONNX")
    parser.add_argument("--model", default="maia3-5m", help="Model alias (maia3-5m, maia3-23m, maia3-79m)")
    parser.add_argument("--opset", type=int, default=14, help="ONNX opset version")
    parser.add_argument("--output-dir", default=None, help="Output directory for ONNX file")
    args = parser.parse_args()

    export(args.model, args.opset, args.output_dir)


if __name__ == "__main__":
    main()
