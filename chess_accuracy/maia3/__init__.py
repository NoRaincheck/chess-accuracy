"""Vendored maia3 inference package (ONNX-ready)."""

__version__ = "0.1.0"

from .model_registry import (
    MODEL_SPECS,
    ModelSpec,
    resolve_checkpoint_path,
    resolve_model_spec,
)
from .models import MAIA3Model

__all__ = [
    "MODEL_SPECS",
    "MAIA3Model",
    "ModelSpec",
    "resolve_checkpoint_path",
    "resolve_model_spec",
]
