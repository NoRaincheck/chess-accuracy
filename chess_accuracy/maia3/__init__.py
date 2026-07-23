"""Vendored maia3 inference package (ONNX-ready)."""

__version__ = "0.1.0"

from .models import MAIA3Model
from .model_registry import (
    MODEL_SPECS,
    ModelSpec,
    resolve_checkpoint_path,
    resolve_model_spec,
)

__all__ = [
    "MAIA3Model",
    "ModelSpec",
    "MODEL_SPECS",
    "resolve_model_spec",
    "resolve_checkpoint_path",
]
