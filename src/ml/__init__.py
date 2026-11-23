"""Optional PyTorch segmentation components."""

from src.ml.inference import analyze_ml
from src.ml.model import CompactUNet, require_torch

__all__ = ["CompactUNet", "analyze_ml", "require_torch"]
