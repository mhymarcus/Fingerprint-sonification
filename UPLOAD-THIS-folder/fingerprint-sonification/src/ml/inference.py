"""Checkpoint-backed ML ridge segmentation and shared geometry extraction."""

from pathlib import Path
import warnings

import numpy as np

from src.geometry.analyzer import analyze_ridge_mask
from src.ml.model import CompactUNet, require_torch


def analyze_ml(
    processed_image,
    checkpoint_path,
    orientation_block_size=16,
    minimum_ridge_length=8.0,
    threshold=0.5,
    device="cpu",
):
    """Run genuine checkpoint inference; never falls back silently to CV."""
    require_torch()
    import torch

    checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if checkpoint_path is None or not checkpoint_path.is_file():
        raise FileNotFoundError(
            "ML analysis requires a valid trained checkpoint. "
            f"Checkpoint not found: {checkpoint_path}"
        )
    # weights_only=True keeps checkpoint loading from executing arbitrary code:
    # the checkpoint only ever contains tensors and plain scalars.
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if "model_state_dict" not in checkpoint:
        raise ValueError("Checkpoint does not contain model_state_dict")
    base_channels = int(checkpoint.get("base_channels", 24))
    model = CompactUNet(base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    image = np.asarray(processed_image.grayscale, dtype=np.float32)
    height, width = image.shape
    # The U-Net is fully convolutional, so it runs at the pipeline's analysis
    # resolution. Rescaling the input to the checkpoint's training size was
    # destroying ridge detail: at 128 px the ridges of a phone photo are only
    # ~2.4 px apart and even the CV baseline fails there.
    trained_size = int(checkpoint.get("image_size", 0) or 0)
    if trained_size and max(height, width) > 2 * trained_size:
        warnings.warn(
            f"Checkpoint was trained at {trained_size}px but analysis runs at "
            f"{max(height, width)}px; very different ridge scale can degrade "
            "the mask. Consider --analysis-size closer to the training size.",
            UserWarning,
        )
    tensor = torch.from_numpy(image[None, None]).to(device)
    padding_height = (-height) % 8
    padding_width = (-width) % 8
    if padding_height or padding_width:
        tensor = torch.nn.functional.pad(
            tensor, (0, padding_width, 0, padding_height), mode="reflect"
        )
    with torch.no_grad():
        probability = torch.sigmoid(model(tensor))[0, 0, :height, :width].cpu().numpy()
    ridge_mask = probability >= threshold
    ridge_mask &= processed_image.foreground_mask
    if not ridge_mask.any():
        raise ValueError("ML checkpoint predicted an empty ridge mask")
    return analyze_ridge_mask(
        processed_image,
        ridge_mask,
        mode="ml",
        orientation_block_size=orientation_block_size,
        minimum_ridge_length=minimum_ridge_length,
    )
