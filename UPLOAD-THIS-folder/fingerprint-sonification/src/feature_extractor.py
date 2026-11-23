"""Deterministic pixel-feature extraction using a two-dimensional snake scan."""

from dataclasses import dataclass
import math
import warnings

import numpy as np
from scipy import ndimage

from src.image_processor import ProcessedImage


@dataclass(frozen=True)
class FeatureSequence:
    scan_position: np.ndarray
    x_position: np.ndarray
    y_position: np.ndarray
    ridge_darkness: np.ndarray
    local_contrast: np.ndarray
    ridge_density: np.ndarray
    foreground_coverage: np.ndarray

    def __len__(self):
        return len(self.scan_position)


def _scan_regions(height, width, num_segments):
    """Yield exactly num_segments cells in spatially continuous snake order."""
    rows = max(1, round(math.sqrt(num_segments * height / width)))
    rows = min(rows, num_segments)
    base_columns, extra_columns = divmod(num_segments, rows)
    row_edges = np.linspace(0, height, rows + 1, dtype=int)

    emitted = 0
    for row_index in range(rows):
        columns = base_columns + (1 if row_index < extra_columns else 0)
        if columns == 0:
            continue
        column_edges = np.linspace(0, width, columns + 1, dtype=int)
        column_indices = range(columns)
        if row_index % 2:
            column_indices = reversed(range(columns))
        for column_index in column_indices:
            yield (
                row_edges[row_index],
                row_edges[row_index + 1],
                column_edges[column_index],
                column_edges[column_index + 1],
            )
            emitted += 1
    if emitted != num_segments:
        raise RuntimeError("Internal scan-grid error")


def extract_features(processed_image: ProcessedImage, num_segments=144):
    """Extract brightness, contrast, and ridge density in snake-scan order.

    A two-dimensional snake scan was selected instead of simple full-height
    columns because it preserves local vertical structure while keeping
    consecutive sound events spatially adjacent.
    """
    if num_segments < 4:
        raise ValueError("num_segments must be at least 4")

    image = np.asarray(processed_image.grayscale, dtype=np.float32)
    mask = np.asarray(processed_image.foreground_mask, dtype=bool)
    if image.shape != mask.shape:
        raise ValueError("Processed grayscale image and foreground mask differ in shape")

    gradient_x = ndimage.sobel(image, axis=1, mode="reflect")
    gradient_y = ndimage.sobel(image, axis=0, mode="reflect")
    gradient_magnitude = np.hypot(gradient_x, gradient_y)
    active_gradients = gradient_magnitude[mask]
    gradient_scale = float(np.percentile(active_gradients, 95)) if active_gradients.size else 0
    if gradient_scale > np.finfo(np.float32).eps:
        normalized_gradient = np.clip(gradient_magnitude / gradient_scale, 0.0, 1.0)
    else:
        normalized_gradient = np.zeros_like(gradient_magnitude)

    height, width = image.shape
    features = {
        "x_position": [],
        "y_position": [],
        "ridge_darkness": [],
        "local_contrast": [],
        "ridge_density": [],
        "foreground_coverage": [],
    }
    regions = list(_scan_regions(height, width, num_segments))
    for top, bottom, left, right in regions:
        image_block = image[top:bottom, left:right]
        mask_block = mask[top:bottom, left:right]
        gradient_block = normalized_gradient[top:bottom, left:right]
        coverage = float(mask_block.mean()) if mask_block.size else 0.0

        if mask_block.any():
            pixels = image_block[mask_block]
            darkness = float(np.clip(1.0 - pixels.mean(), 0.0, 1.0))
            contrast = float(np.clip(pixels.std() / 0.25, 0.0, 1.0))
            ridge_density = float(np.clip(gradient_block[mask_block].mean(), 0.0, 1.0))
        else:
            darkness = 0.0
            contrast = 0.0
            ridge_density = 0.0

        features["x_position"].append(((left + right) * 0.5) / width)
        features["y_position"].append(((top + bottom) * 0.5) / height)
        features["ridge_darkness"].append(darkness)
        features["local_contrast"].append(contrast)
        features["ridge_density"].append(ridge_density)
        features["foreground_coverage"].append(coverage)

    sequence = FeatureSequence(
        scan_position=np.linspace(0.0, 1.0, num_segments, dtype=np.float32),
        **{
            name: np.asarray(values, dtype=np.float32)
            for name, values in features.items()
        },
    )
    activity = sequence.local_contrast.mean() + sequence.ridge_density.mean()
    if activity < 0.02:
        warnings.warn(
            "The image contains very little usable ridge variation; audio may be simple",
            UserWarning,
        )
    return sequence
