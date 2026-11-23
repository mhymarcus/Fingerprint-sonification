"""Synthetic ridge patterns with exact masks and orientation ground truth."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticFingerprint:
    image: np.ndarray
    roi_mask: np.ndarray
    ridge_mask: np.ndarray
    orientation_radians: np.ndarray


def generate_synthetic_fingerprint(
    size=128,
    angle_degrees=35.0,
    ridge_spacing=11.0,
    curvature=0.08,
    noise_sigma=3.0,
    seed=0,
):
    """Create deterministic curved ridges and pixel-level geometric labels."""
    if size < 32 or ridge_spacing < 4:
        raise ValueError("Synthetic size must be >= 32 and spacing >= 4")
    coordinates = np.linspace(-1.0, 1.0, size, dtype=np.float32)
    y, x = np.meshgrid(coordinates, coordinates, indexing="ij")
    angle = np.deg2rad(angle_degrees)
    along = x * np.cos(angle) + y * np.sin(angle)
    normal = -x * np.sin(angle) + y * np.cos(angle)
    phase_coordinate = normal + curvature * along**2
    period = ridge_spacing * 2.0 / size
    ridge_mask = np.cos(2.0 * np.pi * phase_coordinate / period) > 0.42
    roi_mask = (x / 0.82) ** 2 + (y / 0.96) ** 2 <= 1.0
    ridge_mask &= roi_mask

    normal_x = -np.sin(angle) + 2.0 * curvature * along * np.cos(angle)
    normal_y = np.cos(angle) + 2.0 * curvature * along * np.sin(angle)
    orientation = np.arctan2(-normal_x, normal_y) % np.pi
    orientation = np.where(roi_mask, orientation, np.nan).astype(np.float32)

    softened = cv2.GaussianBlur(ridge_mask.astype(np.float32), (3, 3), 0.55)
    image = 238.0 - 190.0 * softened
    generator = np.random.default_rng(seed)
    image += generator.normal(0.0, noise_sigma, image.shape)
    image[~roi_mask] = 255.0
    return SyntheticFingerprint(
        image=np.clip(image, 0, 255).astype(np.uint8),
        roi_mask=roi_mask,
        ridge_mask=ridge_mask,
        orientation_radians=orientation,
    )
