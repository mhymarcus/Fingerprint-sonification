"""Traditional-CV and shared ML-assisted fingerprint geometry analysis."""

from dataclasses import dataclass

import cv2
import numpy as np
from skimage.morphology import remove_small_objects, skeletonize

from src.geometry.tracing import prune_skeleton_spurs, trace_ridge_segments
from src.image_processor import ProcessedImage
from utils import orientation


@dataclass(frozen=True)
class GeometryAnalysis:
    mode: str
    processed_image: ProcessedImage
    enhanced: np.ndarray
    roi_mask: np.ndarray
    ridge_mask: np.ndarray
    skeleton: np.ndarray
    orientation_angles: np.ndarray
    orientation_coherence: np.ndarray
    orientation_image: np.ndarray
    density_map: np.ndarray
    segments: tuple
    endpoints: tuple
    bifurcations: tuple


def enhance_fingerprint(grayscale, roi_mask):
    """Improve local contrast without synthesizing new ridge structures."""
    image_u8 = np.round(np.clip(grayscale, 0.0, 1.0) * 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=1.6, tileGridSize=(8, 8))
    enhanced = clahe.apply(image_u8)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0.55)
    enhanced = enhanced.copy()
    enhanced[~roi_mask] = 255
    return enhanced


def segment_ridges_cv(enhanced, roi_mask):
    """Return a ridge-level mask; this is distinct from the foreground ROI."""
    block_size = max(15, ((min(enhanced.shape) // 12) | 1))
    ridge_mask = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size,
        5,
    ) > 0
    ridge_mask &= roi_mask
    minimum_object_size = max(6, enhanced.size // 9000)
    ridge_mask = remove_small_objects(ridge_mask, min_size=minimum_object_size)
    return ridge_mask


def _roi_bounds(roi_mask, shape):
    """Inclusive pixel bounds of the fingerprint ROI, or the full frame."""
    rows, columns = np.nonzero(roi_mask)
    if not rows.size:
        return (0, 0, shape[1] - 1, shape[0] - 1)
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def analyze_ridge_mask(
    processed_image,
    ridge_mask,
    mode="cv",
    orientation_block_size=16,
    minimum_ridge_length=8.0,
    spur_length=6,
):
    """Run shared geometry extraction from a CV or genuine ML ridge mask."""
    roi_mask = np.asarray(processed_image.foreground_mask, dtype=bool)
    ridge_mask = np.asarray(ridge_mask, dtype=bool)
    if ridge_mask.shape != roi_mask.shape:
        raise ValueError("Ridge mask and processed image must have the same shape")
    ridge_mask &= roi_mask
    enhanced = enhance_fingerprint(processed_image.grayscale, roi_mask)
    skeleton = prune_skeleton_spurs(skeletonize(ridge_mask), max_spur_length=spur_length)

    angles, coherence = orientation.estimate_orientation_field(
        enhanced,
        roi_mask,
        block_size=orientation_block_size,
        smooth=True,
    )
    orientation_image = orientation.render_orientation_field(
        enhanced.shape,
        angles,
        coherence,
        block_size=orientation_block_size,
        mask=roi_mask,
    )
    density_window = max(7, (orientation_block_size | 1))
    density_map = cv2.blur(
        ridge_mask.astype(np.float32), (density_window, density_window)
    )
    segments, endpoints, bifurcations = trace_ridge_segments(
        skeleton,
        density_map,
        coherence,
        orientation_block_size=orientation_block_size,
        minimum_length=minimum_ridge_length,
        roi_bounds=_roi_bounds(roi_mask, roi_mask.shape),
        roi_mask=roi_mask,
    )
    return GeometryAnalysis(
        mode=mode,
        processed_image=processed_image,
        enhanced=enhanced,
        roi_mask=roi_mask,
        ridge_mask=ridge_mask,
        skeleton=skeleton,
        orientation_angles=angles,
        orientation_coherence=coherence,
        orientation_image=orientation_image,
        density_map=density_map,
        segments=tuple(segments),
        endpoints=tuple(endpoints),
        bifurcations=tuple(bifurcations),
    )


def analyze_cv(processed_image, orientation_block_size=16, minimum_ridge_length=8.0):
    roi_mask = np.asarray(processed_image.foreground_mask, dtype=bool)
    enhanced = enhance_fingerprint(processed_image.grayscale, roi_mask)
    ridge_mask = segment_ridges_cv(enhanced, roi_mask)
    return analyze_ridge_mask(
        processed_image,
        ridge_mask,
        mode="cv",
        orientation_block_size=orientation_block_size,
        minimum_ridge_length=minimum_ridge_length,
    )
