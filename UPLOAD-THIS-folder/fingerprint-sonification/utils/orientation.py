"""Fingerprint ridge-orientation estimation and rendering utilities."""

import cv2 as cv
import numpy as np


def estimate_orientation_field(image, mask, block_size=16, smooth=True):
    """Estimate ridge direction and confidence for each image block.

    Angles use radians in the range [0, pi): 0 points to the right and pi/2
    points down. Invalid blocks are represented by NaN angles and zero
    coherence.
    """
    image = np.asarray(image)
    mask = np.asarray(mask) > 0
    if image.ndim != 2:
        raise ValueError("image must be a two-dimensional grayscale array")
    if mask.shape != image.shape:
        raise ValueError("mask and image must have the same shape")
    if block_size < 4:
        raise ValueError("block_size must be at least 4")

    image_float = cv.GaussianBlur(image.astype(np.float32), (0, 0), 1.0)
    gradient_x = cv.Sobel(image_float, cv.CV_32F, 1, 0, ksize=3)
    gradient_y = cv.Sobel(image_float, cv.CV_32F, 0, 1, ksize=3)
    tensor_x = gradient_x * gradient_x - gradient_y * gradient_y
    tensor_y = 2.0 * gradient_x * gradient_y
    energy = gradient_x * gradient_x + gradient_y * gradient_y

    height, width = image.shape
    block_rows = (height + block_size - 1) // block_size
    block_columns = (width + block_size - 1) // block_size
    angles = np.full((block_rows, block_columns), np.nan, dtype=np.float32)
    coherence = np.zeros((block_rows, block_columns), dtype=np.float32)

    for block_row in range(block_rows):
        row_start = block_row * block_size
        row_end = min(row_start + block_size, height)
        for block_column in range(block_columns):
            column_start = block_column * block_size
            column_end = min(column_start + block_size, width)
            block_mask = mask[row_start:row_end, column_start:column_end]
            if block_mask.mean() < 0.35:
                continue

            block_tensor_x = tensor_x[
                row_start:row_end, column_start:column_end
            ][block_mask].sum()
            block_tensor_y = tensor_y[
                row_start:row_end, column_start:column_end
            ][block_mask].sum()
            block_energy = energy[
                row_start:row_end, column_start:column_end
            ][block_mask].sum()
            if block_energy <= np.finfo(np.float32).eps:
                continue

            # The structure tensor gives the gradient direction. Adding pi/2
            # rotates it so the result follows the fingerprint ridge itself.
            ridge_angle = (
                0.5 * np.arctan2(block_tensor_y, block_tensor_x) + np.pi / 2
            ) % np.pi
            angles[block_row, block_column] = ridge_angle
            coherence[block_row, block_column] = np.clip(
                np.hypot(block_tensor_x, block_tensor_y) / block_energy,
                0.0,
                1.0,
            )

    if smooth:
        angles = smooth_orientation_field(angles, coherence)
    return angles, coherence


def smooth_orientation_field(angles, coherence):
    """Smooth axial angles without treating 0 and pi as different axes."""
    valid = np.isfinite(angles)
    weights = np.where(valid, coherence, 0.0).astype(np.float32)
    safe_angles = np.where(valid, angles, 0.0)
    vector_x = np.cos(2.0 * safe_angles).astype(np.float32) * weights
    vector_y = np.sin(2.0 * safe_angles).astype(np.float32) * weights

    smoothed_weights = cv.GaussianBlur(weights, (3, 3), 0.8)
    smoothed_x = cv.GaussianBlur(vector_x, (3, 3), 0.8)
    smoothed_y = cv.GaussianBlur(vector_y, (3, 3), 0.8)

    result = np.full(angles.shape, np.nan, dtype=np.float32)
    usable = valid & (smoothed_weights > np.finfo(np.float32).eps)
    result[usable] = (
        0.5 * np.arctan2(smoothed_y[usable], smoothed_x[usable])
    ) % np.pi
    return result


def render_orientation_field(
    image_shape,
    angles,
    coherence,
    block_size=16,
    mask=None,
    minimum_coherence=0.12,
):
    """Render the orientation field as dark line segments on white."""
    height, width = image_shape
    canvas = np.full((height, width), 255, dtype=np.uint8)
    half_length = max(2.0, block_size * 0.38)

    for block_row in range(angles.shape[0]):
        center_y = min(int((block_row + 0.5) * block_size), height - 1)
        for block_column in range(angles.shape[1]):
            angle = angles[block_row, block_column]
            if not np.isfinite(angle):
                continue
            if coherence[block_row, block_column] < minimum_coherence:
                continue

            center_x = min(int((block_column + 0.5) * block_size), width - 1)
            if mask is not None and not mask[center_y, center_x]:
                continue
            delta_x = np.cos(angle) * half_length
            delta_y = np.sin(angle) * half_length
            start = (round(center_x - delta_x), round(center_y - delta_y))
            end = (round(center_x + delta_x), round(center_y + delta_y))
            shade = round(90 - 55 * coherence[block_row, block_column])
            cv.line(canvas, start, end, int(shade), 1, cv.LINE_AA)
    return canvas


# Backwards-compatible wrappers for code that used the original API.
def calculate_angles(image, W, smoth=False):
    mask = np.ones(image.shape, dtype=np.uint8)
    angles, _ = estimate_orientation_field(image, mask, W, smooth=smoth)
    return np.nan_to_num(angles)


def visualize_angles(image, mask, angles, W):
    coherence = np.ones(angles.shape, dtype=np.float32)
    rendered = render_orientation_field(
        image.shape, angles, coherence, block_size=W, mask=mask
    )
    return cv.cvtColor(rendered, cv.COLOR_GRAY2RGB)
