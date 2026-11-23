"""Fingerprint image loading and preprocessing."""

from dataclasses import dataclass
from pathlib import Path
import warnings

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class ProcessedImage:
    source_path: Path
    original: np.ndarray
    grayscale: np.ndarray
    foreground_mask: np.ndarray


def load_grayscale_image(image_path):
    """Load a supported image as an 8-bit grayscale array."""
    image_path = Path(image_path)
    if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported image format. Expected one of: {supported}")
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image does not exist: {image_path}")

    try:
        with Image.open(image_path) as image:
            image.load()
            grayscale = ImageOps.exif_transpose(image).convert("L")
            array = np.asarray(grayscale, dtype=np.uint8)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Input image is unreadable or corrupted: {image_path}") from error

    if array.ndim != 2 or min(array.shape) < 16:
        raise ValueError("Input image must be a readable image at least 16 x 16 pixels")
    return array


def _border_pixels(image):
    return np.concatenate((image[0], image[-1], image[:, 0], image[:, -1]))


def _crop_large_uniform_background(image):
    """Crop to the largest non-background region while preserving generic images."""
    background = float(np.median(_border_pixels(image)))
    deviation = np.abs(image.astype(np.float32) - background)
    threshold = max(10.0, float(np.percentile(deviation, 70)) * 0.45)
    active = (deviation > threshold).astype(np.uint8)

    kernel_size = max(3, (min(image.shape) // 40) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    active = cv2.morphologyEx(active, cv2.MORPH_CLOSE, kernel)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(active, 8)
    if count <= 1:
        return image

    largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, width, height, area = stats[largest_label]
    if area < image.size * 0.04:
        return image

    padding = max(2, round(max(width, height) * 0.04))
    left = max(0, x - padding)
    top = max(0, y - padding)
    right = min(image.shape[1], x + width + padding)
    bottom = min(image.shape[0], y + height + padding)
    return image[top:bottom, left:right]


def _resize_with_padding(image, width, height):
    """Fit the image into a square canvas and report where the real pixels are.

    The padding is synthetic, so callers must be able to exclude it: an adaptive
    threshold treats the seam between photo and padding as a strong edge and
    reports it as ridges.
    """
    background = int(np.median(_border_pixels(image)))
    pil_image = Image.fromarray(image)
    fitted = ImageOps.contain(pil_image, (width, height), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (width, height), color=background)
    offset = ((width - fitted.width) // 2, (height - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return np.asarray(canvas, dtype=np.uint8), (offset[0], offset[1], fitted.width, fitted.height)


def _normalize_brightness(image):
    image_float = image.astype(np.float32)
    low, high = np.percentile(image_float, (1.0, 99.0))
    if high - low < 2.0:
        warnings.warn(
            "Image has almost uniform brightness; sonic variation will be limited",
            UserWarning,
        )
        return np.full(image.shape, 0.5, dtype=np.float32)
    normalized = np.clip((image_float - low) / (high - low), 0.0, 1.0)

    # Standardize polarity: a light background and dark ridges make mappings
    # consistent across scans with inverted colors.
    if np.median(_border_pixels(normalized)) < 0.5:
        normalized = 1.0 - normalized
    return normalized.astype(np.float32)


def _create_foreground_mask(grayscale):
    """Find the subject as the region that carries ridge texture.

    Two obvious rules fail on photographs. "The dark thing is the print" picks
    the shadow cast by the finger. "Everything that differs from the background
    colour" picks up vignetting and the shadow as well, and ends up as the whole
    frame.

    The question that survives both is where ridge texture actually exists: a
    fingertip is full of fine ridges, a shadow and a paper background are not.
    That makes the mask self-consistent with what the analysis does next.
    """
    image = np.round(np.clip(np.asarray(grayscale, dtype=np.float32), 0, 1) * 255.0)
    image = image.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    block = max(15, ((min(enhanced.shape) // 12) | 1))
    ridges = (
        cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            block,
            5,
        )
        > 0
    )
    window = max(15, (min(enhanced.shape) // 10) | 1)
    density = cv2.blur(ridges.astype(np.float32), (window, window))
    subject = (density > 0.18).astype(np.uint8)

    kernel_size = max(5, (min(enhanced.shape) // 20) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    subject = cv2.morphologyEx(subject, cv2.MORPH_CLOSE, kernel)
    subject = cv2.morphologyEx(subject, cv2.MORPH_OPEN, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(subject, 8)
    if count <= 1:
        return np.ones(grayscale.shape, dtype=bool)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = labels == index
    fraction = float(mask.mean())
    # Nothing convincing: behave exactly as before rather than crop to a sliver.
    if fraction < 0.05 or fraction > 0.85:
        return np.ones(grayscale.shape, dtype=bool)
    # The frame's own edges always look like texture. Keep them out.
    inset = max(2, int(round(min(mask.shape) * 0.06)))
    mask[:inset, :] = False
    mask[-inset:, :] = False
    mask[:, :inset] = False
    mask[:, -inset:] = False
    return mask


def _apply_crop(image, crop):
    """Keep a fractional rectangle: (left, top, right, bottom) in 0..1."""
    left, top, right, bottom = crop
    height, width = image.shape
    left_px = int(np.clip(left, 0.0, 0.95) * width)
    right_px = int(np.clip(right, 0.05, 1.0) * width)
    top_px = int(np.clip(top, 0.0, 0.95) * height)
    bottom_px = int(np.clip(bottom, 0.05, 1.0) * height)
    if right_px - left_px < 32 or bottom_px - top_px < 32:
        raise ValueError("Crop is too small; keep at least 32 x 32 pixels")
    return image[top_px:bottom_px, left_px:right_px]


def preprocess_image(
    image_path,
    analysis_width=256,
    analysis_height=256,
    enhance_contrast=True,
    crop=None,
):
    """Load, crop, resize, normalize, enhance, and denoise a fingerprint.

    ``crop`` is an optional fractional rectangle ``(left, top, right, bottom)``.
    It matters for photographs: a finger against a textured background gives the
    background as much detail as the ridges, so framing the print by hand is
    often the difference between a usable mask and a crescent.
    """
    original = load_grayscale_image(image_path)
    if crop is not None:
        original = _apply_crop(original, crop)
    cropped = _crop_large_uniform_background(original)
    resized, content = _resize_with_padding(cropped, analysis_width, analysis_height)
    normalized = _normalize_brightness(resized)

    normalized_u8 = np.round(normalized * 255).astype(np.uint8)
    if enhance_contrast and np.ptp(normalized_u8) > 4:
        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        normalized_u8 = clahe.apply(normalized_u8)
    denoised = cv2.GaussianBlur(normalized_u8, (3, 3), 0.65)
    grayscale = denoised.astype(np.float32) / 255.0
    foreground_mask = _create_foreground_mask(grayscale)
    # Never analyse the padded border: it is invented, not photographed.
    left, top, content_width, content_height = content
    if (content_width, content_height) != (analysis_width, analysis_height):
        content_mask = np.zeros_like(foreground_mask)
        content_mask[top : top + content_height, left : left + content_width] = True
        foreground_mask &= content_mask
    return ProcessedImage(
        source_path=Path(image_path),
        original=original,
        grayscale=grayscale,
        foreground_mask=foreground_mask,
    )
