"""Find the fingerprint in a photograph and choose a sensible resolution.

A photo taken on a phone has no reliable orientation, no guarantee that the
finger fills the frame, and no metadata about ridge scale. Both of those can be
measured:

* the print is a large, compact, roughly centred subject that differs from the
  border — whichever way round (a bright finger on a dark surface, or dark ink
  on light paper);
* the ridge spacing can be read from the dominant period of the texture, and
  the analysis resolution then follows from it.

Nothing here is specific to one camera, one skin tone or one background. It
measures the image it is given.
"""

import cv2
import numpy as np


MIN_SIZE = 256
MAX_SIZE = 768
TARGET_SPACING = 10.0  # pixels between ridges at the analysis resolution
MARGIN = 0.06          # breathing room around the detected subject


def _largest_component(mask):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 8
    )
    if count <= 1:
        return None, None, None
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    component = labels == index
    return component, stats[index], centroids[index]


def _border_touch(bbox, shape):
    x, y, width, height = bbox
    image_h, image_w = shape
    edges = 0
    edges += 1 if x <= 1 else 0
    edges += 1 if y <= 1 else 0
    edges += 1 if x + width >= image_w - 1 else 0
    edges += 1 if y + height >= image_h - 1 else 0
    return edges


def _subject_score(bbox, area, shape):
    """Prefer large, solid, centred subjects that do not hug the frame."""
    image_h, image_w = shape
    x, y, width, height = bbox
    area_fraction = area / float(image_h * image_w)
    if not 0.03 <= area_fraction <= 0.92:
        return -1.0
    solidity = area / float(max(width * height, 1))
    centre_x = (x + width / 2) / image_w - 0.5
    centre_y = (y + height / 2) / image_h - 0.5
    centre_distance = float(np.hypot(centre_x, centre_y))
    return (
        area_fraction
        * solidity
        * max(0.15, 1.0 - centre_distance)
        / (1.0 + _border_touch(bbox, shape))
    )


def detect_subject(image, working=256):
    """Return the subject's bounding box as fractions (left, top, right, bottom).

    Falls back to the whole frame when nothing convincing is found, which is the
    safe choice: the caller then behaves exactly as before.
    """
    image_h, image_w = image.shape
    scale = working / float(max(image_h, image_w))
    small = cv2.resize(
        image, (max(8, int(image_w * scale)), max(8, int(image_h * scale))),
        interpolation=cv2.INTER_AREA,
    )
    small = cv2.GaussianBlur(small.astype(np.uint8), (5, 5), 0)
    _, binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    best = None
    for mask in (binary, 255 - binary):
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        closed = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        _, stats, _ = _largest_component(closed)
        if stats is None:
            continue
        bbox = (
            int(stats[cv2.CC_STAT_LEFT]),
            int(stats[cv2.CC_STAT_TOP]),
            int(stats[cv2.CC_STAT_WIDTH]),
            int(stats[cv2.CC_STAT_HEIGHT]),
        )
        area = float(stats[cv2.CC_STAT_AREA])
        score = _subject_score(bbox, area, closed.shape)
        if score > 0 and (best is None or score > best[0]):
            best = (score, bbox)

    if best is None:
        return (0.0, 0.0, 1.0, 1.0)

    x, y, width, height = best[1]
    small_h, small_w = small.shape
    left = x / small_w
    right = (x + width) / small_w
    top = y / small_h
    bottom = (y + height) / small_h
    return (
        max(0.0, left - MARGIN),
        max(0.0, top - MARGIN),
        min(1.0, right + MARGIN),
        min(1.0, bottom + MARGIN),
    )


def estimate_ridge_spacing(patch, working=512, minimum=3.0, maximum=96.0):
    """Dominant ridge period of a patch, in that patch's own pixels.

    A high-pass first, because the slow shading of a finger carries far more
    energy than its ridges and would otherwise dominate the spectrum.
    """
    if patch is None or patch.size < 32 * 32:
        return None
    patch_h, patch_w = patch.shape
    # Measure at native scale: downscaling a 3000 px photo averages the ridges
    # away and the spectrum then reports the fabric or nothing at all.
    if working and max(patch_h, patch_w) > working:
        centre_y, centre_x = patch_h // 2, patch_w // 2
        half = working // 2
        patch = patch[
            max(0, centre_y - half):centre_y + half,
            max(0, centre_x - half):centre_x + half,
        ]
    small = patch.astype(np.float32)
    high_pass = small - cv2.GaussianBlur(small, (0, 0), 4.0)
    if float(high_pass.std()) < 1e-3:
        return None

    window = np.outer(np.hanning(small.shape[0]), np.hanning(small.shape[1]))
    spectrum = np.abs(np.fft.fftshift(np.fft.fft2(high_pass * window)))
    spectrum = np.log1p(spectrum)

    centre_y, centre_x = small.shape[0] // 2, small.shape[1] // 2
    yy, xx = np.mgrid[0 : small.shape[0], 0 : small.shape[1]]
    radius = np.hypot(yy - centre_y, xx - centre_x).astype(np.int32)

    lowest = max(2, int(np.floor(max(small.shape) / maximum)))
    highest = min(centre_y, centre_x) - 1
    if highest <= lowest:
        return None

    radial = np.bincount(radius.ravel(), weights=spectrum.ravel(),
                         minlength=highest + 2)
    counts = np.bincount(radius.ravel(), minlength=highest + 2)
    radial = radial[: highest + 1] / np.maximum(counts[: highest + 1], 1)

    band = radial[lowest:highest]
    if band.size == 0:
        return None
    peak = int(np.argmax(band)) + lowest
    # A real ridge texture has one clear dominant period; flat or noisy patches
    # do not, and guessing from them would pick the wrong resolution.
    if band.max() < float(np.median(band)) * 1.5:
        return None

    return max(small.shape) / float(peak)


def plan(image, target_spacing=TARGET_SPACING):
    """Return ``(crop_fractions, analysis_size)`` for one photograph."""
    crop = detect_subject(image)
    image_h, image_w = image.shape
    left = int(crop[0] * image_w)
    right = int(crop[2] * image_w)
    top = int(crop[1] * image_h)
    bottom = int(crop[3] * image_h)
    if right - left < 32 or bottom - top < 32:
        return (0.0, 0.0, 1.0, 1.0), 512

    patch = image[top:bottom, left:right]
    spacing = estimate_ridge_spacing(patch)
    crop_width = max(1, right - left)

    if spacing and spacing > 0.5:
        size = target_spacing * crop_width / spacing
    else:
        size = 512.0
    # Never invent more pixels than the crop actually has: upscaling a 96 px
    # scanner image to 768 buys nothing but time.
    size = min(size, max(float(MIN_SIZE), 2.0 * crop_width))
    size = int(np.clip(round(size / 8.0) * 8, MIN_SIZE, MAX_SIZE))
    return crop, size
