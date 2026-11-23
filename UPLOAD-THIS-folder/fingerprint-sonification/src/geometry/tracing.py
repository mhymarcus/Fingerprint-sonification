"""Skeleton graph tracing and geometric ridge measurements."""

from dataclasses import asdict, dataclass, replace
import math

import numpy as np
from scipy import ndimage


NEIGHBOUR_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


@dataclass(frozen=True)
class RidgeSegment:
    segment_id: int
    points: tuple
    length_pixels: float
    length_normalized: float
    centroid_x: float
    centroid_y: float
    orientation_radians: float
    orientation_normalized: float
    density: float
    curvature: float
    confidence: float
    spacing_pixels: float = 0.0
    tightness: float = 0.0

    def to_dict(self, include_points=True):
        result = asdict(self)
        if not include_points:
            result.pop("points")
        return result


def _neighbours(point, skeleton):
    row, column = point
    height, width = skeleton.shape
    result = []
    for row_delta, column_delta in NEIGHBOUR_OFFSETS:
        neighbour = (row + row_delta, column + column_delta)
        if (
            0 <= neighbour[0] < height
            and 0 <= neighbour[1] < width
            and skeleton[neighbour]
        ):
            result.append(neighbour)
    return result


def _edge(first, second):
    return tuple(sorted((first, second)))


def _degree_map(skeleton):
    neighbour_kernel = np.ones((3, 3), dtype=np.uint8)
    neighbour_kernel[1, 1] = 0
    return ndimage.convolve(skeleton.astype(np.uint8), neighbour_kernel, mode="constant")


def prune_skeleton_spurs(skeleton, max_spur_length=6, max_iterations=3):
    """Remove short skeleton branches that terminate on a junction.

    Binarisation noise leaves hair-like spurs hanging off the ridge skeleton.
    Left in place they dominate the segment list, so paths are traced only
    after the spurs have been trimmed.
    """
    skeleton = np.asarray(skeleton, dtype=bool).copy()
    if not skeleton.any() or max_spur_length < 2:
        return skeleton

    for _ in range(max(1, max_iterations)):
        degree = _degree_map(skeleton)
        endpoints = np.argwhere(skeleton & (degree == 1))
        removed = 0
        for endpoint in endpoints:
            start = (int(endpoint[0]), int(endpoint[1]))
            if not skeleton[start]:
                continue  # already trimmed by an earlier spur in this pass
            path = [start]
            previous = None
            current = start
            reached_junction = False
            for _ in range(max_spur_length):
                options = [
                    point for point in _neighbours(current, skeleton) if point != previous
                ]
                if len(options) != 1:
                    break
                previous, current = current, options[0]
                path.append(current)
                if degree[current] >= 3:
                    reached_junction = True
                    break
                if degree[current] == 1:
                    break
            if reached_junction and len(path) <= max_spur_length:
                for pixel in path[:-1]:
                    skeleton[pixel] = False
                removed += 1
        if not removed:
            break
    return skeleton


def _trace_from(start, first_step, skeleton, degree, visited_edges):
    path = [start, first_step]
    visited_edges.add(_edge(start, first_step))
    previous = start
    current = first_step
    while degree[current] == 2:
        options = [point for point in _neighbours(current, skeleton) if point != previous]
        if not options:
            break
        next_point = options[0]
        next_edge = _edge(current, next_point)
        if next_edge in visited_edges:
            break
        path.append(next_point)
        visited_edges.add(next_edge)
        previous, current = current, next_point
    return path


def _path_length(points):
    if len(points) < 2:
        return 0.0
    coordinates = np.asarray(points, dtype=np.float32)
    steps = np.diff(coordinates, axis=0)
    return float(np.hypot(steps[:, 0], steps[:, 1]).sum())


def _path_orientation_and_curvature(points):
    coordinates = np.asarray(points, dtype=np.float32)
    if len(coordinates) < 3:
        vector = coordinates[-1] - coordinates[0]
        angle = math.atan2(float(vector[0]), float(vector[1])) % math.pi
        return angle, 0.0

    windowed_vectors = coordinates[2:] - coordinates[:-2]
    valid = np.hypot(windowed_vectors[:, 0], windowed_vectors[:, 1]) > 0
    windowed_vectors = windowed_vectors[valid]
    if not len(windowed_vectors):
        return 0.0, 0.0
    angles = np.arctan2(windowed_vectors[:, 0], windowed_vectors[:, 1])
    mean_angle = 0.5 * math.atan2(
        float(np.sin(2.0 * angles).mean()),
        float(np.cos(2.0 * angles).mean()),
    )
    mean_angle %= math.pi
    unwrapped = np.unwrap(2.0 * angles) / 2.0
    angular_change = np.abs(np.diff(unwrapped))
    curvature = float(np.clip(angular_change.mean() / (math.pi / 4), 0.0, 1.0)) if len(angular_change) else 0.0
    return mean_angle, curvature


def _segment_from_path(
    segment_id,
    path,
    image_shape,
    density_map,
    orientation_coherence,
    orientation_block_size,
    roi_bounds,
    spacing_map,
):
    coordinates = np.asarray(path, dtype=np.int32)
    length = _path_length(path)
    orientation, curvature = _path_orientation_and_curvature(path)
    rows = coordinates[:, 0]
    columns = coordinates[:, 1]
    density = float(np.mean(density_map[rows, columns]))

    if spacing_map is None:
        spacing = 0.0
    else:
        local = spacing_map[rows, columns]
        local = local[np.isfinite(local)]
        spacing = float(np.mean(local)) if local.size else 0.0

    # Positions are normalised against the fingerprint ROI, not the padded
    # canvas. Normalising against the canvas crushed every ridge into the middle
    # third of the timeline and left the piece's head and tail silent.
    left, top, right, bottom = roi_bounds
    span_x = max(int(right) - int(left), 1)
    span_y = max(int(bottom) - int(top), 1)
    diagonal = max(math.hypot(span_x, span_y), 1.0)

    block_rows = np.clip(rows // orientation_block_size, 0, orientation_coherence.shape[0] - 1)
    block_columns = np.clip(columns // orientation_block_size, 0, orientation_coherence.shape[1] - 1)
    confidence = float(np.mean(orientation_coherence[block_rows, block_columns]))
    return RidgeSegment(
        segment_id=segment_id,
        points=tuple((int(row), int(column)) for row, column in path),
        length_pixels=length,
        length_normalized=length / diagonal,
        centroid_x=float(np.clip((columns.mean() - left) / span_x, 0.0, 1.0)),
        centroid_y=float(np.clip((rows.mean() - top) / span_y, 0.0, 1.0)),
        orientation_radians=orientation,
        orientation_normalized=orientation / math.pi,
        density=density,
        curvature=curvature,
        confidence=confidence,
        spacing_pixels=spacing,
        tightness=0.0,
    )


def _ridge_spacing_map(skeleton, roi_mask, window):
    """Estimate local ridge spacing in pixels from skeleton line density.

    Inside a window the skeleton contributes one pixel of length per unit of
    ridge spacing, so ``roi_area / skeleton_length`` is a spacing estimate. It
    is resolution dependent but stable because preprocessing fits the print to
    the analysis canvas.
    """
    if roi_mask is None:
        roi_mask = np.ones(skeleton.shape, dtype=bool)
    skeleton_sum = (
        ndimage.uniform_filter(skeleton.astype(np.float32), window) * window * window
    )
    roi_sum = ndimage.uniform_filter(
        np.asarray(roi_mask, dtype=np.float32), window
    ) * window * window
    spacing = np.full(skeleton.shape, np.nan, dtype=np.float32)
    usable = skeleton_sum > 0.5
    spacing[usable] = roi_sum[usable] / np.maximum(skeleton_sum[usable], 1.0)
    return spacing


def _assign_tightness(segments):
    """Convert raw spacing into an image-relative 0..1 tightness score.

    Ridge-pixel density is almost constant across a binarised print, so the
    sound layers are driven by spacing instead: 1.0 means the tightest ridge
    packing in this print, 0.0 the widest.
    """
    spacings = np.array(
        [segment.spacing_pixels for segment in segments], dtype=np.float64
    )
    finite = spacings[np.isfinite(spacings) & (spacings > 0)]
    if not finite.size:
        return segments
    log_spacing = np.log(np.where(spacings > 0, spacings, np.nan))
    low, high = np.percentile(np.log(finite), (5.0, 95.0))
    if not np.isfinite(low) or not np.isfinite(high) or high - low < 1e-6:
        return [
            replace(segment, tightness=0.5) if segment.spacing_pixels > 0 else segment
            for segment in segments
        ]
    scored = []
    for segment, value in zip(segments, log_spacing):
        if not np.isfinite(value):
            scored.append(segment)
            continue
        tightness = float(np.clip((high - value) / (high - low), 0.0, 1.0))
        scored.append(replace(segment, tightness=tightness))
    return scored


def trace_ridge_segments(
    skeleton,
    density_map,
    orientation_coherence,
    orientation_block_size=16,
    minimum_length=8.0,
    roi_bounds=None,
    roi_mask=None,
):
    """Trace skeleton paths between endpoints/branches and measure arc length.

    ``roi_bounds`` is ``(left, top, right, bottom)`` in pixel coordinates. When
    omitted the full image frame is used, which reproduces the original
    whole-frame normalisation. ``roi_mask`` enables the ridge-spacing estimate.
    """
    skeleton = np.asarray(skeleton, dtype=bool)
    height, width = skeleton.shape
    if roi_bounds is None:
        roi_bounds = (0, 0, width - 1, height - 1)
    degree = _degree_map(skeleton)
    spacing_map = _ridge_spacing_map(
        skeleton, roi_mask, max(9, (orientation_block_size | 1))
    )
    nodes = skeleton & (degree != 2)
    visited_edges = set()
    paths = []

    for start_array in np.argwhere(nodes):
        start = tuple(map(int, start_array))
        for neighbour in _neighbours(start, skeleton):
            if _edge(start, neighbour) not in visited_edges:
                paths.append(
                    _trace_from(start, neighbour, skeleton, degree, visited_edges)
                )

    # Closed ridge loops have no degree != 2 node, so trace remaining edges.
    for point_array in np.argwhere(skeleton):
        point = tuple(map(int, point_array))
        for neighbour in _neighbours(point, skeleton):
            if _edge(point, neighbour) not in visited_edges:
                paths.append(
                    _trace_from(point, neighbour, skeleton, degree, visited_edges)
                )

    segments = []
    for path in paths:
        if _path_length(path) < minimum_length:
            continue
        segment = _segment_from_path(
            len(segments),
            path,
            skeleton.shape,
            density_map,
            orientation_coherence,
            orientation_block_size,
            roi_bounds,
            spacing_map,
        )
        segments.append(segment)

    segments = _assign_tightness(segments)
    endpoints = [tuple(map(int, point)) for point in np.argwhere(skeleton & (degree == 1))]
    bifurcations = [tuple(map(int, point)) for point in np.argwhere(skeleton & (degree >= 3))]
    return segments, endpoints, bifurcations
