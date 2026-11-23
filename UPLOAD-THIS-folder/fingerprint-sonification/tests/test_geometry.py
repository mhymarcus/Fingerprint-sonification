import math

import numpy as np

from src.geometry import (
    analyze_ridge_mask,
    prune_skeleton_spurs,
    trace_ridge_segments,
)
from src.image_processor import ProcessedImage
from src.ml.synthetic import generate_synthetic_fingerprint


def _trace(skeleton, roi_bounds=None, minimum_length=1):
    density = np.ones(skeleton.shape, dtype=np.float32) * 0.25
    coherence = np.ones(
        (max(1, skeleton.shape[0] // 16), max(1, skeleton.shape[1] // 16)),
        dtype=np.float32,
    )
    return trace_ridge_segments(
        skeleton,
        density,
        coherence,
        orientation_block_size=16,
        minimum_length=minimum_length,
        roi_bounds=roi_bounds,
    )


def test_tracing_measures_horizontal_arc_length_and_orientation():
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[24, 10:51] = True
    density = np.ones((64, 64), dtype=np.float32) * 0.25
    coherence = np.ones((4, 4), dtype=np.float32)

    segments, endpoints, bifurcations = trace_ridge_segments(
        skeleton,
        density,
        coherence,
        orientation_block_size=16,
        minimum_length=1,
    )

    assert len(segments) == 1
    assert math.isclose(segments[0].length_pixels, 40.0)
    assert math.isclose(segments[0].centroid_x, 30 / 63)
    assert min(segments[0].orientation_radians, math.pi - segments[0].orientation_radians) < 1e-5
    assert segments[0].curvature < 1e-5
    assert len(endpoints) == 2
    assert not bifurcations


def test_positions_are_normalised_against_the_fingerprint_roi():
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[24, 10:51] = True

    whole_frame = _trace(skeleton)[0][0]
    within_roi = _trace(skeleton, roi_bounds=(20, 10, 40, 40))[0][0]

    # Centroid column is 30: 30/63 against the canvas, (30-20)/20 inside the ROI.
    assert math.isclose(whole_frame.centroid_x, 30 / 63)
    assert math.isclose(within_roi.centroid_x, 0.5)
    # A 40 px arc inside a 20 x 30 px ROI is normalised by the ROI diagonal.
    assert math.isclose(within_roi.length_normalized, 40.0 / math.hypot(20, 30))


def test_spur_pruning_removes_short_branches_but_keeps_the_ridge():
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[24, 10:51] = True
    skeleton[20:24, 30] = True  # 4 px spur meeting the ridge at a junction

    pruned = prune_skeleton_spurs(skeleton, max_spur_length=6)

    assert pruned[24, 10:51].all()
    # Under 8-connectivity the pixel next to the ridge is itself a junction, so
    # the spur is trimmed down to that single contact pixel and no further.
    assert not pruned[20:23, 30].any()
    assert pruned[20:24, 30].sum() <= 1


def test_spur_pruning_keeps_branches_longer_than_the_limit():
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[40, 10:51] = True
    skeleton[20:40, 30] = True  # 20 px branch, far longer than the limit

    pruned = prune_skeleton_spurs(skeleton, max_spur_length=6)

    assert pruned[20:40, 30].all()


def test_synthetic_ground_truth_runs_shared_geometry_interface():
    synthetic = generate_synthetic_fingerprint(size=96, angle_degrees=30, seed=7)
    processed = ProcessedImage(
        source_path=None,
        original=synthetic.image,
        grayscale=synthetic.image.astype(np.float32) / 255.0,
        foreground_mask=synthetic.roi_mask,
    )

    analysis = analyze_ridge_mask(
        processed,
        synthetic.ridge_mask,
        mode="synthetic",
        orientation_block_size=12,
        minimum_ridge_length=5,
    )

    assert analysis.mode == "synthetic"
    assert len(analysis.segments) > 5
    assert all(segment.length_pixels >= 5 for segment in analysis.segments)
    assert all(0 <= segment.centroid_x <= 1 for segment in analysis.segments)
    assert all(0 <= segment.orientation_normalized <= 1 for segment in analysis.segments)
    assert np.isfinite(analysis.orientation_angles).any()
    # Ridge spacing is measured in pixels; tightness is its image-relative score.
    assert all(segment.spacing_pixels > 0 for segment in analysis.segments)
    assert all(0 <= segment.tightness <= 1 for segment in analysis.segments)
    assert max(segment.tightness for segment in analysis.segments) > 0.5
