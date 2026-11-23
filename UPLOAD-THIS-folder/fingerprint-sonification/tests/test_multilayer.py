from dataclasses import replace

import numpy as np

from config import SonificationConfig
from src.geometry import analyze_ridge_mask
from src.image_processor import ProcessedImage
from src.ml.synthetic import generate_synthetic_fingerprint
from src.sonification import synthesize_multilayer


def make_analysis(curvature=0.12):
    synthetic = generate_synthetic_fingerprint(size=96, curvature=curvature, seed=11)
    processed = ProcessedImage(
        source_path=None,
        original=synthetic.image,
        grayscale=synthetic.image.astype(np.float32) / 255.0,
        foreground_mask=synthetic.roi_mask,
    )
    return analyze_ridge_mask(
        processed,
        synthetic.ridge_mask,
        mode="synthetic",
        orientation_block_size=12,
        minimum_ridge_length=5,
    )


def test_multilayer_stems_are_aligned_deterministic_and_safely_mixed():
    analysis = make_analysis()
    config = SonificationConfig(
        duration=0.3,
        fade_duration=0.04,
        orientation_block_size=12,
        max_segment_events=12,
    )

    first = synthesize_multilayer(analysis, config)
    second = synthesize_multilayer(analysis, config)

    expected_length = round(config.duration * config.sample_rate)
    assert set(first.stems) == {
        "ridge_length",
        "ridge_position",
        "ridge_orientation",
        "ridge_texture",
    }
    assert all(len(stem) == expected_length for stem in first.stems.values())
    assert len(first.mix) == expected_length
    assert np.array_equal(first.mix, second.mix)
    assert all(np.isfinite(stem).all() for stem in first.stems.values())
    assert first.peak <= 10 ** (-3 / 20) + 1e-6
    assert all(np.max(np.abs(stem)) > 0 for stem in first.stems.values())


def test_layers_can_be_switched_off_independently():
    analysis = make_analysis()
    config = replace(
        SonificationConfig(duration=0.2, fade_duration=0.03),
        enable_texture_layer=False,
    )

    result = synthesize_multilayer(analysis, config)

    assert np.all(result.stems["ridge_texture"] == 0)
    assert np.max(np.abs(result.stems["ridge_length"])) > 0


def test_master_is_normalised_to_minus_three_dbfs():
    config = SonificationConfig(duration=0.3, fade_duration=0.04)
    result = synthesize_multilayer(make_analysis(), config)

    assert abs(result.peak - 10 ** (-3 / 20)) < 1e-4


def test_melodic_events_spread_across_the_timeline():
    config = SonificationConfig(
        duration=2.0, orientation_block_size=12, max_segment_events=24
    )
    result = synthesize_multilayer(make_analysis(), config)
    starts = [
        event["start_seconds"]
        for event in result.events
        if event["layer"] in ("ridge_length", "ridge_position", "ridge_texture")
    ]

    assert starts
    assert max(starts) - min(starts) > 0.5


def test_orientation_pitch_glide_scales_with_actual_rotation():
    config = SonificationConfig(
        duration=1.0, orientation_block_size=12, max_segment_events=12
    )

    def octaves(curvature):
        result = synthesize_multilayer(make_analysis(curvature), config)
        event = next(
            item for item in result.events if item["layer"] == "ridge_orientation"
        )
        return event["pitch_octaves"], event["pitch_min_hz"], event["pitch_max_hz"]

    flat_octaves, flat_low, flat_high = octaves(0.0)
    curved_octaves, curved_low, curved_high = octaves(0.12)

    # A print whose ridge flow does not rotate stays near the base pitch ...
    assert flat_octaves < 0.05
    assert flat_high / flat_low < 1.05
    # ... while real rotation produces a proportionally larger glide.
    assert curved_octaves > 3 * flat_octaves
    assert curved_high > flat_high * 1.05
