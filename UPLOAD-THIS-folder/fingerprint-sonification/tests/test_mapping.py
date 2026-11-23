import numpy as np

from config import SonificationConfig
from src.feature_extractor import FeatureSequence
from src.sound_mapper import map_features_to_sound


def make_features():
    values = np.linspace(0, 1, 8, dtype=np.float32)
    return FeatureSequence(
        scan_position=values,
        x_position=values,
        y_position=values[::-1],
        ridge_darkness=values,
        local_contrast=values[::-1],
        ridge_density=values,
        foreground_coverage=np.ones(8, dtype=np.float32),
    )


def test_mapping_stays_within_configured_ranges():
    config = SonificationConfig(duration=0.2)
    parameters = map_features_to_sound(make_features(), config)

    assert np.all(parameters.frequency >= config.min_frequency)
    assert np.all(parameters.frequency <= config.max_frequency)
    assert np.all(parameters.amplitude >= config.min_amplitude)
    assert np.all(parameters.amplitude <= config.max_amplitude)
    assert np.all((parameters.harmonic_mix >= 0) & (parameters.harmonic_mix <= 1))


def test_mapping_is_deterministic():
    config = SonificationConfig(duration=0.2)
    first = map_features_to_sound(make_features(), config)
    second = map_features_to_sound(make_features(), config)

    assert np.array_equal(first.frequency, second.frequency)
    assert np.array_equal(first.amplitude, second.amplitude)
    assert np.array_equal(first.harmonic_mix, second.harmonic_mix)
