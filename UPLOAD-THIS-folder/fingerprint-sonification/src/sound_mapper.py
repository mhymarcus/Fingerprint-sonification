"""Explicit and reproducible image-feature to sound-parameter mapping."""

from dataclasses import dataclass

import numpy as np

from config import SonificationConfig
from src.feature_extractor import FeatureSequence


@dataclass(frozen=True)
class AudioParameters:
    scan_position: np.ndarray
    frequency: np.ndarray
    amplitude: np.ndarray
    harmonic_mix: np.ndarray


def map_features_to_sound(features: FeatureSequence, config: SonificationConfig):
    """Map image features to pitch, amplitude, and harmonic complexity.

    Equations:
      frequency = f_min * (f_max / f_min) ** ridge_darkness
      amplitude = lerp(a_min, a_max, local_contrast)
                  * (0.25 + 0.75 * sqrt(foreground_coverage))
      harmonic_mix = ridge_density

    The exponential pitch equation distributes image brightness perceptually
    across the configured frequency range. All mappings are deterministic.
    """
    config.validate()
    frequency = config.min_frequency * (
        config.max_frequency / config.min_frequency
    ) ** np.clip(features.ridge_darkness, 0.0, 1.0)
    amplitude = config.min_amplitude + (
        config.max_amplitude - config.min_amplitude
    ) * np.clip(features.local_contrast, 0.0, 1.0)
    amplitude *= 0.25 + 0.75 * np.sqrt(
        np.clip(features.foreground_coverage, 0.0, 1.0)
    )
    harmonic_mix = np.clip(features.ridge_density, 0.0, 1.0)

    return AudioParameters(
        scan_position=features.scan_position.copy(),
        frequency=np.asarray(frequency, dtype=np.float32),
        amplitude=np.asarray(amplitude, dtype=np.float32),
        harmonic_mix=np.asarray(harmonic_mix, dtype=np.float32),
    )
