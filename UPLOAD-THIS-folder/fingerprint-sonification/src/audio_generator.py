"""Click-free deterministic harmonic audio synthesis."""

import numpy as np

from config import SonificationConfig
from src.sound_mapper import AudioParameters


def _interpolate_parameter(values, positions, sample_positions):
    return np.interp(sample_positions, positions, values).astype(np.float64)


def generate_audio(parameters: AudioParameters, config: SonificationConfig):
    """Synthesize normalized mono floating-point audio in the range [-1, 1]."""
    config.validate()
    sample_count = round(config.duration * config.sample_rate)
    if sample_count < 2:
        raise ValueError("Duration and sample rate produce too few audio samples")
    sample_positions = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)

    frequency = _interpolate_parameter(
        parameters.frequency, parameters.scan_position, sample_positions
    )
    amplitude = _interpolate_parameter(
        parameters.amplitude, parameters.scan_position, sample_positions
    )
    harmonic_mix = _interpolate_parameter(
        parameters.harmonic_mix, parameters.scan_position, sample_positions
    )

    # Integrating instantaneous frequency keeps phase continuous during pitch
    # changes and prevents clicks between scan cells.
    phase = 2.0 * np.pi * np.cumsum(frequency) / config.sample_rate
    fundamental_weight = 1.0 - 0.30 * harmonic_mix
    second_weight = 0.22 * harmonic_mix
    third_weight = 0.10 * harmonic_mix
    signal = amplitude * (
        fundamental_weight * np.sin(phase)
        + second_weight * np.sin(2.0 * phase)
        + third_weight * np.sin(3.0 * phase)
    )

    fade_samples = min(
        round(config.fade_duration * config.sample_rate), sample_count // 2
    )
    if fade_samples:
        fade = np.sin(np.linspace(0.0, np.pi / 2, fade_samples)) ** 2
        signal[:fade_samples] *= fade
        signal[-fade_samples:] *= fade[::-1]

    signal -= signal.mean()
    if not np.all(np.isfinite(signal)):
        raise ValueError("Audio synthesis produced NaN or infinite samples")
    peak = float(np.max(np.abs(signal)))
    if peak < 1e-8:
        raise ValueError("Audio synthesis produced a silent signal")
    signal *= 0.92 / peak
    return signal.astype(np.float32)
