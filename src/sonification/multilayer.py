"""Four-layer deterministic synthesis driven by fingerprint geometry."""

from dataclasses import dataclass
import math

import numpy as np

from config import SonificationConfig
from src.geometry.analyzer import GeometryAnalysis


@dataclass(frozen=True)
class MultiLayerAudio:
    mix: np.ndarray
    stems: dict
    events: tuple
    peak: float


def _event_envelope(sample_count, attack_fraction=0.18, release_fraction=0.35):
    envelope = np.ones(sample_count, dtype=np.float64)
    attack = min(sample_count // 2, max(1, round(sample_count * attack_fraction)))
    release = min(sample_count // 2, max(1, round(sample_count * release_fraction)))
    envelope[:attack] = np.sin(np.linspace(0.0, np.pi / 2, attack)) ** 2
    envelope[-release:] = np.cos(np.linspace(0.0, np.pi / 2, release)) ** 2
    return envelope


def _add_tone(buffer, sample_rate, start, duration, frequency, amplitude, timbre="warm"):
    start_sample = max(0, round(start * sample_rate))
    end_sample = min(len(buffer), start_sample + max(2, round(duration * sample_rate)))
    if end_sample <= start_sample:
        return
    count = end_sample - start_sample
    time = np.arange(count, dtype=np.float64) / sample_rate
    phase = 2.0 * np.pi * frequency * time
    if timbre == "bell":
        tone = (
            np.sin(phase)
            + 0.38 * np.sin(2.01 * phase)
            + 0.16 * np.sin(3.97 * phase)
        )
        envelope = np.exp(-4.2 * time / max(duration, 1e-3))
        envelope *= _event_envelope(count, attack_fraction=0.04, release_fraction=0.25)
    elif timbre == "texture":
        modulation = 0.7 * np.sin(2.0 * np.pi * 3.0 * time)
        tone = np.sin(phase + modulation) + 0.18 * np.sin(2.0 * phase)
        envelope = _event_envelope(count, attack_fraction=0.12, release_fraction=0.45)
    else:
        tone = np.sin(phase) + 0.24 * np.sin(2.0 * phase) + 0.08 * np.sin(3.0 * phase)
        envelope = _event_envelope(count)
    buffer[start_sample:end_sample] += amplitude * envelope * tone


def _select_segments(segments, maximum, score):
    selected = sorted(segments, key=score, reverse=True)[:maximum]
    return sorted(selected, key=lambda segment: (segment.centroid_x, segment.centroid_y))


def _length_layer(analysis, config, sample_count):
    stem = np.zeros(sample_count, dtype=np.float64)
    events = []
    selected = _select_segments(
        analysis.segments,
        config.max_segment_events,
        lambda segment: segment.length_pixels * (0.4 + segment.confidence),
    )
    for segment in selected:
        start = segment.centroid_x * max(0.0, config.duration - 0.8)
        duration = float(np.clip(0.20 + 3.2 * segment.length_normalized, 0.22, 2.6))
        frequency = config.min_frequency + (
            config.max_frequency - config.min_frequency
        ) * 0.25 * (1.0 - segment.centroid_y)
        amplitude = 0.055 + 0.075 * segment.confidence
        _add_tone(stem, config.sample_rate, start, duration, frequency, amplitude, "warm")
        events.append(
            {
                "layer": "ridge_length",
                "segment_id": segment.segment_id,
                "start_seconds": start,
                "duration_seconds": duration,
                "frequency_hz": frequency,
                "source_value": segment.length_pixels,
                "source_unit": "pixels_arc_length",
            }
        )
    return stem, events


def _position_layer(analysis, config, sample_count):
    stem = np.zeros(sample_count, dtype=np.float64)
    events = []
    minimum_midi = math.ceil(69 + 12 * math.log2(config.min_frequency / 440.0))
    maximum_midi = math.floor(69 + 12 * math.log2(config.max_frequency / 440.0))
    scale_midi = np.array(
        [note for note in range(minimum_midi, maximum_midi + 1) if note % 12 in {0, 2, 4, 7, 9}],
        dtype=int,
    )
    if not len(scale_midi):
        scale_midi = np.array([round((minimum_midi + maximum_midi) / 2)], dtype=int)
    candidates = _select_segments(
        analysis.segments,
        config.max_segment_events,
        lambda segment: segment.confidence * (0.5 + segment.length_normalized),
    )
    last_start = -math.inf
    for segment in candidates:
        start = segment.centroid_x * max(0.0, config.duration - 0.6)
        if start - last_start < config.minimum_event_spacing:
            continue
        scale_index = round((1.0 - segment.centroid_y) * (len(scale_midi) - 1))
        midi = int(scale_midi[np.clip(scale_index, 0, len(scale_midi) - 1)])
        frequency = 440.0 * 2.0 ** ((midi - 69) / 12.0)
        duration = 0.30 + 0.45 * segment.tightness
        amplitude = 0.035 + 0.065 * segment.confidence
        _add_tone(stem, config.sample_rate, start, duration, frequency, amplitude, "bell")
        last_start = start
        events.append(
            {
                "layer": "ridge_position",
                "segment_id": segment.segment_id,
                "start_seconds": start,
                "duration_seconds": duration,
                "frequency_hz": frequency,
                "x_normalized": segment.centroid_x,
                "y_normalized": segment.centroid_y,
            }
        )
    return stem, events


def _orientation_controls(analysis):
    angles = analysis.orientation_angles
    coherence = analysis.orientation_coherence
    x_positions = []
    angle_values = []
    confidence_values = []
    for column in range(angles.shape[1]):
        valid = np.isfinite(angles[:, column]) & (coherence[:, column] > 0.05)
        if not valid.any():
            continue
        weights = coherence[valid, column]
        column_angles = angles[valid, column]
        mean_angle = 0.5 * np.arctan2(
            np.sum(weights * np.sin(2.0 * column_angles)),
            np.sum(weights * np.cos(2.0 * column_angles)),
        )
        x_positions.append((column + 0.5) / angles.shape[1])
        angle_values.append(float(mean_angle % np.pi))
        confidence_values.append(float(np.mean(weights)))
    if not x_positions:
        return np.array([0.0, 1.0]), np.zeros(2), np.zeros(2)
    return (
        np.asarray(x_positions),
        np.asarray(angle_values),
        np.asarray(confidence_values),
    )


def _orientation_layer(analysis, config, sample_count):
    """Turn the print's ridge-flow rotation into a continuous glissando.

    The per-column mean ridge angle is unwrapped across the doubled-angle
    representation, so a 0/pi crossing is treated as continued rotation rather
    than a full-range pitch jump. The print's own angular range then spans two
    octaves above the layer's base frequency.
    """
    x_positions, angles, confidence = _orientation_controls(analysis)
    continuous = 0.5 * np.unwrap(2.0 * angles)
    angle_span = float(continuous.max() - continuous.min())

    timeline = np.linspace(0.0, 1.0, sample_count, dtype=np.float64)
    if angle_span < 1e-6:
        pitch_track = np.full(sample_count, 0.5, dtype=np.float64)
        rotation_rate = np.zeros(sample_count, dtype=np.float64)
    else:
        pitch_control = (continuous - continuous.min()) / angle_span
        pitch_track = np.interp(timeline, x_positions, pitch_control)
        if len(x_positions) > 1:
            rotation = np.abs(np.gradient(continuous, x_positions))
            rotation_control = np.clip(rotation / np.pi, 0.0, 1.0)
            rotation_rate = np.interp(timeline, x_positions, rotation_control)
        else:
            rotation_rate = np.zeros(sample_count, dtype=np.float64)
    smooth_confidence = np.interp(timeline, x_positions, confidence)

    # The size of the glide is proportional to how far the ridge flow actually
    # rotates: a quarter turn becomes one octave, a half turn two. A print with
    # near-uniform orientation therefore stays near the base pitch instead of
    # being stretched into a dramatic sweep by min-max normalisation.
    octaves = float(np.clip(angle_span / (0.5 * np.pi), 0.0, 3.0))
    frequency = config.min_frequency * (2.0 ** (octaves * pitch_track))
    # Integrating the instantaneous frequency keeps the glide phase-continuous.
    phase = 2.0 * np.pi * np.cumsum(frequency) / config.sample_rate
    harmonic_two = 0.10 + 0.22 * rotation_rate
    harmonic_three = 0.06 + 0.14 * (1.0 - rotation_rate)
    amplitude = 0.045 * (0.35 + 0.65 * smooth_confidence)
    stem = amplitude * (
        np.sin(phase)
        + harmonic_two * np.sin(2.0 * phase + pitch_track * np.pi)
        + harmonic_three * np.sin(3.0 * phase - pitch_track * np.pi)
    )
    events = [
        {
            "layer": "ridge_orientation",
            "start_seconds": 0.0,
            "duration_seconds": config.duration,
            "control_points": int(len(x_positions)),
            "orientation_unit": "radians_modulo_pi",
            "unwrapped_span_radians": angle_span,
            "pitch_octaves": octaves,
            "pitch_start_hz": float(frequency[0]),
            "pitch_end_hz": float(frequency[-1]),
            "pitch_min_hz": float(np.min(frequency)),
            "pitch_max_hz": float(np.max(frequency)),
        }
    ]
    return stem, events


def _texture_layer(analysis, config, sample_count):
    stem = np.zeros(sample_count, dtype=np.float64)
    events = []
    selected = _select_segments(
        analysis.segments,
        max(1, config.max_segment_events // 2),
        lambda segment: segment.curvature
        * (0.35 + 0.65 * segment.tightness)
        * (0.25 + segment.confidence),
    )
    last_start = -math.inf
    for segment in selected:
        start = segment.centroid_x * max(0.0, config.duration - 0.5)
        if start - last_start < config.minimum_event_spacing * 1.5:
            continue
        duration = 0.14 + 0.42 * segment.tightness
        frequency = config.min_frequency + (
            config.max_frequency - config.min_frequency
        ) * segment.curvature
        amplitude = 0.016 + 0.050 * segment.tightness * (0.3 + 0.7 * segment.curvature)
        _add_tone(stem, config.sample_rate, start, duration, frequency, amplitude, "texture")
        last_start = start
        events.append(
            {
                "layer": "ridge_texture",
                "segment_id": segment.segment_id,
                "start_seconds": start,
                "duration_seconds": duration,
                "frequency_hz": frequency,
                "density": segment.density,
                "curvature": segment.curvature,
                "tightness": segment.tightness,
                "spacing_pixels": segment.spacing_pixels,
            }
        )
    return stem, events


def synthesize_multilayer(analysis: GeometryAnalysis, config: SonificationConfig):
    """Create four time-aligned stems and a safely mixed -3 dBFS master."""
    config.validate()
    sample_count = round(config.duration * config.sample_rate)
    builders = {
        "ridge_length": (_length_layer, config.enable_length_layer, config.length_layer_gain),
        "ridge_position": (_position_layer, config.enable_position_layer, config.position_layer_gain),
        "ridge_orientation": (
            _orientation_layer,
            config.enable_orientation_layer,
            config.orientation_layer_gain,
        ),
        "ridge_texture": (_texture_layer, config.enable_texture_layer, config.texture_layer_gain),
    }
    stems = {}
    events = []
    for name, (builder, enabled, gain) in builders.items():
        if enabled:
            stem, layer_events = builder(analysis, config, sample_count)
            stems[name] = stem * gain
            events.extend(layer_events)
        else:
            stems[name] = np.zeros(sample_count, dtype=np.float64)

    fade_samples = min(round(config.fade_duration * config.sample_rate), sample_count // 2)
    if fade_samples:
        fade = np.sin(np.linspace(0.0, np.pi / 2, fade_samples)) ** 2
        for stem in stems.values():
            stem[:fade_samples] *= fade
            stem[-fade_samples:] *= fade[::-1]

    mix = sum(stems.values()) * config.global_gain
    mix -= mix.mean()
    target_peak = 10.0 ** (-3.0 / 20.0)
    peak = float(np.max(np.abs(mix)))
    # Normalise to a constant -3 dBFS so a render is never left unusably quiet.
    # Previously the master was only attenuated, which produced -12 dBFS mixes.
    master_scale = target_peak / peak if peak > 1e-9 else 1.0
    for name in stems:
        stems[name] = (stems[name] * config.global_gain * master_scale).astype(np.float32)
    mix = (mix * master_scale).astype(np.float32)
    if not np.all(np.isfinite(mix)) or max((np.max(np.abs(stem)) for stem in stems.values()), default=0) > 1:
        raise ValueError("Multi-layer synthesis produced invalid or clipped audio")
    return MultiLayerAudio(
        mix=mix,
        stems=stems,
        events=tuple(events),
        peak=float(np.max(np.abs(mix))),
    )
