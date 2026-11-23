import numpy as np
from scipy.io import wavfile
import pytest

from config import SonificationConfig
from src.audio_exporter import export_wav
from src.audio_generator import generate_audio
from src.feature_extractor import FeatureSequence
from src.sound_mapper import map_features_to_sound


def make_features(reverse=False):
    values = np.linspace(0.05, 0.95, 12, dtype=np.float32)
    if reverse:
        values = values[::-1]
    return FeatureSequence(
        scan_position=np.linspace(0, 1, 12, dtype=np.float32),
        x_position=np.linspace(0, 1, 12, dtype=np.float32),
        y_position=np.linspace(0, 1, 12, dtype=np.float32),
        ridge_darkness=values,
        local_contrast=np.clip(values * 0.8, 0, 1),
        ridge_density=np.clip(1 - values * 0.7, 0, 1),
        foreground_coverage=np.ones(12, dtype=np.float32),
    )


def test_audio_is_deterministic_finite_non_silent_and_does_not_clip():
    config = SonificationConfig(duration=0.25)
    parameters = map_features_to_sound(make_features(), config)

    first = generate_audio(parameters, config)
    second = generate_audio(parameters, config)

    assert np.array_equal(first, second)
    assert len(first) == round(config.duration * config.sample_rate)
    assert np.isfinite(first).all()
    assert 0.01 < np.max(np.abs(first)) <= 1.0


def test_different_patterns_produce_different_audio():
    config = SonificationConfig(duration=0.2)
    forward = generate_audio(map_features_to_sound(make_features(), config), config)
    reverse = generate_audio(
        map_features_to_sound(make_features(reverse=True), config), config
    )

    assert not np.array_equal(forward, reverse)
    assert np.mean(np.abs(forward - reverse)) > 0.05


def test_exported_wav_has_required_pcm_format_and_duration(tmp_path):
    config = SonificationConfig(duration=0.2)
    audio = generate_audio(map_features_to_sound(make_features(), config), config)
    output = tmp_path / "fingerprint.wav"

    export_wav(output, audio, config.sample_rate)
    sample_rate, pcm = wavfile.read(output)

    assert sample_rate == 44100
    assert pcm.dtype == np.int16
    assert pcm.ndim == 1
    assert len(pcm) == round(config.duration * config.sample_rate)
    assert np.max(np.abs(pcm.astype(np.int32))) <= np.iinfo(np.int16).max
    assert np.max(np.abs(pcm.astype(np.int32))) > 100


def test_export_does_not_overwrite_silently(tmp_path):
    path = tmp_path / "existing.wav"
    path.write_bytes(b"existing")
    audio = np.zeros(100, dtype=np.float32)

    with pytest.raises(FileExistsError, match="--overwrite"):
        export_wav(path, audio)
