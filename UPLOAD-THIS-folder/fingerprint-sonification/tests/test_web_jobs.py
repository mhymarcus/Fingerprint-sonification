import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from config import SonificationConfig
from src.ml.synthetic import generate_synthetic_fingerprint
from src.web_jobs import MAX_UPLOAD_BYTES, sonify, upload_digest


def sample_bytes(seed=3, size=192, fmt="PNG"):
    synthetic = generate_synthetic_fingerprint(size=size, seed=seed)
    buffer = io.BytesIO()
    Image.fromarray(synthetic.image).save(buffer, format=fmt)
    return buffer.getvalue()


@pytest.fixture
def config():
    return SonificationConfig(duration=0.4, fade_duration=0.05)


def test_sonify_returns_audio_stems_figures_and_geometry(config):
    result = sonify(sample_bytes(), "print.png", config)

    assert result.mix_wav[:4] == b"RIFF"
    assert set(result.stem_wavs) == {
        "ridge_length",
        "ridge_position",
        "ridge_orientation",
        "ridge_texture",
    }
    for payload in result.stem_wavs.values():
        assert payload[:4] == b"RIFF"

    with zipfile.ZipFile(io.BytesIO(result.stems_zip)) as archive:
        names = archive.namelist()
    assert len(names) == 4
    assert all(name.endswith(".wav") for name in names)

    assert set(result.figures) == {"analysis", "layers", "spectrogram"}
    for payload in result.figures.values():
        assert payload[:8] == b"\x89PNG\r\n\x1a\n"

    assert result.summary["segments"] > 0
    assert result.summary["duration_seconds"] == 0.4
    assert result.summary["peak_dbfs"] <= -2.9

    document = json.loads(result.geometry_json())
    assert document["analysis_mode"] == "cv"
    assert document["segment_count"] == result.summary["segments"]
    assert document["sound_events"]


def test_bmp_uploads_are_accepted(config):
    result = sonify(sample_bytes(fmt="BMP"), "print.bmp", config)
    assert result.mix_wav[:4] == b"RIFF"


def test_unsupported_extension_is_rejected(config):
    with pytest.raises(ValueError, match="Unsupported image format"):
        sonify(sample_bytes(), "print.gif", config)


def test_empty_upload_is_rejected(config):
    with pytest.raises(ValueError, match="No image data"):
        sonify(b"", "print.png", config)


def test_corrupt_upload_is_rejected(config):
    with pytest.raises(ValueError, match="unreadable or corrupted"):
        sonify(b"not really an image", "print.png", config)


def test_unknown_analysis_mode_is_rejected(config):
    with pytest.raises(ValueError, match="Unsupported analysis mode"):
        sonify(sample_bytes(), "print.png", config, analysis_mode="pixel")


def test_upload_limit_is_exposed_and_reasonable():
    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024


def test_digest_is_stable_and_content_addressed():
    payload = sample_bytes()
    assert upload_digest(payload) == upload_digest(payload)
    assert upload_digest(payload) != upload_digest(payload + b"\x00")
