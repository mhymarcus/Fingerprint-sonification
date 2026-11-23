from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from src.image_processor import load_grayscale_image, preprocess_image


def synthetic_fingerprint(size=96, horizontal=False):
    y, x = np.mgrid[:size, :size]
    coordinate = y if horizontal else x
    ridges = 220 - 150 * (0.5 + 0.5 * np.cos(coordinate * 0.55 + y * 0.025))
    return np.clip(ridges, 0, 255).astype(np.uint8)


@pytest.mark.parametrize("extension", [".png", ".jpg", ".jpeg", ".bmp"])
def test_supported_image_formats_are_processed(tmp_path, extension):
    path = tmp_path / f"fingerprint{extension}"
    Image.fromarray(synthetic_fingerprint()).save(path)

    result = preprocess_image(path, analysis_width=128, analysis_height=96)

    assert result.grayscale.shape == (96, 128)
    assert result.foreground_mask.shape == (96, 128)
    assert np.isfinite(result.grayscale).all()
    assert 0 <= result.grayscale.min() <= result.grayscale.max() <= 1


def test_missing_and_unsupported_images_have_clear_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="does not exist"):
        load_grayscale_image(tmp_path / "missing.png")

    unsupported = tmp_path / "fingerprint.gif"
    unsupported.write_bytes(b"not an image")
    with pytest.raises(ValueError, match="Unsupported image format"):
        load_grayscale_image(unsupported)


def test_corrupted_image_is_rejected(tmp_path):
    path = tmp_path / "corrupted.png"
    path.write_bytes(b"not a real png")
    with pytest.raises(ValueError, match="unreadable or corrupted"):
        load_grayscale_image(path)


def test_uniform_image_is_handled_with_warning(tmp_path):
    path = tmp_path / "uniform.png"
    Image.fromarray(np.full((64, 64), 128, dtype=np.uint8)).save(path)
    with pytest.warns(UserWarning, match="uniform brightness"):
        result = preprocess_image(path, analysis_width=64, analysis_height=64)
    assert np.isfinite(result.grayscale).all()
