import csv

import numpy as np
import pytest

from src.ml.dataset import read_manifest
from src.ml.model import CompactUNet

def test_supervised_training_rejects_manifest_without_real_masks(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(
            manifest_file,
            fieldnames=("image_path", "ridge_mask_path", "subject_id", "split"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "image_path": "fingerprint.png",
                "ridge_mask_path": "",
                "subject_id": "1",
                "split": "train",
            }
        )

    with pytest.raises(ValueError, match="requires real ridge_mask_path"):
        read_manifest(manifest, require_masks=True)


def test_compact_unet_forward_shape_when_torch_is_available():
    torch = pytest.importorskip("torch")
    model = CompactUNet(base_channels=4)
    inputs = torch.zeros((2, 1, 64, 64))

    outputs = model(inputs)

    assert outputs.shape == (2, 1, 64, 64)
    assert torch.isfinite(outputs).all()


def test_ml_inference_runs_at_the_analysis_resolution(tmp_path):
    torch = pytest.importorskip("torch")

    from src.image_processor import ProcessedImage
    from src.ml.inference import analyze_ml
    from src.ml.model import CompactUNet
    from src.ml.synthetic import generate_synthetic_fingerprint

    model = CompactUNet(base_channels=4)
    with torch.no_grad():
        # Deterministic prediction: every pixel is ridge.
        model.output.weight.zero_()
        model.output.bias.fill_(2.0)
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_channels": 4,
            "image_size": 64,
        },
        checkpoint,
    )

    synthetic = generate_synthetic_fingerprint(size=96, seed=3)
    processed = ProcessedImage(
        source_path=None,
        original=synthetic.image,
        grayscale=synthetic.image.astype(np.float32) / 255.0,
        foreground_mask=synthetic.roi_mask,
    )

    analysis = analyze_ml(
        processed, checkpoint, orientation_block_size=12, minimum_ridge_length=3
    )

    assert analysis.mode == "ml"
    # The net is fully convolutional: it must keep the analysis resolution
    # instead of rescaling to the checkpoint's training size, which destroys
    # ridge detail on higher-resolution inputs.
    assert analysis.ridge_mask.shape == processed.grayscale.shape
    assert analysis.ridge_mask.any()


def test_ml_inference_warns_on_a_large_resolution_mismatch(tmp_path):
    torch = pytest.importorskip("torch")

    from src.image_processor import ProcessedImage
    from src.ml.inference import analyze_ml
    from src.ml.model import CompactUNet
    from src.ml.synthetic import generate_synthetic_fingerprint

    model = CompactUNet(base_channels=4)
    with torch.no_grad():
        model.output.weight.zero_()
        model.output.bias.fill_(2.0)
    checkpoint = tmp_path / "weights.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "base_channels": 4,
            "image_size": 32,
        },
        checkpoint,
    )

    synthetic = generate_synthetic_fingerprint(size=128, seed=3)
    processed = ProcessedImage(
        source_path=None,
        original=synthetic.image,
        grayscale=synthetic.image.astype(np.float32) / 255.0,
        foreground_mask=synthetic.roi_mask,
    )

    with pytest.warns(UserWarning, match="trained at 32px"):
        analyze_ml(processed, checkpoint, orientation_block_size=16, minimum_ridge_length=3)
