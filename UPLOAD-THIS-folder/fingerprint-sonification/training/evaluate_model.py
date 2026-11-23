"""Evaluate a real segmentation checkpoint with Dice and IoU."""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.dataset import FingerprintSegmentationDataset, read_manifest
from src.ml.model import CompactUNet, require_torch
from training.train_segmentation import _run_epoch


def _save_previews(model, dataset, directory, count, device):
    """Write image / reference / prediction / disagreement tiles."""
    import torch

    directory.mkdir(parents=True, exist_ok=True)
    model.eval()
    tiles = []
    with torch.no_grad():
        for index in range(min(count, len(dataset))):
            image, mask = dataset[index]
            logits = model(image[None].to(device))
            prediction = torch.sigmoid(logits)[0, 0].cpu().numpy() >= 0.5
            reference = mask[0].numpy() >= 0.5
            disagreement = np.zeros((*prediction.shape, 3), dtype=np.uint8)
            disagreement[reference & prediction] = (255, 255, 255)
            disagreement[reference & ~prediction] = (60, 60, 220)   # missed ridge
            disagreement[~reference & prediction] = (220, 180, 60)  # false ridge
            panels = [
                cv2.cvtColor((image[0].numpy() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor((reference * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                cv2.cvtColor((prediction * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR),
                disagreement,
            ]
            tiles.append(np.hstack(panels))
    if not tiles:
        return None
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i : i + 2]) for i in range(0, len(tiles), 2)])
    path = directory / "predictions.png"
    if not cv2.imwrite(str(path), grid):
        raise OSError(f"Could not write {path}")
    return path


def main(argv=None):
    require_torch()
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Evaluate ridge segmentation checkpoint")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--preview-dir",
        help="Directory for a prediction preview grid (image/reference/prediction/diff)",
    )
    parser.add_argument("--preview-count", type=int, default=6)
    parser.add_argument("--output-json", help="Write the metrics to this JSON file")
    arguments = parser.parse_args(argv)
    checkpoint_path = Path(arguments.checkpoint)
    if not checkpoint_path.is_file():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(
        checkpoint_path, map_location=arguments.device, weights_only=True
    )
    model = CompactUNet(base_channels=int(checkpoint.get("base_channels", 24))).to(arguments.device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dataset = FingerprintSegmentationDataset(
        arguments.manifest,
        arguments.split,
        image_size=int(checkpoint.get("image_size", 128)),
        augment=False,
    )
    metrics = _run_epoch(
        model, DataLoader(dataset, batch_size=arguments.batch_size), None, arguments.device
    )
    label_sources = sorted(
        {
            record.get("label_source") or "unspecified"
            for record in read_manifest(arguments.manifest, require_masks=True)
        }
    )
    print(f"split={arguments.split} images={len(dataset)} metrics={metrics}")
    print(f"label_source={label_sources}")
    if arguments.preview_dir:
        path = _save_previews(
            model,
            dataset,
            Path(arguments.preview_dir),
            arguments.preview_count,
            arguments.device,
        )
        if path:
            print(f"preview={path}")
    if arguments.output_json:
        payload = {
            "split": arguments.split,
            "images": len(dataset),
            "metrics": metrics,
            "label_sources": label_sources,
            "checkpoint": str(checkpoint_path),
            "checkpoint_metrics": checkpoint.get("validation_metrics"),
        }
        Path(arguments.output_json).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
