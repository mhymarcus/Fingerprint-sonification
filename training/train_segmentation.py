"""Explicit compact U-Net ridge-segmentation training command."""

import argparse
import random
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from src.ml.dataset import FingerprintSegmentationDataset, read_manifest
from src.ml.model import CompactUNet, require_torch


def _metrics(logits, targets):
    import torch

    predictions = torch.sigmoid(logits) >= 0.5
    targets = targets >= 0.5
    intersection = (predictions & targets).sum(dim=(1, 2, 3)).float()
    union = (predictions | targets).sum(dim=(1, 2, 3)).float()
    total = predictions.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = ((2 * intersection + 1) / (total + 1)).mean()
    iou = ((intersection + 1) / (union + 1)).mean()
    return dice, iou


def _run_epoch(model, loader, optimizer, device):
    import torch

    training = optimizer is not None
    model.train(training)
    bce = torch.nn.BCEWithLogitsLoss()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "batches": 0}
    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            logits = model(images)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * masks).sum(dim=(1, 2, 3))
            dice_loss = 1 - ((2 * intersection + 1) / (probabilities.sum(dim=(1, 2, 3)) + masks.sum(dim=(1, 2, 3)) + 1)).mean()
            loss = bce(logits, masks) + dice_loss
            if training:
                loss.backward()
                optimizer.step()
        dice, iou = _metrics(logits.detach(), masks)
        totals["loss"] += float(loss.detach())
        totals["dice"] += float(dice)
        totals["iou"] += float(iou)
        totals["batches"] += 1
    return {key: totals[key] / totals["batches"] for key in ("loss", "dice", "iou")}


def main(argv=None):
    require_torch()
    import torch
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Train ridge segmentation U-Net")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.epochs < 1:
        parser.error("epochs must be positive")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)

    try:
        records = read_manifest(arguments.manifest, require_masks=True)
        label_sources = sorted(
            {record.get("label_source") or "unspecified" for record in records}
        )
        train_dataset = FingerprintSegmentationDataset(
            arguments.manifest, "train", arguments.image_size, augment=True
        )
        validation_dataset = FingerprintSegmentationDataset(
            arguments.manifest, "validation", arguments.image_size, augment=False
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))
    print(f"label_source={label_sources}")
    if any(source.startswith("cv_baseline") for source in label_sources):
        print(
            "Note: masks are algorithm-produced references, not hand-verified "
            "ground truth. The checkpoint distils the CV baseline."
        )
    train_loader = DataLoader(train_dataset, batch_size=arguments.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=arguments.batch_size)
    model = CompactUNet(base_channels=arguments.base_channels).to(arguments.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=arguments.learning_rate)
    best_dice = -1.0
    checkpoint_path = Path(arguments.checkpoint)
    if checkpoint_path.exists() and not arguments.overwrite:
        parser.error(
            f"Checkpoint already exists: {checkpoint_path}. Use --overwrite to replace it."
        )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, arguments.epochs + 1):
        train_metrics = _run_epoch(model, train_loader, optimizer, arguments.device)
        validation_metrics = _run_epoch(model, validation_loader, None, arguments.device)
        print(f"epoch={epoch} train={train_metrics} validation={validation_metrics}")
        if validation_metrics["dice"] > best_dice:
            best_dice = validation_metrics["dice"]
            torch.save(
                {
                    "task": "ridge_segmentation",
                    "model_state_dict": model.state_dict(),
                    "base_channels": arguments.base_channels,
                    "image_size": arguments.image_size,
                    "validation_metrics": validation_metrics,
                    "label_sources": label_sources,
                    "seed": arguments.seed,
                },
                checkpoint_path,
            )
    print(f"Best checkpoint: {checkpoint_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
