"""Manifest-backed segmentation dataset with strict annotation validation."""

import csv
from pathlib import Path

import cv2
import numpy as np

from src.ml.model import require_torch


REQUIRED_COLUMNS = {"image_path", "ridge_mask_path", "subject_id", "split"}


def read_manifest(manifest_path, require_masks=True):
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        reader = csv.DictReader(manifest_file)
        missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")
        records = list(reader)
    if not records:
        raise ValueError("Manifest contains no records")
    if require_masks:
        missing_masks = [record["image_path"] for record in records if not record["ridge_mask_path"]]
        if missing_masks:
            raise ValueError(
                "Supervised segmentation requires real ridge_mask_path labels; "
                f"{len(missing_masks)} records are unlabeled"
            )
    return records


class FingerprintSegmentationDataset:
    def __init__(self, manifest_path, split, image_size=128, augment=False):
        require_torch()
        import torch

        self.torch = torch
        self.records = [
            record
            for record in read_manifest(manifest_path, require_masks=True)
            if record["split"] == split
        ]
        if not self.records:
            raise ValueError(f"Manifest has no records for split: {split}")
        self.image_size = image_size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(record["image_path"], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(record["ridge_mask_path"], cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Unreadable image/mask pair: {record['image_path']}")
        if image.shape != mask.shape:
            raise ValueError(f"Image/mask shape mismatch: {record['image_path']}")
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            # Draw per access so each epoch sees a different flip. Seed the
            # global NumPy RNG (or pass a generator) to make runs reproducible.
            if np.random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()
        image_tensor = self.torch.from_numpy(image.astype(np.float32)[None] / 255.0)
        mask_tensor = self.torch.from_numpy((mask > 127).astype(np.float32)[None])
        return image_tensor, mask_tensor
