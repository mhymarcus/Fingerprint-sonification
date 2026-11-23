"""Export deterministic fingerprint-like images with exact geometry labels."""

import argparse
import csv
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ml.synthetic import generate_synthetic_fingerprint


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate labeled synthetic ridge patterns")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.count < 1:
        parser.error("count must be positive")

    output = Path(arguments.output)
    if output.exists() and any(output.iterdir()) and not arguments.overwrite:
        parser.error(
            f"Synthetic output directory is not empty: {output}. "
            "Use --overwrite to replace matching files."
        )
    image_directory = output / "images"
    mask_directory = output / "ridge_masks"
    orientation_directory = output / "orientation"
    for directory in (image_directory, mask_directory, orientation_directory):
        directory.mkdir(parents=True, exist_ok=True)

    rows = []
    for index in range(arguments.count):
        angle = (index * 137.507764) % 180.0
        curvature = 0.02 + 0.16 * ((index * 37) % 101) / 100.0
        sample = generate_synthetic_fingerprint(
            size=arguments.size,
            angle_degrees=angle,
            curvature=curvature,
            seed=arguments.seed + index,
        )
        stem = f"synthetic_{index:05d}"
        image_path = image_directory / f"{stem}.png"
        mask_path = mask_directory / f"{stem}.png"
        orientation_path = orientation_directory / f"{stem}.npy"
        Image.fromarray(sample.image).save(image_path)
        Image.fromarray(sample.ridge_mask.astype(np.uint8) * 255).save(mask_path)
        np.save(orientation_path, sample.orientation_radians)
        ratio = index / arguments.count
        split = "train" if ratio < 0.70 else "validation" if ratio < 0.85 else "test"
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "ridge_mask_path": str(mask_path.resolve()),
                "orientation_path": str(orientation_path.resolve()),
                "dataset_source": "synthetic_known_geometry",
                "subject_id": stem,
                "group_id": stem,
                "split": split,
                "licence": "generated_locally",
            }
        )

    manifest_path = output / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Synthetic dataset: {output}")
    print(f"Samples: {len(rows)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
