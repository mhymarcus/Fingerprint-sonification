"""Build a paired ridge-segmentation dataset from fingerprint images.

SOCOFing (and every other common fingerprint dataset) ships images without
ridge masks. This script turns a directory of images into a training-ready set
of aligned ``image`` / ``ridge_mask`` pairs at the exact resolution the
segmentation model uses, plus a manifest.

The masks are produced by the CV baseline in ``src.geometry``. They are a
*reference* produced by an algorithm, not hand-verified ground truth, and the
manifest records that in ``label_source`` so no downstream report can mistake
them for annotation. The resulting model is a distilled version of the CV
baseline: it reproduces it on unseen subjects, faster and with gaps bridged.
It is not evidence that ML segmentation beats the baseline.
"""

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.analyzer import enhance_fingerprint, segment_ridges_cv
from src.image_processor import preprocess_image


MANIFEST_FIELDS = (
    "image_path",
    "ridge_mask_path",
    "subject_id",
    "split",
    "label_source",
    "difficulty",
    "source_image",
)


def _subject_of(path):
    return path.stem.split("__", 1)[0]


def _difficulty_of(path, root):
    parts = path.relative_to(root).parts
    if parts and parts[0].lower() == "altered" and len(parts) > 2:
        return parts[1].replace("Altered-", "")
    return "Real"


def _scan(root):
    """Collect image paths with their subject and alteration difficulty."""
    records = []
    candidates = []
    if (root / "Real").is_dir() and (root / "Altered").is_dir():
        candidates.extend(sorted((root / "Real").glob("*.BMP")))
        candidates.extend(sorted((root / "Altered").glob("*/*.BMP")))
    else:
        for pattern in ("*.bmp", "*.BMP", "*.png", "*.jpg", "*.jpeg"):
            candidates.extend(sorted(root.rglob(pattern)))
    for path in candidates:
        if not path.is_file():
            continue
        subject = _subject_of(path)
        if not subject:
            continue
        records.append(
            {
                "path": path,
                "subject_id": subject,
                "difficulty": _difficulty_of(path, root),
            }
        )
    return records


def _split_subjects(records, seed, train_ratio, validation_ratio):
    subjects = sorted({record["subject_id"] for record in records})
    random.Random(seed).shuffle(subjects)
    train_end = round(len(subjects) * train_ratio)
    validation_end = train_end + round(len(subjects) * validation_ratio)
    assignment = {}
    for index, subject in enumerate(subjects):
        if index < train_end:
            assignment[subject] = "train"
        elif index < validation_end:
            assignment[subject] = "validation"
        else:
            assignment[subject] = "test"
    for record in records:
        record["split"] = assignment[record["subject_id"]]
    return assignment


def _sample(records, split, count, seed):
    """Stratified sample across alteration difficulty so all scans are covered."""
    pool = [record for record in records if record["split"] == split]
    buckets = defaultdict(list)
    for record in pool:
        buckets[record["difficulty"]].append(record)
    generator = random.Random(seed + len(split))
    for bucket in buckets.values():
        generator.shuffle(bucket)
    ordered = sorted(buckets)
    per_bucket = max(1, count // max(len(ordered), 1))
    chosen = []
    for name in ordered:
        chosen.extend(buckets[name][:per_bucket])
    if len(chosen) < count:
        remainder = [record for record in pool if record not in chosen]
        generator.shuffle(remainder)
        chosen.extend(remainder[: count - len(chosen)])
    chosen.sort(key=lambda record: str(record["path"]))
    return chosen[:count]


def _ridge_mask(processed):
    enhanced = enhance_fingerprint(
        processed.grayscale, processed.foreground_mask
    )
    return segment_ridges_cv(enhanced, processed.foreground_mask)


def _write_pair(record, output, image_size):
    processed = preprocess_image(
        record["path"], image_size, image_size, enhance_contrast=True
    )
    mask = _ridge_mask(processed)
    coverage = float(mask.sum()) / mask.size
    if coverage < 0.02 or coverage > 0.85:
        return None  # unusable label: empty or nearly solid
    stem = f"{record['subject_id']}_{record['difficulty']}_{record['path'].stem}"
    image_path = output / "images" / f"{stem}.png"
    mask_path = output / "ridge_masks" / f"{stem}.png"
    image = np.round(np.clip(processed.grayscale, 0.0, 1.0) * 255).astype(np.uint8)
    if not cv2.imwrite(str(image_path), image):
        raise OSError(f"Could not write {image_path}")
    if not cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255)):
        raise OSError(f"Could not write {mask_path}")
    return {
        "image_path": str(image_path.resolve()),
        "ridge_mask_path": str(mask_path.resolve()),
        "subject_id": record["subject_id"],
        "split": record["split"],
        "label_source": "cv_baseline",
        "difficulty": record["difficulty"],
        "source_image": str(record["path"].resolve()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build paired image/ridge-mask data for ML training"
    )
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--validation", type=int, default=80)
    parser.add_argument("--test", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)

    root = Path(arguments.images_root)
    if not root.is_dir():
        parser.error(f"Images root does not exist: {root}")
    if arguments.image_size < 32 or arguments.image_size % 8:
        parser.error("--image-size must be at least 32 and a multiple of 8")

    output = Path(arguments.output)
    manifest_path = output / "manifest.csv"
    if manifest_path.exists() and not arguments.overwrite:
        parser.error(f"{manifest_path} already exists. Use --overwrite to replace it.")
    (output / "images").mkdir(parents=True, exist_ok=True)
    (output / "ridge_masks").mkdir(parents=True, exist_ok=True)

    records = _scan(root)
    if not records:
        parser.error("No fingerprint images were found")
    _split_subjects(
        records, arguments.seed, arguments.train_ratio, arguments.validation_ratio
    )
    selection = {
        "train": _sample(records, "train", arguments.train, arguments.seed),
        "validation": _sample(
            records, "validation", arguments.validation, arguments.seed
        ),
        "test": _sample(records, "test", arguments.test, arguments.seed),
    }

    written = []
    for split, chosen in selection.items():
        kept = 0
        for record in chosen:
            row = _write_pair(record, output, arguments.image_size)
            if row is not None:
                written.append(row)
                kept += 1
        print(f"{split}: requested {len(chosen)}, wrote {kept} usable pairs")

    if not written:
        parser.error("No usable image/mask pairs were produced")
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(written)

    counts = defaultdict(int)
    for row in written:
        counts[row["split"]] += 1
    print(f"Manifest: {manifest_path}")
    print(f"Pairs: {len(written)} {dict(counts)}")
    print(f"Resolution: {arguments.image_size} x {arguments.image_size}")
    print(
        "label_source=CV baseline masks. These are algorithm-produced references, "
        "not hand-verified ground truth."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
