"""Create a subject-disjoint manifest without fabricating missing labels."""

import argparse
import csv
import hashlib
from pathlib import Path
import random
import re

from PIL import Image


SUPPORTED_EXTENSIONS = {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}
SOCOFING_PATTERN = re.compile(
    r"^(?P<subject>\d+)__(?P<gender>[MF])_(?P<hand>Left|Right)_"
    r"(?P<finger>thumb|index|middle|ring|little)_finger"
    r"(?:_(?P<alteration>CR|Obl|Zcut))?$",
    re.IGNORECASE,
)


def _pixel_hash(path):
    with Image.open(path) as image:
        pixels = image.convert("L").tobytes()
    return hashlib.sha256(pixels).hexdigest()


def _find_mask(image_path, image_root, mask_root):
    if mask_root is None:
        return ""
    relative = image_path.relative_to(image_root)
    candidates = [
        mask_root / relative.with_suffix(extension)
        for extension in (".png", ".bmp", ".tif", ".tiff")
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return ""


def _scan_socofing(root, masks_root, source, licence):
    image_paths = list((root / "Real").glob("*.BMP"))
    image_paths += list((root / "Altered").glob("*/*.BMP"))
    records = []
    for path in sorted(image_paths):
        match = SOCOFING_PATTERN.match(path.stem)
        if not match:
            continue
        labels = match.groupdict()
        relative = path.relative_to(root)
        difficulty = relative.parts[1].replace("Altered-", "") if relative.parts[0] == "Altered" else ""
        base_name = path.stem.rsplit("_", 1)[0] + path.suffix if labels["alteration"] else path.name
        paired_real = root / "Real" / base_name
        with Image.open(path) as image:
            width, height = image.size
        records.append(
            {
                "image_path": str(path.resolve()),
                "ridge_mask_path": _find_mask(path, root, masks_root),
                "orientation_path": "",
                "dataset_source": source,
                "subject_id": labels["subject"],
                "group_id": labels["subject"],
                "gender": labels["gender"],
                "hand": labels["hand"],
                "finger": labels["finger"],
                "alteration": labels["alteration"] or "",
                "difficulty": difficulty,
                "paired_real_path": str(paired_real.resolve()) if paired_real.is_file() else "",
                "width": width,
                "height": height,
                "pixel_hash": _pixel_hash(path),
                "licence": licence,
                "split": "",
            }
        )
    return records


def _scan_generic(root, masks_root, source, licence):
    records = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        with Image.open(path) as image:
            width, height = image.size
        subject = path.stem.split("__", 1)[0]
        records.append(
            {
                "image_path": str(path.resolve()),
                "ridge_mask_path": _find_mask(path, root, masks_root),
                "orientation_path": "",
                "dataset_source": source,
                "subject_id": subject,
                "group_id": subject,
                "gender": "",
                "hand": "",
                "finger": "",
                "alteration": "",
                "difficulty": "",
                "paired_real_path": "",
                "width": width,
                "height": height,
                "pixel_hash": _pixel_hash(path),
                "licence": licence,
                "split": "",
            }
        )
    return records


def _subject_disjoint_splits(records, seed, train_ratio, validation_ratio):
    subjects = sorted({record["subject_id"] for record in records})
    generator = random.Random(seed)
    generator.shuffle(subjects)
    train_end = round(len(subjects) * train_ratio)
    validation_end = train_end + round(len(subjects) * validation_ratio)
    subject_split = {
        subject: "train" if index < train_end else "validation" if index < validation_end else "test"
        for index, subject in enumerate(subjects)
    }
    for record in records:
        record["split"] = subject_split[record["subject_id"]]

    # Flag exact cross-subject duplicates. They are not silently re-labelled;
    # users can quarantine them before training.
    hashes = {}
    duplicate_hashes = set()
    for record in records:
        previous = hashes.setdefault(record["pixel_hash"], record["subject_id"])
        if previous != record["subject_id"]:
            duplicate_hashes.add(record["pixel_hash"])
    for record in records:
        if record["pixel_hash"] in duplicate_hashes:
            record["quality_flags"] = "cross_subject_exact_duplicate"
            record["split"] = "quarantine"
        else:
            record["quality_flags"] = ""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Create a fingerprint dataset manifest")
    parser.add_argument("--images-root", required=True)
    parser.add_argument("--masks-root")
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset-source", default="unknown")
    parser.add_argument("--licence", default="unknown")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    parser.add_argument("--overwrite", action="store_true")
    arguments = parser.parse_args(argv)

    root = Path(arguments.images_root)
    if not root.is_dir():
        parser.error(f"Images root does not exist: {root}")
    masks_root = Path(arguments.masks_root) if arguments.masks_root else None
    if masks_root is not None and not masks_root.is_dir():
        parser.error(f"Masks root does not exist: {masks_root}")
    if arguments.train_ratio <= 0 or arguments.validation_ratio <= 0 or arguments.train_ratio + arguments.validation_ratio >= 1:
        parser.error("Split ratios must be positive and leave a non-empty test ratio")

    is_socofing = (root / "Real").is_dir() and (root / "Altered").is_dir()
    scanner = _scan_socofing if is_socofing else _scan_generic
    records = scanner(root, masks_root, arguments.dataset_source, arguments.licence)
    if not records:
        parser.error("No supported fingerprint images were found")
    _subject_disjoint_splits(
        records, arguments.seed, arguments.train_ratio, arguments.validation_ratio
    )

    output = Path(arguments.output)
    if output.exists() and not arguments.overwrite:
        parser.error(f"Manifest already exists: {output}. Use --overwrite to replace it.")
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with output.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    labeled = sum(bool(record["ridge_mask_path"]) for record in records)
    print(f"Manifest: {output}")
    print(f"Images: {len(records)}")
    print(f"Ridge masks: {labeled}")
    print("No missing annotation was fabricated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
