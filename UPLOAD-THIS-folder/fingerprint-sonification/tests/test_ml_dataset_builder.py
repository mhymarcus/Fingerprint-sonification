import csv

import numpy as np
from PIL import Image

from src.ml.synthetic import generate_synthetic_fingerprint
from training.build_ml_dataset import main as build_dataset


def test_builder_writes_aligned_pairs_and_records_provenance(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        synthetic = generate_synthetic_fingerprint(size=96, seed=index)
        Image.fromarray(synthetic.image).save(
            source / f"{index}__M_Left_index_finger.png"
        )

    output = tmp_path / "built"
    exit_code = build_dataset(
        [
            "--images-root",
            str(source),
            "--output",
            str(output),
            "--image-size",
            "128",
            "--train",
            "3",
            "--validation",
            "1",
            "--test",
            "1",
            "--overwrite",
        ]
    )
    assert exit_code == 0

    manifest_path = output / "manifest.csv"
    assert manifest_path.is_file()
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8")))
    # 6 subjects split 4/1/1; the requested sample sizes are 3/1/1.
    assert len(rows) == 5
    assert {row["split"] for row in rows} == {"train", "validation", "test"}

    for row in rows:
        image = np.asarray(Image.open(row["image_path"]))
        mask = np.asarray(Image.open(row["ridge_mask_path"]))
        # Pairs must line up pixel for pixel, at the requested model resolution.
        assert image.shape == mask.shape == (128, 128)
        # Provenance travels with every record.
        assert row["label_source"] == "cv_baseline"
        assert set(np.unique(mask)) <= {0, 255}
        assert 0 < (mask > 127).mean() < 1


def test_builder_refuses_to_overwrite_a_manifest(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    synthetic = generate_synthetic_fingerprint(size=96, seed=0)
    Image.fromarray(synthetic.image).save(source / "0__M_Left_index_finger.png")

    output = tmp_path / "built"
    arguments = [
        "--images-root",
        str(source),
        "--output",
        str(output),
        "--image-size",
        "128",
        "--train",
        "1",
        "--validation",
        "1",
        "--test",
        "1",
    ]
    build_dataset([*arguments, "--overwrite"])

    try:
        build_dataset(arguments)
    except SystemExit as error:
        assert error.code == 2
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("builder overwrote an existing manifest")
