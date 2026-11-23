import csv
import http.server
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request

import cv2
import numpy as np
from PIL import Image
import pytest

from src.ml.synthetic import generate_synthetic_fingerprint
from training import correct_masks
from training.build_ml_dataset import main as build_dataset


@pytest.fixture
def dataset(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for index in range(6):
        synthetic = generate_synthetic_fingerprint(size=96, seed=index)
        Image.fromarray(synthetic.image).save(
            source / f"{index}__M_Left_index_finger.png"
        )
    output = tmp_path / "built"
    build_dataset(
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
    return output


@pytest.fixture
def server(dataset):
    session = correct_masks.Session(dataset, splits=("train", "validation"))
    httpd = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), correct_masks.make_handler(session)
    )
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", session
    finally:
        httpd.shutdown()
        httpd.server_close()


def _png(array):
    ok, buffer = cv2.imencode(".png", array)
    assert ok
    return buffer.tobytes()


def test_serves_the_tool_and_lists_only_review_splits(server):
    base, session = server
    with urllib.request.urlopen(base + "/") as response:
        assert response.status == 200
        assert b"Ridge mask corrector" in response.read()

    with urllib.request.urlopen(base + "/api/items") as response:
        items = json.load(response)

    assert len(items) == len(session.order)
    assert {item["split"] for item in items} <= {"train", "validation"}
    assert all(item["status"] == "pending" for item in items)


def test_serves_decodable_image_and_mask(server):
    base, session = server
    identifier = session.order[0]
    for path in (f"/api/image/{identifier}", f"/api/mask/{identifier}"):
        with urllib.request.urlopen(base + path) as response:
            payload = response.read()
        decoded = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        assert decoded is not None
        assert decoded.shape == (128, 128)


def test_accept_marks_the_sample_without_writing_a_mask(server):
    base, session = server
    identifier = session.order[0]
    request = urllib.request.Request(
        base + f"/api/accept/{identifier}", data=b"", method="POST"
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)

    assert payload["status"] == "accepted"
    assert payload["counts"]["accepted"] == 1
    assert not session.manual_path(identifier).exists()
    # Accepted means "the CV mask is right", so it stays the current mask.
    assert session.current_mask_path(identifier) == Path(
        session.records[identifier]["ridge_mask_path"]
    )


def test_save_writes_a_binary_mask_and_becomes_current(server):
    base, session = server
    identifier = session.order[0]
    original = cv2.imread(
        str(session.current_mask_path(identifier)), cv2.IMREAD_GRAYSCALE
    )
    edited = original.copy()
    edited[:8, :8] = 255

    request = urllib.request.Request(
        base + f"/api/save/{identifier}",
        data=_png(edited),
        method="POST",
        headers={"Content-Type": "image/png"},
    )
    with urllib.request.urlopen(request) as response:
        payload = json.load(response)

    assert payload["status"] == "edited"
    written = cv2.imread(str(session.manual_path(identifier)), cv2.IMREAD_GRAYSCALE)
    assert written.shape == original.shape
    assert set(np.unique(written)) <= {0, 255}
    assert (written[:8, :8] == 255).all()
    assert session.current_mask_path(identifier) == session.manual_path(identifier)


def test_save_rejects_a_mask_of_the_wrong_size(server):
    base, session = server
    identifier = session.order[0]
    wrong = np.zeros((32, 32), dtype=np.uint8)

    request = urllib.request.Request(
        base + f"/api/save/{identifier}",
        data=_png(wrong),
        method="POST",
        headers={"Content-Type": "image/png"},
    )
    with pytest.raises(urllib.error.HTTPError) as error:
        urllib.request.urlopen(request)
    assert error.value.code == 400
    assert not session.manual_path(identifier).exists()


def test_unknown_sample_and_action_are_rejected(server):
    base, session = server
    with pytest.raises(urllib.error.HTTPError) as missing:
        urllib.request.urlopen(base + "/api/mask/does_not_exist")
    assert missing.value.code == 404

    request = urllib.request.Request(
        base + f"/api/destroy/{session.order[0]}", data=b"", method="POST"
    )
    with pytest.raises(urllib.error.HTTPError) as unknown:
        urllib.request.urlopen(request)
    assert unknown.value.code == 404


def test_cli_refuses_to_open_the_test_split(dataset):
    with pytest.raises(SystemExit):
        correct_masks.main(
            ["--dataset", str(dataset), "--splits", "train,test", "--no-browser"]
        )


def test_export_attributes_each_row_to_how_its_mask_was_obtained(
    server, dataset, tmp_path
):
    base, session = server
    accepted, edited = session.order[0], session.order[1]

    urllib.request.urlopen(
        urllib.request.Request(
            base + f"/api/accept/{accepted}", data=b"", method="POST"
        )
    ).close()
    original = cv2.imread(str(session.records[edited]["ridge_mask_path"]), 0)
    urllib.request.urlopen(
        urllib.request.Request(
            base + f"/api/save/{edited}",
            data=_png(original),
            method="POST",
            headers={"Content-Type": "image/png"},
        )
    ).close()

    target = tmp_path / "labeled.csv"
    correct_masks.export_manifest(dataset, target)
    rows = list(csv.DictReader(target.open(encoding="utf-8")))
    sources = {row["label_source"] for row in rows}

    assert "manual" in sources
    assert "cv_baseline_accepted" in sources
    manual_row = next(row for row in rows if row["label_source"] == "manual")
    assert "ridge_masks_manual" in manual_row["ridge_mask_path"]
    assert Path(manual_row["ridge_mask_path"]).is_file()
