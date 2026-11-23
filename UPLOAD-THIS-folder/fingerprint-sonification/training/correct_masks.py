"""Correct CV-generated ridge masks in a local browser.

The correction loop is deliberately short: open the page, look at the CV mask
laid over the print, paint over what is wrong, press save, and the next print
loads. Nothing is installed and nothing leaves the machine — this is a stdlib
HTTP server bound to 127.0.0.1.

    python training/correct_masks.py --dataset datasets/ml_ridge
    python training/correct_masks.py --dataset datasets/ml_ridge --export

Only the train and validation splits are offered by default. Correcting the
test split would make it worthless as held-out data.
"""

import argparse
import csv
from datetime import datetime, timezone
import http.server
from http import HTTPStatus
import json
from pathlib import Path
import socket
import sys
import threading
import webbrowser

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "labeler"
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
}
MAX_BODY_BYTES = 8 * 1024 * 1024


class Session:
    """Owns the manifest, the review state and the corrected mask files."""

    def __init__(self, dataset, splits=("train", "validation"), limit=None):
        self.dataset = Path(dataset)
        manifest_path = self.dataset / "manifest.csv"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
            rows = list(csv.DictReader(manifest_file))
        if not rows:
            raise ValueError("Manifest contains no records")
        wanted = [row for row in rows if (row.get("split") or "") in splits]
        if not wanted:
            raise ValueError(f"Manifest has no records for splits: {sorted(splits)}")
        if limit:
            wanted = wanted[:limit]
        self.records = {}
        self.order = []
        for row in wanted:
            identifier = Path(row["ridge_mask_path"]).stem
            self.records[identifier] = row
            self.order.append(identifier)
        self.manual_dir = self.dataset / "ridge_masks_manual"
        self.manual_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.dataset / "review_state.json"
        self.state = {}
        if self.state_path.is_file():
            self.state = json.loads(self.state_path.read_text(encoding="utf-8"))

    # -- state ------------------------------------------------------------
    def status(self, identifier):
        return (self.state.get(identifier) or {}).get("status", "pending")

    def mark(self, identifier, status):
        self.state[identifier] = {
            "status": status,
            "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.save_state()

    def save_state(self):
        self.state_path.write_text(
            json.dumps(self.state, indent=2, sort_keys=True), encoding="utf-8"
        )

    def counts(self):
        result = {"pending": 0, "edited": 0, "accepted": 0, "skipped": 0}
        for identifier in self.order:
            status = self.status(identifier)
            result[status] = result.get(status, 0) + 1
        return result

    # -- files ------------------------------------------------------------
    def manual_path(self, identifier):
        return self.manual_dir / f"{identifier}.png"

    def current_mask_path(self, identifier):
        manual = self.manual_path(identifier)
        if self.status(identifier) == "edited" and manual.is_file():
            return manual
        return Path(self.records[identifier]["ridge_mask_path"])

    def image_path(self, identifier):
        return Path(self.records[identifier]["image_path"])

    def expected_shape(self, identifier):
        reference = cv2.imread(str(self.current_mask_path(identifier)), cv2.IMREAD_GRAYSCALE)
        if reference is None:
            raise ValueError(f"Unreadable mask for {identifier}")
        return reference.shape

    def store_mask(self, identifier, payload):
        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("Body is not a decodable image")
        if image.ndim == 3:
            if image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected = self.expected_shape(identifier)
        if image.shape != expected:
            raise ValueError(
                f"Mask shape {image.shape} does not match reference {expected}"
            )
        binary = (image > 127).astype(np.uint8) * 255
        if not cv2.imwrite(str(self.manual_path(identifier)), binary):
            raise OSError("Could not write the corrected mask")
        self.mark(identifier, "edited")
        return binary


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "MaskLabeler/1.0"
    bound_session = None

    def log_message(self, format, *args):  # keep the console readable
        return

    def _send(self, status, body=b"", content_type="application/json; charset=utf-8"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _json(self, payload, status=HTTPStatus.OK):
        self._send(status, json.dumps(payload).encode("utf-8"))

    def _error(self, status, message):
        self._json({"error": message}, status)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path in STATIC_FILES:
            name, content_type = STATIC_FILES[path]
            body = (STATIC_DIR / name).read_bytes()
            self._send(HTTPStatus.OK, body, content_type)
            return
        if path == "/api/items":
            self._json(self._items())
            return
        if path == "/api/meta":
            self._json(
                {
                    "counts": self.bound_session.counts(),
                    "total": len(self.bound_session.order),
                }
            )
            return
        if path.startswith("/api/image/") or path.startswith("/api/mask/"):
            identifier = path.rsplit("/", 1)[-1]
            if identifier not in self.bound_session.records:
                self._error(HTTPStatus.NOT_FOUND, "Unknown sample")
                return
            source = (
                self.bound_session.image_path(identifier)
                if path.startswith("/api/image/")
                else self.bound_session.current_mask_path(identifier)
            )
            try:
                body = Path(source).read_bytes()
            except OSError as error:
                self._error(HTTPStatus.NOT_FOUND, str(error))
                return
            self._send(HTTPStatus.OK, body, "image/png")
            return
        self._error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        parts = path.strip("/").split("/")
        if len(parts) != 3 or parts[0] != "api":
            self._error(HTTPStatus.NOT_FOUND, "Not found")
            return
        action, identifier = parts[1], parts[2]
        if identifier not in self.bound_session.records:
            self._error(HTTPStatus.NOT_FOUND, "Unknown sample")
            return
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self._error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Payload too large")
            return
        payload = self.rfile.read(length) if length else b""
        try:
            if action == "save":
                self.bound_session.store_mask(identifier, payload)
            elif action == "accept":
                self.bound_session.mark(identifier, "accepted")
            elif action == "skip":
                self.bound_session.mark(identifier, "skipped")
            else:
                self._error(HTTPStatus.NOT_FOUND, "Unknown action")
                return
        except (ValueError, OSError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._json(
            {
                "ok": True,
                "status": self.bound_session.status(identifier),
                "counts": self.bound_session.counts(),
            }
        )

    def _items(self):
        items = []
        for identifier in self.bound_session.order:
            record = self.bound_session.records[identifier]
            items.append(
                {
                    "id": identifier,
                    "status": self.bound_session.status(identifier),
                    "split": record.get("split", ""),
                    "difficulty": record.get("difficulty", ""),
                    "subject": record.get("subject_id", ""),
                    "source": Path(record.get("source_image") or record["image_path"]).name,
                }
            )
        return items


def export_manifest(dataset, output=None):
    """Write a manifest that attributes every row to how its mask was obtained."""
    session = Session(dataset, splits=("train", "validation", "test"))
    rows = []
    for identifier in session.order:
        record = dict(session.records[identifier])
        status = session.status(identifier)
        if status == "edited":
            record["ridge_mask_path"] = str(session.manual_path(identifier).resolve())
            record["label_source"] = "manual"
        elif status == "accepted":
            record["label_source"] = "cv_baseline_accepted"
        else:
            record["label_source"] = "cv_baseline"
        rows.append(record)
    target = Path(output) if output else Path(dataset) / "manifest_labeled.csv"
    fieldnames = list(rows[0].keys())
    with target.open("w", newline="", encoding="utf-8") as manifest_file:
        writer = csv.DictWriter(manifest_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for row in rows:
        counts[row["label_source"]] = counts.get(row["label_source"], 0) + 1
    print(f"Wrote {target}")
    print(f"label_source counts: {counts}")
    manual = sum(1 for row in rows if row["label_source"] == "manual")
    if manual == 0:
        print(
            "No masks were edited, so this manifest adds no human-verified data. "
            "Training on it still yields a distilled model."
        )
    return target


def _free_port(preferred):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", preferred))
            return preferred
        except OSError:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]


def make_handler(session):
    """Bind one session to a handler class so there is no module-level state."""

    class BoundHandler(Handler):
        bound_session = session

    return BoundHandler


def main(argv=None):
    parser = argparse.ArgumentParser(description="Interactively correct ridge masks")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=0, help="Review at most N samples")
    parser.add_argument(
        "--splits",
        default="train,validation",
        help="Comma-separated splits to review (never correct the test split)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--export", action="store_true", help="Write manifest_labeled.csv and exit")
    parser.add_argument("--output", help="Path for --export (default manifest_labeled.csv)")
    arguments = parser.parse_args(argv)

    if arguments.export:
        export_manifest(arguments.dataset, arguments.output)
        return 0

    splits = tuple(part.strip() for part in arguments.splits.split(",") if part.strip())
    if "test" in splits:
        parser.error(
            "Refusing to correct the test split: it must stay untouched to keep "
            "evaluation honest."
        )
    try:
        session = Session(
            arguments.dataset,
            splits=splits,
            limit=arguments.limit or None,
        )
    except (FileNotFoundError, ValueError) as error:
        parser.error(str(error))

    port = _free_port(arguments.port)
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(session)
    )
    url = f"http://127.0.0.1:{port}/"
    print(f"Reviewing {len(session.order)} samples from {session.dataset}")
    print(f"Open {url}")
    print("Keys: B brush · E erase · A accept · S skip · N next · P previous · U undo · [ ] size")
    print("Ctrl-C to stop.")
    if not arguments.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
        print(f"Counts: {session.counts()}")
        print("Run with --export to write manifest_labeled.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
