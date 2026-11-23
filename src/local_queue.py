"""Hand each finished piece to a folder on this machine.

The installation runs website and TouchDesigner on the same laptop, so the
simplest hand-off is a directory. TouchDesigner watches ``latest/`` and always
reads the same four filenames, which means the patch never has to be re-pathed
between visitors.

Publishing is off unless ``RIDGE_QUEUE_DIR`` is set, so the public deployment
never tries to write anything.
"""

import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

ENV_VAR = "RIDGE_QUEUE_DIR"

# Fixed names TouchDesigner can hard-code.
WANTED = {
    "_mix.wav": "mix.wav",
    "_controls.wav": "controls.wav",
    "_events.json": "events.json",
    "mask.png": "mask.png",
    "roi.png": "roi.png",
}
KEEP_ARCHIVE = 20


def queue_directory():
    """The configured hand-off folder, or None when publishing is disabled."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    return Path(value).expanduser()


def _members(visual_zip):
    """Map the package's files onto the fixed names TouchDesigner expects."""
    found = {}
    with zipfile.ZipFile(io.BytesIO(visual_zip)) as archive:
        for name in archive.namelist():
            for suffix, target in WANTED.items():
                if name == suffix or name.endswith(suffix):
                    found[target] = archive.read(name)
    return found


def publish(visual_zip, source_name, directory=None):
    """Write the package to ``<directory>/latest`` and archive a copy.

    Returns the path written to, or None when publishing is disabled. Files are
    written under temporary names and moved into place, so TouchDesigner never
    reads a half-written WAV.
    """
    directory = Path(directory).expanduser() if directory else queue_directory()
    if directory is None:
        return None

    members = _members(visual_zip)
    if "mix.wav" not in members or "controls.wav" not in members:
        raise ValueError("visual package is missing its audio")

    latest = directory / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    staging = directory / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

    written = []
    for name, payload in members.items():
        temporary = staging / f"{name}.part"
        temporary.write_bytes(payload)
        written.append((temporary, latest / name))

    meta = {
        "source": source_name,
        "published_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "files": sorted(members),
    }
    (staging / "meta.json.part").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    written.append((staging / "meta.json.part", latest / "meta.json"))

    # Swap everything in one pass so the folder is never half-updated.
    for temporary, target in written:
        os.replace(temporary, target)

    archive_root = directory / "archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    destination = archive_root / stamp
    suffix = 1
    while destination.exists():  # keep every print, even within one second
        destination = archive_root / f"{stamp}-{suffix}"
        suffix += 1
    shutil.copytree(latest, destination, dirs_exist_ok=True)
    _prune(archive_root)
    return latest


def _prune(archive_root):
    """Keep only the most recent prints so a long day cannot fill the disk."""
    entries = sorted(
        (path for path in archive_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )
    for old in entries[:-KEEP_ARCHIVE]:
        shutil.rmtree(old, ignore_errors=True)
