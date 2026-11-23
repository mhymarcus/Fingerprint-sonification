"""A folder queue between the website and TouchDesigner.

Two folders, two meanings:

``pending/``  pieces that have been generated but not shown yet. Finishing a
              piece never changes what is on screen, so the previous visitor's
              work keeps playing while the next person is made ready.
``latest/``   the piece TouchDesigner loads. It only changes when the operator
              advances the queue.
``played/``   the last few promoted pieces, kept so a show can be re-run.

Fixed filenames inside ``latest/`` mean a patch never has to be re-pathed
between visitors.
"""

import io
import json
import os
from pathlib import Path
import shutil
import time
import zipfile

ENV_VAR = "RIDGE_QUEUE_DIR"
KEEP_PLAYED = 20
STAMP = "%Y%m%d-%H%M%S"

WANTED = {
    "_mix.wav": "mix.wav",
    "_controls.wav": "controls.wav",
    "_events.json": "events.json",
    "mask.png": "mask.png",
    "roi.png": "roi.png",
}


def queue_directory():
    """The configured queue root, or None when the feature is disabled."""
    value = os.environ.get(ENV_VAR)
    if not value:
        return None
    return Path(value).expanduser()


def _root(directory):
    return Path(directory).expanduser() if directory else queue_directory()


def _members(visual_zip):
    """Map the package's files onto the fixed names a patch expects."""
    found = {}
    with zipfile.ZipFile(io.BytesIO(visual_zip)) as archive:
        for name in archive.namelist():
            for suffix, target in WANTED.items():
                if name == suffix or name.endswith(suffix):
                    found[target] = archive.read(name)
    return found


def _unique(folder, stamp):
    candidate = folder / stamp
    suffix = 1
    while candidate.exists():
        candidate = folder / f"{stamp}-{suffix}"
        suffix += 1
    return candidate


def _write(folder, members, meta):
    folder.mkdir(parents=True, exist_ok=True)
    for name, payload in members.items():
        temporary = folder / f"{name}.part"
        temporary.write_bytes(payload)
        os.replace(temporary, folder / name)
    (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def publish(visual_zip, source_name, directory=None):
    """Add a finished piece to the queue. Does not change what is playing."""
    root = _root(directory)
    if root is None:
        return None

    members = _members(visual_zip)
    if "mix.wav" not in members or "controls.wav" not in members:
        raise ValueError("visual package is missing its audio")

    target = _unique(root / "pending", time.strftime(STAMP))
    _write(
        target,
        members,
        {
            "source": source_name,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "waiting",
        },
    )
    return target


def pending(directory=None):
    """Waiting pieces, oldest first."""
    root = _root(directory)
    if root is None:
        return []
    folder = root / "pending"
    if not folder.is_dir():
        return []
    return sorted(
        (path for path in folder.iterdir() if path.is_dir()),
        key=lambda path: path.name,
    )


def current(directory=None):
    """Metadata of the piece on screen, or None."""
    root = _root(directory)
    if root is None:
        return None
    meta = root / "latest" / "meta.json"
    if not meta.is_file():
        return None
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def promote(directory=None, identifier=None):
    """Move a waiting piece into ``latest/``. Returns the path, or None.

    The copy is staged and swapped in, so TouchDesigner cannot read a
    half-written WAV if someone presses its button mid-advance.
    """
    root = _root(directory)
    if root is None:
        return None
    items = pending(root)
    if not items:
        return None
    chosen = next((path for path in items if path.name == identifier), items[0])

    staging = root / ".latest-staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(chosen, staging)

    meta = json.loads((staging / "meta.json").read_text(encoding="utf-8"))
    meta["status"] = "showing"
    meta["shown_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (staging / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    latest = root / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    staging.rename(latest)

    played = root / "played"
    played.mkdir(parents=True, exist_ok=True)
    # A promoted folder frees its name again, so a later piece in the same
    # second can want the same destination. Never move into an existing folder.
    shutil.move(str(chosen), str(_unique(played, chosen.name)))
    _prune(played)
    return latest


def _prune(folder):
    entries = sorted(
        (path for path in folder.iterdir() if path.is_dir()), key=lambda path: path.name
    )
    for old in entries[:-KEEP_PLAYED]:
        shutil.rmtree(old, ignore_errors=True)
