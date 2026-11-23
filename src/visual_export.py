"""Export a TouchDesigner-ready visual package for one sonification.

TouchDesigner should not have to guess what the sound is doing. The pipeline
already knows which layer fired, where on the print it came from, and how the
ridges are oriented, so this module writes that knowledge out in two forms:

``controls.wav``
    Six channels of 0..1 control signal at 1000 Hz, sample-aligned with the
    mixture. Loading it into an Audio File In CHOP gives TouchDesigner curves
    that cannot drift out of sync with the sound the audience is hearing.

``events.json``
    The discrete events, each carrying the ridge segment that produced it, so a
    patch can place an emitter at the exact spot on the print that made the
    sound.

Everything is deterministic: the same print always produces the same package.
"""

from dataclasses import dataclass
import io
import json
import zipfile

import cv2
import numpy as np
from scipy.io import wavfile

from src.sonification.multilayer import orientation_flow_track


SCHEMA = "fingerprint-visual/1"
CONTROL_RATE = 1000
CONTROL_CHANNELS = (
    "ridge_length",
    "ridge_position",
    "ridge_orientation",
    "ridge_texture",
    "flow",
    "energy",
)


def _envelope(audio, sample_rate):
    """RMS envelope at roughly 100 Hz, normalised to its own peak."""
    hop = max(1, sample_rate // 100)
    frames = len(audio) // hop
    if frames < 1:
        return np.zeros(1), np.zeros(1)
    blocks = np.asarray(audio[: frames * hop], dtype=np.float64).reshape(frames, hop)
    rms = np.sqrt(np.mean(blocks**2, axis=1))
    peak = float(rms.max())
    if peak > 1e-9:
        rms = rms / peak
    times = (np.arange(frames) + 0.5) * hop / sample_rate
    return times, np.clip(rms, 0.0, 1.0)


def _to_control_grid(times, values, frame_count):
    grid = np.arange(frame_count) / CONTROL_RATE
    return np.interp(grid, times, values, left=values[0], right=values[-1])


def control_channels(analysis, multilayer, config):
    """Build the 0..1 control matrix, shape ``(6, frames)``."""
    frames = max(2, int(round(config.duration * CONTROL_RATE)))
    sample_rate = config.sample_rate

    rows = []
    for name in CONTROL_CHANNELS[:4]:
        times, values = _envelope(multilayer.stems[name], sample_rate)
        rows.append(_to_control_grid(times, values, frames))

    flow, _, _, _, _ = orientation_flow_track(analysis, frames)
    rows.append(np.clip(np.asarray(flow, dtype=np.float64), 0.0, 1.0))

    times, values = _envelope(multilayer.mix, sample_rate)
    rows.append(_to_control_grid(times, values, frames))

    matrix = np.vstack(rows)
    if matrix.shape != (len(CONTROL_CHANNELS), frames):
        raise ValueError("control matrix has an unexpected shape")
    # A gentle fade at both ends keeps a patch from snapping on the first frame.
    fade = max(2, int(0.02 * CONTROL_RATE))
    ramp = np.linspace(0.0, 1.0, fade)
    matrix[:, :fade] *= ramp
    matrix[:, -fade:] *= ramp[::-1]
    return np.clip(matrix, 0.0, 1.0)


def encode_controls(matrix):
    """16-bit PCM is enough for control data and is read by every host."""
    payload = np.round(np.clip(matrix, 0.0, 1.0).T * np.iinfo(np.int16).max)
    buffer = io.BytesIO()
    wavfile.write(buffer, CONTROL_RATE, payload.astype(np.int16))
    return buffer.getvalue()


def _mask_png(mask):
    """White ridges on black, so a patch can sample positions straight off it."""
    payload = (np.asarray(mask, dtype=bool).astype(np.uint8)) * 255
    ok, buffer = cv2.imencode(".png", payload)
    if not ok:
        raise ValueError("could not encode the ridge mask")
    return buffer.tobytes()


def events(analysis, multilayer):
    """Discrete events, each resolved back to the ridge segment behind it."""
    by_id = {segment.segment_id: segment for segment in analysis.segments}
    out = []
    for event in multilayer.events:
        segment = by_id.get(event.get("segment_id"))
        if segment is None:
            continue
        out.append(
            {
                "layer": event["layer"],
                "t": round(float(event["start_seconds"]), 4),
                "dur": round(float(event["duration_seconds"]), 4),
                "x": round(segment.centroid_x, 4),
                "y": round(segment.centroid_y, 4),
                "angle": round(segment.orientation_radians, 4),
                "spacing": round(segment.spacing_pixels, 3),
                "tightness": round(segment.tightness, 4),
                "curvature": round(segment.curvature, 4),
                "confidence": round(segment.confidence, 4),
            }
        )
    out.sort(key=lambda item: item["t"])
    return out


def describe(analysis, multilayer, config, source_name):
    spacings = [s.spacing_pixels for s in analysis.segments if s.spacing_pixels > 0]
    return {
        "schema": SCHEMA,
        "source": source_name,
        "duration": float(config.duration),
        "control_rate": CONTROL_RATE,
        "channels": list(CONTROL_CHANNELS),
        "analysis": {
            "mode": analysis.mode,
            "segments": len(analysis.segments),
            "endpoints": len(analysis.endpoints),
            "spacing_min": round(min(spacings), 3) if spacings else None,
            "spacing_max": round(max(spacings), 3) if spacings else None,
        },
        "events": events(analysis, multilayer),
    }


README = """fingerprint visual package
==========================

Files
-----
mix.wav          stereo-free mono master, 44.1 kHz. Play this to the audience.
controls.wav     6-channel control signal at {rate} Hz, aligned to mix.wav
                 from sample 0. Channels, in order:
                   0 ridge_length   long-form structure / decay
                   1 ridge_position timed events / spatial triggers
                   2 ridge_orientation slow harmonic movement
                   3 ridge_texture  detail and turbulence
                   4 flow           the print's ridge rotation, 0..1
                   5 energy         overall envelope of the mix
events.json      discrete events, each with the ridge segment behind it
mask.png         the ridge mask, white on black, at the analysis resolution.
                 Sample it to scatter particles along the actual ridges.
roi.png          the foreground outline of the print, white on black
stems/           the four layers as separate files, sample-aligned

Using it in TouchDesigner
-------------------------
1. Audio File In CHOP  -> mix.wav          (play the piece)
2. Audio File In CHOP  -> controls.wav     (channels 0..5 above)
3. Both files are the same length and start at frame 0, so once the transport
   is running the picture cannot drift out of sync with the sound.
4. Read events.json with the Python snippet in docs/TOUCHDESIGNER.md to drive
   one-shot bursts at exact positions on the print.

Coordinates in events.json are normalised: x and y are 0..1 across the
fingerprint's bounding box, angle is radians modulo pi, spacing is in pixels at
the analysis resolution.

Generated on {source}. Deterministic: the same print always renders the same.
"""


@dataclass(frozen=True)
class VisualPackage:
    controls_wav: bytes
    document: dict
    source_name: str
    mask_png: bytes = b""
    roi_png: bytes = b""

    def events_json(self):
        return json.dumps(self.document, indent=1).encode("utf-8")

    def zip_bytes(self, mix_wav, stems, sample_rate):
        buffer = io.BytesIO()
        stem_name = self.source_name.rsplit(".", 1)[0] or "fingerprint"
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(f"{stem_name}_mix.wav", mix_wav)
            archive.writestr(f"{stem_name}_controls.wav", self.controls_wav)
            archive.writestr(f"{stem_name}_events.json", self.events_json())
            if self.mask_png:
                archive.writestr("mask.png", self.mask_png)
            if self.roi_png:
                archive.writestr("roi.png", self.roi_png)
            for layer, payload in stems.items():
                archive.writestr(f"stems/{layer}.wav", payload)
            archive.writestr(
                "README.txt",
                README.format(rate=CONTROL_RATE, source=self.source_name),
            )
        return buffer.getvalue()


def build_visual_package(analysis, multilayer, config, source_name="fingerprint"):
    matrix = control_channels(analysis, multilayer, config)
    return VisualPackage(
        controls_wav=encode_controls(matrix),
        document=describe(analysis, multilayer, config, source_name),
        source_name=source_name,
        mask_png=_mask_png(analysis.ridge_mask),
        roi_png=_mask_png(analysis.roi_mask),
    )
