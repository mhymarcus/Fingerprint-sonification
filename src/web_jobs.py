"""One sonification job: uploaded bytes in, browser-ready artifacts out.

This is the shared core of the web app. It deliberately knows nothing about
Streamlit (or any other framework) so it can be unit tested, reused by a future
HTTP API, or called from a notebook.
"""

from dataclasses import dataclass, replace
import hashlib
import io
import json
from pathlib import Path
import tempfile
import zipfile

import numpy as np

from src.audio_exporter import wav_bytes
from src.diagnostics import (
    build_analysis_figure,
    build_geometry_document,
    build_layer_figure,
    build_spectrogram_figure,
    figure_to_png,
)
from src.autoframe import plan as plan_frame
from src.geometry import analyze_cv
from src.image_processor import load_grayscale_image, preprocess_image
from src.sonification import synthesize_multilayer
from src.visual_export import build_visual_package


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


@dataclass(frozen=True)
class WebResult:
    mix_wav: bytes
    stem_wavs: dict
    stems_zip: bytes
    visual_zip: bytes
    figures: dict
    geometry: dict
    summary: dict

    def geometry_json(self):
        return json.dumps(self.geometry, indent=2)


def upload_digest(image_bytes):
    return hashlib.sha256(image_bytes).hexdigest()


def _validated_suffix(image_bytes, filename):
    if not image_bytes:
        raise ValueError("No image data was received.")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        limit = MAX_UPLOAD_BYTES // (1024 * 1024)
        raise ValueError(f"Image is larger than {limit} MB.")
    suffix = Path(filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise ValueError(f"Unsupported image format. Expected one of: {supported}")
    return suffix


def _processed_from_bytes(
    image_bytes,
    filename,
    width,
    height,
    enhance_contrast,
    crop=None,
    auto_frame=False,
):
    """Write the upload to a temp file so the CLI loader can validate it."""
    suffix = _validated_suffix(image_bytes, filename)
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        handle.write(image_bytes)
        handle.flush()
        handle.close()
        if auto_frame:
            # Find the print and decide the resolution from the ridge spacing
            # actually present in this photograph.
            crop, width = plan_frame(load_grayscale_image(handle.name))
            height = width
        processed = preprocess_image(
            handle.name,
            width,
            height,
            enhance_contrast=enhance_contrast,
            crop=crop,
        )
    finally:
        Path(handle.name).unlink(missing_ok=True)
    # The temp file is gone; keep the uploaded name for figures and messages.
    return replace(processed, source_path=Path(filename))


def _summary(analysis, multilayer, config):
    spacings = [
        float(segment.spacing_pixels)
        for segment in analysis.segments
        if segment.spacing_pixels > 0
    ]
    peak = float(np.max(np.abs(multilayer.mix)))
    return {
        "analysis_mode": analysis.mode,
        "segments": len(analysis.segments),
        "endpoints": len(analysis.endpoints),
        "bifurcations": len(analysis.bifurcations),
        "spacing_min": min(spacings) if spacings else None,
        "spacing_median": float(np.median(spacings)) if spacings else None,
        "spacing_max": max(spacings) if spacings else None,
        "events": len(multilayer.events),
        "analysis_size": int(config.analysis_width),
        "duration_seconds": float(config.duration),
        "sample_rate": int(config.sample_rate),
        "peak_dbfs": float(20.0 * np.log10(peak)) if peak > 0 else None,
    }


def sonify(
    image_bytes,
    filename,
    config,
    analysis_mode="cv",
    checkpoint=None,
    crop=None,
    auto_frame=False,
):
    """Run the full pipeline and return everything the UI needs."""
    config.validate()
    processed = _processed_from_bytes(
        image_bytes,
        filename,
        config.analysis_width,
        config.analysis_height,
        config.enhance_contrast,
        crop=crop,
        auto_frame=auto_frame,
    )
    if auto_frame:
        # _processed_from_bytes resized its own crop; keep the config truthful so
        # the summary and the geometry report the size that was really used.
        config = replace(
            config,
            analysis_width=processed.grayscale.shape[1],
            analysis_height=processed.grayscale.shape[0],
        )

    if analysis_mode == "ml":
        from src.ml.inference import analyze_ml

        analysis = analyze_ml(
            processed,
            checkpoint,
            orientation_block_size=config.orientation_block_size,
            minimum_ridge_length=config.minimum_ridge_length,
        )
    elif analysis_mode == "cv":
        analysis = analyze_cv(
            processed,
            orientation_block_size=config.orientation_block_size,
            minimum_ridge_length=config.minimum_ridge_length,
        )
    else:
        raise ValueError(f"Unsupported analysis mode: {analysis_mode}")

    multilayer = synthesize_multilayer(analysis, config)
    stem_wavs = {
        name: wav_bytes(audio, config.sample_rate)
        for name, audio in multilayer.stems.items()
    }

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, payload in stem_wavs.items():
            archive.writestr(f"{Path(filename).stem}_{name}.wav", payload)

    figures = {
        "analysis": figure_to_png(build_analysis_figure(analysis), dpi=140),
        "layers": figure_to_png(
            build_layer_figure(multilayer, config.sample_rate), dpi=130
        ),
        "spectrogram": figure_to_png(
            build_spectrogram_figure(multilayer, config.sample_rate), dpi=130
        ),
    }

    visual = build_visual_package(analysis, multilayer, config, source_name=filename)

    return WebResult(
        mix_wav=wav_bytes(multilayer.mix, config.sample_rate),
        stem_wavs=stem_wavs,
        stems_zip=buffer.getvalue(),
        visual_zip=visual.zip_bytes(
            wav_bytes(multilayer.mix, config.sample_rate), stem_wavs, config.sample_rate
        ),
        figures=figures,
        geometry=build_geometry_document(analysis, events=multilayer.events),
        summary=_summary(analysis, multilayer, config),
    )
