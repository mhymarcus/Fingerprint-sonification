"""Export explainable geometry measurements and visual diagnostics."""

import csv
import io
import json
from pathlib import Path

import numpy as np


def _pyplot():
    """Import matplotlib lazily on a headless backend.

    Geometry and sonification must stay importable in environments without
    matplotlib (for example when embedding them in TouchDesigner).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


SEGMENT_FIELDS = (
    "segment_id",
    "length_pixels",
    "length_normalized",
    "centroid_x",
    "centroid_y",
    "orientation_radians",
    "orientation_normalized",
    "density",
    "curvature",
    "confidence",
    "spacing_pixels",
    "tightness",
)


def build_analysis_figure(analysis):
    """Six-panel figure of the geometry stages. Caller owns (and closes) it."""
    plt = _pyplot()
    figure, axes = plt.subplots(2, 3, figsize=(13, 9))
    panels = (
        (analysis.processed_image.original, "Raw image"),
        (analysis.enhanced, "Enhanced grayscale"),
        (analysis.roi_mask, "Foreground ROI mask"),
        (analysis.ridge_mask, "Ridge-level mask"),
        (analysis.skeleton, "Skeleton"),
        (analysis.orientation_image, "Orientation field"),
    )
    for axis, (image, title) in zip(axes.flat, panels):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(f"Fingerprint geometry analysis ({analysis.mode})")
    figure.tight_layout()
    return figure


def _save_analysis_figure(analysis, path):
    figure = build_analysis_figure(analysis)
    figure.savefig(path, dpi=170, bbox_inches="tight")
    _pyplot().close(figure)


def figure_to_png(figure, dpi=150):
    """Render a figure to PNG bytes and release it. Used by the web app."""
    plt = _pyplot()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def build_geometry_document(analysis, events=None):
    """The explainable JSON record of one analysis."""
    document = {
        "analysis_mode": analysis.mode,
        "image_width": int(analysis.ridge_mask.shape[1]),
        "image_height": int(analysis.ridge_mask.shape[0]),
        "length_unit": "pixels_along_skeleton_arc",
        "position_unit": "normalized_0_to_1_within_roi_bounds",
        "orientation_unit": "radians_modulo_pi",
        "density_unit": "local_ridge_pixel_fraction_0_to_1",
        "curvature_unit": "normalized_mean_local_turn_0_to_1",
        "spacing_unit": "pixels_between_adjacent_ridges",
        "tightness_unit": "ridge_spacing_relative_to_this_image_0_to_1",
        "segment_count": len(analysis.segments),
        "endpoint_count": len(analysis.endpoints),
        "bifurcation_pixel_count": len(analysis.bifurcations),
        "segments": [
            segment.to_dict(include_points=True) for segment in analysis.segments
        ],
    }
    if events is not None:
        document["sound_events"] = list(events)
    return document


def export_geometry_diagnostics(analysis, output_directory, events=None):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "analysis_figure": output_directory / "geometry_analysis.png",
        "geometry_json": output_directory / "geometry.json",
        "segments_csv": output_directory / "ridge_segments.csv",
    }
    _save_analysis_figure(analysis, paths["analysis_figure"])

    geometry = build_geometry_document(analysis, events=events)
    paths["geometry_json"].write_text(
        json.dumps(geometry, indent=2), encoding="utf-8"
    )

    with paths["segments_csv"].open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=SEGMENT_FIELDS)
        writer.writeheader()
        for segment in analysis.segments:
            record = segment.to_dict(include_points=False)
            writer.writerow({field: record[field] for field in SEGMENT_FIELDS})
    return paths


def build_layer_figure(multilayer_audio, sample_rate):
    """Aligned waveform of the mix and every stem. Caller owns the figure."""
    plt = _pyplot()
    figure, axes = plt.subplots(
        len(multilayer_audio.stems) + 1, 1, figsize=(13, 10), sharex=True
    )
    series = [("final_mix", multilayer_audio.mix), *multilayer_audio.stems.items()]
    for axis, (name, audio) in zip(axes, series):
        step = max(1, len(audio) // 25000)
        times = np.arange(0, len(audio), step) / sample_rate
        axis.plot(times, audio[::step], linewidth=0.55)
        axis.set_ylabel(name.replace("ridge_", ""))
        axis.set_ylim(-0.75, 0.75)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (seconds)")
    figure.suptitle("Time-aligned geometry sound layers")
    figure.tight_layout()
    return figure


def build_spectrogram_figure(multilayer_audio, sample_rate):
    """Spectrogram of the final mix. Caller owns the figure."""
    plt = _pyplot()
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.specgram(
        multilayer_audio.mix,
        NFFT=2048,
        Fs=sample_rate,
        noverlap=1536,
        cmap="magma",
    )
    axis.set_ylim(0, min(5000, sample_rate / 2))
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title("Final mix spectrogram")
    figure.tight_layout()
    return figure


def export_multilayer_audio_diagnostics(multilayer_audio, sample_rate, output_directory):
    """Save aligned stem waveforms and the final mix spectrogram."""
    plt = _pyplot()
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    waveform_path = output_directory / "stems_waveform.png"
    spectrogram_path = output_directory / "mix_spectrogram.png"

    figure = build_layer_figure(multilayer_audio, sample_rate)
    figure.savefig(waveform_path, dpi=160, bbox_inches="tight")
    plt.close(figure)

    figure = build_spectrogram_figure(multilayer_audio, sample_rate)
    figure.savefig(spectrogram_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return {"stems_waveform": waveform_path, "mix_spectrogram": spectrogram_path}
