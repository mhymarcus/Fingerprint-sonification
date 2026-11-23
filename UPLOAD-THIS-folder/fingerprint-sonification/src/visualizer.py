"""Optional diagnostic images and CSV export for sonification."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.feature_extractor import FeatureSequence
from src.image_processor import ProcessedImage
from src.sound_mapper import AudioParameters


def _save_preprocessing(processed, path):
    figure, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(processed.original, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original fingerprint")
    axes[1].imshow(processed.grayscale, cmap="gray", vmin=0, vmax=1)
    axes[1].contour(processed.foreground_mask, levels=[0.5], colors=["cyan"], linewidths=0.5)
    axes[1].set_title("Preprocessed + foreground mask")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_features(features, parameters, path):
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(features.scan_position, features.ridge_darkness, label="Ridge darkness")
    axes[0].plot(features.scan_position, features.local_contrast, label="Local contrast")
    axes[0].plot(features.scan_position, features.ridge_density, label="Ridge density")
    axes[0].plot(
        features.scan_position,
        features.foreground_coverage,
        label="Foreground coverage",
        alpha=0.65,
    )
    axes[0].set_ylabel("Normalized image feature")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(ncol=2)
    axes[0].grid(alpha=0.25)

    axes[1].plot(parameters.scan_position, parameters.frequency, label="Frequency (Hz)")
    amplitude_scaled = parameters.amplitude * parameters.frequency.max()
    axes[1].plot(
        parameters.scan_position,
        amplitude_scaled,
        label="Amplitude (scaled for display)",
        alpha=0.75,
    )
    axes[1].set_xlabel("Snake-scan progression")
    axes[1].set_ylabel("Mapped audio parameter")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_waveform(audio, sample_rate, path):
    max_points = 30000
    step = max(1, len(audio) // max_points)
    samples = audio[::step]
    times = np.arange(0, len(audio), step)[: len(samples)] / sample_rate
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.plot(times, samples, linewidth=0.6)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Amplitude")
    axis.set_ylim(-1.05, 1.05)
    axis.set_title("Generated waveform")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_spectrogram(audio, sample_rate, path):
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.specgram(audio, NFFT=2048, Fs=sample_rate, noverlap=1536, cmap="magma")
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_ylim(0, min(5000, sample_rate / 2))
    axis.set_title("Audio spectrogram")
    figure.tight_layout()
    figure.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _save_mapping_csv(features, parameters, duration, path):
    fieldnames = [
        "time_seconds",
        "scan_position",
        "x_position",
        "y_position",
        "ridge_darkness",
        "local_contrast",
        "ridge_density",
        "foreground_coverage",
        "mapped_frequency_hz",
        "mapped_amplitude",
        "harmonic_mix",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(len(features)):
            writer.writerow(
                {
                    "time_seconds": float(features.scan_position[index] * duration),
                    "scan_position": float(features.scan_position[index]),
                    "x_position": float(features.x_position[index]),
                    "y_position": float(features.y_position[index]),
                    "ridge_darkness": float(features.ridge_darkness[index]),
                    "local_contrast": float(features.local_contrast[index]),
                    "ridge_density": float(features.ridge_density[index]),
                    "foreground_coverage": float(features.foreground_coverage[index]),
                    "mapped_frequency_hz": float(parameters.frequency[index]),
                    "mapped_amplitude": float(parameters.amplitude[index]),
                    "harmonic_mix": float(parameters.harmonic_mix[index]),
                }
            )


def save_debug_outputs(
    output_directory,
    processed_image: ProcessedImage,
    features: FeatureSequence,
    parameters: AudioParameters,
    audio,
    sample_rate,
    duration,
):
    """Save all optional diagnostics and return their paths."""
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "preprocessing": output_directory / "preprocessing.png",
        "features": output_directory / "features_and_mapping.png",
        "waveform": output_directory / "waveform.png",
        "spectrogram": output_directory / "spectrogram.png",
        "mapping_csv": output_directory / "mapping.csv",
    }
    _save_preprocessing(processed_image, paths["preprocessing"])
    _save_features(features, parameters, paths["features"])
    _save_waveform(audio, sample_rate, paths["waveform"])
    _save_spectrogram(audio, sample_rate, paths["spectrogram"])
    _save_mapping_csv(features, parameters, duration, paths["mapping_csv"])
    return paths
