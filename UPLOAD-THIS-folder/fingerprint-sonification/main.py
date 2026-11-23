"""Command-line entry point for fingerprint geometry sonification."""

import argparse
from dataclasses import replace
from pathlib import Path
import sys

from config import SonificationConfig
from src.audio_exporter import export_wav
from src.audio_generator import generate_audio
from src.diagnostics import (
    export_geometry_diagnostics,
    export_multilayer_audio_diagnostics,
)
from src.feature_extractor import extract_features
from src.geometry import analyze_cv
from src.image_processor import preprocess_image
from src.ml.inference import analyze_ml
from src.sonification import synthesize_multilayer
from src.sound_mapper import map_features_to_sound


LAYER_NAMES = ("length", "position", "orientation", "texture")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert explainable fingerprint geometry into deterministic audio."
    )
    parser.add_argument("--input", required=True, help="Input PNG, JPG, JPEG, or BMP image")
    parser.add_argument("--output", required=True, help="Final mixed .wav path")
    parser.add_argument(
        "--analysis",
        choices=("cv", "ml", "pixel"),
        default="cv",
        help="cv geometry baseline, genuine checkpoint-backed ml, or legacy pixel mode",
    )
    parser.add_argument(
        "--checkpoint", help="Required trained ridge-segmentation checkpoint for --analysis ml"
    )
    parser.add_argument("--duration", type=float, default=30.0, help="Audio duration in seconds")
    parser.add_argument(
        "--min-frequency", type=float, default=150.0, help="Minimum synthesis frequency"
    )
    parser.add_argument(
        "--max-frequency", type=float, default=1200.0, help="Maximum synthesis frequency"
    )
    parser.add_argument("--analysis-size", type=int, default=256)
    parser.add_argument("--segments", type=int, default=144, help="Legacy pixel-mode regions")
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--orientation-block-size", type=int, default=16)
    parser.add_argument("--minimum-ridge-length", type=float, default=8.0)
    parser.add_argument("--max-events", type=int, default=48)
    parser.add_argument("--length-gain", type=float, default=0.32)
    parser.add_argument("--position-gain", type=float, default=0.32)
    parser.add_argument("--orientation-gain", type=float, default=0.55)
    parser.add_argument("--texture-gain", type=float, default=0.45)
    parser.add_argument("--global-gain", type=float, default=1.4)
    parser.add_argument(
        "--disable-layer",
        action="append",
        default=[],
        choices=LAYER_NAMES,
        help="Disable one geometry sound layer; may be repeated",
    )
    parser.add_argument(
        "--export-stems", action="store_true", help="Export four time-aligned layer WAVs"
    )
    parser.add_argument(
        "--no-contrast-enhancement", action="store_true"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Export analysis and audio diagnostic PNGs"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Explicitly replace existing outputs"
    )
    return parser


def _apply_disabled_layers(config, disabled_layers):
    return replace(
        config,
        enable_length_layer="length" not in disabled_layers,
        enable_position_layer="position" not in disabled_layers,
        enable_orientation_layer="orientation" not in disabled_layers,
        enable_texture_layer="texture" not in disabled_layers,
    )


def _ensure_output_available(path, overwrite, description):
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"{description} already exists: {path}. Use --overwrite to replace it."
        )


def _run_legacy_pixel_mode(input_path, output_path, config, debug, overwrite):
    processed = preprocess_image(
        input_path,
        config.analysis_width,
        config.analysis_height,
        config.enhance_contrast,
    )
    features = extract_features(processed, config.num_segments)
    parameters = map_features_to_sound(features, config)
    audio = generate_audio(parameters, config)
    wav_path = export_wav(output_path, audio, config.sample_rate, overwrite)
    debug_paths = {}
    if debug:
        # Imported here so the geometry/CV path does not require matplotlib.
        from src.visualizer import save_debug_outputs

        debug_directory = output_path.parent / f"{output_path.stem}_debug"
        _ensure_output_available(debug_directory, overwrite, "Debug directory")
        debug_paths = save_debug_outputs(
            debug_directory,
            processed,
            features,
            parameters,
            audio,
            config.sample_rate,
            config.duration,
        )
    return {
        "wav": wav_path,
        "stems": {},
        "analysis": None,
        "audio": audio,
        "diagnostics": debug_paths,
        "analysis_mode": "pixel",
    }


def run_sonification(
    input_path,
    output_path,
    config,
    analysis_mode="cv",
    checkpoint=None,
    export_stems=False,
    debug=False,
    overwrite=False,
):
    """Run CV, genuine ML, or preserved legacy pixel sonification."""
    config.validate()
    output_path = Path(output_path)
    _ensure_output_available(output_path, overwrite, "Output WAV")
    if analysis_mode == "pixel":
        return _run_legacy_pixel_mode(input_path, output_path, config, debug, overwrite)

    analysis_directory = output_path.parent / f"{output_path.stem}_analysis"
    _ensure_output_available(analysis_directory, overwrite, "Analysis directory")
    processed = preprocess_image(
        input_path,
        config.analysis_width,
        config.analysis_height,
        config.enhance_contrast,
    )
    if analysis_mode == "cv":
        analysis = analyze_cv(
            processed,
            orientation_block_size=config.orientation_block_size,
            minimum_ridge_length=config.minimum_ridge_length,
        )
    elif analysis_mode == "ml":
        analysis = analyze_ml(
            processed,
            checkpoint,
            orientation_block_size=config.orientation_block_size,
            minimum_ridge_length=config.minimum_ridge_length,
        )
    else:
        raise ValueError(f"Unknown analysis mode: {analysis_mode}")

    multilayer = synthesize_multilayer(analysis, config)
    wav_path = export_wav(
        output_path, multilayer.mix, sample_rate=config.sample_rate, overwrite=overwrite
    )
    stem_paths = {}
    if export_stems:
        for name, audio in multilayer.stems.items():
            stem_path = output_path.with_name(f"{output_path.stem}_{name}.wav")
            _ensure_output_available(stem_path, overwrite, f"{name} stem")
            stem_paths[name] = export_wav(
                stem_path, audio, sample_rate=config.sample_rate, overwrite=overwrite
            )

    diagnostics = export_geometry_diagnostics(
        analysis, analysis_directory, events=multilayer.events
    )
    if debug:
        diagnostics.update(
            export_multilayer_audio_diagnostics(
                multilayer, config.sample_rate, analysis_directory
            )
        )
    return {
        "wav": wav_path,
        "stems": stem_paths,
        "analysis": analysis,
        "audio": multilayer,
        "diagnostics": diagnostics,
        "analysis_mode": analysis_mode,
    }


def main(argv=None):
    parser = build_parser()
    arguments = parser.parse_args(argv)
    config = SonificationConfig(
        analysis_width=arguments.analysis_size,
        analysis_height=arguments.analysis_size,
        num_segments=arguments.segments,
        duration=arguments.duration,
        min_frequency=arguments.min_frequency,
        max_frequency=arguments.max_frequency,
        sample_rate=arguments.sample_rate,
        enhance_contrast=not arguments.no_contrast_enhancement,
        orientation_block_size=arguments.orientation_block_size,
        minimum_ridge_length=arguments.minimum_ridge_length,
        max_segment_events=arguments.max_events,
        length_layer_gain=arguments.length_gain,
        position_layer_gain=arguments.position_gain,
        orientation_layer_gain=arguments.orientation_gain,
        texture_layer_gain=arguments.texture_gain,
        global_gain=arguments.global_gain,
    )
    config = _apply_disabled_layers(config, set(arguments.disable_layer))
    if arguments.analysis == "ml" and not arguments.checkpoint:
        parser.error("--analysis ml requires --checkpoint with real trained weights")
    try:
        result = run_sonification(
            arguments.input,
            arguments.output,
            config,
            analysis_mode=arguments.analysis,
            checkpoint=arguments.checkpoint,
            export_stems=arguments.export_stems,
            debug=arguments.debug,
            overwrite=arguments.overwrite,
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Analysis mode: {result['analysis_mode']}")
    print(f"Final WAV: {result['wav']}")
    print(
        f"Format: mono 16-bit PCM, {config.sample_rate} Hz, "
        f"{config.duration:.3f} seconds"
    )
    if result["analysis"] is not None:
        print(f"Ridge segments: {len(result['analysis'].segments)}")
    for name, path in result["stems"].items():
        print(f"Stem {name}: {path}")
    for name, path in result["diagnostics"].items():
        print(f"Diagnostic {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
