"""Streamlit front end: turn a fingerprint image into a four-layer composition.

Run locally:

    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

from config import SonificationConfig
from src.web_jobs import MAX_UPLOAD_BYTES, sonify

CHECKPOINT_ENV = "RIDGE_CHECKPOINT"
DEFAULT_CHECKPOINT = Path("checkpoints/ridge_segmentation.pt")
SAMPLE_PATH = Path("images/sample_fingerprint.png")

LAYER_LABELS = {
    "ridge_length": "Ridge length",
    "ridge_position": "Ridge position",
    "ridge_orientation": "Orientation",
    "ridge_texture": "Texture",
}


def available_checkpoint():
    """Return a checkpoint path only if the weights actually exist."""
    candidate = Path(os.environ.get(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT)))
    return candidate if candidate.is_file() else None


def torch_available():
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def build_config():
    """Sidebar controls mapped onto SonificationConfig."""
    with st.sidebar:
        st.header("Composition")
        duration = st.slider(
            "Duration (seconds)", min_value=5, max_value=60, value=30, step=1
        )
        analysis_size = st.select_slider(
            "Analysis resolution",
            options=[128, 192, 256, 320],
            value=256,
            help=(
                "Ridge detail needs resolution. Below ~128 px the ridges of a "
                "phone photo are only a few pixels apart and the masks degrade."
            ),
        )
        global_gain = st.slider("Master gain", 0.5, 2.0, 1.4, 0.05)

        st.header("Layers")
        enabled = {}
        gains = {}
        defaults = {
            "ridge_length": ("length", 0.32),
            "ridge_position": ("position", 0.32),
            "ridge_orientation": ("orientation", 0.55),
            "ridge_texture": ("texture", 0.45),
        }
        for key, (slug, default_gain) in defaults.items():
            enabled[key] = st.checkbox(LAYER_LABELS[key], value=True, key=f"on_{key}")
            gains[key] = st.slider(
                f"{LAYER_LABELS[key]} gain",
                0.0,
                1.5,
                default_gain,
                0.05,
                key=f"gain_{key}",
                disabled=not enabled[key],
            )

        with st.expander("Advanced"):
            min_frequency = st.number_input(
                "Minimum frequency (Hz)", 60, 400, 150, step=10
            )
            max_frequency = st.number_input(
                "Maximum frequency (Hz)", 200, 4000, 1200, step=50
            )
            max_events = st.slider("Maximum events per layer", 8, 96, 48, 4)
            enhance = st.checkbox("Contrast enhancement", value=True)

    config = SonificationConfig(
        analysis_width=analysis_size,
        analysis_height=analysis_size,
        duration=float(duration),
        global_gain=float(global_gain),
        min_frequency=float(min_frequency),
        max_frequency=float(max_frequency),
        max_segment_events=int(max_events),
        length_layer_gain=float(gains["ridge_length"]),
        position_layer_gain=float(gains["ridge_position"]),
        orientation_layer_gain=float(gains["ridge_orientation"]),
        texture_layer_gain=float(gains["ridge_texture"]),
        enable_length_layer=enabled["ridge_length"],
        enable_position_layer=enabled["ridge_position"],
        enable_orientation_layer=enabled["ridge_orientation"],
        enable_texture_layer=enabled["ridge_texture"],
        enhance_contrast=bool(enhance),
    )
    return config


def render_summary(summary, filename):
    stem = Path(filename).stem
    columns = st.columns(4)
    columns[0].metric("Ridge segments", summary["segments"])
    columns[1].metric("Sound events", summary["events"])
    columns[2].metric("Duration", f"{summary['duration_seconds']:.0f} s")
    peak = summary.get("peak_dbfs")
    columns[3].metric("Peak", f"{peak:.1f} dBFS" if peak is not None else "–")
    if summary.get("spacing_median"):
        st.caption(
            f"Ridge spacing {summary['spacing_min']:.1f}–{summary['spacing_max']:.1f} px "
            f"(median {summary['spacing_median']:.1f}). "
            f"Analysis mode: {summary['analysis_mode'].upper()}."
        )
    return stem


def render_result(result, filename):
    stem = Path(filename).stem
    st.subheader("Result")
    render_summary(result.summary, filename)

    audio_tab, geometry_tab, layers_tab, data_tab = st.tabs(
        ["Audio", "Geometry", "Layers", "Data"]
    )

    with audio_tab:
        st.audio(result.mix_wav, format="audio/wav")
        download_columns = st.columns(3)
        download_columns[0].download_button(
            "Download mix (.wav)",
            result.mix_wav,
            file_name=f"{stem}_mix.wav",
            mime="audio/wav",
            width="stretch",
        )
        download_columns[1].download_button(
            "Download stems (.zip)",
            result.stems_zip,
            file_name=f"{stem}_stems.zip",
            mime="application/zip",
            width="stretch",
        )
        download_columns[2].download_button(
            "Download geometry (.json)",
            result.geometry_json().encode("utf-8"),
            file_name=f"{stem}_geometry.json",
            mime="application/json",
            width="stretch",
        )
        st.caption(
            "Stems are sample-aligned mono 16-bit WAV. The mix is normalised to "
            "-3 dBFS; every stem is scaled by the same factor."
        )

    with geometry_tab:
        st.image(result.figures["analysis"], width="stretch")
        st.caption(
            "Foreground ROI and ridge mask are different things: the ROI is the "
            "finger area, the ridge mask is the individual ridges inside it."
        )

    with layers_tab:
        st.image(result.figures["layers"], width="stretch")
        st.image(result.figures["spectrogram"], width="stretch")

    with data_tab:
        st.caption(
            "Every sound event points back at the ridge segment that produced it."
        )
        segments = result.geometry["segments"]
        st.dataframe(
            [
                {
                    "id": segment["segment_id"],
                    "length_px": round(segment["length_pixels"], 1),
                    "x": round(segment["centroid_x"], 3),
                    "y": round(segment["centroid_y"], 3),
                    "orientation_deg": round(
                        segment["orientation_radians"] * 180 / 3.141592653589793, 1
                    ),
                    "spacing_px": round(segment["spacing_pixels"], 2),
                    "tightness": round(segment["tightness"], 3),
                    "curvature": round(segment["curvature"], 3),
                    "confidence": round(segment["confidence"], 3),
                }
                for segment in segments
            ],
            width="stretch",
            hide_index=True,
        )
        st.download_button(
            "Download full geometry JSON",
            result.geometry_json().encode("utf-8"),
            file_name=f"{stem}_geometry.json",
            mime="application/json",
        )


def main():
    st.set_page_config(
        page_title="Fingerprint Sonification",
        page_icon="🎧",
        layout="wide",
    )

    st.title("Fingerprint → four-layer composition")
    st.markdown(
        "Upload a fingerprint photo. The app extracts ridge geometry — length, "
        "position, orientation, spacing — and turns each into its own sound layer, "
        "then mixes them into one WAV you can download alongside the stems."
    )

    config = build_config()

    checkpoint = available_checkpoint()
    analysis_mode = "cv"
    if checkpoint and torch_available():
        with st.sidebar:
            analysis_mode = st.radio(
                "Ridge analysis",
                ["cv", "ml"],
                horizontal=True,
                help="ML uses a trained segmentation checkpoint instead of the CV baseline.",
            )

    uploaded = st.file_uploader(
        "Fingerprint image",
        type=["png", "jpg", "jpeg", "bmp"],
        help=f"PNG, JPG or BMP, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
    )

    sample_available = SAMPLE_PATH.is_file()
    use_sample = st.checkbox(
        "No print handy? Use the built-in sample (synthetic, nothing is uploaded)",
        value=False,
        disabled=not sample_available,
        help="A generated ridge pattern that ships with the app, so you can hear "
        "the mapping without uploading a real fingerprint.",
    )

    image_bytes = None
    filename = "fingerprint.png"
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        filename = uploaded.name
        st.image(
            image_bytes,
            caption=f"{filename} · {len(image_bytes) / 1024:.0f} KB",
            width=220,
        )
    elif use_sample and sample_available:
        image_bytes = SAMPLE_PATH.read_bytes()
        filename = SAMPLE_PATH.name
        st.image(image_bytes, caption=f"{filename} (built-in sample)", width=220)

    generate = st.button(
        "Generate sound",
        type="primary",
        disabled=image_bytes is None,
    )

    if generate and image_bytes is not None:
        with st.spinner("Analyzing ridges and synthesizing layers…"):
            try:
                result = sonify(
                    image_bytes,
                    filename,
                    config,
                    analysis_mode=analysis_mode,
                    checkpoint=checkpoint,
                )
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
                st.error(f"Could not process this image: {error}")
            else:
                st.session_state["result"] = result
                st.session_state["filename"] = filename

    result = st.session_state.get("result")
    if result is not None:
        render_result(result, st.session_state.get("filename", "fingerprint.wav"))
    elif image_bytes is None:
        st.info(
            "Pick an image and press **Generate sound**, or tick the sample option. "
            "A sharp, well-lit photo with the print filling most of the frame works best."
        )

    st.divider()
    st.caption(
        "**Privacy.** Images are processed on the server to produce audio and are "
        "not stored by this app. Upload only fingerprints you have permission to use. "
        "This is an artistic analysis tool: it does not identify people, and it is "
        "not authentication software."
    )


main()
