"""Streamlit front end: turn a fingerprint image into a four-layer composition.

The page is built as a printed spread rather than a dashboard: one tall hero,
hairline section rules, monochrome figures, and no chrome of its own.

Run locally:

    streamlit run app.py
"""

import os
from pathlib import Path

import streamlit as st

from config import SonificationConfig
from src import ui_theme
from src.local_queue import publish
from src.web_jobs import MAX_UPLOAD_BYTES, sonify

CHECKPOINT_ENV = "RIDGE_CHECKPOINT"
DEFAULT_CHECKPOINT = Path("checkpoints/ridge_segmentation.pt")
SAMPLE_PATH = Path("images/sample_fingerprint.png")

TITLE = "fingerprints"
ARTIST = "Marcus Mak"
YEAR = "2026"

LAYER_LABELS = {
    "ridge_length": ("length", 0.32),
    "ridge_position": ("position", 0.32),
    "ridge_orientation": ("orientation", 0.55),
    "ridge_texture": ("texture", 0.45),
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


def masthead():
    st.markdown(
        ui_theme.masthead(
            title=TITLE,
            credit=ARTIST,
            left="fingerprints / 001",
            right="sys.live · 100%",
            timecode="t:00:00:00",
            middle="ridge · flow · grain",
        ),
        unsafe_allow_html=True,
    )


def section(number, label):
    st.markdown(ui_theme.section(number, label), unsafe_allow_html=True)


def build_config():
    """Duration, resolution and the layer mix."""
    controls = st.columns([1, 1], gap="large")
    with controls[0]:
        duration = st.slider("duration / seconds", 5, 60, 30, 1)
    with controls[1]:
        analysis_size = st.select_slider(
            "analysis resolution",
            options=[128, 192, 256, 320, 384, 512],
            value=512,
            help="Ridge detail needs resolution. For a photo of a finger, 256 "
            "is often too coarse and 512 is what makes the ridges appear.",
        )

    enabled, gains, min_frequency, max_frequency, max_events, enhance = (
        {},
        {},
        None,
        None,
        None,
        None,
    )
    with st.expander("mix / layer balance", expanded=False):
        for key, (slug, default_gain) in LAYER_LABELS.items():
            row = st.columns([1, 2], gap="large")
            with row[0]:
                enabled[key] = st.checkbox(slug, value=True, key=f"on_{key}")
            with row[1]:
                gains[key] = st.slider(
                    f"{slug} gain",
                    0.0,
                    1.5,
                    default_gain,
                    0.05,
                    key=f"gain_{key}",
                    disabled=not enabled[key],
                )
        st.markdown(ui_theme.rule(), unsafe_allow_html=True)
        advanced = st.columns(3, gap="large")
        with advanced[0]:
            min_frequency = st.number_input("min frequency", 60, 400, 150, step=10)
        with advanced[1]:
            max_frequency = st.number_input("max frequency", 200, 4000, 1200, step=50)
        with advanced[2]:
            max_events = st.slider("max events", 8, 96, 48, 4)
        enhance = st.checkbox("contrast enhancement", value=True)

    return SonificationConfig(
        analysis_width=analysis_size,
        analysis_height=analysis_size,
        duration=float(duration),
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


def render_output(result, filename):
    stem = Path(filename).stem
    summary = result.summary

    section("03", "output")
    st.markdown(
        ui_theme.micro(
            f"{stem} · {summary['analysis_mode']} · ridge spacing "
            f"{summary['spacing_min']:.1f}–{summary['spacing_max']:.1f} px"
            if summary.get("spacing_min")
            else f"{stem} · {summary['analysis_mode']}"
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        ui_theme.readouts(
            [
                ("segments", f"{summary['segments']}"),
                ("events", f"{summary['events']}"),
                ("duration", f"{summary['duration_seconds']:.0f} s"),
                (
                    "peak",
                    f"{summary['peak_dbfs']:.1f} dB"
                    if summary.get("peak_dbfs") is not None
                    else "—",
                ),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.audio(result.mix_wav, format="audio/wav")
    st.write("")

    downloads = st.columns(4, gap="small")
    downloads[0].download_button(
        "mix .wav",
        result.mix_wav,
        file_name=f"{stem}_mix.wav",
        mime="audio/wav",
        width="stretch",
    )
    downloads[1].download_button(
        "stems .zip",
        result.stems_zip,
        file_name=f"{stem}_stems.zip",
        mime="application/zip",
        width="stretch",
    )
    downloads[2].download_button(
        "geometry .json",
        result.geometry_json().encode("utf-8"),
        file_name=f"{stem}_geometry.json",
        mime="application/json",
        width="stretch",
    )
    published = st.session_state.get("published")
    if published:
        st.caption(
            f"added to the queue · {published} — the previous piece keeps playing "
            "until the operator advances"
        )

    downloads[3].download_button(
        "visual .zip",
        result.visual_zip,
        file_name=f"{stem}_visual.zip",
        mime="application/zip",
        width="stretch",
        help="TouchDesigner package: mix, controls.wav and events.json.",
    )

    section("04", "geometry")
    st.image(result.figures["analysis"], width="stretch")
    st.caption(
        "Foreground ROI and ridge mask are different things — the ROI is the "
        "finger area, the ridge mask is the individual ridges inside it."
    )

    section("05", "layers")
    st.image(result.figures["layers"], width="stretch")
    st.image(result.figures["spectrogram"], width="stretch")
    st.caption(
        "Four layers, one timeline. The stems are sample-aligned; the mix is "
        "normalised to -3 dBFS."
    )

    section("06", "data")
    with st.expander("ridge segments", expanded=False):
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
                for segment in result.geometry["segments"]
            ],
            width="stretch",
            hide_index=True,
        )


def main():
    st.set_page_config(
        page_title=f"Fingerprints — {ARTIST}",
        page_icon="◐",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(ui_theme.style_block(), unsafe_allow_html=True)

    masthead()
    st.markdown(
        ui_theme.lede(
            "An image of a fingerprint is read as geometry — the length of each "
            "ridge, where it sits, which way it turns, how tightly the ridges are "
            "packed — and each of those measurements drives one layer of a "
            "composition. Nothing here identifies anyone."
        ),
        unsafe_allow_html=True,
    )

    section("01", "source")
    uploaded = st.file_uploader(
        "fingerprint image",
        type=["png", "jpg", "jpeg", "bmp"],
        help=f"PNG, JPG or BMP, up to {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.",
        label_visibility="collapsed",
    )
    sample_available = SAMPLE_PATH.is_file()
    use_sample = st.checkbox(
        "use the built-in sample (synthetic, nothing is uploaded)",
        value=False,
        disabled=not sample_available,
    )

    image_bytes, filename = None, "fingerprint.png"
    if uploaded is not None:
        image_bytes, filename = uploaded.getvalue(), uploaded.name
    elif use_sample and sample_available:
        image_bytes, filename = SAMPLE_PATH.read_bytes(), SAMPLE_PATH.name

    if image_bytes is not None:
        preview, meta = st.columns([1, 4], gap="large")
        with preview:
            st.image(image_bytes, width="stretch")
        with meta:
            st.markdown(
                ui_theme.micro(
                    f"{filename} · {len(image_bytes) / 1024:.0f} kb"
                ),
                unsafe_allow_html=True,
            )
            st.caption("Ready. Set the parameters below, then generate.")

    crop = None
    if image_bytes is not None:
        with st.expander("framing / crop", expanded=False):
            st.caption(
                "If the mask comes out wrong — a crescent, or ridges everywhere — "
                "pull these in tight around the finger only. Leave them alone when "
                "the photo already fills the frame."
            )
            box = st.columns(4, gap="small")
            left = box[0].slider("left %", 0, 90, 0, 1)
            top = box[1].slider("top %", 0, 90, 0, 1)
            right = box[2].slider("right %", 10, 100, 100, 1)
            bottom = box[3].slider("bottom %", 10, 100, 100, 1)
            if (left, top, right, bottom) != (0, 0, 100, 100):
                crop = (left / 100, top / 100, right / 100, bottom / 100)

    section("02", "parameters")
    config = build_config()

    checkpoint = available_checkpoint()
    analysis_mode = "cv"
    if checkpoint and torch_available():
        analysis_mode = st.radio(
            "ridge analysis", ["cv", "ml"], horizontal=True
        )

    st.write("")
    generate = st.button(
        "generate sound", type="primary", width="stretch", disabled=image_bytes is None
    )

    if generate and image_bytes is not None:
        with st.spinner("reading ridges · synthesising layers"):
            try:
                result = sonify(
                    image_bytes,
                    filename,
                    config,
                    analysis_mode=analysis_mode,
                    checkpoint=checkpoint,
                    crop=crop,
                )
            except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
                st.error(f"could not process this image: {error}")
            else:
                st.session_state["result"] = result
                st.session_state["filename"] = filename
                try:
                    published = publish(result.visual_zip, filename)
                except (ValueError, OSError) as error:
                    st.warning(f"could not write to the projector folder: {error}")
                else:
                    st.session_state["published"] = (
                        str(published) if published else None
                    )

    result = st.session_state.get("result")
    if result is not None:
        render_output(result, st.session_state.get("filename", "fingerprint.wav"))
    elif image_bytes is None:
        st.info(
            "choose an image above, or use the sample — a sharp, well-lit photo "
            "with the print filling most of the frame works best."
        )

    st.markdown(
        ui_theme.footer(
            "Images are processed on the server to produce audio and are not "
            "stored. Upload only fingerprints you have permission to use. This is "
            "an artistic analysis tool: it does not identify people and is not "
            "authentication software.",
            f"Fingerprints — {ARTIST}, {YEAR}",
        ),
        unsafe_allow_html=True,
    )


main()
