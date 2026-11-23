"""16-bit PCM WAV export, to disk or to memory."""

import io
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def _to_pcm(audio):
    audio = np.asarray(audio)
    if audio.ndim != 1:
        raise ValueError("Audio must be a one-dimensional mono signal")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or infinite samples")
    return np.round(
        np.clip(audio, -1.0, 1.0) * np.iinfo(np.int16).max
    ).astype(np.int16)


def wav_bytes(audio, sample_rate=44100):
    """Encode a mono float signal as 16-bit PCM WAV and return the bytes.

    Shared by the CLI (which writes the bytes to disk) and the web app (which
    streams them to the browser) so both produce identical files.
    """
    if sample_rate < 8000:
        raise ValueError("Sample rate must be at least 8000 Hz")
    buffer = io.BytesIO()
    wavfile.write(buffer, sample_rate, _to_pcm(audio))
    return buffer.getvalue()


def export_wav(output_path, audio, sample_rate=44100, overwrite=False):
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".wav":
        raise ValueError("Output path must use the .wav extension")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it."
        )
    payload = wav_bytes(audio, sample_rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_path.write_bytes(payload)
    except OSError as error:
        raise OSError(f"Could not write WAV file: {output_path}") from error
    return output_path
