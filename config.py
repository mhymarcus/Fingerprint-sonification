"""Configuration for the fingerprint sonification pipeline."""

from dataclasses import dataclass


@dataclass(frozen=True)
class SonificationConfig:
    analysis_width: int = 256
    analysis_height: int = 256
    num_segments: int = 144
    duration: float = 30.0
    min_frequency: float = 150.0
    max_frequency: float = 1200.0
    min_amplitude: float = 0.06
    max_amplitude: float = 0.55
    sample_rate: int = 44100
    fade_duration: float = 0.15
    enhance_contrast: bool = True
    orientation_block_size: int = 16
    minimum_ridge_length: float = 8.0
    max_segment_events: int = 48
    minimum_event_spacing: float = 0.08
    length_layer_gain: float = 0.32
    position_layer_gain: float = 0.32
    orientation_layer_gain: float = 0.55
    texture_layer_gain: float = 0.45
    global_gain: float = 1.4
    enable_length_layer: bool = True
    enable_position_layer: bool = True
    enable_orientation_layer: bool = True
    enable_texture_layer: bool = True

    def validate(self):
        if self.analysis_width < 32 or self.analysis_height < 32:
            raise ValueError("Analysis width and height must both be at least 32")
        if self.num_segments < 4:
            raise ValueError("Number of scanning segments must be at least 4")
        if self.duration <= 0 or self.duration > 600:
            raise ValueError("Duration must be greater than 0 and no more than 600 seconds")
        if self.min_frequency < 20:
            raise ValueError("Minimum frequency must be at least 20 Hz")
        if self.max_frequency <= self.min_frequency:
            raise ValueError("Maximum frequency must be greater than minimum frequency")
        nyquist_limit = self.sample_rate * 0.45
        if self.max_frequency * 4 >= nyquist_limit:
            raise ValueError(
                "Maximum frequency is too high for safe four-harmonic synthesis "
                f"at {self.sample_rate} Hz; use less than {nyquist_limit / 4:.1f} Hz"
            )
        if not 0 <= self.min_amplitude < self.max_amplitude <= 1:
            raise ValueError("Amplitude range must satisfy 0 <= min < max <= 1")
        if self.sample_rate < 8000:
            raise ValueError("Sample rate must be at least 8000 Hz")
        if self.fade_duration < 0:
            raise ValueError("Fade duration must be non-negative")
        if self.orientation_block_size < 4:
            raise ValueError("Orientation block size must be at least 4")
        if self.minimum_ridge_length < 1:
            raise ValueError("Minimum ridge length must be at least 1 pixel")
        if self.max_segment_events < 1:
            raise ValueError("Maximum segment events must be positive")
        if self.minimum_event_spacing < 0:
            raise ValueError("Minimum event spacing must be non-negative")
        gains = (
            self.length_layer_gain,
            self.position_layer_gain,
            self.orientation_layer_gain,
            self.texture_layer_gain,
            self.global_gain,
        )
        if any(gain < 0 or gain > 2 for gain in gains):
            raise ValueError("Layer and global gains must be between 0 and 2")
