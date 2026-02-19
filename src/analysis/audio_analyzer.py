"""Audio analysis for crowd excitement and whistle/buzzer detection."""

from __future__ import annotations

import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.config import AudioConfig

logger = logging.getLogger(__name__)


@dataclass
class AudioEvent:
    """A detected audio event with timing and score."""

    event_type: str  # "crowd_excitement", "whistle", "buzzer"
    start_sec: float
    end_sec: float
    score: float  # normalized 0-1


def extract_audio(video_path: Path, output_path: Path | None = None, sample_rate: int = 22050) -> Path:
    """Extract audio track from video file using FFmpeg.

    Args:
        video_path: Path to the video file.
        output_path: Where to write the WAV file. Uses a temp file if None.
        sample_rate: Target sample rate.

    Returns:
        Path to the extracted WAV file.
    """
    if output_path is None:
        output_path = Path(tempfile.mktemp(suffix=".wav"))

    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        str(output_path),
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def analyze_crowd_excitement(
    audio_path: Path,
    config: AudioConfig | None = None,
) -> list[AudioEvent]:
    """Detect crowd excitement peaks using mel-spectrogram energy analysis.

    Computes energy in the frequency band where crowd noise is most
    prominent (typically 500-4000 Hz), normalizes against the baseline,
    and identifies windows that exceed the excitement threshold.

    Args:
        audio_path: Path to a WAV audio file.
        config: Audio analysis settings.

    Returns:
        List of AudioEvent objects for detected crowd excitement peaks.
    """
    config = config or AudioConfig()
    logger.info("Analyzing crowd excitement in %s", audio_path)

    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)
    duration = librosa.get_duration(y=y, sr=sr)

    hop_length = int(config.hop_sec * sr)
    n_fft = int(config.window_sec * sr)

    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=config.n_mels,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Map frequency bounds to mel bin indices
    mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmax=sr / 2)
    low_bin = np.searchsorted(mel_freqs, config.crowd_freq_low_hz)
    high_bin = np.searchsorted(mel_freqs, config.crowd_freq_high_hz)

    # Compute per-window energy in the crowd frequency band
    crowd_energy = np.mean(mel_db[low_bin:high_bin, :], axis=0)

    # Normalize to 0-1 range
    e_min, e_max = crowd_energy.min(), crowd_energy.max()
    if e_max - e_min > 0:
        excitement_scores = (crowd_energy - e_min) / (e_max - e_min)
    else:
        excitement_scores = np.zeros_like(crowd_energy)

    # Find contiguous regions above threshold
    above = excitement_scores >= config.excitement_threshold
    events = _contiguous_regions(above, excitement_scores, config.hop_sec, "crowd_excitement")

    logger.info(
        "Found %d crowd excitement events in %.1fs of audio", len(events), duration,
    )
    return events


def detect_whistles(
    audio_path: Path,
    config: AudioConfig | None = None,
) -> list[AudioEvent]:
    """Detect referee whistles and buzzers via spectral peak analysis.

    Whistles produce sharp tonal energy in a narrow high-frequency band
    (typically 2000-4500 Hz). We detect onset events in that band.

    Args:
        audio_path: Path to a WAV audio file.
        config: Audio analysis settings.

    Returns:
        List of AudioEvent objects for detected whistles/buzzers.
    """
    config = config or AudioConfig()
    logger.info("Detecting whistles in %s", audio_path)

    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)

    # Bandpass filter around whistle frequencies
    y_whistle = _bandpass(y, sr, config.whistle_freq_low_hz, config.whistle_freq_high_hz)

    # Compute spectral flux for onset detection
    hop_length = int(0.01 * sr)  # 10ms hop for fine resolution
    onset_env = librosa.onset.onset_strength(y=y_whistle, sr=sr, hop_length=hop_length)

    # Normalize
    if onset_env.max() > 0:
        onset_norm = onset_env / onset_env.max()
    else:
        onset_norm = onset_env

    # Find peaks above threshold
    above = onset_norm >= config.whistle_energy_threshold
    hop_sec = hop_length / sr
    events = _contiguous_regions(above, onset_norm, hop_sec, "whistle")

    # Filter out very short detections (< 0.1s) — whistles have sustained tone
    events = [e for e in events if (e.end_sec - e.start_sec) >= 0.1]

    logger.info("Found %d whistle events", len(events))
    return events


def analyze_audio(video_path: Path, config: AudioConfig | None = None) -> list[AudioEvent]:
    """Run full audio analysis pipeline on a video file.

    Extracts audio, then runs crowd excitement and whistle detection.

    Args:
        video_path: Path to the video file.
        config: Audio analysis settings.

    Returns:
        Combined list of all detected audio events, sorted by time.
    """
    config = config or AudioConfig()
    audio_path = extract_audio(video_path, sample_rate=config.sample_rate)

    try:
        crowd_events = analyze_crowd_excitement(audio_path, config)
        whistle_events = detect_whistles(audio_path, config)
    finally:
        audio_path.unlink(missing_ok=True)

    all_events = crowd_events + whistle_events
    all_events.sort(key=lambda e: e.start_sec)
    return all_events


def _bandpass(y: np.ndarray, sr: int, low_hz: int, high_hz: int) -> np.ndarray:
    """Apply a simple FFT-based bandpass filter."""
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    fft[~mask] = 0
    return np.fft.irfft(fft, n=len(y))


def _contiguous_regions(
    mask: np.ndarray,
    scores: np.ndarray,
    hop_sec: float,
    event_type: str,
) -> list[AudioEvent]:
    """Find contiguous True regions in a boolean mask and build AudioEvents."""
    events = []
    in_region = False
    start_idx = 0

    for i, val in enumerate(mask):
        if val and not in_region:
            start_idx = i
            in_region = True
        elif not val and in_region:
            peak_score = float(np.max(scores[start_idx:i]))
            events.append(
                AudioEvent(
                    event_type=event_type,
                    start_sec=start_idx * hop_sec,
                    end_sec=i * hop_sec,
                    score=peak_score,
                )
            )
            in_region = False

    # Handle region that extends to the end
    if in_region:
        peak_score = float(np.max(scores[start_idx:]))
        events.append(
            AudioEvent(
                event_type=event_type,
                start_sec=start_idx * hop_sec,
                end_sec=len(mask) * hop_sec,
                score=peak_score,
            )
        )

    return events
