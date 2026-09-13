"""Audio analysis for crowd excitement and whistle/buzzer detection."""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from src.analysis.video_io import run_ffmpeg
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

    args = [
        "-y",
        "-i", str(video_path),
        "-vn",  # no video
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",  # mono
        str(output_path),
    ]
    logger.info("Extracting audio from %s", Path(video_path).name)
    run_ffmpeg(args, description="audio extraction")
    return output_path


def crowd_band_energy(
    audio_path: Path,
    config: AudioConfig | None = None,
) -> tuple[np.ndarray, float]:
    """Per-window mean log-energy in the crowd band (absolute dB, not per-file relative).

    Returns ``(energy_db, hop_sec)``.  Absolute reference matters: energies
    from different clips must be comparable so that one normalisation can
    span a whole game.
    """
    config = config or AudioConfig()
    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)

    hop_length = int(config.hop_sec * sr)
    n_fft = int(config.window_sec * sr)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=config.n_mels,
    )
    mel_db = librosa.power_to_db(mel_spec, ref=1.0, top_db=None)

    mel_freqs = librosa.mel_frequencies(n_mels=config.n_mels, fmax=sr / 2)
    low_bin = np.searchsorted(mel_freqs, config.crowd_freq_low_hz)
    high_bin = np.searchsorted(mel_freqs, config.crowd_freq_high_hz)

    energy = np.mean(mel_db[low_bin:high_bin, :], axis=0)
    return energy, config.hop_sec


def crowd_norm_from_energies(
    energies: list[np.ndarray], config: AudioConfig | None = None,
) -> tuple[float, float] | None:
    """Game-wide normalisation bounds (low/high percentiles of all windows)."""
    config = config or AudioConfig()
    joined = np.concatenate([e for e in energies if e is not None and len(e)]) if energies else np.array([])
    if joined.size < 10:
        return None
    lo = float(np.percentile(joined, config.crowd_norm_low_pct))
    hi = float(np.percentile(joined, config.crowd_norm_high_pct))
    if hi - lo <= 1e-6:
        return None
    return lo, hi


def analyze_crowd_excitement(
    audio_path: Path,
    config: AudioConfig | None = None,
    norm: tuple[float, float] | None = None,
) -> list[AudioEvent]:
    """Detect crowd excitement peaks using mel-spectrogram energy analysis.

    Computes energy in the frequency band where crowd noise is most
    prominent (typically 500-4000 Hz) and scores each window between 0 and 1.

    Args:
        audio_path: Path to a WAV audio file.
        config: Audio analysis settings.
        norm: ``(low_db, high_db)`` bounds mapping to scores 0 and 1 — pass
            the game-wide bounds from :func:`crowd_norm_from_energies` so a
            quiet clip's loudest moment does not score 1.0.  None falls back
            to this file's own min/max (legacy per-clip behaviour).

    Returns:
        List of AudioEvent objects for detected crowd excitement peaks.
    """
    config = config or AudioConfig()
    logger.info("Analyzing crowd excitement in %s", audio_path)

    crowd_energy, hop_sec = crowd_band_energy(audio_path, config)
    duration = len(crowd_energy) * hop_sec

    if norm is not None:
        lo, hi = norm
    else:
        lo, hi = float(crowd_energy.min()), float(crowd_energy.max())
    if hi - lo > 0:
        excitement_scores = np.clip((crowd_energy - lo) / (hi - lo), 0.0, 1.0)
    else:
        excitement_scores = np.zeros_like(crowd_energy)

    above = excitement_scores >= config.excitement_threshold
    events = _contiguous_regions(above, excitement_scores, hop_sec, "crowd_excitement")

    logger.info(
        "Found %d crowd excitement events in %.1fs of audio (%s normalisation)",
        len(events), duration, "game" if norm is not None else "clip",
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


def detect_rim_impacts(
    audio_path: Path,
    config: AudioConfig | None = None,
) -> list[AudioEvent]:
    """Detect short broadband transients — a ball hitting rim or backboard.

    Spectral-flux onsets in the 1–6 kHz band at 5 ms resolution, scored by
    prominence over a local median/MAD baseline so that a loud gym does not
    hide them and a quiet one does not manufacture them.  Sustained bursts
    (crowd, whistles) are rejected by ``impact_max_duration_sec``.  ``score``
    is the prominence mapped to 0–1 (saturating at 3× the minimum).
    """
    config = config or AudioConfig()
    y, sr = librosa.load(str(audio_path), sr=config.sample_rate)
    if y.size == 0:
        return []

    # Band-limited spectral flux: positive log-magnitude change summed over the
    # 1-6 kHz bins, at a 5 ms hop.  (librosa's onset_strength with a median
    # aggregate over a band-passed signal collapses to zero — most mel bands
    # are empty — so the flux is computed explicitly.)
    hop = max(1, int(config.impact_hop_sec * sr))
    n_fft = 1024
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    band = (freqs >= config.impact_freq_low_hz) & (freqs <= config.impact_freq_high_hz)
    log_s = np.log1p(S[band])
    flux = np.sum(np.maximum(log_s[:, 1:] - log_s[:, :-1], 0.0), axis=0)
    onset = np.concatenate([[0.0], flux])
    if onset.size < 8:
        return []
    hop_sec = hop / sr

    # Local baseline: rolling median and MAD
    win = max(3, int(config.impact_local_window_sec / hop_sec) | 1)
    pad = win // 2
    padded = np.pad(onset, pad, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, win)
    med = np.median(windows, axis=1)
    mad = np.median(np.abs(windows - med[:, None]), axis=1) + 1e-6
    prominence = (onset - med) / mad

    above = prominence >= config.impact_min_prominence
    events = _contiguous_regions(above, prominence, hop_sec, "rim_impact")
    out = []
    for e in events:
        if (e.end_sec - e.start_sec) > config.impact_max_duration_sec:
            continue
        e.score = float(min(1.0, e.score / (3.0 * config.impact_min_prominence)))
        out.append(e)
    logger.info("Found %d rim/backboard impact transients", len(out))
    return out


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
        impact_events = detect_rim_impacts(audio_path, config)
    finally:
        audio_path.unlink(missing_ok=True)

    all_events = crowd_events + whistle_events + impact_events
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
