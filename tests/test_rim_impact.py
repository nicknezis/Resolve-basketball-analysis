"""Rim/backboard impact transient detection."""

from __future__ import annotations

import numpy as np
import soundfile as sf

from src.analysis.audio_analyzer import detect_rim_impacts
from src.config import AudioConfig


def _write(tmp_path, y, sr=22050):
    p = tmp_path / "a.wav"
    sf.write(str(p), y.astype(np.float32), sr)
    return p


def test_click_in_gym_noise_is_detected(tmp_path):
    sr = 22050
    rng = np.random.default_rng(0)
    y = 0.02 * rng.standard_normal(sr * 4)  # 4 s of low-level broadband noise
    # 30 ms metallic burst at t=2.0 s
    t = np.arange(int(0.03 * sr)) / sr
    burst = 0.6 * np.sin(2 * np.pi * 2500 * t) * np.exp(-t / 0.01)
    y[int(2.0 * sr): int(2.0 * sr) + burst.size] += burst
    events = detect_rim_impacts(_write(tmp_path, y, sr), AudioConfig())
    assert events, "impact not detected"
    hit = min(events, key=lambda e: abs(e.start_sec - 2.0))
    assert abs(hit.start_sec - 2.0) < 0.05
    assert 0.0 < hit.score <= 1.0


def test_steady_noise_has_no_impacts(tmp_path):
    sr = 22050
    rng = np.random.default_rng(1)
    y = 0.05 * rng.standard_normal(sr * 3)
    events = detect_rim_impacts(_write(tmp_path, y, sr), AudioConfig())
    assert len(events) <= 1  # an occasional noise spike is tolerable, a storm is not
