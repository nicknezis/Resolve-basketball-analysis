"""Game-wide crowd normalisation."""

from __future__ import annotations

import numpy as np

from src.analysis.audio_analyzer import crowd_norm_from_energies
from src.config import AudioConfig


def test_norm_spans_all_clips():
    quiet = np.full(200, -60.0)
    loud = np.linspace(-60.0, -10.0, 200)
    norm = crowd_norm_from_energies([quiet, loud], AudioConfig())
    assert norm is not None
    lo, hi = norm
    assert lo < -55 and hi > -12  # bounds come from the whole game, not the quiet clip alone


def test_norm_none_for_flat_or_empty():
    assert crowd_norm_from_energies([], AudioConfig()) is None
    assert crowd_norm_from_energies([np.full(50, -40.0)], AudioConfig()) is None
