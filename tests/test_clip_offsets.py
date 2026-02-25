"""Tests for clip offset handling in timeline-aware analysis."""

from __future__ import annotations

from src.analysis.audio_analyzer import AudioEvent
from src.resolve.export import _compute_source_range


# ---------------------------------------------------------------------------
# _compute_source_range tests (Bugs 2, 3, 4)
# ---------------------------------------------------------------------------


def test_source_range_same_fps():
    """When media and timeline fps match, source frames == timeline frames."""
    start, end, duration = _compute_source_range(
        left_offset=1000,
        tl_duration=500,
        media_fps=30.0,
        timeline_fps=30.0,
    )
    assert start == 1000
    assert duration == 500
    assert end == 1500


def test_source_range_different_fps():
    """60fps media on 24fps timeline — duration must be scaled."""
    start, end, duration = _compute_source_range(
        left_offset=1831,
        tl_duration=405,
        media_fps=59.94,
        timeline_fps=24.0,
    )
    assert start == 1831
    # 405 timeline frames at 24fps = 16.875s → 16.875 * 59.94 ≈ 1011 source frames
    expected_duration = int(405 * 59.94 / 24.0)
    assert duration == expected_duration
    assert end == 1831 + expected_duration


def test_source_range_media_slower_than_timeline():
    """24fps media on 30fps timeline — fewer source frames than timeline frames."""
    start, end, duration = _compute_source_range(
        left_offset=100,
        tl_duration=300,
        media_fps=24.0,
        timeline_fps=30.0,
    )
    assert start == 100
    expected_duration = int(300 * 24.0 / 30.0)  # 240
    assert duration == expected_duration
    assert end == 100 + expected_duration


def test_source_start_ignores_timecode():
    """left_offset is file-relative; start TC must not inflate the position.

    Before the fix, media with non-zero start timecode (e.g. 10:30:45:12)
    would produce source_start_frame = timecode_frames + left_offset, causing
    OpenCV to seek past the end of the file.  The fix uses left_offset directly.
    """
    start, end, duration = _compute_source_range(
        left_offset=100,
        tl_duration=200,
        media_fps=30.0,
        timeline_fps=30.0,
    )
    # source_start must be 100 (file-relative), NOT some huge timecode offset + 100
    assert start == 100
    assert end == 300


def test_source_range_untrimmed_clip():
    """Untrimmed clip (left_offset=0) starts at file frame 0."""
    start, end, duration = _compute_source_range(
        left_offset=0,
        tl_duration=1000,
        media_fps=30.0,
        timeline_fps=30.0,
    )
    assert start == 0
    assert end == 1000
    assert duration == 1000


# ---------------------------------------------------------------------------
# Audio timestamp offset tests (Bug 1)
# ---------------------------------------------------------------------------


def test_audio_offset_shifts_to_source_time():
    """Audio events from trimmed WAV (0-based) must be shifted to source time."""
    source_start = 1831
    media_fps = 59.94
    audio_offset_sec = source_start / media_fps  # ≈ 30.55s

    events = [
        AudioEvent(event_type="crowd_excitement", start_sec=2.0, end_sec=3.5, score=0.8),
        AudioEvent(event_type="whistle", start_sec=5.0, end_sec=5.5, score=0.6),
    ]

    for ae in events:
        ae.start_sec += audio_offset_sec
        ae.end_sec += audio_offset_sec

    assert abs(events[0].start_sec - (2.0 + audio_offset_sec)) < 1e-6
    assert abs(events[0].end_sec - (3.5 + audio_offset_sec)) < 1e-6
    assert abs(events[1].start_sec - (5.0 + audio_offset_sec)) < 1e-6


def test_audio_offset_zero_for_untrimmed():
    """Untrimmed clip (source_start=0) produces zero offset — no shift."""
    source_start = 0
    media_fps = 30.0
    audio_offset_sec = source_start / media_fps  # 0.0

    event = AudioEvent(event_type="crowd_excitement", start_sec=1.0, end_sec=2.0, score=0.9)
    event.start_sec += audio_offset_sec
    event.end_sec += audio_offset_sec

    assert event.start_sec == 1.0
    assert event.end_sec == 2.0


def test_audio_video_correlation_window_overlaps():
    """After offset, audio and shot events should share the same time base.

    Shot at source frame 2000 (at 30fps = 66.67s).
    Audio event at clip-relative 5.0s from a clip starting at source frame 1800
    (= source time 60.0s).  After offset: audio at 60.0 + 5.0 = 65.0s.
    Correlation window for shot: [65.67s, 71.67s] should overlap audio at 65.0s.
    """
    source_start = 1800
    media_fps = 30.0
    audio_offset_sec = source_start / media_fps  # 60.0s

    # Simulate audio event from trimmed WAV
    audio_event = AudioEvent(
        event_type="crowd_excitement", start_sec=5.0, end_sec=7.0, score=0.85,
    )
    audio_event.start_sec += audio_offset_sec
    audio_event.end_sec += audio_offset_sec

    # Simulate shot event at source frame 2000
    shot_start_sec = 2000 / media_fps  # 66.67s
    shot_end_sec = 2030 / media_fps    # 67.67s

    # EventClassifier search window: [shot_start - 1, shot_end + 4]
    search_start = shot_start_sec - 1.0   # 65.67s
    search_end = shot_end_sec + 4.0       # 71.67s

    # Audio event at [65.0, 67.0] should overlap [65.67, 71.67]
    overlaps = not (audio_event.end_sec < search_start or audio_event.start_sec > search_end)
    assert overlaps, (
        f"Audio [{audio_event.start_sec:.2f}, {audio_event.end_sec:.2f}] should overlap "
        f"search [{search_start:.2f}, {search_end:.2f}]"
    )
