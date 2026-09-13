"""Unit tests for EventClassifier fusion, gating and merging."""

from __future__ import annotations

from src.analysis.audio_analyzer import AudioEvent
from src.analysis.ball_tracker import BallPosition, ShotEvent
from src.analysis.event_classifier import (
    VIDEO_CONF_MADE,
    VIDEO_CONF_MISS,
    VIDEO_CONF_NO_HOOP,
    EventClassifier,
    GameEvent,
)
from src.config import EventConfig

FPS = 60.0


def _shot(start: int, end: int, made: bool | None, **kw) -> ShotEvent:
    positions = [BallPosition(frame_idx=f, x=450, y=200, predicted=False) for f in range(start, end + 1)]
    return ShotEvent(
        start_frame=start, end_frame=end, made=made, ball_positions=positions,
        arc_height_px=kw.pop("arc_height_px", 120), **kw,
    )


class TestFusion:
    def test_made_shot_passes_without_audio(self):
        clf = EventClassifier(EventConfig(), fps=FPS)
        events = clf.classify([_shot(0, 60, True)], [], [])
        assert [e.event_type for e in events] == ["made_shot"]
        assert events[0].video_confidence == VIDEO_CONF_MADE
        assert events[0].audio_confidence == 0.0
        assert events[0].confidence == VIDEO_CONF_MADE

    def test_miss_with_hoop_evidence_passes_without_audio(self):
        clf = EventClassifier(EventConfig(), fps=FPS)
        events = clf.classify([_shot(0, 60, False)], [], [])
        assert [e.event_type for e in events] == ["shot_attempt"]
        assert events[0].details["made"] is False
        assert events[0].confidence >= VIDEO_CONF_MISS

    def test_arc_without_hoop_needs_audio(self):
        clf = EventClassifier(EventConfig(), fps=FPS)
        assert clf.classify([_shot(0, 60, None)], [], []) == []

        loud = [AudioEvent("crowd_excitement", start_sec=1.0, end_sec=3.0, score=1.0)]
        events = clf.classify([_shot(0, 60, None)], loud, [])
        assert len(events) == 1
        assert events[0].details["made"] is None
        assert events[0].video_confidence == VIDEO_CONF_NO_HOOP

    def test_audio_only_raises_confidence(self):
        clf = EventClassifier(EventConfig(), fps=FPS)
        quiet = clf.classify([_shot(0, 60, False)], [], [])[0]
        loud = clf.classify(
            [_shot(0, 60, False)],
            [AudioEvent("crowd_excitement", 1.0, 3.0, 1.0)], [],
        )[0]
        assert loud.confidence > quiet.confidence
        assert loud.confidence <= 1.0
        assert loud.video_confidence == quiet.video_confidence

    def test_min_video_confidence_gate(self):
        clf = EventClassifier(EventConfig(min_video_confidence=0.9), fps=FPS)
        loud = [AudioEvent("crowd_excitement", 1.0, 3.0, 1.0)]
        events = clf.classify([_shot(0, 60, False)], loud, [])
        # The shot is gated out; only the standalone crowd event survives
        assert [e.event_type for e in events] == ["crowd_excitement"]


class TestMerging:
    def test_shots_close_together_are_not_merged(self):
        clf = EventClassifier(EventConfig(merge_gap_sec=2.0), fps=FPS)
        shots = [_shot(0, 40, True), _shot(100, 140, False)]  # 1 s apart
        events = clf.classify(shots, [], [])
        assert len(events) == 2

    def test_same_type_shots_are_not_merged(self):
        clf = EventClassifier(EventConfig(merge_gap_sec=2.0), fps=FPS)
        shots = [_shot(0, 40, False), _shot(100, 140, False)]
        events = clf.classify(shots, [], [])
        assert len(events) == 2

    def test_adjacent_crowd_events_are_merged(self):
        clf = EventClassifier(EventConfig(merge_gap_sec=2.0), fps=FPS)
        a = GameEvent("crowd_excitement", 0, 60, 0.0, 1.0, 0.9, 0.0, 0.9, {"source": "audio_only"})
        b = GameEvent("crowd_excitement", 90, 150, 1.5, 2.5, 0.8, 0.0, 0.8, {"source": "audio_only"})
        merged = clf._merge_nearby([a, b])
        assert len(merged) == 1
        assert merged[0].end_sec == 2.5
        assert merged[0].confidence == 0.9
