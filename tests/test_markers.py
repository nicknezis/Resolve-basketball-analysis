"""Marker planning for the Resolve import (no Resolve needed)."""

from __future__ import annotations

import json

from src.resolve.markers import event_passes, marker_note, plan_markers


def _shot(start, end, made, via, conf=0.9, **details):
    d = {"made": made, "made_via": via, "source_clip": "C20260912_6415.MP4", **details}
    return {"type": "made_shot" if made else "shot_attempt", "start_frame": start, "end_frame": end,
            "start_sec": start / 60, "end_sec": end / 60, "confidence": conf, "details": d}


def _crowd(start, end, conf):
    return {"type": "crowd_excitement", "start_frame": start, "end_frame": end,
            "start_sec": start / 60, "end_sec": end / 60, "confidence": conf, "details": {"source": "audio_only"}}


def test_note_is_short_and_readable():
    ev = _shot(836, 962, True, "through_net", conf=0.97, shot_distance_ft=19.6, shot_zone="two", rim_impact=0.52,
               fit_rmse=19.1, ball_positions_count=45, source_file="/very/long/path.mp4")
    note = marker_note(ev)
    assert note == "Made Shot 97% · through net · 20 ft two · rim impact (audio) · C20260912_6415"
    assert "path" not in note and "fit_rmse" not in note


def test_note_for_miss_and_unknown_and_crowd():
    assert marker_note(_shot(10, 40, False, "rim_out", conf=0.88)) == "Shot Attempt 88% · rim out · C20260912_6415"
    unknown = _shot(10, 40, False, None, conf=0.84)
    unknown["details"]["made"] = None
    assert marker_note(unknown) == "Shot Attempt 84% · no rim in view · C20260912_6415"
    assert marker_note(_crowd(0, 270, 0.92)) == "Crowd Reaction 92% · 4.5 s"


def test_crowd_has_its_own_threshold():
    assert event_passes(_crowd(0, 60, 0.8), 0.0, 0.85) is False
    assert event_passes(_crowd(0, 60, 0.9), 0.0, 0.85) is True
    assert event_passes(_shot(0, 60, False, "short", conf=0.8), 0.0, 0.85) is True


def test_plan_nudges_collisions_and_keeps_shot_on_its_frame():
    events = [_crowd(100, 400, 0.95), _shot(100, 140, True, "through_net")]
    markers, skipped = plan_markers(events, 60.0, 60.0)
    assert [m["name"] for m in markers] == ["Made Shot", "Crowd Reaction"]
    assert markers[0]["frame"] == 100 and "nudged_from" not in markers[0]
    assert markers[1]["frame"] == 101 and markers[1]["nudged_from"] == 100
    assert markers[0]["color"] == "Blue" and markers[1]["color"] == "Cyan"
    assert skipped == {}


def test_plan_filters_and_reports_by_type():
    events = [_crowd(0, 60, 0.7), _crowd(200, 260, 0.9), _shot(500, 540, False, "short", conf=0.75)]
    markers, skipped = plan_markers(events, 60.0, 60.0, min_confidence=0.0, crowd_min_confidence=0.85)
    assert [m["frame"] for m in markers] == [200, 500]
    assert skipped == {"crowd_excitement": 1}
    custom = json.loads(markers[1]["custom_data"])
    assert custom["details"]["made_via"] == "short" and custom["start_frame"] == 500


def test_fps_rescale():
    markers, _ = plan_markers([_shot(600, 660, True, "rim_entry")], timeline_fps=29.97, source_fps=59.94)
    assert markers[0]["frame"] == 300 and markers[0]["duration"] == 30
