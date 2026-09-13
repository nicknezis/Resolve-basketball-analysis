"""ByteTrack-based player tracking and team clustering."""

from __future__ import annotations

import numpy as np

from src.analysis.object_detector import Detection, FrameDetections
from src.analysis.player_tracker import PlayerTracker
from src.config import TrackingConfig


def _frame_with_players(frame_idx: int, boxes: list[tuple[int, int, int, int]]) -> FrameDetections:
    fd = FrameDetections(frame_idx=frame_idx)
    for b in boxes:
        fd.players.append(Detection("player", 0.9, b, frame_idx))
    return fd


def _image(colors: list[tuple[tuple[int, int, int, int], tuple[int, int, int]]]) -> np.ndarray:
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    for (x1, y1, x2, y2), bgr in colors:
        img[y1:y2, x1:x2] = bgr
    return img


class TestPlayerTracker:
    def test_ids_are_stable_across_frames(self):
        tr = PlayerTracker(TrackingConfig(track_min_hits=1), fps=30.0)
        ids = []
        for i in range(10):
            boxes = [(100 + i * 3, 200, 160 + i * 3, 360), (700, 200, 760, 360)]
            ft = tr.update(_image([]), _frame_with_players(i * 2, boxes))
            ids.append(sorted(p["track_id"] for p in ft.players))
        assert ids[-1] == ids[3]  # same two ids once confirmed
        assert len(tr.get_tracked_players()) == 2

    def test_empty_frames_do_not_crash(self):
        tr = PlayerTracker(TrackingConfig(), fps=30.0)
        ft = tr.update(_image([]), FrameDetections(frame_idx=0))
        assert ft.players == []

    def test_two_jersey_colours_become_two_teams(self):
        cfg = TrackingConfig(track_min_hits=1, jersey_sample_every_frames=1, jersey_min_crops=2)
        tr = PlayerTracker(cfg, fps=30.0)
        red_boxes = [(100, 200, 160, 360), (300, 200, 360, 360)]
        teal_boxes = [(700, 200, 760, 360), (900, 200, 960, 360)]
        for i in range(6):
            img = _image([(b, (40, 40, 220)) for b in red_boxes] + [(b, (200, 180, 20)) for b in teal_boxes])
            tr.update(img, _frame_with_players(i * 2, red_boxes + teal_boxes))
        tr.classify_teams()
        players = tr.get_tracked_players()
        teams = {}
        for p in players.values():
            x = p.positions[-1][1]
            teams.setdefault("red" if x < 500 else "teal", set()).add(p.team)
        assert len(teams["red"]) == 1 and len(teams["teal"]) == 1
        assert teams["red"] != teams["teal"]
        assert None not in teams["red"] | teams["teal"]
