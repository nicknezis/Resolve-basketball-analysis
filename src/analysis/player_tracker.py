"""Player tracking with team classification via jersey color."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

from src.analysis.object_detector import Detection, FrameDetections
from src.config import TrackingConfig

logger = logging.getLogger(__name__)


@dataclass
class TrackedPlayer:
    """A player tracked across frames with team assignment."""

    track_id: int
    team: str | None  # "team_a", "team_b", or None if unclassified
    positions: list[tuple[int, int, int]]  # (frame_idx, center_x, center_y)
    dominant_color_hsv: tuple[int, int, int] | None = None


@dataclass
class FrameTracking:
    """Tracking state for a single frame."""

    frame_idx: int
    players: list[dict] = field(default_factory=list)  # [{track_id, bbox, team}]


class PlayerTracker:
    """Tracks players across frames and classifies them by team color.

    Uses DeepSORT for multi-object tracking (maintains identity across
    frames using appearance and motion features). Team classification
    is done by analyzing the dominant jersey color in HSV space and
    clustering into two groups.
    """

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self._tracker = DeepSort(
            max_age=self.config.deepsort_max_age,
            n_init=self.config.deepsort_n_init,
        )
        self._color_samples: dict[int, list[np.ndarray]] = {}  # track_id -> HSV samples
        self._tracks: dict[int, TrackedPlayer] = {}

    def reset(self) -> None:
        """Reset tracker for a new video or scene."""
        self._tracker = DeepSort(
            max_age=self.config.deepsort_max_age,
            n_init=self.config.deepsort_n_init,
        )
        self._color_samples = {}
        self._tracks = {}

    def update(
        self,
        frame: np.ndarray,
        frame_detections: FrameDetections,
    ) -> FrameTracking:
        """Process one frame of player detections.

        Args:
            frame: BGR image for jersey color extraction.
            frame_detections: Player detections from ObjectDetector.

        Returns:
            FrameTracking with tracked player states.
        """
        frame_idx = frame_detections.frame_idx
        ft = FrameTracking(frame_idx=frame_idx)

        if not frame_detections.players:
            # Still call update with empty detections to age out old tracks
            self._tracker.update_tracks([], frame=frame)
            return ft

        # Prepare detections for DeepSORT: ([x1, y1, w, h], confidence, class)
        raw_detections = []
        for det in frame_detections.players:
            x1, y1, x2, y2 = det.bbox
            raw_detections.append(
                ([x1, y1, x2 - x1, y2 - y1], det.confidence, "player")
            )

        tracks = self._tracker.update_tracks(raw_detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            # Sample jersey color from upper body region
            self._sample_jersey_color(frame, track_id, x1, y1, x2, y2)

            # Update tracked player
            if track_id not in self._tracks:
                self._tracks[track_id] = TrackedPlayer(
                    track_id=track_id, team=None, positions=[],
                )
            self._tracks[track_id].positions.append((frame_idx, cx, cy))

            ft.players.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "team": self._tracks[track_id].team,
            })

        return ft

    def classify_teams(self) -> None:
        """Classify all tracked players into two teams based on jersey color.

        Uses k-means clustering (k=2) on the median HSV jersey color
        of each player. Should be called after processing all frames.
        """
        if len(self._color_samples) < 2:
            logger.warning("Not enough tracked players to classify teams")
            return

        # Compute median HSV for each player
        player_colors = {}
        for track_id, samples in self._color_samples.items():
            if len(samples) >= 3:
                stacked = np.stack(samples)
                median_hsv = np.median(stacked, axis=0).astype(np.float32)
                player_colors[track_id] = median_hsv

        if len(player_colors) < 2:
            return

        # K-means with k=2
        track_ids = list(player_colors.keys())
        color_matrix = np.array([player_colors[tid] for tid in track_ids], dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(
            color_matrix, 2, None, criteria, 10, cv2.KMEANS_PP_CENTERS,
        )

        for i, track_id in enumerate(track_ids):
            team = "team_a" if labels[i][0] == 0 else "team_b"
            if track_id in self._tracks:
                self._tracks[track_id].team = team
                self._tracks[track_id].dominant_color_hsv = tuple(
                    int(v) for v in player_colors[track_id]
                )

        logger.info(
            "Classified %d players into teams (cluster sizes: %d / %d)",
            len(track_ids),
            int(np.sum(labels == 0)),
            int(np.sum(labels == 1)),
        )

    def get_tracked_players(self) -> dict[int, TrackedPlayer]:
        """Return all tracked players after processing."""
        return dict(self._tracks)

    def _sample_jersey_color(
        self,
        frame: np.ndarray,
        track_id: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> None:
        """Extract the dominant jersey color from the upper body region."""
        h, w = frame.shape[:2]
        # Upper 40% of bounding box approximates the torso/jersey
        body_y1 = max(0, y1)
        body_y2 = min(h, y1 + int((y2 - y1) * 0.4))
        body_x1 = max(0, x1)
        body_x2 = min(w, x2)

        if body_y2 <= body_y1 or body_x2 <= body_x1:
            return

        crop = frame[body_y1:body_y2, body_x1:body_x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        median_hsv = np.median(hsv.reshape(-1, 3), axis=0)

        if track_id not in self._color_samples:
            self._color_samples[track_id] = []
        self._color_samples[track_id].append(median_hsv)
