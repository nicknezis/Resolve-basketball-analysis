"""Player tracking (ByteTrack) with team classification.

ByteTrack (Roboflow's ``trackers`` package) associates boxes by predicted
position/IoU only — no appearance embedder — which is what a 60 fps sideline
camera needs: players move little between analysed frames, and the re-ID
network DeepSORT ran on every box was ~40% of the per-frame budget while
adding nothing the IoU match didn't already know.

Team classification collects jersey crops per track during the pass and
clusters them once at the end (see :meth:`PlayerTracker.classify_teams`).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import supervision as sv
from trackers import ByteTrackTracker

from src.analysis.object_detector import FrameDetections
from src.config import TrackingConfig

logger = logging.getLogger(__name__)

CROP_SIZE = (64, 96)  # (w, h) of stored jersey crops


@dataclass
class TrackedPlayer:
    """A player tracked across frames with team assignment."""

    track_id: int
    team: str | None  # "team_a", "team_b", or None if unclassified
    positions: list[tuple[int, int, int]]  # (frame_idx, center_x, center_y)
    dominant_color_hsv: tuple[int, int, int] | None = None
    crops: list[np.ndarray] = field(default_factory=list, repr=False)  # torso crops, BGR
    on_court: bool | None = None  # None = unknown (no court homography)


@dataclass
class FrameTracking:
    """Tracking state for a single frame."""

    frame_idx: int
    players: list[dict] = field(default_factory=list)  # [{track_id, bbox, team}]


class PlayerTracker:
    """Tracks players across frames and classifies them into two teams."""

    def __init__(self, config: TrackingConfig | None = None, device: str = "auto", fps: float = 30.0):
        self.config = config or TrackingConfig()
        self.fps = fps
        self._tracker = self._create_tracker()
        self._tracks: dict[int, TrackedPlayer] = {}
        self._last_crop_frame: dict[int, int] = {}
        self._frame_tracks: dict[int, list[tuple[int, tuple[int, int, int, int]]]] = {}
        self._team_info: dict = {}

    def _create_tracker(self) -> ByteTrackTracker:
        cfg = self.config
        return ByteTrackTracker(
            lost_track_buffer=cfg.track_lost_buffer_frames,
            frame_rate=float(self.fps),
            track_activation_threshold=cfg.track_activation_threshold,
            minimum_consecutive_frames=cfg.track_min_hits,
            minimum_iou_threshold=cfg.track_min_iou,
            high_conf_det_threshold=cfg.track_high_conf_threshold,
        )

    def reset(self) -> None:
        """Reset tracker for a new video or scene."""
        self._tracker = self._create_tracker()
        self._tracks = {}
        self._last_crop_frame = {}
        self._frame_tracks = {}
        self._team_info = {}

    def update(
        self,
        frame: np.ndarray,
        frame_detections: FrameDetections,
    ) -> FrameTracking:
        """Process one frame of player detections.

        Args:
            frame: BGR image, in the same coordinate space as the boxes.
            frame_detections: Player detections from ObjectDetector.
        """
        frame_idx = frame_detections.frame_idx
        ft = FrameTracking(frame_idx=frame_idx)

        players = frame_detections.players
        if players:
            dets = sv.Detections(
                xyxy=np.array([d.bbox for d in players], dtype=np.float32),
                confidence=np.array([d.confidence for d in players], dtype=np.float32),
                class_id=np.zeros(len(players), dtype=int),
            )
        else:
            dets = sv.Detections.empty()
        tracked = self._tracker.update(dets)

        if tracked.tracker_id is None:
            return ft
        for xyxy, tid in zip(tracked.xyxy, tracked.tracker_id):
            track_id = int(tid)
            if track_id < 0:
                continue  # not yet confirmed (fewer than track_min_hits matches)
            x1, y1, x2, y2 = (int(v) for v in xyxy)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            player = self._tracks.get(track_id)
            if player is None:
                player = TrackedPlayer(track_id=track_id, team=None, positions=[])
                self._tracks[track_id] = player
            player.positions.append((frame_idx, cx, cy))

            self._maybe_store_crop(frame, player, frame_idx, x1, y1, x2, y2)

            ft.players.append({
                "track_id": track_id,
                "bbox": (x1, y1, x2, y2),
                "team": player.team,
            })
            self._frame_tracks.setdefault(frame_idx, []).append((track_id, (x1, y1, x2, y2)))

        return ft

    def _maybe_store_crop(
        self, frame: np.ndarray, player: TrackedPlayer, frame_idx: int,
        x1: int, y1: int, x2: int, y2: int,
    ) -> None:
        """Keep a few evenly spaced torso crops per track for team clustering."""
        cfg = self.config
        last = self._last_crop_frame.get(player.track_id)
        if last is not None and frame_idx - last < cfg.jersey_sample_every_frames:
            return
        if len(player.crops) >= cfg.jersey_crops_per_track:
            return
        h, w = frame.shape[:2]
        bh = y2 - y1
        bw = x2 - x1
        if bh < 40 or bw < 16:
            return
        # Central torso: skip the head (top 15%) and legs (bottom 45%), trim the
        # sides so an adjacent player's shirt doesn't leak in.
        ty1 = max(0, y1 + int(bh * 0.15))
        ty2 = min(h, y1 + int(bh * 0.55))
        tx1 = max(0, x1 + int(bw * 0.2))
        tx2 = min(w, x2 - int(bw * 0.2))
        if ty2 - ty1 < 10 or tx2 - tx1 < 6:
            return
        crop = cv2.resize(frame[ty1:ty2, tx1:tx2], CROP_SIZE, interpolation=cv2.INTER_AREA)
        player.crops.append(crop)
        self._last_crop_frame[player.track_id] = frame_idx

    def classify_teams(self) -> None:
        """Split tracked players into two teams from their jersey crops.

        ``team_method="siglip"`` embeds the crops with SigLIP and clusters
        them (see :mod:`team_classifier`); referees and spectators that sit
        far from both clusters stay unassigned.  ``"hsv"`` is the legacy
        median-colour k-means, also the fallback if the model can't load.
        """
        cfg = self.config
        ids = [
            tid for tid, p in self._tracks.items()
            if len(p.crops) >= cfg.jersey_min_crops and p.on_court is not False
        ]
        if len(ids) < 2:
            logger.warning("Not enough tracked players to classify teams")
            return

        for tid in ids:
            p = self._tracks[tid]
            hsv = [np.median(cv2.cvtColor(c, cv2.COLOR_BGR2HSV).reshape(-1, 3), axis=0) for c in p.crops]
            p.dominant_color_hsv = tuple(int(v) for v in np.median(np.stack(hsv), axis=0))

        feats = None
        method = cfg.team_method
        if method == "siglip":
            try:
                from src.analysis.team_classifier import JerseyEmbedder

                embedder = JerseyEmbedder(cfg.team_embed_model)
                crops, owners = [], []
                for tid in ids:
                    for c in self._tracks[tid].crops:
                        crops.append(c)
                        owners.append(tid)
                emb = embedder.embed(crops)
                feats = np.stack([emb[[i for i, o in enumerate(owners) if o == tid]].mean(axis=0) for tid in ids])
                feats /= np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9
            except Exception as exc:  # noqa: BLE001
                logger.warning("SigLIP team classification unavailable (%s); falling back to HSV", exc)
                method = "hsv"
        if feats is None:
            feats = np.array([self._tracks[tid].dominant_color_hsv for tid in ids], dtype=np.float32)

        from src.analysis.team_classifier import cluster_two_teams

        labels, info = cluster_two_teams(feats, outlier_std=cfg.team_outlier_std)
        for tid, lab in zip(ids, labels):
            self._tracks[tid].team = None if lab < 0 else ("team_a" if lab == 0 else "team_b")
        self._team_info = {"method": method, **info}
        logger.info(
            "Classified %d players into teams via %s (sizes %s, %d unassigned, separation ratio %.2f)",
            len(ids), method, info.get("sizes"), info.get("outliers", 0), info.get("separation_ratio", 0.0),
        )

    def get_frame_tracks(self) -> dict[int, list[tuple[int, tuple[int, int, int, int]]]]:
        """Per-frame ``[(track_id, bbox)]`` for confirmed tracks (for review overlays)."""
        return self._frame_tracks

    def get_tracked_players(self) -> dict[int, TrackedPlayer]:
        """Return all tracked players after processing."""
        return dict(self._tracks)
