"""Ball tracking with Kalman filter and shot detection logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from src.analysis.object_detector import FrameDetections
from src.config import TrackingConfig

logger = logging.getLogger(__name__)


@dataclass
class BallPosition:
    """Tracked ball position at a frame."""

    frame_idx: int
    x: int
    y: int
    predicted: bool  # True if interpolated/predicted rather than detected


@dataclass
class ShotEvent:
    """A detected shot attempt or made shot."""

    start_frame: int
    end_frame: int
    made: bool
    ball_positions: list[BallPosition]
    arc_height_px: float
    hoop_x: int | None = None
    hoop_y: int | None = None


class BallTracker:
    """Tracks basketball position across frames and detects shot attempts.

    Uses a Kalman filter to smooth the ball trajectory and interpolate
    across frames where detection is missing. Shot detection looks for
    an arc trajectory that passes through or near the hoop position.
    """

    def __init__(self, config: TrackingConfig | None = None):
        self.config = config or TrackingConfig()
        self._kf = self._init_kalman()
        self._positions: list[BallPosition] = []
        self._last_detection_frame: int = -1

    def _init_kalman(self) -> KalmanFilter:
        """Initialize a 2D position+velocity Kalman filter."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # frame-based time step

        # State transition: [x, y, vx, vy]
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement function: we observe [x, y]
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Covariances
        q = self.config.kalman_process_noise
        kf.Q *= q
        r = self.config.kalman_measurement_noise
        kf.R = np.eye(2) * r
        kf.P *= 10.0

        return kf

    def reset(self) -> None:
        """Reset tracker state for a new video or scene."""
        self._kf = self._init_kalman()
        self._positions = []
        self._last_detection_frame = -1

    def update(self, frame_detections: FrameDetections) -> BallPosition | None:
        """Process detections for one frame and return tracked ball position.

        Args:
            frame_detections: Detections for this frame from ObjectDetector.

        Returns:
            BallPosition if ball is being tracked, None if lost.
        """
        frame_idx = frame_detections.frame_idx

        if frame_detections.balls:
            # Use highest-confidence ball detection
            best_ball = max(frame_detections.balls, key=lambda d: d.confidence)
            cx, cy = best_ball.center

            if self._last_detection_frame < 0:
                # First detection — initialize state
                self._kf.x = np.array([cx, cy, 0, 0], dtype=float)
            else:
                self._kf.predict()
                self._kf.update(np.array([cx, cy], dtype=float))

            self._last_detection_frame = frame_idx
            pos = BallPosition(frame_idx=frame_idx, x=cx, y=cy, predicted=False)

        elif (
            self._last_detection_frame >= 0
            and (frame_idx - self._last_detection_frame) <= self.config.max_ball_gap_frames
        ):
            # Ball not detected but within interpolation window — predict
            self._kf.predict()
            px, py = int(self._kf.x[0]), int(self._kf.x[1])
            pos = BallPosition(frame_idx=frame_idx, x=px, y=py, predicted=True)
        else:
            return None

        self._positions.append(pos)
        return pos

    def detect_shots(
        self,
        all_detections: list[FrameDetections],
    ) -> list[ShotEvent]:
        """Analyze full detection sequence to find shot attempts and makes.

        A shot is identified by:
        1. Ball moving upward (arc), reaching a peak, then descending
        2. Arc height exceeds minimum threshold
        3. If ball passes through/near hoop bbox → made shot

        Args:
            all_detections: Full sequence of per-frame detections.

        Returns:
            List of detected ShotEvent objects.
        """
        self.reset()

        # First pass: track ball through all frames
        for fd in all_detections:
            self.update(fd)

        if len(self._positions) < 5:
            return []

        # Build hoop position lookup (use median hoop position if stable)
        hoop_positions = []
        for fd in all_detections:
            if fd.hoops:
                best_hoop = max(fd.hoops, key=lambda d: d.confidence)
                hoop_positions.append(best_hoop.center)

        median_hoop = None
        if hoop_positions:
            hx = int(np.median([p[0] for p in hoop_positions]))
            hy = int(np.median([p[1] for p in hoop_positions]))
            median_hoop = (hx, hy)

        # Second pass: find arc trajectories
        shots = self._find_arcs(median_hoop)

        logger.info("Detected %d shot events", len(shots))
        return shots

    def _find_arcs(self, hoop_pos: tuple[int, int] | None) -> list[ShotEvent]:
        """Find ball arc trajectories that look like shot attempts."""
        shots = []
        positions = self._positions

        # Sliding window to find vertical arc patterns
        # Ball goes up (y decreasing in image coords), peaks, then comes down
        i = 0
        while i < len(positions) - 4:
            # Look for start of upward motion
            arc_start = i
            peak_idx = None
            min_y = positions[i].y

            j = i + 1
            while j < len(positions):
                curr_y = positions[j].y
                if curr_y < min_y:
                    min_y = curr_y
                    peak_idx = j
                elif peak_idx is not None and curr_y > min_y + self.config.shot_min_arc_height_px:
                    # Ball has descended significantly past the peak — end of arc
                    arc_end = j
                    arc_height = positions[arc_start].y - min_y

                    if arc_height >= self.config.shot_min_arc_height_px:
                        arc_positions = positions[arc_start : arc_end + 1]

                        made = False
                        if hoop_pos:
                            made = self._check_through_hoop(arc_positions, hoop_pos)

                        shots.append(
                            ShotEvent(
                                start_frame=positions[arc_start].frame_idx,
                                end_frame=positions[arc_end].frame_idx,
                                made=made,
                                ball_positions=arc_positions,
                                arc_height_px=arc_height,
                                hoop_x=hoop_pos[0] if hoop_pos else None,
                                hoop_y=hoop_pos[1] if hoop_pos else None,
                            )
                        )

                    i = arc_end
                    break
                j += 1
            i += 1

        return shots

    def _check_through_hoop(
        self, positions: list[BallPosition], hoop: tuple[int, int]
    ) -> bool:
        """Check if the ball trajectory passes through/near the hoop."""
        hx, hy = hoop
        proximity = self.config.hoop_proximity_px

        for pos in positions:
            dist = ((pos.x - hx) ** 2 + (pos.y - hy) ** 2) ** 0.5
            if dist <= proximity:
                return True

        return False
