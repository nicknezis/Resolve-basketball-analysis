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
class HoopObservation:
    """A hoop detection at a specific frame, including its bounding box."""

    frame_idx: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[int, int]
    confidence: float


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
    hoop_bbox: tuple[int, int, int, int] | None = None
    hoop_x_distance: float | None = None  # pixels between descent median x and hoop x
    descent_ratio: float | None = None  # descent_height / arc_height


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
        self._hoop_observations: list[HoopObservation] = []

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
        self._hoop_observations = []

    def update(self, frame_detections: FrameDetections) -> BallPosition | None:
        """Process detections for one frame and return tracked ball position.

        Uses spatially-gated detection: candidates are scored by a blend of
        confidence and proximity to the Kalman filter's predicted position.
        Detections beyond ``max_ball_jump_px`` are rejected unless the tracker
        is re-acquiring after an extended gap.

        Args:
            frame_detections: Detections for this frame from ObjectDetector.

        Returns:
            BallPosition if ball is being tracked, None if lost.
        """
        frame_idx = frame_detections.frame_idx

        if frame_detections.balls:
            if self._last_detection_frame < 0:
                # First detection ever — pick highest confidence, initialize
                best_ball = max(frame_detections.balls, key=lambda d: d.confidence)
                cx, cy = best_ball.center
                self._kf.x = np.array([cx, cy, 0, 0], dtype=float)
            else:
                # Predict where ball should be
                self._kf.predict()
                pred_x, pred_y = float(self._kf.x[0]), float(self._kf.x[1])

                gap = frame_idx - self._last_detection_frame
                reacquiring = gap > self.config.reacquire_after_gap_frames

                best_ball = None
                best_score = -1.0
                max_jump = self.config.max_ball_jump_px

                for det in frame_detections.balls:
                    dcx, dcy = det.center
                    dist = ((dcx - pred_x) ** 2 + (dcy - pred_y) ** 2) ** 0.5

                    if not reacquiring and dist > max_jump:
                        continue

                    proximity = max(0.0, 1.0 - dist / max_jump) if max_jump > 0 else 1.0
                    w = self.config.ball_gate_weight
                    score = (1.0 - w) * det.confidence + w * proximity

                    if score > best_score:
                        best_score = score
                        best_ball = det

                if best_ball is None:
                    # All detections rejected — use prediction if within gap window
                    if gap <= self.config.max_ball_gap_frames:
                        px, py = int(self._kf.x[0]), int(self._kf.x[1])
                        pos = BallPosition(frame_idx=frame_idx, x=px, y=py, predicted=True)
                        self._positions.append(pos)
                        return pos
                    return None

                cx, cy = best_ball.center
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

    def set_hoop_positions(self, hoop_positions: list[tuple[int, int]]) -> None:
        """Store hoop positions collected during frame-by-frame processing.

        Computes the median hoop position for use in shot detection.
        Call this after accumulating ball positions via update(), before find_shots().

        Args:
            hoop_positions: List of (x, y) hoop center coordinates.
        """
        if hoop_positions:
            hx = int(np.median([p[0] for p in hoop_positions]))
            hy = int(np.median([p[1] for p in hoop_positions]))
            self._median_hoop = (hx, hy)
        else:
            self._median_hoop = None

    def set_hoop_observations(self, observations: list[HoopObservation]) -> None:
        """Store per-frame hoop observations with bounding boxes.

        Enables bbox-based made-shot detection.  Also computes the median
        hoop center for backward-compatible fallback.

        Args:
            observations: List of HoopObservation collected during detection.
        """
        self._hoop_observations = observations
        # Also compute median hoop for fallback
        if observations:
            hx = int(np.median([o.center[0] for o in observations]))
            hy = int(np.median([o.center[1] for o in observations]))
            self._median_hoop = (hx, hy)
        else:
            self._median_hoop = None

    def find_shots(self) -> list[ShotEvent]:
        """Find shot events from already-accumulated ball positions.

        Unlike detect_shots(), this does not reset or re-track — it uses
        positions already collected via update() calls and the hoop position
        set via set_hoop_positions() or set_hoop_observations().

        Returns:
            List of detected ShotEvent objects.
        """
        if len(self._positions) < 5:
            return []

        median_hoop = getattr(self, "_median_hoop", None)
        shots = self._find_arcs(median_hoop)
        logger.info("Detected %d shot events", len(shots))
        return shots

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
        """Find ball arc trajectories that look like shot attempts.

        Applies multiple validation gates beyond the basic arc height check:
        - Maximum arc duration (rejects overly long arcs from tracking noise)
        - Minimum descent ratio (ball must fall significantly after peak)
        - Hoop-directed descent (ball's descent must be near the hoop horizontally)
        """
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
                        arc_len = arc_end - arc_start + 1

                        # Gate A: Maximum arc duration
                        if arc_len > self.config.shot_max_arc_frames:
                            logger.debug(
                                "Arc rejected (Gate A: duration): arc_len=%d > max=%d (frames %d-%d)",
                                arc_len, self.config.shot_max_arc_frames,
                                positions[arc_start].frame_idx, positions[arc_end].frame_idx,
                            )
                            i = arc_end
                            break

                        # Gate B: Minimum descent ratio
                        descent_height = curr_y - min_y
                        d_ratio = descent_height / arc_height if arc_height > 0 else 0.0
                        if d_ratio < self.config.shot_min_descent_ratio:
                            logger.debug(
                                "Arc rejected (Gate B: descent_ratio): d_ratio=%.3f < min=%.3f "
                                "(arc_height=%d, descent=%d, frames %d-%d)",
                                d_ratio, self.config.shot_min_descent_ratio,
                                arc_height, descent_height,
                                positions[arc_start].frame_idx, positions[arc_end].frame_idx,
                            )
                            i = arc_end
                            break

                        # Gate C: Hoop-directed descent
                        hoop_x_dist: float | None = None
                        if hoop_pos is not None:
                            hoop_x = hoop_pos[0]
                            descent_positions = positions[peak_idx : arc_end + 1]
                            if descent_positions:
                                descent_median_x = int(np.median([p.x for p in descent_positions]))
                                hoop_x_dist = float(abs(descent_median_x - hoop_x))
                                # Estimate frame width from tracked positions
                                all_x = [p.x for p in positions]
                                frame_w = max(max(all_x) - min(all_x), 640)
                                max_x_dist = frame_w * self.config.shot_hoop_x_range_ratio
                                if hoop_x_dist > max_x_dist:
                                    logger.debug(
                                        "Arc rejected (Gate C: hoop-directed): hoop_x_dist=%.1f > max=%.1f "
                                        "(frames %d-%d)",
                                        hoop_x_dist, max_x_dist,
                                        positions[arc_start].frame_idx, positions[arc_end].frame_idx,
                                    )
                                    i = arc_end
                                    break

                        # Tighten event window: start at most shot_pre_peak_frames before peak
                        effective_start = max(arc_start, peak_idx - self.config.shot_pre_peak_frames)
                        arc_positions = positions[effective_start : arc_end + 1]

                        # Extend window for made-shot check to capture backboard
                        # bounces where the ball goes through the hoop after the
                        # main arc descent triggers
                        made_end = min(
                            arc_end + self.config.shot_post_arc_frames,
                            len(positions) - 1,
                        )
                        made_positions = positions[effective_start : made_end + 1]

                        # Try bbox-based check first, fall back to center proximity
                        made = False
                        hoop_bbox = None
                        if self._hoop_observations:
                            made, hoop_bbox = self._check_ball_through_hoop_bbox(
                                made_positions, self._hoop_observations,
                            )
                        elif hoop_pos:
                            made = self._check_through_hoop(made_positions, hoop_pos)

                        # Include extended positions in event if made via
                        # the post-arc window (e.g. backboard bounce)
                        if made and made_end > arc_end:
                            arc_positions = made_positions
                            arc_end = made_end

                        logger.debug(
                            "Arc accepted: frames %d-%d, arc_height=%d, d_ratio=%.3f, made=%s",
                            positions[effective_start].frame_idx, positions[arc_end].frame_idx,
                            arc_height, d_ratio, made,
                        )
                        shots.append(
                            ShotEvent(
                                start_frame=positions[effective_start].frame_idx,
                                end_frame=positions[arc_end].frame_idx,
                                made=made,
                                ball_positions=arc_positions,
                                arc_height_px=arc_height,
                                hoop_x=hoop_pos[0] if hoop_pos else None,
                                hoop_y=hoop_pos[1] if hoop_pos else None,
                                hoop_bbox=hoop_bbox,
                                hoop_x_distance=hoop_x_dist,
                                descent_ratio=d_ratio,
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

    def _check_ball_through_hoop_bbox(
        self,
        positions: list[BallPosition],
        hoop_observations: list[HoopObservation],
    ) -> tuple[bool, tuple[int, int, int, int] | None]:
        """Check if the ball descends through a hoop bounding box.

        Looks at the descent phase of the arc (after the peak) and checks:
        1. Ball is horizontally within the hoop bbox (expanded by tolerance)
        2. Ball transitions from above hoop top to below it

        Returns:
            (made, hoop_bbox) — True and the matched hoop bbox if a through-
            hoop transition was detected, otherwise (False, None).
        """
        if not positions or not hoop_observations:
            return False, None

        # Find the peak of the arc (minimum y in image coords)
        peak_idx = min(range(len(positions)), key=lambda k: positions[k].y)
        descent = positions[peak_idx:]
        if len(descent) < 2:
            return False, None

        # Find the hoop observation nearest in time to the descent phase
        descent_start_frame = descent[0].frame_idx
        descent_end_frame = descent[-1].frame_idx

        best_obs = None
        best_dist = float("inf")
        for obs in hoop_observations:
            # Prefer observations during the descent
            mid_frame = (descent_start_frame + descent_end_frame) / 2
            d = abs(obs.frame_idx - mid_frame)
            if d < best_dist:
                best_dist = d
                best_obs = obs

        if best_obs is None:
            return False, None

        hx1, hy1, hx2, hy2 = best_obs.bbox
        hoop_w = hx2 - hx1
        tolerance = hoop_w * self.config.hoop_x_tolerance_ratio
        margin = self.config.hoop_entry_y_margin_px

        # Check for a ball position above the hoop followed by one at/below it.
        # In image coords, smaller y = higher in the frame.
        above = False
        for pos in descent:
            in_x = (hx1 - tolerance) <= pos.x <= (hx2 + tolerance)
            if not in_x:
                continue

            if pos.y <= hy1 - margin:
                # Ball is clearly above the hoop top
                above = True
            elif above and pos.y >= hy1:
                # Ball was above hoop top and has now reached/crossed it
                return True, best_obs.bbox

        return False, None
