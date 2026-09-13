"""Ball tracking with Kalman filter and shot detection logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from filterpy.kalman import KalmanFilter

from src.analysis.object_detector import Detection, FrameDetections
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
class _ConsensusCandidate:
    """A candidate ball detection pending consensus confirmation."""

    frame_idx: int
    x: int
    y: int
    confidence: float


@dataclass
class HoopObservation:
    """A hoop detection at a specific frame, including its bounding box."""

    frame_idx: int
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    center: tuple[int, int]
    confidence: float


@dataclass
class ShotEvent:
    """A detected shot attempt or made shot.

    ``made`` is tri-state: ``True`` / ``False`` when a usable hoop observation
    was available to adjudicate the descent, ``None`` when no hoop was seen
    near the shot (so "unknown", not "miss").
    """

    start_frame: int
    end_frame: int
    made: bool | None
    ball_positions: list[BallPosition]
    arc_height_px: float
    hoop_x: int | None = None
    hoop_y: int | None = None
    hoop_bbox: tuple[int, int, int, int] | None = None
    hoop_x_distance: float | None = None  # pixels between descent median x and hoop x
    descent_ratio: float | None = None  # descent_height / arc_height
    made_via: str | None = None  # "bbox", "polygon", "proximity", "ball_in_basket"
    peak_frame: int | None = None


class BallTracker:
    """Tracks basketball position across frames and detects shot attempts.

    Uses a Kalman filter to smooth the ball trajectory and interpolate
    across frames where detection is missing. Shot detection looks for
    an arc trajectory that passes through or near the hoop position.

    Args:
        config: Tracking thresholds.
        fps: Source frame rate — converts the second-based shot windows in
            the config into frame counts.  Frame indices passed to
            :meth:`update` are *source* frame numbers, so frame skipping does
            not change this value.
        frame_size: ``(width, height)`` of the analysed frames, used by the
            hoop-directed gate.  Can be set later with :meth:`set_frame_size`.
    """

    def __init__(
        self,
        config: TrackingConfig | None = None,
        fps: float = 30.0,
        frame_size: tuple[int, int] | None = None,
    ):
        self.config = config or TrackingConfig()
        self.fps = fps if fps and fps > 0 else 30.0
        self._frame_width: int | None = frame_size[0] if frame_size else None
        self._kf = self._init_kalman()
        self._positions: list[BallPosition] = []
        self._last_detection_frame: int = -1
        self._hoop_observations: list[HoopObservation] = []
        self._median_hoop: tuple[int, int] | None = None
        self._ball_in_basket_frames: list[int] = []
        self._consensus_buffer: list[_ConsensusCandidate] = []
        self._consensus_confirmed: bool = False

    def _init_kalman(self) -> KalmanFilter:
        """Initialize a 2D position+velocity Kalman filter."""
        kf = KalmanFilter(dim_x=4, dim_z=2)
        dt = 1.0  # one analysed frame per step

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
        self._median_hoop = None
        self._ball_in_basket_frames = []
        self._consensus_buffer = []
        self._consensus_confirmed = False

    def set_frame_size(self, width: int, height: int) -> None:
        """Record the analysed frame size (for the hoop-directed gate)."""
        self._frame_width = int(width)

    def _seed_filter(self, x: int, y: int) -> None:
        """(Re)start the Kalman filter at a position with fresh uncertainty.

        Re-inflating ``P`` matters on re-acquisition: after a long track the
        covariance has converged small, and without a reset the filter would
        be over-confident about a velocity that no longer applies.
        """
        self._kf.x = np.array([x, y, 0, 0], dtype=float)
        self._kf.P = np.eye(4) * 10.0

    def _select_best_ball(
        self, frame_detections: FrameDetections, frame_idx: int,
    ) -> Detection | None:
        """Select the best ball detection using confidence+proximity scoring.

        Returns None if all detections are rejected by the distance gate.
        """
        if not frame_detections.balls:
            return None

        if self._last_detection_frame < 0:
            return max(frame_detections.balls, key=lambda d: d.confidence)

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

        return best_ball

    def _handle_consensus(
        self, frame_idx: int, cx: int, cy: int, confidence: float,
    ) -> BallPosition | None:
        """Build consensus before committing to a ball track.

        Collects candidates in a rolling window.  Once N out of M frames
        have detections within ``consensus_max_spread_px`` of each other,
        consensus is confirmed and the Kalman filter is initialized.
        """
        self._consensus_buffer.append(
            _ConsensusCandidate(frame_idx=frame_idx, x=cx, y=cy, confidence=confidence)
        )

        # Expire candidates by frame age, not buffer length, so consensus
        # reflects "N detections within the last M frames".  A count-based
        # trim would let sparse detections spread across arbitrarily many
        # frames (e.g. 0, 100, 200) confirm consensus.
        window = self.config.consensus_window
        self._consensus_buffer = [
            c for c in self._consensus_buffer if frame_idx - c.frame_idx < window
        ]

        candidates = self._consensus_buffer
        if len(candidates) >= self.config.consensus_required:
            xs = [c.x for c in candidates]
            ys = [c.y for c in candidates]
            spread = max(max(xs) - min(xs), max(ys) - min(ys))

            if spread <= self.config.consensus_max_spread_px:
                self._consensus_confirmed = True
                best = max(candidates, key=lambda c: c.confidence)
                self._seed_filter(best.x, best.y)
                self._last_detection_frame = frame_idx

                for c in candidates:
                    self._positions.append(BallPosition(
                        frame_idx=c.frame_idx, x=c.x, y=c.y, predicted=False,
                    ))

                self._consensus_buffer = []
                # The current frame's candidate is already in the candidates
                # list, so return the last appended position.
                return self._positions[-1]

        return None

    def update(self, frame_detections: FrameDetections) -> BallPosition | None:
        """Process detections for one frame and return tracked ball position.

        Uses spatially-gated detection: candidates are scored by a blend of
        confidence and proximity to the Kalman filter's predicted position.
        Detections beyond ``max_ball_jump_px`` are rejected unless the tracker
        is re-acquiring after an extended gap.

        When ``consensus_required > 1``, new ball tracks must be confirmed by
        multiple consistent detections before the Kalman filter is initialized.

        Args:
            frame_detections: Detections for this frame from ObjectDetector.

        Returns:
            BallPosition if ball is being tracked, None if lost.
        """
        frame_idx = frame_detections.frame_idx

        if frame_detections.balls_in_basket:
            self._ball_in_basket_frames.append(frame_idx)

        if frame_detections.balls:
            best_ball = self._select_best_ball(frame_detections, frame_idx)

            if best_ball is None:
                # All detections rejected by distance gate
                if (
                    self._last_detection_frame >= 0
                    and (frame_idx - self._last_detection_frame) <= self.config.max_ball_gap_frames
                ):
                    px, py = int(self._kf.x[0]), int(self._kf.x[1])
                    pos = BallPosition(frame_idx=frame_idx, x=px, y=py, predicted=True)
                    self._positions.append(pos)
                    return pos
                return None

            cx, cy = best_ball.center

            # Consensus gate: require multiple detections before committing
            if not self._consensus_confirmed:
                return self._handle_consensus(frame_idx, cx, cy, best_ball.confidence)

            # Normal tracking — consensus already confirmed
            if self._last_detection_frame < 0:
                self._seed_filter(cx, cy)
            else:
                # _select_best_ball() already advanced the filter with
                # predict(); only apply the measurement update here.
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
            # Ball lost — reset consensus so re-acquisition must re-confirm
            if self._last_detection_frame >= 0:
                self._consensus_confirmed = False
                self._consensus_buffer = []
                self._last_detection_frame = -1
            return None

        self._positions.append(pos)
        return pos

    # ------------------------------------------------------------------
    # Hoop context
    # ------------------------------------------------------------------

    def set_hoop_positions(self, hoop_positions: list[tuple[int, int]]) -> None:
        """Store hoop center positions (legacy API without bounding boxes).

        Computes a median hoop position used by the center-proximity
        made-shot fallback.  Prefer :meth:`set_hoop_observations`.
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
        """
        self._hoop_observations = list(observations)
        if observations:
            hx = int(np.median([o.center[0] for o in observations]))
            hy = int(np.median([o.center[1] for o in observations]))
            self._median_hoop = (hx, hy)
        else:
            self._median_hoop = None

    def set_ball_in_basket_frames(self, frames: list[int]) -> None:
        """Record frames where the detector saw a ``ball-in-basket`` box."""
        self._ball_in_basket_frames = sorted(set(frames))

    def _select_hoop_observation(
        self,
        positions: list[BallPosition],
        observations: list[HoopObservation] | None = None,
    ) -> HoopObservation | None:
        """Pick the hoop observation that should adjudicate these positions.

        Only observations within ``hoop_obs_max_age_frames`` of the window
        are eligible — with a panning camera a rim seen 20 s earlier says
        nothing about where the rim is now.  When two rims are in view,
        prefer the one horizontally closest to the ball's descent, then the
        observation nearest in time.
        """
        obs_list = self._hoop_observations if observations is None else observations
        if not positions or not obs_list:
            return None

        first, last = positions[0].frame_idx, positions[-1].frame_idx
        mid_frame = (first + last) / 2
        half_window = (last - first) / 2
        max_age = self.config.hoop_obs_max_age_frames + half_window
        median_x = float(np.median([p.x for p in positions]))

        candidates = [o for o in obs_list if abs(o.frame_idx - mid_frame) <= max_age]
        if not candidates:
            return None

        def key(o: HoopObservation) -> tuple[bool, float]:
            hoop_w = max(1, o.bbox[2] - o.bbox[0])
            far = abs(o.center[0] - median_x) > 2 * hoop_w
            return (far, abs(o.frame_idx - mid_frame))

        return min(candidates, key=key)

    # ------------------------------------------------------------------
    # Shot detection
    # ------------------------------------------------------------------

    def find_shots(self) -> list[ShotEvent]:
        """Find shot events from already-accumulated ball positions.

        Uses positions collected via update() calls and the hoop context set
        via set_hoop_positions() / set_hoop_observations().
        """
        if len(self._positions) < 5:
            return []

        shots = self._find_arcs(self._median_hoop)
        logger.info("Detected %d shot events", len(shots))
        return shots

    def detect_shots(
        self,
        all_detections: list[FrameDetections],
    ) -> list[ShotEvent]:
        """Analyze a full detection sequence to find shot attempts and makes."""
        self.reset()

        for fd in all_detections:
            self.update(fd)

        if len(self._positions) < 5:
            return []

        observations = []
        for fd in all_detections:
            if fd.hoops:
                best_hoop = max(fd.hoops, key=lambda d: d.confidence)
                observations.append(HoopObservation(
                    frame_idx=fd.frame_idx, bbox=best_hoop.bbox,
                    center=best_hoop.center, confidence=best_hoop.confidence,
                ))
        self.set_hoop_observations(observations)

        shots = self._find_arcs(self._median_hoop)
        logger.info("Detected %d shot events", len(shots))
        return shots

    def _find_arcs(self, hoop_pos: tuple[int, int] | None) -> list[ShotEvent]:
        """Find ball arc trajectories that look like shot attempts.

        Scans the (sparse) position list for up-then-down motion.  All
        duration windows are measured in *source frames* via each position's
        ``frame_idx`` — never in list indices, because the list has holes
        wherever the ball was lost.  A gap longer than ``max_ball_gap_frames``
        between consecutive positions terminates the current arc: a ball that
        was lost and re-acquired somewhere else is not one trajectory.
        """
        shots: list[ShotEvent] = []
        positions = self._positions
        n = len(positions)
        max_gap = self.config.max_ball_gap_frames
        end_descent = self.config.shot_arc_end_descent_px

        i = 0
        while i < n - 4:
            arc_start = i
            peak_idx: int | None = None
            min_y = positions[i].y

            j = i + 1
            while j < n:
                if positions[j].frame_idx - positions[j - 1].frame_idx > max_gap:
                    # Track break — restart the scan at the re-acquired point
                    i = j - 1
                    break

                curr = positions[j]
                if curr.y < min_y:
                    min_y = curr.y
                    peak_idx = j
                elif (
                    peak_idx is not None
                    and not curr.predicted
                    and curr.y > min_y + end_descent
                ):
                    # Ball has descended significantly past the peak — end of
                    # arc.  Only a *detected* position may terminate an arc;
                    # Kalman extrapolation must not manufacture a descent.
                    shot, next_i = self._evaluate_arc(arc_start, peak_idx, j, min_y, hoop_pos)
                    if shot is not None:
                        shots.append(shot)
                    i = next_i
                    break
                j += 1
            i += 1

        return shots

    def _evaluate_arc(
        self,
        arc_start: int,
        peak_idx: int,
        arc_end: int,
        min_y: int,
        hoop_pos: tuple[int, int] | None,
    ) -> tuple[ShotEvent | None, int]:
        """Validate a candidate arc and build a ShotEvent.

        Returns ``(shot_or_None, index_to_resume_scanning_from)``.
        """
        positions = self._positions
        n = len(positions)
        cfg = self.config
        fps = self.fps
        max_gap = cfg.max_ball_gap_frames

        start_frame = positions[arc_start].frame_idx
        end_frame = positions[arc_end].frame_idx
        arc_height = positions[arc_start].y - min_y

        if arc_height < cfg.shot_min_arc_height_px:
            return None, arc_end

        # Gate A: maximum arc duration (wall-clock, via frame indices)
        duration_frames = end_frame - start_frame
        if duration_frames > cfg.shot_max_arc_sec * fps:
            logger.debug(
                "Arc rejected (Gate A: duration): %d frames > %.0f (frames %d-%d)",
                duration_frames, cfg.shot_max_arc_sec * fps, start_frame, end_frame,
            )
            return None, arc_end

        # Gate B: minimum descent ratio
        descent_height = positions[arc_end].y - min_y
        d_ratio = descent_height / arc_height if arc_height > 0 else 0.0
        if d_ratio < cfg.shot_min_descent_ratio:
            logger.debug(
                "Arc rejected (Gate B: descent_ratio): %.3f < %.3f (frames %d-%d)",
                d_ratio, cfg.shot_min_descent_ratio, start_frame, end_frame,
            )
            return None, arc_end

        # Which rim (if any) should judge this arc?
        descent_positions = positions[peak_idx: arc_end + 1]
        sel_obs = self._select_hoop_observation(descent_positions)
        if sel_obs is not None:
            hoop_xy: tuple[int, int] | None = sel_obs.center
        else:
            hoop_xy = hoop_pos

        # Gate C: hoop-directed descent
        hoop_x_dist: float | None = None
        if hoop_xy is not None:
            descent_median_x = int(np.median([p.x for p in descent_positions]))
            hoop_x_dist = float(abs(descent_median_x - hoop_xy[0]))
            if self._frame_width:
                frame_w = self._frame_width
            else:
                all_x = [p.x for p in positions]
                frame_w = max(max(all_x) - min(all_x), 640)
            max_x_dist = frame_w * cfg.shot_hoop_x_range_ratio
            if hoop_x_dist > max_x_dist:
                logger.debug(
                    "Arc rejected (Gate C: hoop-directed): %.1f px > %.1f (frames %d-%d)",
                    hoop_x_dist, max_x_dist, start_frame, end_frame,
                )
                return None, arc_end

        # Gate D: a shot at this rim has to get above it.  Passes, dribbles and
        # hand-offs produce up-and-down arcs too, but they stay below rim
        # height, and a real shot must be higher than the rim where it meets it.
        if sel_obs is not None and cfg.shot_require_peak_above_rim:
            rim_top = sel_obs.bbox[1]
            if min_y > rim_top + cfg.shot_peak_rim_margin_px:
                logger.debug(
                    "Arc rejected (Gate D: peak below rim): peak_y=%d > rim_top=%d+%d (frames %d-%d)",
                    min_y, rim_top, cfg.shot_peak_rim_margin_px, start_frame, end_frame,
                )
                return None, arc_end

        # Tighten the event window to shortly before the peak
        peak_frame = positions[peak_idx].frame_idx
        pre_peak_frames = cfg.shot_pre_peak_sec * fps
        effective_start = arc_start
        while (
            effective_start < peak_idx
            and peak_frame - positions[effective_start].frame_idx > pre_peak_frames
        ):
            effective_start += 1

        # Extend the made-shot window past arc_end to catch backboard bounces
        # that drop through after the descent trigger — but never across a
        # track break.
        post_arc_frames = cfg.shot_post_arc_sec * fps
        made_end = arc_end
        while (
            made_end + 1 < n
            and positions[made_end + 1].frame_idx - end_frame <= post_arc_frames
            and positions[made_end + 1].frame_idx - positions[made_end].frame_idx <= max_gap
        ):
            made_end += 1

        arc_positions = positions[effective_start: arc_end + 1]
        made_positions = positions[effective_start: made_end + 1]

        made: bool | None = None
        made_via: str | None = None
        hoop_bbox: tuple[int, int, int, int] | None = None
        if sel_obs is not None:
            made, hoop_bbox = self._check_ball_through_hoop_bbox(made_positions, [sel_obs])
            made_via = "bbox" if made else None
            if not made and cfg.use_polygon_zone:
                if self._check_ball_in_hoop_zone(made_positions, sel_obs):
                    made, made_via, hoop_bbox = True, "polygon", sel_obs.bbox
        elif hoop_pos is not None and not self._hoop_observations:
            # Legacy center-only hoop data
            made = self._check_through_hoop(made_positions, hoop_pos)
            made_via = "proximity" if made else None

        # A detector-reported ball-in-basket during/just after the descent is
        # direct evidence of a make, regardless of geometry.
        if not made and self._ball_in_basket_frames:
            lo = peak_frame
            hi = positions[made_end].frame_idx + post_arc_frames
            if any(lo <= f <= hi for f in self._ball_in_basket_frames):
                made, made_via = True, "ball_in_basket"

        if made and made_end > arc_end:
            arc_positions = made_positions
            arc_end = made_end

        logger.debug(
            "Arc accepted: frames %d-%d, arc_height=%d, d_ratio=%.3f, made=%s (%s)",
            positions[effective_start].frame_idx, positions[arc_end].frame_idx,
            arc_height, d_ratio, made, made_via,
        )
        shot = ShotEvent(
            start_frame=positions[effective_start].frame_idx,
            end_frame=positions[arc_end].frame_idx,
            made=made,
            ball_positions=arc_positions,
            arc_height_px=arc_height,
            hoop_x=hoop_xy[0] if hoop_xy else None,
            hoop_y=hoop_xy[1] if hoop_xy else None,
            hoop_bbox=hoop_bbox,
            hoop_x_distance=hoop_x_dist,
            descent_ratio=d_ratio,
            made_via=made_via,
            peak_frame=peak_frame,
        )
        return shot, arc_end

    def _check_through_hoop(
        self, positions: list[BallPosition], hoop: tuple[int, int]
    ) -> bool:
        """Check if the ball trajectory passes through/near the hoop center."""
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

        best_obs = self._select_hoop_observation(descent, hoop_observations)
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

    def _check_ball_in_hoop_zone(
        self,
        positions: list[BallPosition],
        hoop_observation: HoopObservation,
    ) -> bool:
        """Check if ball positions fall within a polygon zone around the hoop.

        Constructs a trapezoidal polygon from the hoop bounding box that
        extends downward to account for the net area.  Requires the
        ``supervision`` package; returns False if not available.
        """
        try:
            import supervision as sv
        except ImportError:
            logger.warning("use_polygon_zone requested but 'supervision' is not installed")
            return False

        hx1, hy1, hx2, hy2 = hoop_observation.bbox
        hoop_w = hx2 - hx1
        hoop_h = hy2 - hy1
        margin = hoop_w * self.config.hoop_x_tolerance_ratio
        net_depth = int(hoop_h * 1.5)

        polygon = np.array([
            [hx1 - margin, hy1 - self.config.hoop_entry_y_margin_px],
            [hx2 + margin, hy1 - self.config.hoop_entry_y_margin_px],
            [hx2 + margin * 0.5, hy2 + net_depth],
            [hx1 - margin * 0.5, hy2 + net_depth],
        ], dtype=np.int32)

        zone = sv.PolygonZone(polygon=polygon)

        # Check descent positions (after peak)
        peak_idx = min(range(len(positions)), key=lambda k: positions[k].y)
        descent = positions[peak_idx:]
        if len(descent) < 2:
            return False

        xyxy = np.array([
            [p.x - 5, p.y - 5, p.x + 5, p.y + 5] for p in descent
        ], dtype=np.float32)
        ball_dets = sv.Detections(
            xyxy=xyxy,
            confidence=np.ones(len(descent), dtype=np.float32),
            class_id=np.zeros(len(descent), dtype=int),
        )

        mask = zone.trigger(ball_dets)
        return bool(np.any(mask))
