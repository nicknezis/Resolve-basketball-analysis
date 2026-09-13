"""Offline ball tracking and shot detection.

The analyser works on finished clips, so ball tracking does not have to be
causal.  Instead of committing to one detection per frame as it arrives (and
snapping onto a head or a light the moment the ball is missed), the tracker:

1. **records** every ball candidate per analysed frame during the detection
   pass (:meth:`BallTracker.update`),
2. **links** candidates into *tracklets* by motion consistency once the clip
   is done — a candidate joins a tracklet if it lies where that tracklet's
   velocity (in camera-motion-compensated coordinates) says the ball should
   be, within a gate that widens with the number of missed frames,
3. **selects** the chain of non-overlapping tracklets that best explains one
   ball with a dynamic programme over time; long, motionless tracklets
   (fixtures, signs, heads) are discarded as clutter, and
4. **finds shot arcs per tracklet**, so a break in the track is a break in
   the trajectory rather than a straight line across the gym.

Linking a handful of boxes per frame costs microseconds; runtime is dominated
by the detector, not by anything here.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from src.analysis.object_detector import Detection, FrameDetections
from src.config import TrackingConfig

logger = logging.getLogger(__name__)


@dataclass
class BallPosition:
    """Tracked ball position at a frame."""

    frame_idx: int
    x: int
    y: int
    predicted: bool  # True if interpolated across a gap rather than detected


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


@dataclass
class _Candidate:
    """One ball detection, in raw image coordinates."""

    frame_idx: int
    x: int
    y: int
    size: float  # sqrt(area) — scale check when linking
    confidence: float


@dataclass
class _Frame:
    """Everything the linker needs from one analysed frame."""

    frame_idx: int
    balls: list[_Candidate]
    player_centers: list[tuple[int, int]]
    hoop_center: tuple[int, int] | None


@dataclass
class Tracklet:
    """A run of detections believed to be the same object."""

    dets: list[_Candidate]
    score: float = 0.0
    span_px: float = 0.0  # extent of the tracklet in camera-compensated coordinates
    motion_ratio: float = 0.0  # span_px / motion_full_span_px, capped at 1
    positions: list[BallPosition] = field(default_factory=list)  # filled after selection

    @property
    def start_frame(self) -> int:
        return self.dets[0].frame_idx

    @property
    def end_frame(self) -> int:
        return self.dets[-1].frame_idx

    @property
    def last(self) -> _Candidate:
        return self.dets[-1]


class BallTracker:
    """Records ball candidates per frame and builds the ball trajectory offline.

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
        self._frame_height: int | None = frame_size[1] if frame_size else None
        self._frames: list[_Frame] = []
        self._positions: list[BallPosition] = []
        self._segments: list[list[BallPosition]] = []
        self._tracklets: list[Tracklet] = []
        self._built = False
        self._hoop_observations: list[HoopObservation] = []
        self._median_hoop: tuple[int, int] | None = None
        self._ball_in_basket_frames: list[int] = []

    def reset(self) -> None:
        """Reset tracker state for a new video or scene."""
        self._frames = []
        self._positions = []
        self._segments = []
        self._tracklets = []
        self._built = False
        self._hoop_observations = []
        self._median_hoop = None
        self._ball_in_basket_frames = []

    def set_frame_size(self, width: int, height: int) -> None:
        """Record the analysed frame size (hoop-directed gate, border filtering)."""
        self._frame_width = int(width)
        self._frame_height = int(height)

    # ------------------------------------------------------------------
    # Recording (during the detection pass)
    # ------------------------------------------------------------------

    def update(self, frame_detections: FrameDetections) -> BallPosition | None:
        """Record this frame's ball candidates.

        Returns the highest-confidence candidate as a provisional position so
        the live preview has something to draw; the real trajectory is built
        by :meth:`build_tracks` once the clip is complete.
        """
        frame_idx = frame_detections.frame_idx
        if frame_detections.balls_in_basket:
            self._ball_in_basket_frames.append(frame_idx)

        cands = [
            _Candidate(
                frame_idx=frame_idx, x=d.center[0], y=d.center[1],
                size=float(max(1, d.area)) ** 0.5, confidence=d.confidence,
            )
            for d in frame_detections.balls
            if not self._touches_border(d.center)
        ]
        hoop = None
        if frame_detections.hoops:
            hoop = max(frame_detections.hoops, key=lambda d: d.confidence).center
        self._frames.append(_Frame(
            frame_idx=frame_idx, balls=cands,
            player_centers=[d.center for d in frame_detections.players],
            hoop_center=hoop,
        ))
        self._built = False

        if not cands:
            return None
        best = max(cands, key=lambda c: c.confidence)
        return BallPosition(frame_idx=frame_idx, x=best.x, y=best.y, predicted=False)

    def _touches_border(self, center: tuple[int, int]) -> bool:
        """Boxes centred on the frame edge are clipped objects or fixtures, not the ball."""
        if not self._frame_width or not self._frame_height:
            return False
        m = self.config.edge_margin_px
        x, y = center
        return x < m or y < m or x > self._frame_width - m or y > self._frame_height - m

    # ------------------------------------------------------------------
    # Hoop context
    # ------------------------------------------------------------------

    def set_hoop_positions(self, hoop_positions: list[tuple[int, int]]) -> None:
        """Store hoop center positions (legacy API without bounding boxes)."""
        if hoop_positions:
            hx = int(np.median([p[0] for p in hoop_positions]))
            hy = int(np.median([p[1] for p in hoop_positions]))
            self._median_hoop = (hx, hy)
        else:
            self._median_hoop = None

    def set_hoop_observations(self, observations: list[HoopObservation]) -> None:
        """Store per-frame hoop observations with bounding boxes."""
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
    # Offline track building
    # ------------------------------------------------------------------

    def build_tracks(self) -> list[Tracklet]:
        """Link recorded candidates into tracklets and choose the ball's chain.

        Populates ``_positions`` / ``_segments``.  Safe to call repeatedly.
        """
        if self._built:
            return self._tracklets
        if not self._frames:
            # Positions may have been supplied directly (tests / legacy)
            self._segments = [self._positions] if self._positions else []
            self._built = True
            return []

        cum_shift = self._camera_shifts()
        tracklets = self._link_tracklets(cum_shift)
        self._score_tracklets(tracklets, cum_shift)
        chosen = self._select_chain(tracklets, cum_shift)

        for t in chosen:
            t.positions = self._fill_positions(t)
        self._tracklets = chosen
        self._segments = [t.positions for t in chosen]
        self._positions = [p for seg in self._segments for p in seg]
        self._built = True

        logger.info(
            "Ball tracks: %d tracklets from %d candidates, %d chosen covering %d positions",
            len(tracklets), sum(len(f.balls) for f in self._frames),
            len(chosen), len(self._positions),
        )
        return chosen

    def _camera_shifts(self) -> dict[int, tuple[float, float]]:
        """Cumulative image shift per analysed frame due to camera motion.

        The rim is the ideal anchor (it doesn't move); when it isn't in both
        frames, the median displacement of matched player boxes is used — the
        players' own motion is small and incoherent next to a pan.
        """
        cum: dict[int, tuple[float, float]] = {}
        cx = cy = 0.0
        prev: _Frame | None = None
        for fr in self._frames:
            dx = dy = 0.0
            if prev is not None and self.config.compensate_camera_motion:
                shift = None
                if fr.hoop_center and prev.hoop_center:
                    hx, hy = fr.hoop_center[0] - prev.hoop_center[0], fr.hoop_center[1] - prev.hoop_center[1]
                    if abs(hx) + abs(hy) < 150:  # same rim, not a swap between baskets
                        shift = (hx, hy)
                if shift is None and len(fr.player_centers) >= 3 and len(prev.player_centers) >= 3:
                    dxs, dys = [], []
                    for (px, py) in fr.player_centers:
                        best = min(prev.player_centers, key=lambda q: abs(q[0] - px) + abs(q[1] - py))
                        if abs(best[0] - px) + abs(best[1] - py) < 120:
                            dxs.append(px - best[0])
                            dys.append(py - best[1])
                    if len(dxs) >= 3:
                        shift = (float(np.median(dxs)), float(np.median(dys)))
                if shift is not None:
                    dx, dy = shift
            cx += dx
            cy += dy
            cum[fr.frame_idx] = (cx, cy)
            prev = fr
        return cum

    def _link_tracklets(self, cum_shift: dict[int, tuple[float, float]]) -> list[Tracklet]:
        """Greedy nearest-prediction linking in camera-compensated coordinates."""
        cfg = self.config
        max_speed = cfg.max_ball_speed_px_per_frame
        slack = cfg.link_slack_px
        max_gap = cfg.max_ball_gap_frames
        size_ratio = cfg.tracklet_size_ratio_max

        def comp(c: _Candidate) -> tuple[float, float]:
            sx, sy = cum_shift.get(c.frame_idx, (0.0, 0.0))
            return c.x - sx, c.y - sy

        def velocity(t: Tracklet) -> tuple[float, float]:
            if len(t.dets) < 2:
                return 0.0, 0.0
            a, b = t.dets[-2], t.dets[-1]
            dt = max(1, b.frame_idx - a.frame_idx)
            ax, ay = comp(a)
            bx, by = comp(b)
            # Damp so a single noisy step doesn't fling the prediction
            return 0.8 * (bx - ax) / dt, 0.8 * (by - ay) / dt

        active: list[Tracklet] = []
        finished: list[Tracklet] = []

        for fr in self._frames:
            still: list[Tracklet] = []
            for t in active:
                (finished if fr.frame_idx - t.end_frame > max_gap else still).append(t)
            active = still

            preds = []
            for t in active:
                dt = fr.frame_idx - t.end_frame
                vx, vy = velocity(t)
                lx, ly = comp(t.last)
                gate = min(slack + max_speed * dt, float(cfg.max_link_jump_px))
                preds.append((lx + vx * dt, ly + vy * dt, gate))

            pairs = []
            for i, t in enumerate(active):
                px, py, gate = preds[i]
                for j, c in enumerate(fr.balls):
                    cx, cy = comp(c)
                    d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                    if d > gate:
                        continue
                    ratio = max(c.size, t.last.size) / max(1e-6, min(c.size, t.last.size))
                    if ratio > size_ratio:
                        continue
                    pairs.append((d, i, j))
            pairs.sort()

            used_t: set[int] = set()
            used_c: set[int] = set()
            for d, i, j in pairs:
                if i in used_t or j in used_c:
                    continue
                active[i].dets.append(fr.balls[j])
                used_t.add(i)
                used_c.add(j)
            for j, c in enumerate(fr.balls):
                if j not in used_c:
                    active.append(Tracklet(dets=[c]))

        finished.extend(active)
        return [t for t in finished if len(t.dets) >= cfg.min_tracklet_detections]

    def _score_tracklets(self, tracklets: list[Tracklet], cum_shift: dict) -> None:
        """Score by evidence (count × confidence), discounting things that don't move.

        Motion is the tracklet's *extent* in camera-compensated coordinates,
        not per-step displacement: detection jitter on a tiny fixture is a
        few px every step, but it never goes anywhere.
        """
        full = max(1, self.config.motion_full_span_px)
        for t in tracklets:
            xs, ys = [], []
            for d in t.dets:
                sx, sy = cum_shift.get(d.frame_idx, (0.0, 0.0))
                xs.append(d.x - sx)
                ys.append(d.y - sy)
            t.span_px = max(max(xs) - min(xs), max(ys) - min(ys))
            t.motion_ratio = min(1.0, t.span_px / full)
            mean_conf = float(np.mean([d.confidence for d in t.dets]))
            t.score = len(t.dets) * mean_conf * (0.25 + 0.75 * t.motion_ratio)

    def _select_chain(self, tracklets: list[Tracklet], cum_shift: dict) -> list[Tracklet]:
        """Dynamic programme: best-scoring chain of time-disjoint tracklets.

        Motionless tracklets that last a while are clutter, not a ball in
        play, and are excluded up front so they cannot out-score the real
        ball by sheer length.
        """
        cfg = self.config
        clutter_frames = cfg.static_clutter_sec * self.fps
        usable = [
            t for t in tracklets
            if not (t.span_px < cfg.static_span_px and t.end_frame - t.start_frame >= clutter_frames)
        ]
        usable.sort(key=lambda t: (t.start_frame, t.end_frame))
        n = len(usable)
        if n == 0:
            return []

        def comp(c: _Candidate) -> tuple[float, float]:
            sx, sy = cum_shift.get(c.frame_idx, (0.0, 0.0))
            return c.x - sx, c.y - sy

        best = [0.0] * n
        prev = [-1] * n
        for i, ti in enumerate(usable):
            best[i] = ti.score
            for j in range(i):
                tj = usable[j]
                if tj.end_frame >= ti.start_frame:
                    continue
                gap = ti.start_frame - tj.end_frame
                ax, ay = comp(tj.last)
                bx, by = comp(ti.dets[0])
                dist = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
                # Penalise teleports the ball could not have made in the gap
                excess = max(0.0, dist - cfg.max_ball_speed_px_per_frame * gap)
                cand = best[j] + ti.score - cfg.teleport_penalty_per_px * excess
                if cand > best[i]:
                    best[i] = cand
                    prev[i] = j

        i = int(np.argmax(best))
        chain = []
        while i >= 0:
            chain.append(usable[i])
            i = prev[i]
        chain.reverse()
        return chain

    def _fill_positions(self, t: Tracklet) -> list[BallPosition]:
        """Detections as positions, with gaps linearly interpolated (predicted=True)."""
        step = self._analysis_step()
        out: list[BallPosition] = []
        for a, b in zip(t.dets, t.dets[1:]):
            out.append(BallPosition(a.frame_idx, a.x, a.y, predicted=False))
            gap = b.frame_idx - a.frame_idx
            if gap > step:
                for f in range(a.frame_idx + step, b.frame_idx, step):
                    u = (f - a.frame_idx) / gap
                    out.append(BallPosition(
                        f, int(round(a.x + (b.x - a.x) * u)), int(round(a.y + (b.y - a.y) * u)),
                        predicted=True,
                    ))
        last = t.dets[-1]
        out.append(BallPosition(last.frame_idx, last.x, last.y, predicted=False))
        return out

    def _analysis_step(self) -> int:
        """Source frames between consecutive analysed frames (frame_skip)."""
        if len(self._frames) < 2:
            return 1
        return max(1, self._frames[1].frame_idx - self._frames[0].frame_idx)

    # ------------------------------------------------------------------
    # Shot detection
    # ------------------------------------------------------------------

    def find_shots(self) -> list[ShotEvent]:
        """Find shot events on the built ball trajectory."""
        self.build_tracks()
        shots: list[ShotEvent] = []
        for seg in self._segments:
            if len(seg) >= 5:
                shots.extend(self._find_arcs(seg, self._median_hoop))
        logger.info("Detected %d shot events", len(shots))
        return shots

    def detect_shots(
        self,
        all_detections: list[FrameDetections],
    ) -> list[ShotEvent]:
        """Analyze a full detection sequence to find shot attempts and makes."""
        self.reset()
        observations = []
        for fd in all_detections:
            self.update(fd)
            if fd.hoops:
                best_hoop = max(fd.hoops, key=lambda d: d.confidence)
                observations.append(HoopObservation(
                    frame_idx=fd.frame_idx, bbox=best_hoop.bbox,
                    center=best_hoop.center, confidence=best_hoop.confidence,
                ))
        self.set_hoop_observations(observations)
        return self.find_shots()

    def _find_arcs(
        self, positions: list[BallPosition], hoop_pos: tuple[int, int] | None,
    ) -> list[ShotEvent]:
        """Find up-then-down arcs in one contiguous trajectory segment.

        All duration windows are measured in *source frames* via each
        position's ``frame_idx``.  A gap longer than ``max_ball_gap_frames``
        between consecutive positions still terminates the current arc (a
        safety net for positions supplied directly rather than built here).
        """
        shots: list[ShotEvent] = []
        n = len(positions)
        max_gap = self.config.max_ball_gap_frames
        end_descent = self.config.shot_arc_end_descent_px

        i = 0
        while i < n - 4:
            arc_start = i
            peak_idx: int | None = None
            min_y = positions[i].y
            terminated = False

            j = i + 1
            while j < n:
                if positions[j].frame_idx - positions[j - 1].frame_idx > max_gap:
                    # Track break.  If the ball was on its way down when it
                    # vanished, it may have dropped into the net.
                    shot = self._try_rim_entry(positions, arc_start, peak_idx, j - 1, min_y, hoop_pos)
                    if shot is not None:
                        shots.append(shot)
                    i = j - 1
                    terminated = True
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
                    # Only a *detected* position may terminate an arc;
                    # interpolation must not manufacture a descent.
                    shot, next_i = self._evaluate_arc(positions, arc_start, peak_idx, j, min_y, hoop_pos)
                    if shot is not None:
                        shots.append(shot)
                    i = next_i
                    terminated = True
                    break
                j += 1

            if not terminated:
                # Segment ended without a full descent — same rim-entry check
                shot = self._try_rim_entry(positions, arc_start, peak_idx, n - 1, min_y, hoop_pos)
                if shot is not None:
                    shots.append(shot)
                    break
            i += 1

        return shots

    def _try_rim_entry(
        self,
        positions: list[BallPosition],
        arc_start: int,
        peak_idx: int | None,
        last_idx: int,
        min_y: int,
        hoop_pos: tuple[int, int] | None,
    ) -> ShotEvent | None:
        """A shot whose ball disappeared into the rim before a full descent.

        The detector loses the ball as it enters the net, so a made shot
        from just above the rim often shows only a few descending pixels
        before the track ends.  If the last *detected* position sits inside
        the rim's footprint after a peak, treat the arc as a shot ending
        there; :meth:`_evaluate_arc` decides made/miss with a look-ahead.
        """
        cfg = self.config
        if not cfg.rim_entry_enabled or peak_idx is None or last_idx <= peak_idx:
            return None
        last = positions[last_idx]
        if last.predicted or last.y <= min_y:
            return None
        sel_obs = self._select_hoop_observation(positions[peak_idx: last_idx + 1])
        if sel_obs is None:
            return None
        hx1, hy1, hx2, hy2 = sel_obs.bbox
        tol = (hx2 - hx1) * cfg.hoop_x_tolerance_ratio
        in_x = (hx1 - tol) <= last.x <= (hx2 + tol)
        in_y = (hy1 - cfg.hoop_entry_y_margin_px) <= last.y <= hy2 + (hy2 - hy1)
        if not (in_x and in_y):
            return None
        shot, _ = self._evaluate_arc(
            positions, arc_start, peak_idx, last_idx, min_y, hoop_pos, rim_entry=sel_obs,
        )
        return shot

    def _rim_entry_verdict(self, last: BallPosition, obs: HoopObservation) -> bool:
        """Made unless the ball is re-detected above the rim shortly after vanishing.

        A rim-out bounces back into view above the rim; a make falls through
        the net and is next seen (if at all) below it.
        """
        hy2 = obs.bbox[3]
        horizon = last.frame_idx + self.config.rim_entry_lookahead_sec * self.fps
        for p in self._positions:
            if p.frame_idx <= last.frame_idx or p.predicted:
                continue
            if p.frame_idx > horizon:
                break
            return p.y > hy2
        return True

    def _evaluate_arc(
        self,
        positions: list[BallPosition],
        arc_start: int,
        peak_idx: int,
        arc_end: int,
        min_y: int,
        hoop_pos: tuple[int, int] | None,
        rim_entry: HoopObservation | None = None,
    ) -> tuple[ShotEvent | None, int]:
        """Validate a candidate arc and build a ShotEvent.

        ``rim_entry`` is the rim the ball vanished into when the arc was cut
        short (see :meth:`_try_rim_entry`); the descent-ratio gate is skipped
        and made/miss comes from the look-ahead verdict.

        Returns ``(shot_or_None, index_to_resume_scanning_from)``.
        """
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

        # Gate B: minimum descent ratio (not for a ball that vanished into the rim)
        descent_height = positions[arc_end].y - min_y
        d_ratio = descent_height / arc_height if arc_height > 0 else 0.0
        if rim_entry is None and d_ratio < cfg.shot_min_descent_ratio:
            logger.debug(
                "Arc rejected (Gate B: descent_ratio): %.3f < %.3f (frames %d-%d)",
                d_ratio, cfg.shot_min_descent_ratio, start_frame, end_frame,
            )
            return None, arc_end

        # Which rim (if any) should judge this arc?
        descent_positions = positions[peak_idx: arc_end + 1]
        sel_obs = rim_entry if rim_entry is not None else self._select_hoop_observation(descent_positions)
        hoop_xy: tuple[int, int] | None = sel_obs.center if sel_obs is not None else hoop_pos

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

        # Gate D: a shot at this rim has to get above it.  Passes, dribbles
        # and hand-offs trace up-and-down arcs too, but stay below rim height.
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
        # that drop through after the descent trigger — never across a break.
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
        if rim_entry is not None:
            made = self._rim_entry_verdict(positions[arc_end], rim_entry)
            made_via = "rim_entry" if made else "rim_out"
            hoop_bbox = rim_entry.bbox
        elif sel_obs is not None:
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
        """
        if not positions or not hoop_observations:
            return False, None

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

        above = False
        for pos in descent:
            in_x = (hx1 - tolerance) <= pos.x <= (hx2 + tolerance)
            if not in_x:
                continue
            if pos.y <= hy1 - margin:
                above = True
            elif above and pos.y >= hy1:
                return True, best_obs.bbox

        return False, None

    def _check_ball_in_hoop_zone(
        self,
        positions: list[BallPosition],
        hoop_observation: HoopObservation,
    ) -> bool:
        """Check if descent positions fall within a trapezoidal net zone.

        Requires the ``supervision`` package; returns False if not available.
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
