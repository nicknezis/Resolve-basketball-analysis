"""Unit tests for ball tracker shot detection with hoop bounding boxes."""

from __future__ import annotations

from src.analysis.ball_tracker import (
    BallPosition,
    BallTracker,
    HoopObservation,
    ShotEvent,
)
from src.analysis.object_detector import Detection, FrameDetections
from src.config import TrackingConfig


def _make_arc_positions(
    start_frame: int,
    start_x: int,
    start_y: int,
    peak_y: int,
    end_y: int,
    num_points: int = 10,
    end_x: int | None = None,
) -> list[BallPosition]:
    """Generate a synthetic arc trajectory (up then down).

    The ball moves from (start_x, start_y) up to (mid_x, peak_y) then
    down to (end_x, end_y) over ``num_points`` frames.
    """
    if end_x is None:
        end_x = start_x

    half = num_points // 2
    positions: list[BallPosition] = []
    for i in range(half):
        t = i / max(half - 1, 1)
        x = int(start_x + (end_x - start_x) * t / 2)
        y = int(start_y + (peak_y - start_y) * t)
        positions.append(BallPosition(frame_idx=start_frame + i, x=x, y=y, predicted=False))

    for i in range(half):
        t = i / max(half - 1, 1)
        x = int((start_x + end_x) // 2 + (end_x - (start_x + end_x) // 2) * t)
        y = int(peak_y + (end_y - peak_y) * t)
        positions.append(BallPosition(frame_idx=start_frame + half + i, x=x, y=y, predicted=False))

    return positions


def _make_hoop_observation(
    frame_idx: int = 50,
    x1: int = 400,
    y1: int = 200,
    x2: int = 500,
    y2: int = 230,
    confidence: float = 0.9,
) -> HoopObservation:
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return HoopObservation(
        frame_idx=frame_idx,
        bbox=(x1, y1, x2, y2),
        center=(cx, cy),
        confidence=confidence,
    )


class TestBboxShotDetection:
    """Tests for _check_ball_through_hoop_bbox."""

    def test_made_shot_through_hoop_bbox(self):
        """Ball arc descends through the hoop bounding box -> made shot."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        # Hoop at x=[400,500], y_top=200
        hoop = _make_hoop_observation(frame_idx=55)

        # Arc that goes up to y=100 then comes down through y=200 at x=450
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is True
        assert shots[0].hoop_bbox is not None

    def test_miss_horizontal(self):
        """Ball descends but is too far left of the hoop -> miss."""
        # Wide hoop-direction gate so the arc is *detected* and judged a miss
        config = TrackingConfig(shot_min_arc_height_px=150, shot_hoop_x_range_ratio=0.5)
        tracker = BallTracker(config)

        # Hoop at x=[400,500]
        hoop = _make_hoop_observation(frame_idx=55)

        # Arc at x=200 — far left of hoop
        positions = _make_arc_positions(
            start_frame=50, start_x=200, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is False

    def test_miss_no_descent_through_hoop(self):
        """Ball arc peaks above hoop but never descends through it (lands short)."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        # Hoop at y_top=200
        hoop = _make_hoop_observation(frame_idx=55, y1=200, y2=230)

        # Arc that peaks at y=100 but ends at y=250 far to the right
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12, end_x=700,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        # Ball drifts out of horizontal range during descent
        assert shots[0].made is False

    def test_tolerance_boundary_inside(self):
        """Ball at the edge of hoop_x_tolerance_ratio (just inside) -> made."""
        config = TrackingConfig(
            shot_min_arc_height_px=150,
            hoop_x_tolerance_ratio=0.3,
        )
        tracker = BallTracker(config)

        # Hoop at x=[400,500], width=100, tolerance=30px each side
        # Effective range: [370, 530]
        hoop = _make_hoop_observation(frame_idx=55)

        # Ball at x=520 — inside expanded range (530)
        positions = _make_arc_positions(
            start_frame=50, start_x=520, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is True

    def test_tolerance_boundary_outside(self):
        """Ball just outside hoop_x_tolerance_ratio -> miss."""
        config = TrackingConfig(
            shot_min_arc_height_px=150,
            hoop_x_tolerance_ratio=0.3,
        )
        tracker = BallTracker(config)

        # Hoop at x=[400,500], tolerance=30px, effective range [370,530]
        hoop = _make_hoop_observation(frame_idx=55)

        # Ball at x=550 — outside expanded range
        positions = _make_arc_positions(
            start_frame=50, start_x=550, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is False


class TestSetHoopPositionsBackwardCompat:
    """Ensure the old set_hoop_positions() API still works."""

    def test_set_hoop_positions_still_works(self):
        """set_hoop_positions() should set _median_hoop for fallback detection."""
        config = TrackingConfig(shot_min_arc_height_px=150, hoop_proximity_px=60)
        tracker = BallTracker(config)

        # Arc that passes near (450, 215) — within 60px proximity
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_positions([(445, 210), (455, 220)])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        # Center-proximity check should fire — ball passes near median hoop
        assert shots[0].made is True

    def test_no_hoops_made_is_unknown(self):
        """Without any hoop data the verdict is unknown (None), not a miss."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        positions = _make_arc_positions(
            start_frame=0, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is None
        assert shots[0].made_via is None


class TestEndToEnd:
    """End-to-end test: update() -> set_hoop_observations() -> find_shots()."""

    def test_update_then_find_shots(self):
        """Full pipeline: feed detections via update(), then find shots."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        # Simulate an arc: ball goes from y=350 up to y=100 then back to 350
        # at x=450, over 12 frames
        arc = _make_arc_positions(
            start_frame=0, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )

        # Feed each position as a ball detection at the real analysis cadence
        # (every 2nd source frame, as with frame_skip=2)
        for pos in arc:
            f = pos.frame_idx * 2
            fd = FrameDetections(frame_idx=f)
            fd.balls.append(Detection(
                class_name="sports ball",
                confidence=0.9,
                bbox=(pos.x - 10, pos.y - 10, pos.x + 10, pos.y + 10),
                frame_idx=f,
            ))
            tracker.update(fd)

        # Set hoop observations
        hoop = _make_hoop_observation(frame_idx=12)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is True
        assert shots[0].hoop_bbox == hoop.bbox

    def test_observations_preferred_over_center_proximity(self):
        """When both observations and median hoop exist, bbox check takes priority."""
        config = TrackingConfig(
            shot_min_arc_height_px=150, hoop_proximity_px=1000, shot_hoop_x_range_ratio=0.5,
        )
        tracker = BallTracker(config)

        # Arc near hoop horizontally (passes hoop-directed gate) but doesn't
        # descend through the bbox.  With hoop_proximity_px=1000 the fallback
        # proximity check would say "made", but the bbox check should win.
        positions = _make_arc_positions(
            start_frame=0, start_x=500, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions

        # Hoop bbox offset to the right — ball is close enough for Gate C but
        # outside the bbox horizontal tolerance
        hoop = _make_hoop_observation(frame_idx=6, x1=700, y1=200, x2=800, y2=230)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1
        # Should be False because bbox check runs (not the proximity fallback)
        assert shots[0].made is False


def _make_layup_positions(
    x: int = 450,
    start_y: int = 400,
    peak_y: int = 200,
    end_y: int = 280,
    ascent_frames: int = 15,
    descent_frames: int = 5,
    start_frame: int = 0,
) -> list[BallPosition]:
    """Generate a layup trajectory: long gradual ascent, short descent."""
    positions: list[BallPosition] = []
    for i in range(ascent_frames):
        t = i / max(ascent_frames - 1, 1)
        y = int(start_y + (peak_y - start_y) * t)
        positions.append(BallPosition(frame_idx=start_frame + i, x=x, y=y, predicted=False))
    for i in range(descent_frames):
        t = i / max(descent_frames - 1, 1)
        y = int(peak_y + (end_y - peak_y) * t)
        positions.append(BallPosition(
            frame_idx=start_frame + ascent_frames + i, x=x, y=y, predicted=False,
        ))
    return positions


class TestLayupDetection:
    """Tests for layup-like arcs with asymmetric ascent/descent."""

    def test_layup_detected(self):
        """Layup: long ascent (carried by player), short descent through hoop."""
        config = TrackingConfig(shot_min_arc_height_px=50)
        tracker = BallTracker(config)

        # Ball carried from y=400 up to y=150 (above hoop at y=200), then
        # short descent through hoop to y=280.
        # arc_height = 400 - 150 = 250, d_ratio ~= 65/250 = 0.26
        positions = _make_layup_positions(
            start_y=400, peak_y=150, end_y=280,
            ascent_frames=15, descent_frames=5,
        )
        tracker._positions = positions

        hoop = _make_hoop_observation(frame_idx=15, x1=400, y1=200, x2=500, y2=230)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1, "Layup arc should be detected as a shot"
        assert shots[0].made is True

    def test_layup_rejected_at_old_threshold(self):
        """Same layup arc is rejected with the old 0.4 threshold."""
        config = TrackingConfig(
            shot_min_arc_height_px=50,
            shot_min_descent_ratio=0.4,
        )
        tracker = BallTracker(config)

        positions = _make_layup_positions(
            start_y=400, peak_y=150, end_y=280,
            ascent_frames=15, descent_frames=5,
        )
        tracker._positions = positions

        hoop = _make_hoop_observation(frame_idx=15, x1=400, y1=200, x2=500, y2=230)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) == 0, "Old threshold 0.4 should reject this layup arc"


class TestBackboardShot:
    """Tests for backboard shots where trajectory reverses at the board."""

    @staticmethod
    def _make_backboard_positions(
        x: int = 450,
        start_frame: int = 0,
    ) -> list[BallPosition]:
        """Generate a backboard shot trajectory.

        Ball rises to y=100 (well above hoop at y=200), hits backboard and
        bounces down to y=160 (arc-end triggers at y > 100+50=150). The ball
        is still above the hoop margin (170) at arc_end, so the through-hoop
        transition (y crossing from <=170 to >=200) only appears in the
        post-arc extension window.
        """
        positions: list[BallPosition] = []
        frame = start_frame

        # Ascent toward backboard: y=400 → 100
        for y in range(400, 99, -25):
            positions.append(BallPosition(frame_idx=frame, x=x, y=y, predicted=False))
            frame += 1

        # Backboard bounce: ball descends y=120 → 160
        # Arc-end triggers at y=160 (> 100 + 50 = 150)
        for y in [120, 140, 160]:
            positions.append(BallPosition(frame_idx=frame, x=x, y=y, predicted=False))
            frame += 1

        # Post-arc: ball continues through hoop (y=180 → 300)
        # Through-hoop transition: above (y<=170) → at/below (y>=200)
        for y in [180, 200, 220, 250, 280, 300]:
            positions.append(BallPosition(frame_idx=frame, x=x, y=y, predicted=False))
            frame += 1

        return positions

    def test_backboard_shot_detected_as_made(self):
        """Backboard bounce: arc ends at bounce, extended window catches hoop."""
        config = TrackingConfig(shot_min_arc_height_px=50)
        tracker = BallTracker(config)

        positions = self._make_backboard_positions()
        tracker._positions = positions

        # Hoop at y_top=200 (ball must cross from above 170 to at/below 200)
        hoop = _make_hoop_observation(frame_idx=15, x1=400, y1=200, x2=500, y2=230)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1, "Backboard shot should be detected"
        assert shots[0].made is True, "Extended window should catch through-hoop after bounce"

    def test_backboard_shot_missed_without_extension(self):
        """Without post-arc extension, backboard shot is detected but not made."""
        config = TrackingConfig(shot_min_arc_height_px=50, shot_post_arc_sec=0.0)
        tracker = BallTracker(config)

        positions = self._make_backboard_positions()
        tracker._positions = positions

        hoop = _make_hoop_observation(frame_idx=15, x1=400, y1=200, x2=500, y2=230)
        tracker.set_hoop_observations([hoop])

        shots = tracker.find_shots()
        assert len(shots) >= 1, "Arc should still be detected"
        assert shots[0].made is False, "Without extension, through-hoop is missed"


class TestHoopObservationDataclass:
    """Basic tests for the HoopObservation dataclass."""

    def test_fields(self):
        obs = HoopObservation(
            frame_idx=10,
            bbox=(100, 200, 300, 250),
            center=(200, 225),
            confidence=0.85,
        )
        assert obs.frame_idx == 10
        assert obs.bbox == (100, 200, 300, 250)
        assert obs.center == (200, 225)
        assert obs.confidence == 0.85


class TestSparseTrajectories:
    """Arc finding must reason in source frames, not list indices.

    ``_positions`` has holes wherever the ball was lost, so two unrelated
    arcs can sit next to each other in the list while being many seconds
    apart in the video.
    """

    def test_two_arcs_ten_seconds_apart_are_two_short_shots(self):
        config = TrackingConfig(shot_min_arc_height_px=150, max_ball_gap_frames=10)
        tracker = BallTracker(config, fps=60.0)

        arc1 = _make_arc_positions(
            start_frame=0, start_x=450, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        # 600 frames (10 s) later: a second, unrelated arc
        arc2 = _make_arc_positions(
            start_frame=600, start_x=450, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        tracker._positions = arc1 + arc2

        shots = tracker.find_shots()
        assert len(shots) == 2
        for shot in shots:
            assert shot.end_frame - shot.start_frame <= 12

    def test_gap_inside_arc_breaks_it(self):
        """Ascent, then the ball is lost for 5 s, then a descent: not one arc."""
        config = TrackingConfig(shot_min_arc_height_px=100, max_ball_gap_frames=10)
        tracker = BallTracker(config, fps=60.0)

        ascent = [BallPosition(frame_idx=i, x=450, y=400 - i * 25, predicted=False) for i in range(12)]
        descent = [BallPosition(frame_idx=300 + i, x=450, y=100 + i * 25, predicted=False) for i in range(12)]
        tracker._positions = ascent + descent

        assert tracker.find_shots() == []

    def test_max_arc_duration_uses_frames_not_indices(self):
        """Sparse positions spanning 6 s exceed shot_max_arc_sec even with few samples."""
        config = TrackingConfig(shot_min_arc_height_px=100, max_ball_gap_frames=60, shot_max_arc_sec=3.0)
        tracker = BallTracker(config, fps=60.0)

        # 10 samples, 40 frames apart = 360 frames = 6 s at 60 fps
        ys = [400, 330, 260, 190, 120, 100, 160, 220, 280, 340]
        tracker._positions = [
            BallPosition(frame_idx=i * 40, x=450, y=y, predicted=False) for i, y in enumerate(ys)
        ]

        assert tracker.find_shots() == []

    def test_predicted_positions_cannot_terminate_arc(self):
        """Kalman extrapolation alone must not manufacture a descent."""
        config = TrackingConfig(shot_min_arc_height_px=100)
        tracker = BallTracker(config)

        ascent = [BallPosition(frame_idx=i, x=450, y=400 - i * 30, predicted=False) for i in range(11)]
        fake_descent = [
            BallPosition(frame_idx=11 + i, x=450, y=100 + i * 30, predicted=True) for i in range(6)
        ]
        tracker._positions = ascent + fake_descent

        assert tracker.find_shots() == []


class TestHoopSelection:
    """Which rim judges a shot, and when no rim should."""

    def test_stale_hoop_observation_gives_unknown(self):
        """A rim seen 20 s earlier says nothing about a panned camera now."""
        config = TrackingConfig(shot_min_arc_height_px=150, hoop_obs_max_age_frames=30)
        tracker = BallTracker(config, fps=60.0)

        positions = _make_arc_positions(
            start_frame=1200, start_x=450, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=0)])

        shots = tracker.find_shots()
        assert len(shots) == 1
        assert shots[0].made is None

    def test_nearest_rim_horizontally_is_used(self):
        """With both baskets in frame, the descent is judged against the near one."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        # Ball descends through x=450 — the left rim.  Right rim at x=[1000,1100].
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        tracker._positions = positions
        left = _make_hoop_observation(frame_idx=55, x1=400, x2=500)
        right = _make_hoop_observation(frame_idx=56, x1=1000, x2=1100)  # closer in time
        tracker.set_hoop_observations([right, left])

        shots = tracker.find_shots()
        assert len(shots) == 1
        assert shots[0].made is True
        assert shots[0].hoop_bbox == left.bbox

    def test_gate_c_uses_real_frame_width(self):
        """A descent on the far side of a 640-px frame from the rim is rejected."""
        config = TrackingConfig(shot_min_arc_height_px=150, shot_hoop_x_range_ratio=0.25)
        tracker = BallTracker(config, frame_size=(640, 360))

        # Rim at x≈450, ball arc at x=100: 350 px apart > 0.25 * 640 = 160
        positions = _make_arc_positions(
            start_frame=50, start_x=100, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=55)])

        assert tracker.find_shots() == []

    def test_ball_in_basket_detection_marks_made(self):
        """A detector-reported ball-in-basket during the descent is a make."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        # Arc misses the bbox horizontally (x=550 vs rim 400-500 ±30)
        positions = _make_arc_positions(
            start_frame=50, start_x=550, start_y=350, peak_y=100, end_y=350, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=55)])
        tracker.set_ball_in_basket_frames([59])

        shots = tracker.find_shots()
        assert len(shots) == 1
        assert shots[0].made is True
        assert shots[0].made_via == "ball_in_basket"


class TestPeakAboveRimGate:
    """Gate D: with a rim in view, an arc that never rises above it is not a shot."""

    def test_arc_below_rim_is_rejected(self):
        config = TrackingConfig(shot_min_arc_height_px=100)
        tracker = BallTracker(config)
        # Rim top at y=200; this "arc" peaks at y=300 (a chest pass / dribble)
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=500, peak_y=300, end_y=500, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=55)])
        assert tracker.find_shots() == []

    def test_arc_within_margin_is_kept(self):
        config = TrackingConfig(shot_min_arc_height_px=100, shot_peak_rim_margin_px=20)
        tracker = BallTracker(config)
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=400, peak_y=215, end_y=400, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=55)])
        assert len(tracker.find_shots()) == 1

    def test_gate_skipped_without_rim(self):
        config = TrackingConfig(shot_min_arc_height_px=100)
        tracker = BallTracker(config)
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=500, peak_y=300, end_y=500, num_points=12,
        )
        tracker._positions = positions
        shots = tracker.find_shots()
        assert len(shots) == 1 and shots[0].made is None

    def test_gate_can_be_disabled(self):
        config = TrackingConfig(shot_min_arc_height_px=100, shot_require_peak_above_rim=False)
        tracker = BallTracker(config)
        positions = _make_arc_positions(
            start_frame=50, start_x=450, start_y=500, peak_y=300, end_y=500, num_points=12,
        )
        tracker._positions = positions
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=55)])
        assert len(tracker.find_shots()) == 1


def _ball_fd(frame_idx: int, *centers: tuple[int, int], conf: float = 0.8, size: int = 20,
             players: list[tuple[int, int]] | None = None) -> FrameDetections:
    """FrameDetections with ball boxes centred at each (x, y), plus optional players."""
    fd = FrameDetections(frame_idx=frame_idx)
    h = size // 2
    for (x, y) in centers:
        fd.balls.append(Detection("ball", conf, (x - h, y - h, x + h, y + h), frame_idx))
    for (x, y) in players or []:
        fd.players.append(Detection("player", 0.9, (x - 30, y - 80, x + 30, y + 80), frame_idx))
    return fd


class TestTrackletLinking:
    """Offline linking: candidates -> tracklets -> the ball's chain."""

    def _tracker(self, **cfg) -> BallTracker:
        return BallTracker(TrackingConfig(**cfg), fps=60.0, frame_size=(1280, 720))

    def test_single_stray_detection_forms_no_track(self):
        tr = self._tracker()
        tr.update(_ball_fd(0, (100, 100)))
        for f in range(2, 20, 2):
            tr.update(_ball_fd(f))
        tr.build_tracks()
        assert tr._positions == []

    def test_consecutive_detections_form_one_track(self):
        tr = self._tracker()
        for i in range(10):
            tr.update(_ball_fd(i * 2, (100 + i * 12, 400 - i * 8)))
        chosen = tr.build_tracks()
        assert len(chosen) == 1
        assert len(tr._positions) == 10
        assert not any(p.predicted for p in tr._positions)

    def test_gap_is_bridged_and_interpolated(self):
        tr = self._tracker(max_ball_gap_frames=18)
        for i in range(0, 12):
            if i in (5, 6, 7):
                tr.update(_ball_fd(i * 2))  # ball missed for three analysed frames
            else:
                tr.update(_ball_fd(i * 2, (100 + i * 10, 300)))
        tr.build_tracks()
        frames = [p.frame_idx for p in tr._positions]
        assert frames == [i * 2 for i in range(12)]
        assert [p.predicted for p in tr._positions if p.frame_idx in (10, 12, 14)] == [True, True, True]

    def test_far_stray_detection_during_gap_does_not_hijack_track(self):
        """The old online tracker snapped onto any detection after a short gap."""
        tr = self._tracker()
        for i in range(0, 20):
            f = i * 2
            if 6 <= i <= 8:
                tr.update(_ball_fd(f, (1100, 150)))  # a light fixture, 1000 px away
            else:
                tr.update(_ball_fd(f, (100 + i * 10, 300)))
        tr.build_tracks()
        xs = [p.x for p in tr._positions]
        assert max(xs) < 400, "track must not jump to the far stray detections"
        assert len([p for p in tr._positions if p.predicted]) == 3

    def test_static_clutter_loses_to_moving_ball(self):
        """A motionless 'ball' seen in every frame is clutter; the moving one is the ball."""
        tr = self._tracker()
        for i in range(90):  # 3 s at 60 fps, step 2
            f = i * 2
            centers = [(900, 120)]  # exit sign, always detected
            if 20 <= i < 50:
                centers.append((200 + (i - 20) * 15, 400 - (i - 20) * 6))
            tr.update(_ball_fd(f, *centers))
        chosen = tr.build_tracks()
        assert len(chosen) == 1
        assert all(p.x < 800 for p in tr._positions)
        assert len(tr._positions) == 30

    def test_camera_pan_does_not_make_fixture_look_like_motion(self):
        """During a pan everything shifts; anchored to the players, the sign is still static."""
        tr = self._tracker()
        for i in range(90):
            f = i * 2
            pan = i * 6  # camera pans 6 px per analysed frame
            players = [(300 + pan + k * 90, 450) for k in range(6)]
            centers = [(900 + pan, 120)]
            if 20 <= i < 50:
                centers.append((200 + pan + (i - 20) * 15, 400 - (i - 20) * 6))
            tr.update(_ball_fd(f, *centers, players=players))
        chosen = tr.build_tracks()
        assert len(chosen) == 1
        assert len(tr._positions) == 30

    def test_two_separate_possessions_give_two_segments(self):
        tr = self._tracker()
        for i in range(10):
            tr.update(_ball_fd(i * 2, (100 + i * 10, 300)))
        for i in range(10):
            tr.update(_ball_fd(600 + i * 2, (1000 - i * 10, 300)))
        chosen = tr.build_tracks()
        assert len(chosen) == 2
        assert len(tr._segments) == 2

    def test_update_returns_provisional_position_for_preview(self):
        tr = self._tracker()
        pos = tr.update(_ball_fd(0, (100, 100), (500, 500)))
        assert pos is not None and pos.frame_idx == 0
        assert tr._positions == []  # nothing committed until build_tracks()


class TestRimEntry:
    """A descending ball that vanishes inside the rim footprint is a shot, and
    a make unless it is re-detected above the rim shortly after."""

    def _arc_into_rim(self, start_frame: int = 100) -> list[BallPosition]:
        # Rim box (400,200)-(500,230).  Ascend from y=380 to a peak at y=170,
        # then three descending detections that end at y=205 inside the rim.
        ys_up = [380, 340, 300, 260, 220, 190, 170]
        ys_down = [180, 195, 205]
        pts = []
        f = start_frame
        for y in ys_up + ys_down:
            pts.append(BallPosition(frame_idx=f, x=450, y=y, predicted=False))
            f += 2
        return pts

    def test_vanishing_into_rim_is_a_made_shot(self):
        tracker = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        tracker._positions = self._arc_into_rim()
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=112)])
        shots = tracker.find_shots()
        assert len(shots) == 1
        assert shots[0].made is True
        assert shots[0].made_via == "rim_entry"

    def test_reappearing_above_rim_is_a_rim_out(self):
        tracker = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        pts = self._arc_into_rim()
        last = pts[-1].frame_idx
        # Lost in the rim for 30 frames (> max_ball_gap_frames → track break), then
        # re-detected 0.5 s after vanishing, bouncing up *above* the rim: a rim-out.
        bounce = [BallPosition(frame_idx=last + 30 + i * 2, x=510 + i * 5, y=150 - i * 10, predicted=False)
                  for i in range(5)]
        tracker._positions = pts + bounce
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=112)])
        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is False
        assert shots[0].made_via == "rim_out"

    def test_reappearing_below_rim_is_a_make(self):
        tracker = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        pts = self._arc_into_rim()
        last = pts[-1].frame_idx
        below = [BallPosition(frame_idx=last + 30 + i * 2, x=455, y=300 + i * 20, predicted=False)
                 for i in range(5)]
        tracker._positions = pts + below  # gap of 30 > max_ball_gap_frames → track break
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=112)])
        shots = tracker.find_shots()
        assert len(shots) == 1
        assert shots[0].made is True

    def test_vanishing_away_from_rim_is_not_a_shot(self):
        tracker = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        pts = [BallPosition(frame_idx=100 + i * 2, x=800, y=y, predicted=False)
               for i, y in enumerate([380, 340, 300, 260, 220, 190, 170, 180, 195, 205])]
        tracker._positions = pts
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=112)])
        assert tracker.find_shots() == []

    def test_disabled_flag(self):
        tracker = BallTracker(TrackingConfig(rim_entry_enabled=False), fps=60.0, frame_size=(1280, 720))
        tracker._positions = self._arc_into_rim()
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=112)])
        assert tracker.find_shots() == []


class TestCandidateFiltering:
    def test_border_candidates_are_dropped(self):
        tr = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        for i in range(10):
            tr.update(_ball_fd(i * 2, (480 + i, 5), (300 + i * 10, 400)))  # fixture at top edge + ball
        tr.build_tracks()
        assert all(p.y > 100 for p in tr._positions)

    def test_link_gate_is_capped(self):
        """A fixture missed for a long gap must not reach across the frame."""
        tr = BallTracker(TrackingConfig(max_ball_gap_frames=18, max_link_jump_px=160), fps=60.0, frame_size=(1280, 720))
        # fixture detections, then 16 missing frames, then the ball 430 px below
        for i in range(5):
            tr.update(_ball_fd(i * 2, (470, 60)))
        for f in range(10, 26, 2):
            tr.update(_ball_fd(f))
        for i in range(10):
            tr.update(_ball_fd(26 + i * 2, (470 + i * 8, 490)))
        chosen = tr.build_tracks()
        assert all(len(set(round(p.y / 100) for p in t.positions)) == 1 for t in chosen), \
            "fixture and ball must be separate tracklets"


class TestFreeFlightGate:
    """Gate E: a shot must leave every player box for a while; hand-offs never do."""

    def _feed(self, tracker: BallTracker, boxes_by_frame, arc):
        for pos in arc:
            fd = FrameDetections(frame_idx=pos.frame_idx)
            fd.balls.append(Detection("ball", 0.9, (pos.x - 10, pos.y - 10, pos.x + 10, pos.y + 10), pos.frame_idx))
            for (x1, y1, x2, y2) in boxes_by_frame(pos.frame_idx):
                fd.players.append(Detection("player", 0.9, (x1, y1, x2, y2), pos.frame_idx))
            tracker.update(fd)

    def _arc(self):
        # 20 analysed frames (step 2), rising from y=380 to 100 and back down at x=450
        pts = []
        ys = list(range(380, 90, -30)) + list(range(130, 400, 30))
        for i, y in enumerate(ys):
            pts.append(BallPosition(frame_idx=i * 2, x=450, y=y, predicted=False))
        return pts

    def test_hand_off_inside_player_boxes_is_rejected(self):
        tracker = BallTracker(TrackingConfig(shot_min_arc_height_px=100), fps=60.0, frame_size=(1280, 720))
        # Two players standing together whose boxes cover the whole ball path
        self._feed(tracker, lambda f: [(380, 60, 520, 420), (430, 80, 560, 420)], self._arc())
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=20, y1=90, y2=120)])
        assert tracker.find_shots() == []

    def test_free_flight_passes(self):
        tracker = BallTracker(TrackingConfig(shot_min_arc_height_px=100), fps=60.0, frame_size=(1280, 720))
        # Shooter's box covers only the low part of the arc (y > 300)
        self._feed(tracker, lambda f: [(400, 300, 500, 480)], self._arc())
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=20, y1=90, y2=120)])
        shots = tracker.find_shots()
        assert len(shots) == 1

    def test_gate_skipped_without_player_boxes(self):
        tracker = BallTracker(TrackingConfig(shot_min_arc_height_px=100), fps=60.0, frame_size=(1280, 720))
        self._feed(tracker, lambda f: [], self._arc())
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=20, y1=90, y2=120)])
        assert len(tracker.find_shots()) == 1

    def test_gate_can_be_disabled(self):
        tracker = BallTracker(TrackingConfig(shot_min_arc_height_px=100, require_free_flight=False),
                              fps=60.0, frame_size=(1280, 720))
        self._feed(tracker, lambda f: [(380, 60, 520, 420), (430, 80, 560, 420)], self._arc())
        tracker.set_hoop_observations([_make_hoop_observation(frame_idx=20, y1=90, y2=120)])
        assert len(tracker.find_shots()) == 1


class TestRimPhaseVerdict:
    """Made/miss from the sequence of rim phases and the post-rim speed."""

    RIM = (400, 200, 500, 230)  # 100 wide, 30 tall

    def _tracker(self, **cfg):
        tr = BallTracker(TrackingConfig(**cfg), fps=60.0, frame_size=(1280, 720))
        tr.set_hoop_observations([HoopObservation(frame_idx=30, bbox=self.RIM, center=(450, 215), confidence=0.9)])
        return tr

    @staticmethod
    def _projectile(x: int, y0: float, v0: float, g: float, n: int, t_step: int = 1, post_scale=None, start=0):
        """y(t) = y0 + v0 t + ½ g t² sampled every t_step frames (image coords, +y down)."""
        pts = []
        for i in range(n):
            t = i * t_step
            y = y0 + v0 * t + 0.5 * g * t * t
            pts.append(BallPosition(frame_idx=start + t, x=x, y=int(round(y)), predicted=False))
        return pts

    def test_swish_slowed_by_net_is_made(self):
        tr = self._tracker()
        # Flight: launched upward from y=380 at x=450, g=0.5 px/frame², peak y=124 above the rim
        pts = self._projectile(450, 380, -16.0, 0.5, 70)
        # Truncate once below the rim bottom, then append slow post-net motion
        flight = [p for p in pts if p.y <= 245]
        last = flight[-1]
        slow = [BallPosition(frame_idx=last.frame_idx + k, x=450, y=last.y + 4 * k, predicted=False) for k in range(1, 6)]
        tr._positions = flight + slow
        shots = tr.find_shots()
        assert len(shots) == 1
        assert shots[0].made is True and shots[0].made_via == "through_net"
        assert shots[0].speed_ratio is not None and shots[0].speed_ratio < 0.85

    def test_freefall_through_rim_footprint_is_a_miss(self):
        """Ball crosses the rim's image footprint at full speed: it was behind/in front of the rim."""
        tr = self._tracker()
        pts = self._projectile(450, 380, -16.0, 0.5, 70)
        tr._positions = pts
        shots = tr.find_shots()
        assert len(shots) == 1
        assert shots[0].made is False and shots[0].made_via == "passed_rim"
        assert shots[0].speed_ratio is not None and shots[0].speed_ratio > 0.85

    def test_ball_landing_beside_net_is_rim_out(self):
        tr = self._tracker()
        pts = self._projectile(450, 380, -16.0, 0.5, 70)
        flight = [p for p in pts if p.y <= 245]
        last = flight[-1]
        # Re-emerges well to the right of the net footprint (rim 400-500 ± 30)
        aside = [BallPosition(frame_idx=last.frame_idx + k, x=540 + 30 * k, y=last.y + 10 * k, predicted=False)
                 for k in range(1, 6)]
        tr._positions = flight + aside
        shots = tr.find_shots()
        assert len(shots) == 1
        assert shots[0].made is False and shots[0].made_via == "rim_out"

    def test_add_frames_merges_and_dedups(self):
        tr = BallTracker(TrackingConfig(), fps=60.0, frame_size=(1280, 720))
        tr.update(_ball_fd(0, (100, 100)))
        tr.update(_ball_fd(2, (110, 100)))
        extra = [_ball_fd(1, (105, 100)), _ball_fd(2, (112, 101))]  # frame 2 duplicate within 6 px
        tr.add_frames(extra)
        assert [f.frame_idx for f in tr._frames] == [0, 1, 2]
        assert len(tr._frames[2].balls) == 1
