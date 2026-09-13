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

        # Feed each position as a ball detection
        for pos in arc:
            fd = FrameDetections(frame_idx=pos.frame_idx)
            fd.balls.append(Detection(
                class_name="sports ball",
                confidence=0.9,
                bbox=(pos.x - 10, pos.y - 10, pos.x + 10, pos.y + 10),
                frame_idx=pos.frame_idx,
            ))
            tracker.update(fd)

        # Set hoop observations
        hoop = _make_hoop_observation(frame_idx=6)
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


class TestMultiFrameConsensus:
    """Tests for the multi-frame consensus filter."""

    def test_consensus_blocks_single_spurious_detection(self):
        """A single ball detection should not start tracking when consensus=3."""
        config = TrackingConfig(consensus_required=3, consensus_window=5)
        tracker = BallTracker(config)

        fd = FrameDetections(frame_idx=0)
        fd.balls.append(Detection(
            class_name="sports ball", confidence=0.9,
            bbox=(200, 200, 220, 220), frame_idx=0,
        ))
        result = tracker.update(fd)
        assert result is None

        # Several frames with no ball
        for i in range(1, 5):
            fd_empty = FrameDetections(frame_idx=i)
            result = tracker.update(fd_empty)
            assert result is None

    def test_consensus_confirms_after_n_frames(self):
        """Three consistent detections in 5 frames should confirm tracking."""
        config = TrackingConfig(consensus_required=3, consensus_window=5)
        tracker = BallTracker(config)

        result = None
        for i in range(3):
            fd = FrameDetections(frame_idx=i)
            fd.balls.append(Detection(
                class_name="sports ball", confidence=0.8,
                bbox=(100 + i, 100 + i, 120 + i, 120 + i), frame_idx=i,
            ))
            result = tracker.update(fd)

        assert result is not None
        assert result.predicted is False

    def test_consensus_rejects_spatially_spread_detections(self):
        """Detections spread beyond max_spread_px should not confirm."""
        config = TrackingConfig(
            consensus_required=3, consensus_window=5, consensus_max_spread_px=50,
        )
        tracker = BallTracker(config)

        positions = [(100, 100), (300, 100), (200, 300)]
        for i, (x, y) in enumerate(positions):
            fd = FrameDetections(frame_idx=i)
            fd.balls.append(Detection(
                class_name="sports ball", confidence=0.8,
                bbox=(x - 10, y - 10, x + 10, y + 10), frame_idx=i,
            ))
            result = tracker.update(fd)

        assert result is None

    def test_consensus_rejects_detections_spread_across_too_many_frames(self):
        """Spatially-consistent but temporally sparse detections must not confirm.

        Three detections at the same location but at frames 0, 100, 200 fall
        outside the consensus_window, so they should never confirm a track even
        though their spatial spread is zero.
        """
        config = TrackingConfig(
            consensus_required=3, consensus_window=5, consensus_max_spread_px=50,
            max_ball_gap_frames=1000,
        )
        tracker = BallTracker(config)

        result = None
        for frame in (0, 100, 200):
            fd = FrameDetections(frame_idx=frame)
            fd.balls.append(Detection(
                class_name="sports ball", confidence=0.9,
                bbox=(100, 100, 120, 120), frame_idx=frame,
            ))
            result = tracker.update(fd)

        assert result is None

    def test_default_consensus_is_three(self):
        """Consensus is on by default — a single spurious detection can't seed a track."""
        assert TrackingConfig().consensus_required == 3

    def test_consensus_disabled_with_required_one(self):
        """With consensus_required=1, the first detection is accepted."""
        config = TrackingConfig(consensus_required=1)
        tracker = BallTracker(config)

        fd = FrameDetections(frame_idx=0)
        fd.balls.append(Detection(
            class_name="sports ball", confidence=0.9,
            bbox=(100, 100, 120, 120), frame_idx=0,
        ))
        result = tracker.update(fd)
        assert result is not None

    def test_consensus_resets_after_long_gap(self):
        """After ball is lost for max_ball_gap_frames, consensus must re-establish."""
        config = TrackingConfig(
            consensus_required=3, consensus_window=5, max_ball_gap_frames=5,
        )
        tracker = BallTracker(config)

        # Establish tracking
        for i in range(3):
            fd = FrameDetections(frame_idx=i)
            fd.balls.append(Detection(
                class_name="sports ball", confidence=0.9,
                bbox=(100, 100, 120, 120), frame_idx=i,
            ))
            tracker.update(fd)

        # Gap of 20 frames with no ball — exceeds max_ball_gap_frames
        for i in range(3, 23):
            fd = FrameDetections(frame_idx=i)
            tracker.update(fd)

        # New detection should NOT be immediately accepted
        fd = FrameDetections(frame_idx=23)
        fd.balls.append(Detection(
            class_name="sports ball", confidence=0.9,
            bbox=(500, 500, 520, 520), frame_idx=23,
        ))
        result = tracker.update(fd)
        assert result is None


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


class TestKalmanReacquire:
    def test_covariance_reinflated_on_reseed(self):
        tracker = BallTracker(TrackingConfig(consensus_required=1))
        import numpy as np

        # Converge the filter a bit
        for i in range(20):
            fd = FrameDetections(frame_idx=i)
            fd.balls.append(Detection(
                class_name="ball", confidence=0.9, bbox=(100 + i, 100, 120 + i, 120), frame_idx=i,
            ))
            tracker.update(fd)
        assert tracker._kf.P[0, 0] < 10.0

        tracker._seed_filter(500, 500)
        assert np.allclose(tracker._kf.P, np.eye(4) * 10.0)


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
