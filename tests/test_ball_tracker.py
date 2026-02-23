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
        config = TrackingConfig(shot_min_arc_height_px=150)
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

    def test_no_hoops_no_made(self):
        """Without any hoop data, all shots should be attempts (not made)."""
        config = TrackingConfig(shot_min_arc_height_px=150)
        tracker = BallTracker(config)

        positions = _make_arc_positions(
            start_frame=0, start_x=450, start_y=350, peak_y=100, end_y=350,
            num_points=12,
        )
        tracker._positions = positions

        shots = tracker.find_shots()
        assert len(shots) >= 1
        assert shots[0].made is False


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
        config = TrackingConfig(shot_min_arc_height_px=150, hoop_proximity_px=1000)
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
