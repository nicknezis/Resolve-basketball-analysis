"""Preview windows for visualizing analysis results via OpenCV."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.analysis.ball_tracker import BallPosition, ShotEvent
from src.analysis.color import FrameLUT
from src.analysis.event_classifier import GameEvent
from src.analysis.object_detector import FrameDetections
from src.analysis.player_tracker import FrameTracking

logger = logging.getLogger(__name__)

# BGR color constants
COLOR_BALL = (0, 140, 255)       # orange
COLOR_HOOP = (0, 0, 255)         # red
COLOR_PLAYER = (160, 160, 160)   # gray (unclassified)
COLOR_TEAM_A = (255, 140, 0)     # blue
COLOR_TEAM_B = (0, 200, 0)       # green
COLOR_SHOT_MADE = (0, 200, 0)    # green
COLOR_SHOT_ATTEMPT = (0, 0, 255) # red
COLOR_HUD_BG = (30, 30, 30)      # dark background for text

# BGR colors for event types on the timeline bar
EVENT_TIMELINE_COLORS: dict[str, tuple[int, int, int]] = {
    "made_shot": (0, 200, 0),        # green
    "shot_attempt": (0, 140, 255),   # orange
    "crowd_excitement": (255, 200, 0),  # cyan
    "three_pointer": (0, 220, 220),  # yellow
    "dunk": (0, 0, 255),             # red
    "fast_break": (200, 150, 0),     # teal
    "block": (200, 0, 200),          # magenta
    "steal": (255, 100, 0),          # blue
    "buzzer_beater": (180, 0, 255),  # pink
}
EVENT_TIMELINE_DEFAULT_COLOR = (200, 200, 200)  # fallback gray

MAX_DISPLAY_WIDTH = 1280


def _scale_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Scale frame down if wider than MAX_DISPLAY_WIDTH. Returns (frame, scale)."""
    h, w = frame.shape[:2]
    if w > MAX_DISPLAY_WIDTH:
        scale = MAX_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, (MAX_DISPLAY_WIDTH, int(h * scale)))
        return frame, scale
    return frame, 1.0


def _draw_bbox(
    canvas: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: tuple[int, int, int],
    label: str | None = None,
    scale: float = 1.0,
) -> None:
    """Draw a labeled bounding box on canvas."""
    x1, y1, x2, y2 = (int(v * scale) for v in bbox)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
    if label:
        _put_label(canvas, label, x1, y1 - 6, color)


def _put_label(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
) -> None:
    """Draw text with a dark background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    y = max(th + 4, y)
    cv2.rectangle(canvas, (x, y - th - 4), (x + tw + 4, y + baseline + 2), COLOR_HUD_BG, -1)
    cv2.putText(canvas, text, (x + 2, y), font, font_scale, color, thickness, cv2.LINE_AA)


class FramePreview:
    """Live per-frame preview during analysis.

    Shows raw detections: ball (orange), hoop (red), players (gray),
    ball trail, and a frame counter HUD.
    """

    def __init__(self) -> None:
        self._trail: deque[tuple[int, int]] = deque(maxlen=30)
        self._window_open = False
        self._window_name = "Basketball Analysis - Live Preview"

    def _ensure_window(self) -> bool:
        """Try to open a display window. Returns False if headless."""
        if self._window_open:
            return True
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
            self._window_open = True
            return True
        except cv2.error:
            logger.warning("Cannot open preview window (headless environment?)")
            return False

    def render(
        self,
        frame: np.ndarray,
        detections: FrameDetections,
        ball_position: BallPosition | None,
        tracking: FrameTracking | None,
        frame_idx: int,
        total_frames: int,
    ) -> bool:
        """Render one frame with detection overlays.

        Returns False if the user pressed 'q' (caller should stop rendering).
        """
        if not self._ensure_window():
            return False

        canvas, scale = _scale_frame(frame.copy())

        # Draw player bounding boxes (gray — team colors not available yet)
        for det in detections.players:
            _draw_bbox(canvas, det.bbox, COLOR_PLAYER, scale=scale)

        # Draw hoop bounding boxes
        for det in detections.hoops:
            label = f"hoop {det.confidence:.0%}"
            _draw_bbox(canvas, det.bbox, COLOR_HOOP, label=label, scale=scale)

        # Draw ball bounding boxes
        for det in detections.balls:
            label = f"ball {det.confidence:.0%}"
            _draw_bbox(canvas, det.bbox, COLOR_BALL, label=label, scale=scale)

        # Draw Kalman-filtered ball position and trail
        if ball_position is not None:
            self._trail.append((ball_position.x, ball_position.y))
            bx, by = int(ball_position.x * scale), int(ball_position.y * scale)
            cv2.circle(canvas, (bx, by), 5, COLOR_BALL, -1)

        # Fading trail
        trail_list = list(self._trail)
        for i in range(1, len(trail_list)):
            alpha = i / len(trail_list)
            color = tuple(int(c * alpha) for c in COLOR_BALL)
            p1 = (int(trail_list[i - 1][0] * scale), int(trail_list[i - 1][1] * scale))
            p2 = (int(trail_list[i][0] * scale), int(trail_list[i][1] * scale))
            cv2.line(canvas, p1, p2, color, 2)

        # HUD
        pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
        hud = f"Frame {frame_idx}/{total_frames} ({pct:.1f}%)  |  press q to close"
        _put_label(canvas, hud, 8, canvas.shape[0] - 12, (200, 200, 200))

        cv2.imshow(self._window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            self.close()
            return False
        return True

    def close(self) -> None:
        """Clean up the preview window."""
        if self._window_open:
            cv2.destroyWindow(self._window_name)
            self._window_open = False
            # Pump the event loop so macOS actually processes the destroy
            for _ in range(4):
                cv2.waitKey(1)


class ClipReview:
    """Post-analysis replay of a clip with full result overlays.

    Shows team-colored player boxes, shot arc trajectories,
    event labels, and playback controls.
    """

    def __init__(self) -> None:
        self._window_name = "Basketball Analysis - Clip Review"

    def _prepare(
        self,
        video_path: str,
        start_frame: int,
        all_detections: list[FrameDetections],
        player_tracks: dict,
        ball_positions: list[BallPosition],
        input_lut: Path | None,
    ) -> tuple:
        """Build shared lookups used by both replay() and export()."""
        lut = FrameLUT(input_lut)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.error("Cannot open video: %s", video_path)
            return None
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        det_lookup = {fd.frame_idx: fd for fd in all_detections}
        ball_lookup = {bp.frame_idx: bp for bp in ball_positions}
        hoop_pos = self._compute_median_hoop(all_detections)
        return lut, cap, det_lookup, ball_lookup, hoop_pos

    def _render_overlays(
        self,
        frame: np.ndarray,
        frame_idx: int,
        start_frame: int,
        end_frame: int,
        det_lookup: dict[int, FrameDetections],
        shot_events: list[ShotEvent],
        game_events: list[GameEvent],
        ball_positions: list[BallPosition],
        ball_lookup: dict[int, BallPosition],
        hoop_pos: tuple[int, int] | None,
    ) -> tuple[np.ndarray, float]:
        """Render all analysis overlays onto a frame. Returns (canvas, scale)."""
        canvas, scale = _scale_frame(frame.copy())
        fd = det_lookup.get(frame_idx)

        # Draw median hoop position
        if hoop_pos:
            hx, hy = int(hoop_pos[0] * scale), int(hoop_pos[1] * scale)
            cv2.rectangle(canvas, (hx - 30, hy - 20), (hx + 30, hy + 20), COLOR_HOOP, 2)
            _put_label(canvas, "hoop", hx - 30, hy - 26, COLOR_HOOP)

        # Draw player bounding boxes
        if fd:
            for det in fd.players:
                _draw_bbox(canvas, det.bbox, COLOR_PLAYER, scale=scale)

        # Draw shot arc trajectories
        for shot in shot_events:
            if shot.start_frame <= frame_idx <= shot.end_frame + 30:
                color = COLOR_SHOT_MADE if shot.made else COLOR_SHOT_ATTEMPT
                pts = [
                    (int(bp.x * scale), int(bp.y * scale))
                    for bp in shot.ball_positions
                    if bp.frame_idx <= frame_idx
                ]
                if len(pts) > 1:
                    cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, color, 2)

        # Draw ball trail (last 30 positions leading up to current frame)
        recent_ball = [
            bp for bp in ball_positions
            if bp.frame_idx <= frame_idx and bp.frame_idx > frame_idx - 30
        ]
        for i in range(1, len(recent_ball)):
            alpha = i / len(recent_ball)
            color = tuple(int(c * alpha) for c in COLOR_BALL)
            p1 = (int(recent_ball[i - 1].x * scale), int(recent_ball[i - 1].y * scale))
            p2 = (int(recent_ball[i].x * scale), int(recent_ball[i].y * scale))
            cv2.line(canvas, p1, p2, color, 2)

        # Draw current ball position
        bp = ball_lookup.get(frame_idx)
        if bp:
            cv2.circle(canvas, (int(bp.x * scale), int(bp.y * scale)), 5, COLOR_BALL, -1)

        # Draw event labels during their frame ranges
        y_offset = 30
        for ev in game_events:
            if ev.start_frame <= frame_idx <= ev.end_frame:
                label = f"{ev.event_type.replace('_', ' ').title()} {ev.confidence:.0%}"
                _put_label(canvas, label, 8, y_offset, (255, 255, 255))
                y_offset += 24

        # Event timeline bar at top
        self._draw_timeline_bar(canvas, frame_idx, start_frame, end_frame, game_events, scale)

        return canvas, scale

    def export(
        self,
        export_path: Path,
        video_path: str,
        start_frame: int,
        end_frame: int,
        all_detections: list[FrameDetections],
        shot_events: list[ShotEvent],
        game_events: list[GameEvent],
        player_tracks: dict,
        ball_positions: list[BallPosition],
        fps: float,
        input_lut: Path | None = None,
    ) -> None:
        """Export the review replay as a video file (no display window)."""
        result = self._prepare(
            video_path, start_frame, all_detections, player_tracks,
            ball_positions, input_lut,
        )
        if result is None:
            return
        lut, cap, det_lookup, ball_lookup, hoop_pos = result

        writer = None
        total = end_frame - start_frame
        frame_idx = start_frame
        logger.info("Exporting review video to %s (%d frames)", export_path, total)

        while frame_idx < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            frame = lut.apply(frame)

            canvas, scale = self._render_overlays(
                frame, frame_idx, start_frame, end_frame,
                det_lookup, shot_events, game_events,
                ball_positions, ball_lookup, hoop_pos,
            )

            # HUD (no pause/controls in export)
            pct = (frame_idx - start_frame) / total * 100 if total > 0 else 0
            hud = f"Frame {frame_idx} ({pct:.1f}%)"
            _put_label(canvas, hud, 8, canvas.shape[0] - 12, (200, 200, 200))

            if writer is None:
                h, w = canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(str(export_path), fourcc, fps, (w, h))
            writer.write(canvas)

            frame_idx += 1

        if writer is not None:
            writer.release()
        cap.release()
        logger.info("Review video saved: %s", export_path)

    def replay(
        self,
        video_path: str,
        start_frame: int,
        end_frame: int,
        all_detections: list[FrameDetections],
        shot_events: list[ShotEvent],
        game_events: list[GameEvent],
        player_tracks: dict,
        ball_positions: list[BallPosition],
        fps: float,
        input_lut: Path | None = None,
    ) -> None:
        """Replay a clip's video with full analysis overlays (interactive window)."""
        try:
            cv2.namedWindow(self._window_name, cv2.WINDOW_NORMAL)
        except cv2.error:
            logger.warning("Cannot open review window (headless environment?)")
            return

        result = self._prepare(
            video_path, start_frame, all_detections, player_tracks,
            ball_positions, input_lut,
        )
        if result is None:
            return
        lut, cap, det_lookup, ball_lookup, hoop_pos = result

        delay = max(1, int(1000 / fps)) if fps > 0 else 33
        paused = False
        frame_idx = start_frame

        while frame_idx < end_frame:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = lut.apply(frame)

            canvas, scale = self._render_overlays(
                frame, frame_idx, start_frame, end_frame,
                det_lookup, shot_events, game_events,
                ball_positions, ball_lookup, hoop_pos,
            )

            # HUD
            total = end_frame - start_frame
            progress = frame_idx - start_frame
            pct = progress / total * 100 if total > 0 else 0
            status = "PAUSED" if paused else "PLAYING"
            hud = f"{status}  Frame {frame_idx} ({pct:.1f}%)  |  q=quit  space=pause  arrows=step"
            _put_label(canvas, hud, 8, canvas.shape[0] - 12, (200, 200, 200))

            cv2.imshow(self._window_name, canvas)
            key = cv2.waitKey(0 if paused else delay) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                paused = not paused
            elif key == 81 or key == 2:  # left arrow
                frame_idx = max(start_frame, frame_idx - 1)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frame = lut.apply(frame)
                paused = True
                continue
            elif key == 83 or key == 3:  # right arrow
                if paused:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = lut.apply(frame)
                    frame_idx += 1
                    continue

            if not paused:
                frame_idx += 1

        cap.release()
        cv2.destroyWindow(self._window_name)
        for _ in range(4):
            cv2.waitKey(1)

    @staticmethod
    def _compute_median_hoop(detections: list[FrameDetections]) -> tuple[int, int] | None:
        """Compute median hoop position across all detections."""
        hoop_positions = []
        for fd in detections:
            if fd.hoops:
                best = max(fd.hoops, key=lambda d: d.confidence)
                hoop_positions.append(best.center)
        if not hoop_positions:
            return None
        hx = int(np.median([p[0] for p in hoop_positions]))
        hy = int(np.median([p[1] for p in hoop_positions]))
        return (hx, hy)

    @staticmethod
    def _draw_timeline_bar(
        canvas: np.ndarray,
        current_frame: int,
        start_frame: int,
        end_frame: int,
        events: list[GameEvent],
        scale: float,
    ) -> None:
        """Draw a thin timeline bar at the top with event markers."""
        h, w = canvas.shape[:2]
        bar_y = 8
        bar_h = 6
        total = end_frame - start_frame
        if total <= 0:
            return

        # Background bar
        cv2.rectangle(canvas, (10, bar_y), (w - 10, bar_y + bar_h), (60, 60, 60), -1)

        # Event markers — one color per event type
        seen_types: dict[str, tuple[int, int, int]] = {}
        for ev in events:
            ex1 = 10 + int((ev.start_frame - start_frame) / total * (w - 20))
            ex2 = 10 + int((ev.end_frame - start_frame) / total * (w - 20))
            ex2 = max(ex2, ex1 + 2)
            color = EVENT_TIMELINE_COLORS.get(ev.event_type, EVENT_TIMELINE_DEFAULT_COLOR)
            cv2.rectangle(canvas, (ex1, bar_y), (ex2, bar_y + bar_h), color, -1)
            if ev.event_type not in seen_types:
                seen_types[ev.event_type] = color

        # Playhead
        px = 10 + int((current_frame - start_frame) / total * (w - 20))
        cv2.line(canvas, (px, bar_y - 2), (px, bar_y + bar_h + 2), (255, 255, 255), 2)

        # Legend — only show event types present in this clip
        if seen_types:
            from src.config import EVENT_LABELS

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.4
            legend_x = w - 10
            legend_y = bar_y + bar_h + 14
            for event_type, color in reversed(list(seen_types.items())):
                text = EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())
                (tw, th), _ = cv2.getTextSize(text, font, font_scale, 1)
                swatch_size = th
                legend_x -= tw + 2
                cv2.putText(canvas, text, (legend_x, legend_y), font, font_scale, color, 1, cv2.LINE_AA)
                legend_x -= swatch_size + 6
                cv2.rectangle(
                    canvas,
                    (legend_x, legend_y - swatch_size + 1),
                    (legend_x + swatch_size, legend_y + 1),
                    color, -1,
                )
                legend_x -= 10
