"""Second-pass ball detection in a crop around the rim during shot windows.

The main pass runs the detector on every Nth downscaled frame; near the rim
the ball is small, partly hidden by net and backboard, and moving fast, so
that is exactly where recall is worst — and where made/miss is decided.
This pass re-reads every *source* frame of each shot window, crops a region
around the rim, upscales it and runs the same detector, then hands the extra
ball/hoop boxes (mapped back to analysis coordinates) to the tracker.

Cost: a ~1 s window is ~60 small crops ≈ 1 s of detector time per shot.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.analysis.color import FrameLUT
from src.analysis.object_detector import Detection, FrameDetections, ObjectDetector
from src.config import TrackingConfig

logger = logging.getLogger(__name__)


@dataclass
class RimWindow:
    start_frame: int  # source frames, inclusive
    end_frame: int  # exclusive
    rim_bbox: tuple[int, int, int, int]  # analysis coordinates


class RimRoiRedetector:
    """Runs the detector on rim-centred crops for a set of frame windows."""

    def __init__(
        self,
        detector: ObjectDetector,
        lut: FrameLUT,
        max_resolution: int,
        config: TrackingConfig,
    ) -> None:
        self.detector = detector
        self.lut = lut
        self.max_resolution = max_resolution
        self.config = config

    def crop_box(self, rim: tuple[int, int, int, int], frame_w: int, frame_h: int) -> tuple[int, int, int, int]:
        """Region around the rim: wide enough for the approach, deep enough for the net."""
        cfg = self.config
        x1, y1, x2, y2 = rim
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        cx = (x1 + x2) / 2
        half_w = w * cfg.rim_roi_width_factor / 2
        top = y1 - h * cfg.rim_roi_above_factor
        bottom = y2 + h * cfg.rim_roi_below_factor
        rx1 = int(max(0, cx - half_w))
        rx2 = int(min(frame_w, cx + half_w))
        ry1 = int(max(0, top))
        ry2 = int(min(frame_h, bottom))
        return rx1, ry1, rx2, ry2

    def run(self, video_path: Path | str, windows: list[RimWindow]) -> list[FrameDetections]:
        """Return per-frame detections (analysis coordinates) for all windows."""
        if not windows:
            return []
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logger.warning("Rim ROI pass: cannot open %s", video_path)
            return []

        scale = self.config.rim_roi_scale
        results: dict[int, FrameDetections] = {}
        frames_seen = 0
        balls_found = 0
        try:
            for win in sorted(windows, key=lambda w: w.start_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, win.start_frame)
                for fi in range(win.start_frame, win.end_frame):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if fi in results:
                        continue  # overlapping windows
                    frame = self.lut.apply(frame)
                    h, w = frame.shape[:2]
                    if max(h, w) > self.max_resolution:
                        s = self.max_resolution / max(h, w)
                        frame = cv2.resize(frame, (int(w * s), int(h * s)))
                        h, w = frame.shape[:2]

                    rx1, ry1, rx2, ry2 = self.crop_box(win.rim_bbox, w, h)
                    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
                        continue
                    crop = frame[ry1:ry2, rx1:rx2]
                    if scale != 1.0:
                        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                    fd_crop = self.detector.detect_frame(crop, fi)
                    fd = FrameDetections(frame_idx=fi)
                    for src, dst in ((fd_crop.balls, fd.balls), (fd_crop.hoops, fd.hoops),
                                     (fd_crop.balls_in_basket, fd.balls_in_basket)):
                        for d in src:
                            bx1, by1, bx2, by2 = d.bbox
                            dst.append(Detection(
                                class_name=d.class_name, confidence=d.confidence,
                                bbox=(int(rx1 + bx1 / scale), int(ry1 + by1 / scale),
                                      int(rx1 + bx2 / scale), int(ry1 + by2 / scale)),
                                frame_idx=fi,
                            ))
                    results[fi] = fd
                    frames_seen += 1
                    balls_found += bool(fd.balls)
        finally:
            cap.release()

        logger.info(
            "  Rim ROI pass: %d windows, %d frames, ball seen in %d (%.0f%%)",
            len(windows), frames_seen, balls_found, 100.0 * balls_found / max(1, frames_seen),
        )
        return [results[k] for k in sorted(results)]
