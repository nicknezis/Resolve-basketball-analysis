"""YOLO-based object detection for basketball, hoop, and players."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.analysis.color import FrameLUT
from src.config import VideoConfig

logger = logging.getLogger(__name__)

# Class indices in COCO dataset used by stock YOLO models
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

# Custom-trained model class names (when using a basketball-specific model)
BASKETBALL_CLASSES = {"basketball": 0, "hoop": 1, "player": 2}


@dataclass
class Detection:
    """A single object detection in one frame."""

    class_name: str  # "person", "sports ball", "basketball", "hoop"
    confidence: float
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    frame_idx: int

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class FrameDetections:
    """All detections for a single frame."""

    frame_idx: int
    balls: list[Detection] = field(default_factory=list)
    hoops: list[Detection] = field(default_factory=list)
    players: list[Detection] = field(default_factory=list)


class ObjectDetector:
    """Runs YOLO inference on video frames to detect basketball objects.

    Supports both stock COCO-pretrained models (detects "person" and
    "sports ball") and custom basketball-specific models (detects
    "basketball", "hoop", "player").
    """

    def __init__(self, config: VideoConfig | None = None, device: str = "auto", detect_players: bool = True):
        self.config = config or VideoConfig()
        self.model = YOLO(self.config.yolo_model)
        self._is_custom = self._check_custom_model()
        self._device = self._resolve_device(device)
        self._detect_players = detect_players
        self._hoop_model = None
        if self.config.roboflow_model_id:
            self._hoop_model = self._load_roboflow_model(self.config.roboflow_model_id)
        logger.info(
            "Loaded YOLO model %s (custom=%s, device=%s, hoop_model=%s)",
            self.config.yolo_model, self._is_custom, self._device,
            self.config.roboflow_model_id or "none",
        )

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Pick the best available device."""
        if device != "auto":
            return device
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _check_custom_model(self) -> bool:
        """Check if loaded model has basketball-specific classes."""
        names = self.model.names or {}
        return "basketball" in [str(v).lower() for v in names.values()]

    @staticmethod
    def _load_roboflow_model(model_id: str):
        """Load a Roboflow model via the inference SDK."""
        try:
            from inference import get_model
        except ImportError:
            logger.error(
                "The 'inference' package is required for --roboflow-model. "
                "Install it with: pip install inference"
            )
            raise

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable is required when using "
                "--roboflow-model. Set it with: export ROBOFLOW_API_KEY=your_key"
            )

        model = get_model(model_id=model_id, api_key=api_key)
        logger.info("Loaded Roboflow model: %s", model_id)
        return model

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array.
            frame_idx: Frame index in the video.

        Returns:
            FrameDetections with categorized detections.
        """
        results = self.model(
            frame,
            conf=self.config.yolo_confidence,
            device=self._device,
            verbose=False,
        )
        fd = FrameDetections(frame_idx=frame_idx)

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                bbox = tuple(int(v) for v in boxes.xyxy[i].tolist())
                cls_name = self.model.names[cls_id]

                det = Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                    frame_idx=frame_idx,
                )

                if self._is_custom:
                    self._categorize_custom(det, cls_name.lower(), fd)
                else:
                    self._categorize_coco(det, cls_id, fd)

        # Drop player detections when player tracking is disabled
        if not self._detect_players:
            fd.players.clear()

        if self._hoop_model is not None:
            self._detect_roboflow(frame, frame_idx, fd)

        return fd

    def _detect_roboflow(self, frame: np.ndarray, frame_idx: int, fd: FrameDetections) -> None:
        """Run Roboflow model and append hoop and ball detections."""
        _HOOP_CLASSES = {"hoop", "rim", "basket"}
        _BALL_CLASSES = {"basketball", "ball"}
        try:
            results = self._hoop_model.infer(
                frame, confidence=self.config.roboflow_confidence,
            )
        except Exception:
            logger.debug("Roboflow inference failed on frame %d", frame_idx, exc_info=True)
            return

        # results may be a list or a single response object
        predictions = []
        if isinstance(results, list):
            for r in results:
                predictions.extend(getattr(r, "predictions", []))
        else:
            predictions = getattr(results, "predictions", [])

        for pred in predictions:
            cls_name = getattr(pred, "class_name", "") or ""
            cls_lower = cls_name.lower()

            if cls_lower in _HOOP_CLASSES:
                target_list = fd.hoops
                canonical_name = "hoop"
            elif cls_lower in _BALL_CLASSES:
                target_list = fd.balls
                canonical_name = "basketball"
            else:
                continue

            # Roboflow returns center x/y + width/height
            cx = int(pred.x)
            cy = int(pred.y)
            w = int(pred.width)
            h = int(pred.height)
            x1 = cx - w // 2
            y1 = cy - h // 2
            x2 = cx + w // 2
            y2 = cy + h // 2

            det = Detection(
                class_name=canonical_name,
                confidence=float(pred.confidence),
                bbox=(x1, y1, x2, y2),
                frame_idx=frame_idx,
            )
            target_list.append(det)

    def detect_video(self, video_path: Path) -> list[FrameDetections]:
        """Run detection on all frames of a video (respecting frame_skip).

        Args:
            video_path: Path to video file.

        Returns:
            List of FrameDetections for each analyzed frame.
        """
        lut = FrameLUT(self.config.input_lut)

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        all_detections = []
        frame_idx = 0

        logger.info(
            "Running detection on %s (%d frames, skip=%d)",
            video_path, total_frames, self.config.frame_skip,
        )

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.config.frame_skip == 0:
                frame = lut.apply(frame)
                # Downscale if needed
                h, w = frame.shape[:2]
                if max(h, w) > self.config.max_resolution:
                    scale = self.config.max_resolution / max(h, w)
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

                fd = self.detect_frame(frame, frame_idx)
                all_detections.append(fd)

            frame_idx += 1

        cap.release()
        logger.info("Detection complete: %d frames analyzed", len(all_detections))
        return all_detections

    @staticmethod
    def _categorize_coco(det: Detection, cls_id: int, fd: FrameDetections) -> None:
        """Categorize detection from a COCO-pretrained model."""
        if cls_id == COCO_PERSON:
            fd.players.append(det)
        elif cls_id == COCO_SPORTS_BALL:
            fd.balls.append(det)

    @staticmethod
    def _categorize_custom(det: Detection, cls_name: str, fd: FrameDetections) -> None:
        """Categorize detection from a basketball-specific model."""
        if cls_name in ("basketball", "ball"):
            fd.balls.append(det)
        elif cls_name in ("hoop", "rim", "basket"):
            fd.hoops.append(det)
        elif cls_name in ("player", "person"):
            fd.players.append(det)
