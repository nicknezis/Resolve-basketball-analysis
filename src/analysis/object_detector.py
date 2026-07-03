"""Object detection for basketball, hoop, and players.

Supports YOLO (default) and RF-DETR backends.
"""

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

# Class names from the basketball-player-detection-3 dataset (v6)
# https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo/dataset/6
BASKETBALL_DETECTION_CLASSES = [
    "ball", "ball-in-basket", "number", "player", "player-in-possession",
    "player-jump-shot", "player-layup-dunk", "player-shot-block", "referee", "rim",
]

# COCO class names (80 classes) for RF-DETR COCO-pretrained models
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 4: "airplane",
    5: "bus", 6: "train", 7: "truck", 8: "boat", 9: "traffic light",
    10: "fire hydrant", 11: "stop sign", 12: "parking meter", 13: "bench",
    14: "bird", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow",
    20: "elephant", 21: "bear", 22: "zebra", 23: "giraffe", 24: "backpack",
    25: "umbrella", 26: "handbag", 27: "tie", 28: "suitcase", 29: "frisbee",
    30: "skis", 31: "snowboard", 32: "sports ball", 33: "kite",
    34: "baseball bat", 35: "baseball glove", 36: "skateboard", 37: "surfboard",
    38: "tennis racket", 39: "bottle", 40: "wine glass", 41: "cup", 42: "fork",
    43: "knife", 44: "spoon", 45: "bowl", 46: "banana", 47: "apple",
    48: "sandwich", 49: "orange", 50: "broccoli", 51: "carrot", 52: "hot dog",
    53: "pizza", 54: "donut", 55: "cake", 56: "chair", 57: "couch",
    58: "potted plant", 59: "bed", 60: "dining table", 61: "toilet", 62: "tv",
    63: "laptop", 64: "mouse", 65: "remote", 66: "keyboard", 67: "cell phone",
    68: "microwave", 69: "oven", 70: "toaster", 71: "sink", 72: "refrigerator",
    73: "book", 74: "clock", 75: "vase", 76: "scissors", 77: "teddy bear",
    78: "hair drier", 79: "toothbrush",
}


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
    """Runs object detection on video frames to detect basketball objects.

    Supports YOLO (default) and RF-DETR backends, both with stock
    COCO-pretrained models and custom basketball-specific models.
    """

    def __init__(self, config: VideoConfig | None = None, device: str = "auto", detect_players: bool = True):
        self.config = config or VideoConfig()
        self._detect_players = detect_players
        self._device = self._resolve_device(device)
        self._backend = self.config.detector_backend

        if self._backend == "rfdetr":
            self._rfdetr_model = self._load_rfdetr_model()
            self._is_custom = bool(self.config.rfdetr_weights)
            self.model = None
        else:
            model_path = self._resolve_model_path(self.config.yolo_model)
            self.model = YOLO(model_path)
            self._is_custom = self._check_custom_model()
            self._rfdetr_model = None

        self._hoop_model = None
        if self.config.roboflow_model_id:
            self._hoop_model = self._load_roboflow_model(self.config.roboflow_model_id)

        logger.info(
            "Detector backend=%s, custom=%s, device=%s, hoop_model=%s",
            self._backend, self._is_custom, self._device,
            self.config.roboflow_model_id or "none",
        )

    @staticmethod
    def _resolve_model_path(model: str) -> str:
        """Resolve a bare model filename to the models/ directory."""
        model_p = Path(model)
        if model_p.exists() or "/" in model or "\\" in model:
            return model
        models_dir = Path(__file__).resolve().parents[2] / "models"
        local = models_dir / model
        if local.exists():
            return str(local)
        # Bare name that doesn't exist yet — download into models/
        models_dir.mkdir(parents=True, exist_ok=True)
        return str(local)

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

    def _load_rfdetr_model(self) -> object:
        """Load an RF-DETR model.  Requires the ``rfdetr`` package."""
        try:
            import rfdetr as _rfdetr_module
        except ImportError:
            raise ImportError(
                "The 'rfdetr' package is required when using detector_backend='rfdetr'. "
                "Install it with: pip install rfdetr"
            )

        size_map = {
            "nano": "RFDETRNano",
            "small": "RFDETRSmall",
            "medium": "RFDETRMedium",
            "large": "RFDETRLarge",
            "xlarge": "RFDETRXLarge",
            "2xlarge": "RFDETR2XLarge",
        }
        cls_name = size_map.get(self.config.rfdetr_model_size)
        if cls_name is None:
            raise ValueError(
                f"Unknown rfdetr_model_size: {self.config.rfdetr_model_size!r}. "
                f"Choose from: {list(size_map.keys())}"
            )
        model_cls = getattr(_rfdetr_module, cls_name)

        kwargs: dict = {}
        if self.config.rfdetr_weights:
            kwargs["pretrain_weights"] = self.config.rfdetr_weights
        if self.config.rfdetr_num_classes is not None:
            kwargs["num_classes"] = self.config.rfdetr_num_classes
        if self.config.rfdetr_resolution is not None:
            kwargs["resolution"] = self.config.rfdetr_resolution

        model = model_cls(**kwargs)
        logger.info(
            "Loaded RF-DETR model: %s (weights=%s, num_classes=%s)",
            cls_name,
            self.config.rfdetr_weights or "COCO-pretrained",
            self.config.rfdetr_num_classes or "default",
        )
        return model

    def _get_rfdetr_class_name(self, cls_id: int) -> str:
        """Map a class ID to a class name for RF-DETR detections."""
        if self.config.rfdetr_class_names:
            if cls_id < len(self.config.rfdetr_class_names):
                return self.config.rfdetr_class_names[cls_id]
            return f"class_{cls_id}"
        return COCO_NAMES.get(cls_id, f"class_{cls_id}")

    def detect_frame(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run detection on a single frame.

        Args:
            frame: BGR image as numpy array.
            frame_idx: Frame index in the video.

        Returns:
            FrameDetections with categorized detections.
        """
        if self._backend == "rfdetr":
            fd = self._detect_frame_rfdetr(frame, frame_idx)
        else:
            fd = self._detect_frame_yolo(frame, frame_idx)

        if not self._detect_players:
            fd.players.clear()

        if self._hoop_model is not None:
            self._detect_roboflow(frame, frame_idx, fd)

        if self.config.nms_enabled:
            fd = self._apply_nms(fd, frame_idx)

        return fd

    def _detect_frame_yolo(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run YOLO inference on a single frame."""
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

        return fd

    def _detect_frame_rfdetr(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run RF-DETR inference on a single frame.

        RF-DETR's ``predict()`` returns a ``supervision.Detections`` object
        with ``xyxy``, ``confidence``, and ``class_id`` numpy arrays.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._rfdetr_model.predict(
            frame_rgb, threshold=self.config.yolo_confidence,
        )

        fd = FrameDetections(frame_idx=frame_idx)

        if detections.xyxy is None or len(detections.xyxy) == 0:
            return fd

        for i in range(len(detections.xyxy)):
            cls_id = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            bbox = tuple(int(v) for v in detections.xyxy[i].tolist())

            cls_name = self._get_rfdetr_class_name(cls_id)

            det = Detection(
                class_name=cls_name,
                confidence=conf,
                bbox=bbox,
                frame_idx=frame_idx,
            )

            if self._is_custom and self.config.rfdetr_class_names:
                self._categorize_custom(det, cls_name.lower(), fd)
            else:
                self._categorize_coco(det, cls_id, fd)

        return fd

    def _apply_nms(self, fd: FrameDetections, frame_idx: int) -> FrameDetections:
        """Apply per-category NMS using supervision to deduplicate detections."""
        try:
            import supervision as sv
        except ImportError:
            logger.warning(
                "NMS enabled but 'supervision' is not installed. "
                "Install with: pip install supervision"
            )
            return fd

        result = FrameDetections(frame_idx=frame_idx)

        for category in ("balls", "hoops", "players"):
            det_list = getattr(fd, category)
            if len(det_list) < 2:
                getattr(result, category).extend(det_list)
                continue

            xyxy = np.array([d.bbox for d in det_list], dtype=np.float32)
            confidence = np.array([d.confidence for d in det_list], dtype=np.float32)

            sv_dets = sv.Detections(
                xyxy=xyxy,
                confidence=confidence,
                class_id=np.zeros(len(det_list), dtype=int),
                data={"original_idx": np.arange(len(det_list))},
            )
            filtered = sv_dets.with_nms(threshold=self.config.nms_threshold)
            kept_indices = filtered.data["original_idx"]
            for idx in kept_indices:
                getattr(result, category).append(det_list[int(idx)])

        return result

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
        if cls_name in ("basketball", "ball", "ball-in-basket"):
            fd.balls.append(det)
        elif cls_name in ("hoop", "rim", "basket"):
            fd.hoops.append(det)
        elif cls_name in (
            "player", "person", "player-in-possession",
            "player-jump-shot", "player-layup-dunk", "player-shot-block",
        ):
            fd.players.append(det)
        # "number" and "referee" are ignored
