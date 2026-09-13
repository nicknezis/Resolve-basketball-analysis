"""Object detection for basketball, hoop, and players.

Supports three backends selected by ``VideoConfig.detector_backend``:

* ``yolo`` (default) — Ultralytics YOLO; stock COCO weights or a basketball
  fine-tune.  Stock COCO has **no hoop class**, so made/miss detection needs a
  supplemental Roboflow model or a custom model.
* ``rfdetr`` — Roboflow RF-DETR (``pip install rfdetr``); COCO or fine-tuned.
* ``roboflow`` — a Roboflow Universe / workspace model run locally through the
  ``inference`` package (ONNX).  One model supplies ball, hoop, player and
  referee boxes.

Any backend can additionally be paired with ``roboflow_model_id`` as a
*supplemental* hoop/ball source.

Class names from every backend are normalised to a small set of roles via
:data:`CLASS_ROLE_MAP`, so a model that says ``rim`` and one that says ``hoop``
land in the same bucket.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.config import VideoConfig

logger = logging.getLogger(__name__)

# Class indices in COCO dataset used by stock YOLO models
COCO_PERSON = 0
COCO_SPORTS_BALL = 32

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

# ---------------------------------------------------------------------------
# Class-name → role normalisation
# ---------------------------------------------------------------------------

ROLE_BALL = "ball"
ROLE_BALL_IN_BASKET = "ball_in_basket"
ROLE_HOOP = "hoop"
ROLE_PLAYER = "player"
ROLE_REFEREE = "referee"
ROLE_IGNORE = "ignore"

# Keys are normalised (lower-case, '_' → '-').  Anything not listed is dropped.
CLASS_ROLE_MAP: dict[str, str] = {
    # ball
    "ball": ROLE_BALL, "basketball": ROLE_BALL, "sports ball": ROLE_BALL,
    "sports-ball": ROLE_BALL,
    # ball inside the net — a direct made-shot signal.  Also tracked as a ball.
    "ball-in-basket": ROLE_BALL_IN_BASKET, "made": ROLE_BALL_IN_BASKET,
    # hoop / rim
    "hoop": ROLE_HOOP, "rim": ROLE_HOOP, "basket": ROLE_HOOP,
    "basketball-hoop": ROLE_HOOP, "basketball hoop": ROLE_HOOP, "hoop-rim": ROLE_HOOP,
    # players
    "player": ROLE_PLAYER, "person": ROLE_PLAYER, "shooter": ROLE_PLAYER,
    "player-in-possession": ROLE_PLAYER, "player-jump-shot": ROLE_PLAYER,
    "player-layup-dunk": ROLE_PLAYER, "player-shot-block": ROLE_PLAYER,
    # referees are kept separately so they never feed team classification
    "referee": ROLE_REFEREE, "ref": ROLE_REFEREE,
    # known-but-unused
    "number": ROLE_IGNORE, "backboard": ROLE_IGNORE, "net": ROLE_IGNORE,
    "people": ROLE_IGNORE, "shoot": ROLE_IGNORE,
}


def class_role(class_name: str) -> str | None:
    """Map a raw model class name to a role, or None if unknown."""
    key = str(class_name).strip().lower().replace("_", "-")
    return CLASS_ROLE_MAP.get(key)


@dataclass
class Detection:
    """A single object detection in one frame."""

    class_name: str  # raw model class name, e.g. "sports ball", "rim", "ball-in-basket"
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
    """All detections for a single frame, bucketed by role."""

    frame_idx: int
    balls: list[Detection] = field(default_factory=list)
    hoops: list[Detection] = field(default_factory=list)
    players: list[Detection] = field(default_factory=list)
    referees: list[Detection] = field(default_factory=list)
    # Subset of ``balls`` that the model labelled as inside the net.
    balls_in_basket: list[Detection] = field(default_factory=list)


class ObjectDetector:
    """Runs object detection on video frames to detect basketball objects."""

    def __init__(self, config: VideoConfig | None = None, device: str = "auto", detect_players: bool = True):
        self.config = config or VideoConfig()
        self._detect_players = detect_players
        self._device = self._resolve_device(device)
        self._backend = self.config.detector_backend
        self._roboflow_failures = 0

        self.model = None
        self._rfdetr_model = None
        self._roboflow_model = None  # primary backend model
        self._hoop_model = None  # supplemental Roboflow model

        if self._backend == "rfdetr":
            self._rfdetr_model = self._load_rfdetr_model()
            self._is_custom = bool(self.config.rfdetr_weights)
        elif self._backend == "roboflow":
            if not self.config.roboflow_model_id:
                raise ValueError("detector_backend='roboflow' requires roboflow_model_id")
            self._roboflow_model = self._load_roboflow_model(self.config.roboflow_model_id)
            self._is_custom = True
        elif self._backend == "yolo":
            model_path = self._resolve_model_path(self.config.yolo_model)
            self.model = YOLO(model_path)
            self._is_custom = self._check_custom_model()
        else:
            raise ValueError(
                f"Unknown detector_backend {self._backend!r}; expected yolo, rfdetr or roboflow"
            )

        if self.config.roboflow_model_id and self._backend != "roboflow":
            self._hoop_model = self._load_roboflow_model(self.config.roboflow_model_id)

        logger.info(
            "Detector backend=%s, custom=%s, device=%s, imgsz=%d, supplemental_model=%s",
            self._backend, self._is_custom, self._device, self.config.imgsz,
            self.config.roboflow_model_id if self._hoop_model is not None else "none",
        )
        if not self.hoop_capable:
            logger.warning(
                "Detector cannot produce hoop detections (stock COCO weights have no "
                "hoop/rim class). Made/miss classification is disabled; every shot "
                "will be reported as an attempt. Use --detector roboflow, "
                "--roboflow-model, or a basketball fine-tuned model."
            )

    # ------------------------------------------------------------------
    # Setup helpers
    # ------------------------------------------------------------------

    @property
    def hoop_capable(self) -> bool:
        """True if at least one configured model can emit hoop/rim boxes."""
        return self._is_custom or self._hoop_model is not None

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
        """A YOLO model is basketball-specific if any class maps to the hoop role."""
        names = self.model.names or {}
        return any(class_role(str(v)) == ROLE_HOOP for v in names.values())

    @staticmethod
    def _load_roboflow_model(model_id: str):
        """Load a Roboflow model via the inference SDK (runs locally via ONNX)."""
        try:
            from inference import get_model
        except ImportError:
            logger.error(
                "The 'inference' package is required for Roboflow models. "
                "Install it with: pip install inference"
            )
            raise

        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ROBOFLOW_API_KEY environment variable is required when using "
                "Roboflow models. Set it with: export ROBOFLOW_API_KEY=your_key"
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
        if self.config.rfdetr_weights and not self.config.rfdetr_class_names:
            logger.warning(
                "--rfdetr-weights given without --rfdetr-classes: class IDs will be "
                "interpreted as COCO indices and hoop boxes will be dropped."
            )
        return model

    def _get_rfdetr_class_name(self, cls_id: int) -> str:
        """Map a class ID to a class name for RF-DETR detections."""
        if self.config.rfdetr_class_names:
            if cls_id < len(self.config.rfdetr_class_names):
                return self.config.rfdetr_class_names[cls_id]
            return f"class_{cls_id}"
        return COCO_NAMES.get(cls_id, f"class_{cls_id}")

    # ------------------------------------------------------------------
    # Per-frame inference
    # ------------------------------------------------------------------

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
        elif self._backend == "roboflow":
            fd = FrameDetections(frame_idx=frame_idx)
            self._detect_roboflow(self._roboflow_model, frame, frame_idx, fd)
        else:
            fd = self._detect_frame_yolo(frame, frame_idx)

        if self._hoop_model is not None:
            self._detect_roboflow(self._hoop_model, frame, frame_idx, fd)

        if not self._detect_players:
            fd.players.clear()
            fd.referees.clear()

        if self.config.nms_enabled:
            fd = self._apply_nms(fd, frame_idx)

        return fd

    def _min_conf(self, base: float) -> float:
        """Inference threshold: low enough to let looser hoop boxes through."""
        return min(base, self.config.hoop_confidence) if self._is_custom else base

    def _detect_frame_yolo(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run YOLO inference on a single frame."""
        kwargs: dict = {}
        if not self._is_custom:
            # Stock COCO: only run the two classes we use (saves the NMS work)
            kwargs["classes"] = [COCO_PERSON, COCO_SPORTS_BALL]
        results = self.model(
            frame,
            conf=self._min_conf(self.config.yolo_confidence),
            imgsz=self.config.imgsz,
            device=self._device,
            verbose=False,
            **kwargs,
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
                cls_name = str(self.model.names[cls_id])

                det = Detection(
                    class_name=cls_name,
                    confidence=conf,
                    bbox=bbox,
                    frame_idx=frame_idx,
                )
                self._categorize(det, fd, base_conf=self.config.yolo_confidence)

        return fd

    def _detect_frame_rfdetr(self, frame: np.ndarray, frame_idx: int) -> FrameDetections:
        """Run RF-DETR inference on a single frame.

        RF-DETR's ``predict()`` returns a ``supervision.Detections`` object
        with ``xyxy``, ``confidence``, and ``class_id`` numpy arrays.
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = self._rfdetr_model.predict(
            frame_rgb, threshold=self._min_conf(self.config.yolo_confidence),
        )

        fd = FrameDetections(frame_idx=frame_idx)

        if detections.xyxy is None or len(detections.xyxy) == 0:
            return fd

        for i in range(len(detections.xyxy)):
            cls_id = int(detections.class_id[i])
            conf = float(detections.confidence[i])
            bbox = tuple(int(v) for v in detections.xyxy[i].tolist())

            det = Detection(
                class_name=self._get_rfdetr_class_name(cls_id),
                confidence=conf,
                bbox=bbox,
                frame_idx=frame_idx,
            )
            self._categorize(det, fd, base_conf=self.config.yolo_confidence)

        return fd

    def _detect_roboflow(
        self, model, frame: np.ndarray, frame_idx: int, fd: FrameDetections,
    ) -> None:
        """Run a Roboflow ``inference`` model and merge its boxes into ``fd``."""
        try:
            results = model.infer(
                frame, confidence=min(self.config.roboflow_confidence, self.config.hoop_confidence),
            )
        except Exception:
            self._roboflow_failures += 1
            if self._roboflow_failures == 1:
                logger.warning(
                    "Roboflow inference failed on frame %d (further failures logged at DEBUG)",
                    frame_idx, exc_info=True,
                )
            else:
                logger.debug("Roboflow inference failed on frame %d", frame_idx, exc_info=True)
            return

        # results may be a list or a single response object
        predictions = []
        if isinstance(results, list):
            for r in results:
                predictions.extend(_get(r, "predictions", []))
        else:
            predictions = _get(results, "predictions", [])

        for pred in predictions:
            cls_name = _get(pred, "class_name", None) or _get(pred, "class", "") or ""
            # Roboflow returns center x/y + width/height
            cx = int(_get(pred, "x", 0))
            cy = int(_get(pred, "y", 0))
            w = int(_get(pred, "width", 0))
            h = int(_get(pred, "height", 0))
            det = Detection(
                class_name=str(cls_name),
                confidence=float(_get(pred, "confidence", 0.0)),
                bbox=(cx - w // 2, cy - h // 2, cx + w // 2, cy + h // 2),
                frame_idx=frame_idx,
            )
            self._categorize(det, fd, base_conf=self.config.roboflow_confidence)

    # ------------------------------------------------------------------
    # Post-processing
    # ------------------------------------------------------------------

    def _categorize(self, det: Detection, fd: FrameDetections, base_conf: float) -> None:
        """Bucket a detection by role, applying per-role confidence floors."""
        role = class_role(det.class_name)
        if role is None or role == ROLE_IGNORE:
            return
        if role == ROLE_HOOP:
            if det.confidence >= self.config.hoop_confidence:
                fd.hoops.append(det)
            return
        if det.confidence < base_conf:
            return
        if role == ROLE_BALL:
            fd.balls.append(det)
        elif role == ROLE_BALL_IN_BASKET:
            fd.balls.append(det)
            fd.balls_in_basket.append(det)
        elif role == ROLE_PLAYER:
            fd.players.append(det)
        elif role == ROLE_REFEREE:
            fd.referees.append(det)

    @staticmethod
    def _categorize_custom(det: Detection, cls_name: str, fd: FrameDetections) -> None:
        """Bucket a detection by class name with no confidence filtering.

        Kept for callers/tests that categorize synthetic detections directly.
        """
        role = class_role(cls_name)
        if role == ROLE_HOOP:
            fd.hoops.append(det)
        elif role == ROLE_BALL:
            fd.balls.append(det)
        elif role == ROLE_BALL_IN_BASKET:
            fd.balls.append(det)
            fd.balls_in_basket.append(det)
        elif role == ROLE_PLAYER:
            fd.players.append(det)
        elif role == ROLE_REFEREE:
            fd.referees.append(det)

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

        result = FrameDetections(frame_idx=frame_idx, balls_in_basket=list(fd.balls_in_basket))

        for category in ("balls", "hoops", "players", "referees"):
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


def _get(obj, name: str, default=None):
    """Attribute-or-key lookup for SDK responses that may be objects or dicts."""
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)
