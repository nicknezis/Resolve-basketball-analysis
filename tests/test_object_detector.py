"""Unit tests for ObjectDetector configuration, NMS, and backend selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from src.analysis.object_detector import (
    BASKETBALL_DETECTION_CLASSES,
    COCO_NAMES,
    Detection,
    FrameDetections,
    ObjectDetector,
)
from src.config import VideoConfig


class TestDetectorBackendSelection:
    """Tests for detector backend configuration."""

    def test_default_backend_is_yolo(self):
        config = VideoConfig()
        assert config.detector_backend == "yolo"

    def test_rfdetr_class_names_config(self):
        config = VideoConfig(
            detector_backend="rfdetr",
            rfdetr_class_names=["basketball", "hoop", "player"],
            rfdetr_num_classes=3,
        )
        assert config.rfdetr_class_names == ["basketball", "hoop", "player"]
        assert config.rfdetr_num_classes == 3

    def test_rfdetr_backend_raises_without_package(self):
        """Attempting RF-DETR without the package installed raises ImportError."""
        config = VideoConfig(detector_backend="rfdetr")
        with patch.dict("sys.modules", {"rfdetr": None}):
            try:
                ObjectDetector(config)
                assert False, "Should have raised ImportError"
            except ImportError:
                pass

    def test_basketball_detection_classes_has_10_entries(self):
        assert len(BASKETBALL_DETECTION_CLASSES) == 10
        assert "ball" in BASKETBALL_DETECTION_CLASSES
        assert "rim" in BASKETBALL_DETECTION_CLASSES
        assert "referee" in BASKETBALL_DETECTION_CLASSES
        assert "player-jump-shot" in BASKETBALL_DETECTION_CLASSES

    def test_coco_names_has_80_entries(self):
        assert len(COCO_NAMES) == 80
        assert COCO_NAMES[0] == "person"
        assert COCO_NAMES[32] == "sports ball"


class TestNMSIntegration:
    """Tests for NMS deduplication."""

    def test_nms_disabled_by_default(self):
        config = VideoConfig()
        assert config.nms_enabled is False

    def test_nms_deduplicates_overlapping_balls(self):
        """Two overlapping ball detections should be merged to one."""
        config = VideoConfig(nms_enabled=True, nms_threshold=0.5)

        # Create mock detector to test _apply_nms directly
        with patch.object(ObjectDetector, "__init__", lambda self, *a, **kw: None):
            detector = ObjectDetector.__new__(ObjectDetector)
            detector.config = config

        fd = FrameDetections(frame_idx=0)
        fd.balls.append(Detection(
            class_name="ball", confidence=0.9,
            bbox=(100, 100, 150, 150), frame_idx=0,
        ))
        fd.balls.append(Detection(
            class_name="ball", confidence=0.7,
            bbox=(105, 105, 155, 155), frame_idx=0,
        ))

        # Mock supervision
        mock_sv = MagicMock()
        mock_filtered = MagicMock()
        mock_filtered.data = {"original_idx": np.array([0])}
        mock_filtered.xyxy = np.array([[100, 100, 150, 150]], dtype=np.float32)

        mock_detections_instance = MagicMock()
        mock_detections_instance.with_nms.return_value = mock_filtered
        mock_sv.Detections.return_value = mock_detections_instance

        with patch.dict("sys.modules", {"supervision": mock_sv}):
            result = detector._apply_nms(fd, 0)

        assert len(result.balls) == 1
        assert result.balls[0].confidence == 0.9

    def test_nms_preserves_non_overlapping(self):
        """Two distant detections should both survive NMS."""
        config = VideoConfig(nms_enabled=True, nms_threshold=0.5)

        with patch.object(ObjectDetector, "__init__", lambda self, *a, **kw: None):
            detector = ObjectDetector.__new__(ObjectDetector)
            detector.config = config

        fd = FrameDetections(frame_idx=0)
        fd.balls.append(Detection(
            class_name="ball", confidence=0.9,
            bbox=(100, 100, 150, 150), frame_idx=0,
        ))
        fd.balls.append(Detection(
            class_name="ball", confidence=0.8,
            bbox=(500, 500, 550, 550), frame_idx=0,
        ))

        mock_sv = MagicMock()
        mock_filtered = MagicMock()
        mock_filtered.data = {"original_idx": np.array([0, 1])}
        mock_filtered.xyxy = np.array([
            [100, 100, 150, 150],
            [500, 500, 550, 550],
        ], dtype=np.float32)

        mock_detections_instance = MagicMock()
        mock_detections_instance.with_nms.return_value = mock_filtered
        mock_sv.Detections.return_value = mock_detections_instance

        with patch.dict("sys.modules", {"supervision": mock_sv}):
            result = detector._apply_nms(fd, 0)

        assert len(result.balls) == 2


class TestCategorizeCustomExtended:
    """Tests for the extended _categorize_custom method."""

    def _categorize(self, cls_name: str) -> str | None:
        """Helper: categorize a class name and return which list it goes to."""
        fd = FrameDetections(frame_idx=0)
        det = Detection(class_name=cls_name, confidence=0.9, bbox=(0, 0, 10, 10), frame_idx=0)
        ObjectDetector._categorize_custom(det, cls_name.lower(), fd)
        if fd.balls:
            return "balls"
        if fd.hoops:
            return "hoops"
        if fd.players:
            return "players"
        return None

    def test_ball_classes(self):
        assert self._categorize("ball") == "balls"
        assert self._categorize("basketball") == "balls"
        assert self._categorize("ball-in-basket") == "balls"

    def test_hoop_classes(self):
        assert self._categorize("rim") == "hoops"
        assert self._categorize("hoop") == "hoops"
        assert self._categorize("basket") == "hoops"

    def test_player_classes(self):
        assert self._categorize("player") == "players"
        assert self._categorize("person") == "players"
        assert self._categorize("player-in-possession") == "players"
        assert self._categorize("player-jump-shot") == "players"
        assert self._categorize("player-layup-dunk") == "players"
        assert self._categorize("player-shot-block") == "players"

    def test_ignored_classes(self):
        assert self._categorize("number") is None
        assert self._categorize("referee") is None
