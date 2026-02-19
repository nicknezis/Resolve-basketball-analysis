"""Scene/shot boundary detection using PySceneDetect."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from scenedetect import ContentDetector, SceneManager, open_video

from src.config import SceneConfig

logger = logging.getLogger(__name__)


@dataclass
class Scene:
    """A detected scene boundary."""

    start_frame: int
    end_frame: int
    start_sec: float
    end_sec: float


def detect_scenes(video_path: Path, config: SceneConfig | None = None) -> list[Scene]:
    """Detect shot/scene boundaries in a video file.

    Uses PySceneDetect's ContentDetector which compares frame-to-frame
    changes in HSV color space with an adaptive threshold.

    Args:
        video_path: Path to the video file.
        config: Scene detection settings. Uses defaults if None.

    Returns:
        List of Scene objects with frame and time boundaries.
    """
    config = config or SceneConfig()
    logger.info("Detecting scenes in %s (threshold=%.1f)", video_path, config.content_threshold)

    video = open_video(str(video_path))
    fps = video.frame_rate
    min_scene_len_frames = max(1, int(config.min_scene_len_sec * fps))

    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(
            threshold=config.content_threshold,
            min_scene_len=min_scene_len_frames,
        )
    )

    scene_manager.detect_scenes(video, show_progress=True)
    scene_list = scene_manager.get_scene_list()

    scenes = []
    for start_time, end_time in scene_list:
        scenes.append(
            Scene(
                start_frame=start_time.get_frames(),
                end_frame=end_time.get_frames(),
                start_sec=start_time.get_seconds(),
                end_sec=end_time.get_seconds(),
            )
        )

    logger.info("Detected %d scenes", len(scenes))
    return scenes
