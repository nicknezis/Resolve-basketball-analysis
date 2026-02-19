"""Orchestrator for the full video analysis pipeline."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import cv2

from src.analysis.audio_analyzer import analyze_audio
from src.analysis.ball_tracker import BallTracker
from src.analysis.event_classifier import EventClassifier, GameEvent
from src.analysis.object_detector import ObjectDetector
from src.analysis.player_tracker import PlayerTracker
from src.analysis.scene_detector import Scene, detect_scenes
from src.config import AnalysisConfig

logger = logging.getLogger(__name__)


def analyze_video(
    video_path: Path,
    config: AnalysisConfig | None = None,
) -> dict:
    """Run the full analysis pipeline on a basketball video.

    Pipeline:
    1. Scene/shot boundary detection
    2. Audio analysis (crowd excitement, whistles)
    3. Object detection (ball, hoop, players)
    4. Ball tracking and shot detection
    5. Player tracking and team classification
    6. Multi-modal event classification

    Args:
        video_path: Path to the video file.
        config: Analysis configuration. Uses defaults if None.

    Returns:
        Analysis results dict suitable for JSON serialization.
    """
    config = config or AnalysisConfig()
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    logger.info(
        "Analyzing %s: %.2f fps, %d frames, %dx%d",
        video_path.name, fps, total_frames, width, height,
    )

    # 1. Scene detection
    logger.info("=== Phase 1: Scene Detection ===")
    scenes = detect_scenes(video_path, config.scene)

    # 2. Audio analysis
    logger.info("=== Phase 2: Audio Analysis ===")
    audio_events = analyze_audio(video_path, config.audio)

    # 3. Object detection
    logger.info("=== Phase 3: Object Detection ===")
    detector = ObjectDetector(config.video)
    all_detections = detector.detect_video(video_path)

    # 4. Ball tracking and shot detection
    logger.info("=== Phase 4: Ball Tracking ===")
    ball_tracker = BallTracker(config.tracking)
    shot_events = ball_tracker.detect_shots(all_detections)

    # 5. Player tracking
    logger.info("=== Phase 5: Player Tracking ===")
    player_tracker = PlayerTracker(config.tracking)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    for fd in all_detections:
        # Seek to the correct frame
        while frame_idx < fd.frame_idx:
            cap.read()
            frame_idx += 1
        ret, frame = cap.read()
        frame_idx += 1
        if ret:
            player_tracker.update(frame, fd)

    cap.release()
    player_tracker.classify_teams()

    # 6. Event classification
    logger.info("=== Phase 6: Event Classification ===")
    classifier = EventClassifier(config.events, fps=fps)
    game_events = classifier.classify(shot_events, audio_events, scenes)

    # Build output
    result = _build_output(video_path, fps, total_frames, width, height, scenes, game_events)

    logger.info(
        "Analysis complete: %d events detected across %d scenes",
        len(game_events), len(scenes),
    )

    return result


def save_results(results: dict, output_path: Path) -> Path:
    """Save analysis results to a JSON file.

    Args:
        results: Analysis results dict from analyze_video().
        output_path: Path for the output JSON file.

    Returns:
        Path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)
    return output_path


def _build_output(
    video_path: Path,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    scenes: list[Scene],
    events: list[GameEvent],
) -> dict:
    """Build the output JSON structure."""
    return {
        "analysis_version": "0.1.0",
        "source_file": str(video_path),
        "video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps if fps > 0 else 0,
        },
        "scenes": [
            {
                "start_frame": s.start_frame,
                "end_frame": s.end_frame,
                "start_sec": round(s.start_sec, 3),
                "end_sec": round(s.end_sec, 3),
            }
            for s in scenes
        ],
        "events": [
            {
                "type": e.event_type,
                "start_frame": e.start_frame,
                "end_frame": e.end_frame,
                "start_sec": round(e.start_sec, 3),
                "end_sec": round(e.end_sec, 3),
                "confidence": round(e.confidence, 3),
                "video_confidence": round(e.video_confidence, 3),
                "audio_confidence": round(e.audio_confidence, 3),
                "details": e.details,
            }
            for e in events
        ],
        "summary": {
            "total_events": len(events),
            "event_counts": _count_events(events),
            "total_scenes": len(scenes),
        },
    }


def _count_events(events: list[GameEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1
    return counts
