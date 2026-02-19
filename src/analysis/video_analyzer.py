"""Orchestrator for the full video analysis pipeline.

Supports two input modes:
  1. Single video file: analyze one video, output events with file-relative frames
  2. Timeline export JSON: analyze each clip's media file for only the used
     source range, output events with timeline-relative frames for direct
     marker import back into Resolve
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import cv2

from src.analysis.audio_analyzer import (
    analyze_audio,
    analyze_crowd_excitement,
    detect_whistles,
    extract_audio,
)
from src.analysis.ball_tracker import BallTracker
from src.analysis.event_classifier import EventClassifier, GameEvent
from src.analysis.object_detector import ObjectDetector
from src.analysis.player_tracker import PlayerTracker
from src.analysis.scene_detector import Scene, detect_scenes
from src.config import AnalysisConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single video analysis (original mode)
# ---------------------------------------------------------------------------

def analyze_video(
    video_path: Path,
    config: AnalysisConfig | None = None,
) -> dict:
    """Run the full analysis pipeline on a single basketball video.

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

    scenes, audio_events, game_events = _run_pipeline(video_path, fps, config)

    return _build_single_output(
        video_path, fps, total_frames, width, height, scenes, game_events,
    )


# ---------------------------------------------------------------------------
# Timeline-aware analysis (new mode)
# ---------------------------------------------------------------------------

def analyze_timeline(
    timeline_json_path: Path,
    config: AnalysisConfig | None = None,
) -> dict:
    """Run analysis on a Resolve timeline export, processing each clip.

    For each clip in the timeline:
    1. Opens the clip's media file (proxy if available, else full-res)
    2. Seeks to the source in-point and analyzes only the used range
    3. Maps detected events from source-local frames to timeline frames

    This means the output JSON has events positioned at timeline frame
    numbers, ready for direct marker import into Resolve.

    Args:
        timeline_json_path: Path to the timeline export JSON from
            ``python -m src.resolve.export``.
        config: Analysis configuration. Uses defaults if None.

    Returns:
        Analysis results dict with timeline-relative frame positions.
    """
    config = config or AnalysisConfig()
    timeline_json_path = Path(timeline_json_path)

    with open(timeline_json_path) as f:
        timeline_data = json.load(f)

    tl_info = timeline_data["timeline"]
    tl_fps = tl_info["fps"]
    tl_name = tl_info["name"]

    logger.info("Analyzing timeline '%s' at %.2f fps", tl_name, tl_fps)

    all_events: list[GameEvent] = []
    all_scenes: list[Scene] = []
    clips_analyzed = 0
    clips_skipped = 0

    for track in timeline_data.get("tracks", []):
        track_idx = track["track_index"]
        for clip in track.get("clips", []):
            clip_name = clip.get("clip_name", "unknown")
            analysis_path = clip.get("analysis_path", "") or clip.get("file_path", "")

            if not analysis_path or not Path(analysis_path).exists():
                logger.warning(
                    "Skipping clip '%s' — media file not found: %s",
                    clip_name, analysis_path,
                )
                clips_skipped += 1
                continue

            logger.info(
                "=== Analyzing clip '%s' (track %d) ===", clip_name, track_idx,
            )

            clip_events, clip_scenes = _analyze_clip(
                clip_info=clip,
                timeline_fps=tl_fps,
                config=config,
            )

            all_events.extend(clip_events)
            all_scenes.extend(clip_scenes)
            clips_analyzed += 1

    # Sort all events by timeline position
    all_events.sort(key=lambda e: e.start_sec)

    # Merge events that ended up adjacent across clip boundaries
    classifier = EventClassifier(config.events, fps=tl_fps)
    all_events = classifier._merge_nearby(all_events)

    logger.info(
        "Timeline analysis complete: %d events across %d clips (%d skipped)",
        len(all_events), clips_analyzed, clips_skipped,
    )

    return _build_timeline_output(
        timeline_data, all_scenes, all_events,
        clips_analyzed, clips_skipped,
    )


def _analyze_clip(
    clip_info: dict,
    timeline_fps: float,
    config: AnalysisConfig,
) -> tuple[list[GameEvent], list[Scene]]:
    """Analyze a single clip's source range and map results to timeline frames.

    Args:
        clip_info: Clip dict from the timeline export JSON.
        timeline_fps: The timeline's frame rate.
        config: Analysis settings.

    Returns:
        Tuple of (events mapped to timeline frames, scenes mapped to timeline).
    """
    analysis_path = Path(clip_info.get("analysis_path") or clip_info["file_path"])
    source_start = clip_info["source_start_frame"]
    source_end = clip_info["source_end_frame"]
    tl_start = clip_info["timeline_start_frame"]
    media_fps = clip_info.get("media_fps", timeline_fps)

    logger.info(
        "  Media: %s (source frames %d-%d -> timeline frame %d)",
        analysis_path.name, source_start, source_end, tl_start,
    )

    # --- Video analysis on the source range ---
    detector = ObjectDetector(config.video)
    ball_tracker = BallTracker(config.tracking)
    player_tracker = PlayerTracker(config.tracking)

    cap = cv2.VideoCapture(str(analysis_path))
    if not cap.isOpened():
        logger.error("  Cannot open media file: %s", analysis_path)
        return [], []

    # Seek to source start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)

    all_detections = []
    frame_idx = source_start

    while frame_idx < source_end:
        ret, frame = cap.read()
        if not ret:
            break

        relative_frame = frame_idx - source_start
        if relative_frame % config.video.frame_skip == 0:
            h, w = frame.shape[:2]
            if max(h, w) > config.video.max_resolution:
                scale = config.video.max_resolution / max(h, w)
                frame_resized = cv2.resize(frame, (int(w * scale), int(h * scale)))
            else:
                frame_resized = frame

            fd = detector.detect_frame(frame_resized, frame_idx)
            all_detections.append(fd)

            # Player tracking needs the original frame for color sampling
            player_tracker.update(frame, fd)

        frame_idx += 1

    cap.release()
    player_tracker.classify_teams()

    # Ball tracking + shot detection
    shot_events = ball_tracker.detect_shots(all_detections)

    # --- Audio analysis on the source range ---
    audio_events = []
    try:
        audio_path = extract_audio(analysis_path, sample_rate=config.audio.sample_rate)
        try:
            import librosa
            import soundfile as sf

            y, sr = librosa.load(str(audio_path), sr=config.audio.sample_rate)
            start_sample = int((source_start / media_fps) * sr)
            end_sample = int((source_end / media_fps) * sr)
            y_clip = y[start_sample:end_sample]

            trimmed_path = Path(tempfile.mktemp(suffix=".wav"))
            sf.write(str(trimmed_path), y_clip, sr)

            try:
                crowd_events = analyze_crowd_excitement(trimmed_path, config.audio)
                whistle_events = detect_whistles(trimmed_path, config.audio)
                audio_events = crowd_events + whistle_events
                audio_events.sort(key=lambda e: e.start_sec)
            finally:
                trimmed_path.unlink(missing_ok=True)
        finally:
            audio_path.unlink(missing_ok=True)
    except Exception as e:
        logger.warning("  Audio analysis failed for clip: %s", e)

    # --- Scene detection on the source range ---
    scenes = []
    try:
        all_scenes = detect_scenes(analysis_path, config.scene)
        scenes = [
            s for s in all_scenes
            if s.start_frame >= source_start and s.start_frame < source_end
        ]
    except Exception as e:
        logger.warning("  Scene detection failed for clip: %s", e)

    # --- Event classification ---
    classifier = EventClassifier(config.events, fps=media_fps)
    game_events = classifier.classify(shot_events, audio_events, scenes)

    # --- Map from source frames to timeline frames ---
    fps_ratio = timeline_fps / media_fps if media_fps > 0 else 1.0

    for event in game_events:
        source_offset = event.start_frame - source_start
        event.start_frame = tl_start + int(source_offset * fps_ratio)
        source_offset_end = event.end_frame - source_start
        event.end_frame = tl_start + int(source_offset_end * fps_ratio)
        event.start_sec = event.start_frame / timeline_fps
        event.end_sec = event.end_frame / timeline_fps
        event.details["source_clip"] = clip_info.get("clip_name", "")
        event.details["source_file"] = str(analysis_path)

    mapped_scenes = []
    for s in scenes:
        source_offset = s.start_frame - source_start
        mapped_scenes.append(Scene(
            start_frame=tl_start + int(source_offset * fps_ratio),
            end_frame=tl_start + int((s.end_frame - source_start) * fps_ratio),
            start_sec=(tl_start + source_offset * fps_ratio) / timeline_fps,
            end_sec=(tl_start + (s.end_frame - source_start) * fps_ratio) / timeline_fps,
        ))

    logger.info(
        "  Clip '%s': %d events detected",
        clip_info.get("clip_name", ""), len(game_events),
    )

    return game_events, mapped_scenes


# ---------------------------------------------------------------------------
# Output builders
# ---------------------------------------------------------------------------

def save_results(results: dict, output_path: Path) -> Path:
    """Save analysis results to a JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)
    return output_path


def _build_single_output(
    video_path: Path,
    fps: float,
    total_frames: int,
    width: int,
    height: int,
    scenes: list[Scene],
    events: list[GameEvent],
) -> dict:
    return {
        "analysis_version": "0.2.0",
        "mode": "single_video",
        "source_file": str(video_path),
        "video_info": {
            "fps": fps,
            "total_frames": total_frames,
            "width": width,
            "height": height,
            "duration_sec": total_frames / fps if fps > 0 else 0,
        },
        "scenes": _serialize_scenes(scenes),
        "events": _serialize_events(events),
        "summary": {
            "total_events": len(events),
            "event_counts": _count_events(events),
            "total_scenes": len(scenes),
        },
    }


def _build_timeline_output(
    timeline_data: dict,
    scenes: list[Scene],
    events: list[GameEvent],
    clips_analyzed: int,
    clips_skipped: int,
) -> dict:
    """Build output for timeline mode — events are at timeline frame positions."""
    tl = timeline_data["timeline"]
    total_clips = sum(len(t.get("clips", [])) for t in timeline_data.get("tracks", []))

    return {
        "analysis_version": "0.2.0",
        "mode": "timeline",
        "timeline": {
            "name": tl["name"],
            "fps": tl["fps"],
            "start_frame": tl.get("start_frame", 0),
            "video_track_count": tl.get("video_track_count", 0),
        },
        "clips_summary": {
            "total_clips": total_clips,
            "clips_analyzed": clips_analyzed,
            "clips_skipped": clips_skipped,
        },
        "scenes": _serialize_scenes(scenes),
        "events": _serialize_events(events),
        "summary": {
            "total_events": len(events),
            "event_counts": _count_events(events),
            "total_scenes": len(scenes),
        },
    }


def _serialize_scenes(scenes: list[Scene]) -> list[dict]:
    return [
        {
            "start_frame": s.start_frame,
            "end_frame": s.end_frame,
            "start_sec": round(s.start_sec, 3),
            "end_sec": round(s.end_sec, 3),
        }
        for s in scenes
    ]


def _serialize_events(events: list[GameEvent]) -> list[dict]:
    return [
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
    ]


def _count_events(events: list[GameEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for e in events:
        counts[e.event_type] = counts.get(e.event_type, 0) + 1
    return counts


def _run_pipeline(
    video_path: Path, fps: float, config: AnalysisConfig,
) -> tuple[list[Scene], list, list[GameEvent]]:
    """Run the full analysis pipeline on a single video (all frames)."""
    logger.info("=== Phase 1: Scene Detection ===")
    scenes = detect_scenes(video_path, config.scene)

    logger.info("=== Phase 2: Audio Analysis ===")
    audio_events = analyze_audio(video_path, config.audio)

    logger.info("=== Phase 3: Object Detection ===")
    detector = ObjectDetector(config.video)
    all_detections = detector.detect_video(video_path)

    logger.info("=== Phase 4: Ball Tracking ===")
    ball_tracker = BallTracker(config.tracking)
    shot_events = ball_tracker.detect_shots(all_detections)

    logger.info("=== Phase 5: Player Tracking ===")
    player_tracker = PlayerTracker(config.tracking)
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0

    for fd in all_detections:
        while frame_idx < fd.frame_idx:
            cap.read()
            frame_idx += 1
        ret, frame = cap.read()
        frame_idx += 1
        if ret:
            player_tracker.update(frame, fd)

    cap.release()
    player_tracker.classify_teams()

    logger.info("=== Phase 6: Event Classification ===")
    classifier = EventClassifier(config.events, fps=fps)
    game_events = classifier.classify(shot_events, audio_events, scenes)

    return scenes, audio_events, game_events
