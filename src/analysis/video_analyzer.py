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
import numpy as np

from src.analysis.color import FrameLUT
from src.analysis.audio_analyzer import (
    analyze_audio,
    analyze_crowd_excitement,
    crowd_band_energy,
    crowd_norm_from_energies,
    detect_rim_impacts,
    detect_whistles,
    extract_audio,
)
from src.analysis.ball_tracker import BallTracker, HoopObservation
from src.analysis.event_classifier import EventClassifier, GameEvent
from src.analysis.object_detector import ObjectDetector
from src.analysis.player_tracker import PlayerTracker
from src.analysis.preview import ClipReview, FramePreview
from src.analysis.rim_roi import RimRoiRedetector, RimWindow
from src.analysis.court import CourtModel, CourtTracker, load_court_model
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

    scenes, audio_events, game_events, shot_events, all_detections, \
        player_tracker, ball_tracker, court = _run_pipeline(video_path, fps, total_frames, config)

    # Post-analysis review
    review_kwargs = dict(
        video_path=str(video_path),
        start_frame=0,
        end_frame=total_frames,
        all_detections=all_detections,
        shot_events=shot_events,
        game_events=game_events,
        player_tracks=player_tracker.get_tracked_players() if player_tracker else {},
        frame_tracks=player_tracker.get_frame_tracks() if player_tracker else {},
        court=court if (court is not None and court.available and config.court.overlay) else None,
        ball_positions=ball_tracker._positions,
        fps=fps,
        input_lut=config.video.input_lut,
        max_resolution=config.video.max_resolution,
    )
    if config.review_export:
        config.review_export.mkdir(parents=True, exist_ok=True)
        ClipReview().export(
            export_path=config.review_export / "review.mp4",
            codec=config.video.review_codec, quality=config.video.review_quality,
            **review_kwargs,
        )
    if config.review:
        ClipReview().replay(**review_kwargs)

    return _build_single_output(
        video_path, fps, total_frames, width, height, scenes, game_events,
    )


# ---------------------------------------------------------------------------
# Timeline-aware analysis (new mode)
# ---------------------------------------------------------------------------

def analyze_timeline(
    timeline_json_path: Path,
    config: AnalysisConfig | None = None,
    clip_indices: list[int] | None = None,
    output_path: Path | None = None,
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
        clip_indices: If provided, only process clips at these indices
            (0-based across all tracks). Supports multiple indices.
        output_path: If provided, per-clip JSON files are written to
            a ``clips/`` subdirectory alongside this path.

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

    # Flatten all clips across tracks with their indices for --clip support
    all_clips = []
    for track in timeline_data.get("tracks", []):
        track_idx = track["track_index"]
        for clip in track.get("clips", []):
            all_clips.append((track_idx, clip))

    total_clip_count = len(all_clips)

    if clip_indices is not None:
        selected = []
        for idx in clip_indices:
            if idx < 0 or idx >= total_clip_count:
                logger.warning(
                    "Clip index %d out of range (timeline has %d clips), skipping",
                    idx, total_clip_count,
                )
            else:
                selected.append((idx, all_clips[idx]))
        if not selected:
            return {"error": f"No valid clip indices in {clip_indices} (0-{total_clip_count - 1})"}
        clips_to_process = selected
        logger.info("Processing %d of %d clips: %s", len(selected), total_clip_count, clip_indices)
    else:
        clips_to_process = list(enumerate(all_clips))

    # Set up per-clip output directory
    clips_dir = None
    if output_path is not None:
        clips_dir = Path(output_path).parent / "clips"
        clips_dir.mkdir(parents=True, exist_ok=True)

    all_events: list[GameEvent] = []
    all_scenes: list[Scene] = []
    clips_analyzed = 0
    clips_skipped = 0

    # Audio pre-pass: extract every clip's audio once and derive game-wide
    # crowd-energy bounds, so "loud" means loud for this game rather than
    # for this clip (per-clip min-max made the loudest 2 s of every clip 1.0).
    audio_cache: dict[int, Path] = {}
    crowd_norm: tuple[float, float] | None = None
    if config.audio.crowd_norm == "game":
        energies = []
        for clip_global_idx, (_track_idx, clip) in clips_to_process:
            wav = _extract_clip_audio(clip, tl_fps, config)
            if wav is None:
                continue
            audio_cache[clip_global_idx] = wav
            try:
                energies.append(crowd_band_energy(wav, config.audio)[0])
            except Exception as e:  # noqa: BLE001
                logger.warning("  Crowd energy pre-pass failed for clip %d: %s", clip_global_idx, e)
        crowd_norm = crowd_norm_from_energies(energies, config.audio)
        if crowd_norm:
            logger.info(
                "Game-wide crowd normalisation from %d clips: %.1f dB -> 0, %.1f dB -> 1",
                len(energies), crowd_norm[0], crowd_norm[1],
            )

    for clip_global_idx, (track_idx, clip) in clips_to_process:
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
            "=== Analyzing clip '%s' (track %d, index %d) ===",
            clip_name, track_idx, clip_global_idx,
        )

        clip_events, clip_scenes = _analyze_clip(
            clip_info=clip,
            timeline_fps=tl_fps,
            config=config,
            clip_index=clip_global_idx,
            audio_wav=audio_cache.pop(clip_global_idx, None),
            crowd_norm=crowd_norm,
        )

        all_events.extend(clip_events)
        all_scenes.extend(clip_scenes)
        clips_analyzed += 1

        # Write per-clip JSON immediately
        if clips_dir is not None:
            clip_output = _build_clip_output(
                clip_info=clip,
                clip_index=clip_global_idx,
                events=clip_events,
                scenes=clip_scenes,
                timeline_data=timeline_data,
            )
            clip_path = clips_dir / f"clip_{clip_global_idx}.json"
            save_results(clip_output, clip_path)
            logger.info("Per-clip results saved to %s", clip_path)

    for wav in audio_cache.values():
        wav.unlink(missing_ok=True)

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
    clip_index: int = 0,
    audio_wav: Path | None = None,
    crowd_norm: tuple[float, float] | None = None,
) -> tuple[list[GameEvent], list[Scene]]:
    """Analyze a single clip's source range and map results to timeline frames.

    Args:
        clip_info: Clip dict from the timeline export JSON.
        timeline_fps: The timeline's frame rate.
        config: Analysis settings.
        clip_index: Global clip index (used for review export filenames).
        audio_wav: Pre-extracted, source-range-trimmed WAV (deleted after use).
        crowd_norm: Game-wide crowd-energy bounds from the audio pre-pass.

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
    lut = FrameLUT(config.video.input_lut)
    detector = ObjectDetector(config.video, device=config.device, detect_players=config.tracking.enable_player_tracking)
    ball_tracker = BallTracker(config.tracking, fps=media_fps)
    player_tracker = (
        PlayerTracker(config.tracking, device=config.device, fps=media_fps / config.video.frame_skip)
        if config.tracking.enable_player_tracking else None
    )

    court = _court_tracker_for(config)
    court_every = max(1, int(round(config.court.every_sec * media_fps)))

    cap = cv2.VideoCapture(str(analysis_path))
    if not cap.isOpened():
        logger.error("  Cannot open media file: %s", analysis_path)
        return [], []

    total_source_frames = source_end - source_start

    # Seek to source start frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_start)

    all_detections = []
    hoop_observations: list[HoopObservation] = []
    frame_idx = source_start

    # Set up live preview if requested
    frame_preview = FramePreview() if config.preview else None
    preview_active = config.preview

    while frame_idx < source_end:
        ret, frame = cap.read()
        if not ret:
            break

        relative_frame = frame_idx - source_start
        if relative_frame % config.video.frame_skip == 0:
            frame = lut.apply(frame)
            frame_resized = _downscale(frame, config.video.max_resolution)
            if relative_frame == 0:
                ball_tracker.set_frame_size(frame_resized.shape[1], frame_resized.shape[0])

            fd = detector.detect_frame(frame_resized, frame_idx)
            all_detections.append(fd)
            if court is not None and relative_frame % court_every == 0:
                court.observe(frame_resized, frame_idx)

            # Incremental ball tracking
            ball_pos = ball_tracker.update(fd)

            # Collect hoop observations with bounding boxes
            if fd.hoops:
                best_hoop = max(fd.hoops, key=lambda d: d.confidence)
                hoop_observations.append(HoopObservation(
                    frame_idx=frame_idx,
                    bbox=best_hoop.bbox,
                    center=best_hoop.center,
                    confidence=best_hoop.confidence,
                ))

            # Player tracking samples jersey colour from the frame, so it must
            # see the same (downscaled) frame the bboxes were computed on.
            tracking = player_tracker.update(frame_resized, fd) if player_tracker else None

            # Live preview rendering (use clip-relative indices for display)
            if preview_active and frame_preview is not None:
                preview_active = frame_preview.render(
                    frame=frame_resized,
                    detections=fd,
                    ball_position=ball_pos,
                    tracking=tracking,
                    frame_idx=frame_idx - source_start,
                    total_frames=total_source_frames,
                )

        frame_idx += 1

    cap.release()
    if frame_preview is not None:
        frame_preview.close()

    # Shot detection using incrementally collected data
    _log_hoop_coverage(hoop_observations, len(all_detections), detector.hoop_capable)
    ball_tracker.set_hoop_observations(hoop_observations)
    shot_events = ball_tracker.find_shots()
    shot_events = _rim_roi_refine(
        analysis_path, detector, lut, config, ball_tracker, hoop_observations,
        all_detections, shot_events, media_fps,
    )
    _finish_court(court, ball_tracker, player_tracker, shot_events, config)
    if player_tracker:
        player_tracker.classify_teams()

    # --- Audio analysis on the source range ---
    audio_events = []
    trimmed_path = audio_wav if audio_wav is not None else _extract_clip_audio(clip_info, timeline_fps, config)
    if trimmed_path is not None:
        try:
            crowd_events = analyze_crowd_excitement(trimmed_path, config.audio, norm=crowd_norm)
            whistle_events = detect_whistles(trimmed_path, config.audio)
            impact_events = detect_rim_impacts(trimmed_path, config.audio)
            audio_events = crowd_events + whistle_events + impact_events
            audio_events.sort(key=lambda e: e.start_sec)

            # Shift audio timestamps from clip-relative (0-based from the
            # trimmed WAV) to source-relative so they align with shot event
            # frame numbers for audio-video correlation.
            audio_offset_sec = source_start / media_fps
            for ae in audio_events:
                ae.start_sec += audio_offset_sec
                ae.end_sec += audio_offset_sec
        except Exception as e:  # noqa: BLE001
            logger.warning("  Audio analysis failed for clip: %s", e)
        finally:
            trimmed_path.unlink(missing_ok=True)

    # --- Scene detection on the source range ---
    scenes = []
    try:
        scenes = detect_scenes(analysis_path, config.scene, start_frame=source_start, end_frame=source_end)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Scene detection failed for clip: %s", e)

    # --- Event classification ---
    classifier = EventClassifier(config.events, fps=media_fps)
    game_events = classifier.classify(shot_events, audio_events, scenes)

    # --- Post-analysis review replay (before frame mapping) ---
    review_kwargs = dict(
        video_path=str(analysis_path),
        start_frame=source_start,
        end_frame=source_end,
        all_detections=all_detections,
        shot_events=shot_events,
        game_events=game_events,
        player_tracks=player_tracker.get_tracked_players() if player_tracker else {},
        frame_tracks=player_tracker.get_frame_tracks() if player_tracker else {},
        court=court if (court is not None and court.available and config.court.overlay) else None,
        ball_positions=ball_tracker._positions,
        fps=media_fps,
        input_lut=config.video.input_lut,
        max_resolution=config.video.max_resolution,
    )
    if config.review_export:
        config.review_export.mkdir(parents=True, exist_ok=True)
        ClipReview().export(
            export_path=config.review_export / f"clip_{clip_index}.mp4",
            codec=config.video.review_codec, quality=config.video.review_quality,
            **review_kwargs,
        )
    if config.review:
        ClipReview().replay(**review_kwargs)

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
# Shared helpers
# ---------------------------------------------------------------------------

def _extract_clip_audio(clip_info: dict, timeline_fps: float, config: AnalysisConfig) -> Path | None:
    """Extract the clip's audio and trim it to the source range. Returns a temp WAV."""
    analysis_path = Path(clip_info.get("analysis_path") or clip_info["file_path"])
    source_start = clip_info["source_start_frame"]
    source_end = clip_info["source_end_frame"]
    media_fps = clip_info.get("media_fps", timeline_fps)
    try:
        audio_path = extract_audio(analysis_path, sample_rate=config.audio.sample_rate)
    except Exception as e:  # noqa: BLE001
        logger.warning("  Audio extraction failed for %s: %s", analysis_path.name, e)
        return None
    try:
        import librosa
        import soundfile as sf

        y, sr = librosa.load(str(audio_path), sr=config.audio.sample_rate)
        start_sample = int((source_start / media_fps) * sr)
        end_sample = int((source_end / media_fps) * sr)
        trimmed_path = Path(tempfile.mktemp(suffix=".wav"))
        sf.write(str(trimmed_path), y[start_sample:end_sample], sr)
        return trimmed_path
    except Exception as e:  # noqa: BLE001
        logger.warning("  Audio trim failed for %s: %s", analysis_path.name, e)
        return None
    finally:
        audio_path.unlink(missing_ok=True)


def _downscale(frame, max_resolution: int):
    """Shrink a frame so its long edge is at most ``max_resolution`` px."""
    h, w = frame.shape[:2]
    if max(h, w) > max_resolution:
        scale = max_resolution / max(h, w)
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame


def _rim_roi_refine(
    video_path: Path,
    detector: ObjectDetector,
    lut: FrameLUT,
    config: AnalysisConfig,
    ball_tracker: BallTracker,
    hoop_observations: list[HoopObservation],
    all_detections: list,
    shot_events: list,
    fps: float,
) -> list:
    """Re-detect the ball at full frame rate around the rim for each shot, then re-run shot detection.

    Extra ball/hoop boxes from the crop pass are merged into the tracker and
    the detection list (so the review overlay shows them), the hoop
    observation list is densified, and the arcs are re-evaluated with the
    richer trajectory.
    """
    tcfg = config.tracking
    if not tcfg.rim_roi_redetect or not shot_events:
        return shot_events
    windows = []
    for sh in shot_events:
        rim = sh.hoop_bbox
        if rim is None:
            continue
        start = int((sh.peak_frame if sh.peak_frame is not None else sh.start_frame) - tcfg.rim_roi_pre_sec * fps)
        end = int(sh.end_frame + tcfg.rim_roi_post_sec * fps) + 1
        windows.append(RimWindow(start_frame=max(0, start), end_frame=end, rim_bbox=rim))
    if not windows:
        return shot_events

    extra = RimRoiRedetector(detector, lut, config.video.max_resolution, tcfg).run(video_path, windows)
    if not extra:
        return shot_events

    by_idx = {fd.frame_idx: fd for fd in all_detections}
    for fd in extra:
        if fd.hoops:
            best = max(fd.hoops, key=lambda d: d.confidence)
            hoop_observations.append(HoopObservation(
                frame_idx=fd.frame_idx, bbox=best.bbox, center=best.center, confidence=best.confidence,
            ))
        existing = by_idx.get(fd.frame_idx)
        if existing is None:
            all_detections.append(fd)
            by_idx[fd.frame_idx] = fd
        else:
            existing.balls.extend(fd.balls)
            existing.balls_in_basket.extend(fd.balls_in_basket)
            if not existing.hoops:
                existing.hoops.extend(fd.hoops)
    all_detections.sort(key=lambda fd: fd.frame_idx)
    hoop_observations.sort(key=lambda o: o.frame_idx)

    ball_tracker.add_frames(extra)
    ball_tracker.set_hoop_observations(hoop_observations)
    refined = ball_tracker.find_shots()
    logger.info("  Shots after rim ROI pass: %d (was %d)", len(refined), len(shot_events))
    return refined


def _court_tracker_for(config: AnalysisConfig) -> CourtTracker | None:
    """Build the court tracker (keypoint model + preset), or None if disabled/unavailable."""
    if not config.court.enabled:
        return None
    try:
        model = load_court_model(config.court.model_id)
        return CourtTracker(CourtModel(config.court.preset), config.court, model=model)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Court detection disabled: %s", exc)
        return None


def _finish_court(
    court: CourtTracker | None,
    ball_tracker: BallTracker,
    player_tracker: PlayerTracker | None,
    shot_events: list,
    config: AnalysisConfig,
) -> None:
    """Use the court homography to flag off-court people and place each shot's release point."""
    if court is None or not court.available:
        if court is not None:
            logger.warning("  Court keypoints never fitted a homography in this clip")
        return
    court.shift_at = ball_tracker.camera_shift_at
    ccfg = config.court
    logger.info("  Court homography: %d keyframes, mean reproj %.1f px, mean landmark spread %.2f",
                len(court.keyframes), float(np.mean([k.reproj_px for k in court.keyframes])),
                float(np.mean([k.spread for k in court.keyframes])))

    frame_tracks = player_tracker.get_frame_tracks() if player_tracker else {}
    tracks = player_tracker.get_tracked_players() if player_tracker else {}

    # Off-court people: majority vote over sampled foot points
    votes: dict[int, list[bool]] = {}
    for frame_idx, entries in frame_tracks.items():
        for tid, (x1, y1, x2, y2) in entries:
            pt = court.to_court(np.array([[(x1 + x2) / 2.0, float(y2)]]), frame_idx)
            if pt is None or np.isnan(pt[0][0]):
                continue  # not inside the calibrated area of any keyframe: unknown, not off
            votes.setdefault(tid, []).append(court.court.on_court(pt[0][0], pt[0][1], ccfg.court_margin_cm))
    off = 0
    for tid, v in votes.items():
        if tid in tracks and len(v) >= 3:
            tracks[tid].on_court = sum(v) / len(v) >= 0.5
            off += tracks[tid].on_court is False
    if votes:
        logger.info("  Court filter: %d of %d tracked people (in the calibrated area) are off the court", off, len(votes))
    zoned = 0

    # Shooter and release position per shot
    for sh in shot_events:
        if not sh.ball_positions:
            continue
        release = sh.ball_positions[0]
        best, best_d = None, float("inf")
        for f in range(release.frame_idx - 6, release.frame_idx + 7):
            for tid, (x1, y1, x2, y2) in frame_tracks.get(f, []):
                if tracks.get(tid) is not None and tracks[tid].on_court is False:
                    continue
                w = max(1, x2 - x1)
                inside = (x1 - 0.3 * w) <= release.x <= (x2 + 0.3 * w) and (y1 - 0.5 * w) <= release.y <= y2
                d = 0.0 if inside else min(abs(release.x - x1), abs(release.x - x2)) / w + abs(f - release.frame_idx) * 0.05
                if d < best_d:
                    best, best_d = (tid, f, (x1, y1, x2, y2)), d
        if best is None or best_d > 1.5:
            continue
        tid, f, (x1, y1, x2, y2) = best
        pt = court.to_court(np.array([[(x1 + x2) / 2.0, float(y2)]]), f)
        if pt is None or np.isnan(pt[0][0]):
            continue  # shooter stands outside the calibrated area: zone unknown
        cx, cy = float(pt[0][0]), float(pt[0][1])
        if not court.court.on_court(cx, cy, ccfg.court_margin_cm * 2):
            continue
        sh.shooter_track_id = tid
        sh.zone = court.court.zone(cx, cy, ccfg.three_point_margin_cm)
        sh.distance_ft = court.court.distance_cm(cx, cy) / 30.48
        sh.shot_type = "three" if sh.zone == "three" else ("layup" if sh.distance_ft <= ccfg.layup_max_ft else "jumper")
        zoned += 1
    if shot_events:
        logger.info("  Court zones assigned for %d of %d shots", zoned, len(shot_events))


def _log_hoop_coverage(
    hoop_observations: list[HoopObservation], analysed_frames: int, hoop_capable: bool,
) -> None:
    """Report how often the rim was seen — made/miss depends on it."""
    n = len(hoop_observations)
    pct = 100.0 * n / analysed_frames if analysed_frames else 0.0
    logger.info("  Hoop observations: %d of %d analysed frames (%.0f%%)", n, analysed_frames, pct)
    if n == 0 and hoop_capable:
        logger.warning(
            "  No hoop detected in this clip although the detector supports it — "
            "shots will be reported with made=None. Try lowering --hoop-confidence."
        )


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


def _build_clip_output(
    clip_info: dict,
    clip_index: int,
    events: list[GameEvent],
    scenes: list[Scene],
    timeline_data: dict,
) -> dict:
    """Build output for a single clip within timeline mode."""
    tl = timeline_data["timeline"]
    return {
        "analysis_version": "0.2.0",
        "mode": "timeline_clip",
        "clip_index": clip_index,
        "clip_name": clip_info.get("clip_name", "unknown"),
        "source_file": clip_info.get("analysis_path") or clip_info.get("file_path", ""),
        "timeline": {
            "name": tl["name"],
            "fps": tl["fps"],
        },
        "clip_info": {
            "source_start_frame": clip_info.get("source_start_frame"),
            "source_end_frame": clip_info.get("source_end_frame"),
            "timeline_start_frame": clip_info.get("timeline_start_frame"),
            "media_fps": clip_info.get("media_fps"),
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
    video_path: Path, fps: float, total_frames: int, config: AnalysisConfig,
) -> tuple[list[Scene], list, list[GameEvent], list, list, PlayerTracker, BallTracker, CourtTracker | None]:
    """Run the full analysis pipeline on a single video (all frames).

    Returns (scenes, audio_events, game_events, shot_events,
             all_detections, player_tracker, ball_tracker).
    """
    logger.info("=== Phase 1: Scene Detection ===")
    scenes = detect_scenes(video_path, config.scene)

    logger.info("=== Phase 2: Audio Analysis ===")
    audio_events = analyze_audio(video_path, config.audio)

    logger.info("=== Phase 3: Object Detection + Tracking ===")
    detector = ObjectDetector(config.video, device=config.device, detect_players=config.tracking.enable_player_tracking)
    ball_tracker = BallTracker(config.tracking, fps=fps)
    player_tracker = (
        PlayerTracker(config.tracking, device=config.device, fps=fps / config.video.frame_skip)
        if config.tracking.enable_player_tracking else None
    )

    lut = FrameLUT(config.video.input_lut)
    court = _court_tracker_for(config)
    court_every = max(1, int(round(config.court.every_sec * fps)))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    all_detections = []
    hoop_observations: list[HoopObservation] = []
    frame_idx = 0

    frame_preview = FramePreview() if config.preview else None
    preview_active = config.preview

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % config.video.frame_skip == 0:
            frame = lut.apply(frame)
            frame_resized = _downscale(frame, config.video.max_resolution)
            if frame_idx == 0:
                ball_tracker.set_frame_size(frame_resized.shape[1], frame_resized.shape[0])

            fd = detector.detect_frame(frame_resized, frame_idx)
            all_detections.append(fd)
            if court is not None and frame_idx % court_every == 0:
                court.observe(frame_resized, frame_idx)

            # Incremental ball tracking
            ball_pos = ball_tracker.update(fd)

            # Collect hoop observations with bounding boxes
            if fd.hoops:
                best_hoop = max(fd.hoops, key=lambda d: d.confidence)
                hoop_observations.append(HoopObservation(
                    frame_idx=frame_idx,
                    bbox=best_hoop.bbox,
                    center=best_hoop.center,
                    confidence=best_hoop.confidence,
                ))

            # Player tracking (same coordinate space as the bboxes)
            tracking = player_tracker.update(frame_resized, fd) if player_tracker else None

            # Live preview
            if preview_active and frame_preview is not None:
                preview_active = frame_preview.render(
                    frame=frame_resized,
                    detections=fd,
                    ball_position=ball_pos,
                    tracking=tracking,
                    frame_idx=frame_idx,
                    total_frames=total_frames,
                )

        frame_idx += 1

    cap.release()
    if frame_preview is not None:
        frame_preview.close()

    logger.info("=== Phase 4: Shot Detection ===")
    _log_hoop_coverage(hoop_observations, len(all_detections), detector.hoop_capable)
    ball_tracker.set_hoop_observations(hoop_observations)
    shot_events = ball_tracker.find_shots()
    shot_events = _rim_roi_refine(
        video_path, detector, lut, config, ball_tracker, hoop_observations,
        all_detections, shot_events, fps,
    )
    _finish_court(court, ball_tracker, player_tracker, shot_events, config)
    if player_tracker:
        player_tracker.classify_teams()

    logger.info("=== Phase 5: Event Classification ===")
    classifier = EventClassifier(config.events, fps=fps)
    game_events = classifier.classify(shot_events, audio_events, scenes)

    return scenes, audio_events, game_events, shot_events, all_detections, \
        player_tracker, ball_tracker, court
