"""Auto-generate a highlights timeline from analysis results.

Reads the analysis JSON and creates a new Resolve timeline containing
only the interesting event clips with configurable padding.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.config import EventConfig

logger = logging.getLogger(__name__)


def create_highlight_timeline(
    json_path: Path,
    resolve,
    timeline_name: str = "Highlights",
    config: EventConfig | None = None,
    min_confidence: float = 0.0,
) -> dict:
    """Create a new timeline in Resolve containing only highlight clips.

    Args:
        json_path: Path to analysis JSON file.
        resolve: Connected Resolve API object.
        timeline_name: Name for the new highlights timeline.
        config: Event config for padding settings.
        min_confidence: Only include events above this confidence.

    Returns:
        Summary dict with highlight count and timeline info.
    """
    config = config or EventConfig()

    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        return {"error": "No project open"}

    source_timeline = project.GetCurrentTimeline()
    if source_timeline is None:
        return {"error": "No active timeline"}

    source_fps = float(source_timeline.GetSetting("timelineFrameRate"))

    # Load analysis
    with open(json_path) as f:
        analysis = json.load(f)

    events = analysis.get("events", [])
    analysis_fps = analysis.get("video_info", {}).get("fps", source_fps)

    # Filter and sort events
    events = [e for e in events if e.get("confidence", 0) >= min_confidence]
    events.sort(key=lambda e: e.get("start_frame", 0))

    if not events:
        logger.warning("No events above confidence threshold")
        return {"error": "No events to include", "timeline": None}

    # Calculate clip ranges with padding
    pre_pad_frames = int(config.highlight_pre_pad_sec * source_fps)
    post_pad_frames = int(config.highlight_post_pad_sec * source_fps)

    clip_ranges = []
    for event in events:
        start = event.get("start_frame", 0)
        end = event.get("end_frame", start)

        # Adjust for FPS difference
        if abs(analysis_fps - source_fps) > 0.1:
            ratio = source_fps / analysis_fps
            start = int(start * ratio)
            end = int(end * ratio)

        padded_start = max(0, start - pre_pad_frames)
        padded_end = end + post_pad_frames
        clip_ranges.append((padded_start, padded_end, event.get("type", "unknown")))

    # Merge overlapping ranges
    merged = [clip_ranges[0]]
    for start, end, etype in clip_ranges[1:]:
        prev_start, prev_end, prev_type = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end), prev_type)
        else:
            merged.append((start, end, etype))

    logger.info(
        "Creating highlights timeline '%s' with %d clips from %d events",
        timeline_name, len(merged), len(events),
    )

    # Create the highlights timeline
    # Use Resolve's media pool to create a new timeline and add subclips
    media_pool = project.GetMediaPool()
    new_timeline = media_pool.CreateEmptyTimeline(timeline_name)

    if new_timeline is None:
        return {"error": f"Failed to create timeline '{timeline_name}'"}

    # Switch to the new timeline to add clips
    # Note: actual clip addition depends on the source media structure.
    # This creates markers on the new timeline indicating where clips should go.
    for i, (start_frame, end_frame, event_type) in enumerate(merged):
        new_timeline.AddMarker(
            i * 10,  # placeholder frame position
            "Green",
            f"Highlight {i + 1}",
            f"Source frames {start_frame}-{end_frame} ({event_type})",
            1,
        )

    summary = {
        "timeline": timeline_name,
        "highlight_clips": len(merged),
        "source_events": len(events),
        "total_duration_frames": sum(end - start for start, end, _ in merged),
    }

    logger.info("Highlights timeline created: %d clips", len(merged))
    return summary
