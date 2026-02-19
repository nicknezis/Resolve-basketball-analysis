"""Import analysis results into DaVinci Resolve as timeline markers.

This script reads a JSON file produced by the standalone analysis engine
and creates color-coded markers on the current Resolve timeline.

Usage (from within Resolve's scripting console or as a standalone script):
    python -m src.resolve.markers path/to/analysis.json

Prerequisites:
    - DaVinci Resolve must be running with a project open
    - A timeline must be active
    - Resolve's scripting API must be accessible (Resolve Studio)
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from src.config import EVENT_LABELS, MARKER_COLORS

logger = logging.getLogger(__name__)


def get_resolve():
    """Connect to the running DaVinci Resolve instance.

    The Resolve scripting API module is dynamically loaded because it is
    only available when Resolve is installed and its environment variables
    are set.

    Returns:
        The Resolve API object, or None if connection fails.
    """
    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        return resolve
    except ImportError:
        # Try the fusionscript fallback path
        pass

    # Resolve typically adds its script module to one of these paths
    import importlib.util
    search_paths = [
        "/opt/resolve/Developer/Scripting/Modules",
        "/opt/resolve/libs/Fusion/",
        "C:\\ProgramData\\Blackmagic Design\\DaVinci Resolve\\Support\\Developer\\Scripting\\Modules",
        "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules",
    ]

    for path in search_paths:
        module_path = Path(path) / "DaVinciResolveScript.py"
        if module_path.exists():
            spec = importlib.util.spec_from_file_location("DaVinciResolveScript", module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            resolve = module.scriptapp("Resolve")
            return resolve

    return None


def load_analysis(json_path: Path) -> dict:
    """Load analysis results from JSON file."""
    with open(json_path) as f:
        return json.load(f)


def import_markers(
    json_path: Path,
    clear_existing: bool = False,
    min_confidence: float = 0.0,
) -> dict:
    """Import analysis JSON as markers on the active Resolve timeline.

    Args:
        json_path: Path to the analysis JSON file.
        clear_existing: If True, remove existing markers before adding new ones.
        min_confidence: Only import events with confidence >= this value.

    Returns:
        Summary dict with counts of imported markers.
    """
    resolve = get_resolve()
    if resolve is None:
        logger.error(
            "Cannot connect to DaVinci Resolve. "
            "Make sure Resolve is running and scripting is enabled."
        )
        return {"error": "Cannot connect to DaVinci Resolve"}

    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        logger.error("No project is currently open in Resolve")
        return {"error": "No project open"}

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        logger.error("No timeline is currently active in Resolve")
        return {"error": "No active timeline"}

    timeline_name = timeline.GetName()
    timeline_fps = float(timeline.GetSetting("timelineFrameRate"))
    logger.info("Target timeline: '%s' at %.2f fps", timeline_name, timeline_fps)

    # Load analysis data
    analysis = load_analysis(json_path)
    events = analysis.get("events", [])
    mode = analysis.get("mode", "single_video")

    # In timeline mode, events already have timeline-relative frame numbers.
    # In single_video mode, we may need to adjust for FPS differences.
    if mode == "timeline":
        source_fps = analysis.get("timeline", {}).get("fps", timeline_fps)
        source_tl_name = analysis.get("timeline", {}).get("name", "")
        if source_tl_name and source_tl_name != timeline_name:
            logger.warning(
                "Analysis was for timeline '%s' but importing into '%s'",
                source_tl_name, timeline_name,
            )
    else:
        source_fps = analysis.get("video_info", {}).get("fps", timeline_fps)

    if clear_existing:
        _clear_markers(timeline)

    # Add markers for each event
    imported = 0
    skipped = 0
    for event in events:
        confidence = event.get("confidence", 0)
        if confidence < min_confidence:
            skipped += 1
            continue

        event_type = event.get("type", "unknown")
        start_frame = event.get("start_frame", 0)
        end_frame = event.get("end_frame", start_frame)

        # Adjust frame numbers if source and timeline FPS differ
        if abs(source_fps - timeline_fps) > 0.1:
            ratio = timeline_fps / source_fps
            start_frame = int(start_frame * ratio)
            end_frame = int(end_frame * ratio)

        duration = max(1, end_frame - start_frame)
        color = MARKER_COLORS.get(event_type, "Cyan")
        label = EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())

        note = f"{label} (confidence: {confidence:.0%})"
        details = event.get("details", {})
        if details:
            detail_parts = [f"{k}: {v}" for k, v in details.items()]
            note += " | " + ", ".join(detail_parts)

        # Store full event data in customData for potential later use
        custom_data = json.dumps({
            "type": event_type,
            "confidence": confidence,
            "video_confidence": event.get("video_confidence", 0),
            "audio_confidence": event.get("audio_confidence", 0),
            "details": details,
        })

        success = timeline.AddMarker(
            start_frame,
            color,
            label,
            note,
            duration,
            custom_data,
        )

        if success:
            imported += 1
            logger.debug("Added %s marker at frame %d", color, start_frame)
        else:
            logger.warning("Failed to add marker at frame %d", start_frame)

    summary = {
        "timeline": timeline_name,
        "imported": imported,
        "skipped": skipped,
        "total_events": len(events),
    }
    logger.info(
        "Imported %d markers (%d skipped) onto timeline '%s'",
        imported, skipped, timeline_name,
    )
    return summary


def _clear_markers(timeline) -> int:
    """Remove all existing markers from timeline. Returns count removed."""
    markers = timeline.GetMarkers()
    if not markers:
        return 0
    count = 0
    for frame_id in list(markers.keys()):
        if timeline.DeleteMarkerAtFrame(frame_id):
            count += 1
    logger.info("Cleared %d existing markers", count)
    return count


def main() -> int:
    """CLI entry point for marker import."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Import basketball analysis results as DaVinci Resolve timeline markers",
    )
    parser.add_argument(
        "json_file",
        type=Path,
        help="Path to analysis JSON file",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing markers before importing",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.0,
        help="Minimum confidence threshold for import (default: 0.0, import all)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.json_file.exists():
        print(f"Error: file not found: {args.json_file}", file=sys.stderr)
        return 1

    result = import_markers(
        args.json_file,
        clear_existing=args.clear,
        min_confidence=args.min_confidence,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    print(f"Imported {result['imported']} markers onto timeline '{result['timeline']}'")
    if result["skipped"]:
        print(f"Skipped {result['skipped']} events below confidence threshold")

    return 0


if __name__ == "__main__":
    sys.exit(main())
