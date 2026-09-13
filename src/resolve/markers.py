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
    """Connect to the running DaVinci Resolve instance."""
    from src.resolve.export import get_resolve as _get_resolve
    return _get_resolve()


def load_analysis(json_path: Path) -> dict:
    """Load analysis results from JSON file."""
    with open(json_path) as f:
        return json.load(f)

# Human phrasing for how a shot verdict was reached
_VIA_PHRASE = {
    "through_net": "through net",
    "rim_entry": "into net",
    "ball_in_basket": "ball in basket",
    "polygon": "in net zone",
    "proximity": "near rim",
    "rim_out": "rim out",
    "passed_rim": "missed, passed rim",
    "short": "did not reach rim",
}


def marker_note(event: dict) -> str:
    """Short, readable marker note; the full event stays in customData.

    Examples: ``Made Shot 97% · through net · 20 ft two · rim impact · C20260912_6415``
    ``Shot Attempt 88% · rim out · C20260912_6416``  ``Crowd Reaction 92% · 4.5 s``
    """
    event_type = event.get("type", "unknown")
    label = EVENT_LABELS.get(event_type, event_type.replace("_", " ").title())
    conf = event.get("confidence", 0.0)
    d = event.get("details", {}) or {}
    parts = [f"{label} {conf:.0%}"]

    if event_type in ("made_shot", "shot_attempt", "three_pointer"):
        via = d.get("made_via")
        if via:
            parts.append(_VIA_PHRASE.get(via, via.replace("_", " ")))
        elif d.get("made") is None:
            parts.append("no rim in view")
        if d.get("shot_distance_ft") is not None:
            zone = d.get("shot_zone") or ""
            parts.append(f"{d['shot_distance_ft']:.0f} ft {zone}".strip())
        if d.get("rim_impact") is not None:
            parts.append("rim impact (audio)")
    elif event_type == "crowd_excitement":
        dur = float(event.get("end_sec", 0.0)) - float(event.get("start_sec", 0.0))
        if dur > 0:
            parts.append(f"{dur:.1f} s")

    clip = d.get("source_clip")
    if clip:
        parts.append(Path(str(clip)).stem)
    return " · ".join(parts)


def event_passes(event: dict, min_confidence: float, crowd_min_confidence: float) -> bool:
    """Per-type confidence gate: crowd reactions get their own, usually stricter, floor."""
    conf = float(event.get("confidence", 0.0))
    if event.get("type") == "crowd_excitement":
        return conf >= max(min_confidence, crowd_min_confidence)
    return conf >= min_confidence


def plan_markers(
    events: list[dict],
    timeline_fps: float,
    source_fps: float,
    min_confidence: float = 0.0,
    crowd_min_confidence: float = 0.0,
    max_nudge_frames: int = 5,
) -> tuple[list[dict], dict[str, int]]:
    """Turn events into marker specs, resolving Resolve's one-marker-per-frame rule.

    Returns ``(markers, skipped_by_type)``.  Each marker dict has
    ``frame, duration, color, name, note, custom_data`` and, when its start
    had to move off an occupied frame, ``nudged_from``.
    """
    ratio = timeline_fps / source_fps if abs(source_fps - timeline_fps) > 0.1 and source_fps > 0 else 1.0
    used: set[int] = set()
    markers: list[dict] = []
    skipped: dict[str, int] = {}

    # Shots first so a crowd event never bumps a shot off its frame
    order = sorted(events, key=lambda e: (e.get("type") == "crowd_excitement", e.get("start_frame", 0)))
    for event in order:
        event_type = event.get("type", "unknown")
        if not event_passes(event, min_confidence, crowd_min_confidence):
            skipped[event_type] = skipped.get(event_type, 0) + 1
            continue
        start = int(event.get("start_frame", 0) * ratio)
        end = int(event.get("end_frame", start) * ratio)
        frame = start
        while frame in used and frame - start < max_nudge_frames:
            frame += 1
        if frame in used:
            logger.warning("No free frame near %d for a %s marker; skipping", start, event_type)
            skipped[event_type] = skipped.get(event_type, 0) + 1
            continue
        used.add(frame)
        spec = {
            "frame": frame,
            "duration": max(1, end - frame),
            "color": MARKER_COLORS.get(event_type, "Cyan"),
            "name": EVENT_LABELS.get(event_type, event_type.replace("_", " ").title()),
            "note": marker_note(event),
            "custom_data": json.dumps({
                "type": event_type,
                "confidence": event.get("confidence", 0),
                "video_confidence": event.get("video_confidence", 0),
                "audio_confidence": event.get("audio_confidence", 0),
                "start_frame": event.get("start_frame"),
                "end_frame": event.get("end_frame"),
                "details": event.get("details", {}),
            }),
        }
        if frame != start:
            spec["nudged_from"] = start
        markers.append(spec)
    markers.sort(key=lambda m: m["frame"])
    return markers, skipped


def import_markers(
    json_path: Path,
    clear_existing: bool = False,
    min_confidence: float = 0.0,
    crowd_min_confidence: float = 0.85,
) -> dict:
    """Import analysis JSON as markers on the active Resolve timeline.

    Args:
        json_path: Path to the analysis JSON file.
        clear_existing: If True, remove existing markers before adding new ones.
        min_confidence: Only import events with confidence >= this value.
        crowd_min_confidence: Separate floor for crowd-reaction markers, which
            are far more numerous than shots.

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

    markers, skipped_by_type = plan_markers(
        events, timeline_fps, source_fps,
        min_confidence=min_confidence, crowd_min_confidence=crowd_min_confidence,
    )

    imported = 0
    failed = 0
    for m in markers:
        if "nudged_from" in m:
            logger.info("Marker at frame %d moved to %d (frame already had a marker)", m["nudged_from"], m["frame"])
        if timeline.AddMarker(m["frame"], m["color"], m["name"], m["note"], m["duration"], m["custom_data"]):
            imported += 1
            logger.debug("Added %s marker at frame %d: %s", m["color"], m["frame"], m["note"])
        else:
            failed += 1
            logger.warning("Failed to add %s marker at frame %d", m["color"], m["frame"])

    skipped = sum(skipped_by_type.values())
    summary = {
        "timeline": timeline_name,
        "imported": imported,
        "failed": failed,
        "skipped": skipped,
        "skipped_by_type": skipped_by_type,
        "total_events": len(events),
    }
    logger.info(
        "Imported %d markers (%d skipped, %d failed) onto timeline '%s'",
        imported, skipped, failed, timeline_name,
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
        help="Minimum confidence for any marker (default: 0.0 — the analysis already "
        "filtered shots at its own threshold)",
    )
    parser.add_argument(
        "--crowd-min-confidence",
        type=float,
        default=0.85,
        help="Minimum confidence for crowd-reaction markers (default: 0.85). They are "
        "far more numerous than shots; raise to thin them, 0 to import all.",
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
        crowd_min_confidence=args.crowd_min_confidence,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    print(f"Imported {result['imported']} markers onto timeline '{result['timeline']}'")
    if result["skipped"]:
        by_type = ", ".join(f"{k}: {v}" for k, v in sorted(result["skipped_by_type"].items()))
        print(f"Skipped {result['skipped']} events below the confidence thresholds ({by_type})")
    if result.get("failed"):
        print(f"Resolve rejected {result['failed']} markers (see log)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
