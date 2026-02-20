"""Export timeline structure from DaVinci Resolve to JSON.

Captures the full timeline layout including each clip's timeline position,
source in/out points, associated media file paths (full-res and proxy),
and clip metadata. This JSON is the input to the standalone analysis engine.

Usage (run while Resolve is open with a timeline active):
    python -m src.resolve.export -o my_timeline.json
    python -m src.resolve.export --prefer-proxy -o my_timeline.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def get_resolve():
    """Connect to the running DaVinci Resolve instance.

    Requires DaVinci Resolve to be running with scripting enabled
    (Preferences > General > External scripting using > Local).
    """
    import os
    import sys as _sys

    # Ensure environment variables are set for the native library loader
    if _sys.platform.startswith("darwin"):
        os.environ.setdefault(
            "RESOLVE_SCRIPT_API",
            "/Library/Application Support/Blackmagic Design"
            "/DaVinci Resolve/Developer/Scripting",
        )
        os.environ.setdefault(
            "RESOLVE_SCRIPT_LIB",
            "/Applications/DaVinci Resolve/DaVinci Resolve.app"
            "/Contents/Libraries/Fusion/fusionscript.so",
        )
        _modules_path = (
            "/Library/Application Support/Blackmagic Design"
            "/DaVinci Resolve/Developer/Scripting/Modules"
        )
    elif _sys.platform.startswith("win"):
        os.environ.setdefault(
            "RESOLVE_SCRIPT_API",
            "C:\\ProgramData\\Blackmagic Design\\DaVinci Resolve"
            "\\Support\\Developer\\Scripting",
        )
        os.environ.setdefault(
            "RESOLVE_SCRIPT_LIB",
            "C:\\Program Files\\Blackmagic Design\\DaVinci Resolve"
            "\\fusionscript.dll",
        )
        _modules_path = (
            "C:\\ProgramData\\Blackmagic Design\\DaVinci Resolve"
            "\\Support\\Developer\\Scripting\\Modules"
        )
    else:
        os.environ.setdefault(
            "RESOLVE_SCRIPT_API",
            "/opt/resolve/Developer/Scripting",
        )
        os.environ.setdefault(
            "RESOLVE_SCRIPT_LIB",
            "/opt/resolve/libs/Fusion/fusionscript.so",
        )
        _modules_path = "/opt/resolve/Developer/Scripting/Modules"

    # Add the Resolve scripting modules directory to sys.path so the
    # native fusionscript library is found correctly (importlib file-based
    # loading breaks the module's internal import chain).
    if _modules_path not in _sys.path:
        _sys.path.insert(0, _modules_path)

    try:
        import DaVinciResolveScript as dvr
        resolve = dvr.scriptapp("Resolve")
        if resolve is None:
            logger.error(
                "scriptapp('Resolve') returned None — "
                "is DaVinci Resolve running with scripting enabled?"
            )
        return resolve
    except ImportError:
        logger.error(
            "Could not import DaVinciResolveScript. "
            "Verify that DaVinci Resolve is installed and the scripting "
            "modules exist at: %s",
            _modules_path,
        )
        return None


def _timecode_to_frames(tc: str, fps: float) -> int:
    """Convert a timecode string like '00:01:23:15' to a frame count."""
    if not tc or not isinstance(tc, str):
        return 0
    parts = tc.replace(";", ":").split(":")
    if len(parts) != 4:
        return 0
    h, m, s, f = [int(p) for p in parts]
    return int((h * 3600 + m * 60 + s) * fps + f)


def export_timeline(
    output_path: Path | None = None,
    prefer_proxy: bool = False,
    track_index: int | None = None,
) -> dict:
    """Export the active Resolve timeline structure to a dict.

    Walks every video track (or a specific one) and captures each clip's:
    - Timeline position (start/end frame on the timeline)
    - Source position (which portion of the underlying media file is used)
    - Full-resolution and proxy file paths
    - Clip metadata (fps, resolution, codec, etc.)

    Args:
        output_path: If provided, write JSON to this path.
        prefer_proxy: If True, set the "analysis_path" field to the proxy
            file when available (smaller files = faster analysis).
        track_index: Export only this video track (1-based). If None, export
            all video tracks.

    Returns:
        Timeline export dict.
    """
    resolve = get_resolve()
    if resolve is None:
        return {"error": "Cannot connect to DaVinci Resolve"}

    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        return {"error": "No project open"}

    timeline = project.GetCurrentTimeline()
    if timeline is None:
        return {"error": "No active timeline"}

    timeline_name = timeline.GetName()
    timeline_fps = float(timeline.GetSetting("timelineFrameRate"))
    timeline_start_frame = int(timeline.GetStartFrame())
    track_count = int(timeline.GetTrackCount("video"))

    logger.info(
        "Exporting timeline '%s': %.2f fps, %d video tracks, start frame %d",
        timeline_name, timeline_fps, track_count, timeline_start_frame,
    )

    # Determine which tracks to export
    if track_index is not None:
        track_indices = [track_index]
    else:
        track_indices = list(range(1, track_count + 1))

    tracks = []
    seen_media = {}  # file_path -> media_info dict (dedup)

    for tidx in track_indices:
        track_name = timeline.GetTrackName("video", tidx)
        items = timeline.GetItemListInTrack("video", tidx)

        if not items:
            tracks.append({
                "track_index": tidx,
                "track_name": track_name,
                "clips": [],
            })
            continue

        clips = []
        for item in items:
            clip_info = _extract_clip_info(
                item, timeline_fps, timeline_start_frame, prefer_proxy, seen_media,
            )
            clips.append(clip_info)

        tracks.append({
            "track_index": tidx,
            "track_name": track_name,
            "clips": clips,
        })

        logger.info("  Track %d ('%s'): %d clips", tidx, track_name, len(clips))

    result = {
        "export_version": "1.0",
        "timeline": {
            "name": timeline_name,
            "fps": timeline_fps,
            "start_frame": timeline_start_frame,
            "video_track_count": track_count,
        },
        "tracks": tracks,
        "media_files": list(seen_media.values()),
    }

    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Timeline exported to %s", output_path)

    return result


def _extract_clip_info(
    item,
    timeline_fps: float,
    timeline_start_frame: int,
    prefer_proxy: bool,
    seen_media: dict,
) -> dict:
    """Extract all relevant info from a single TimelineItem."""
    # Timeline position
    tl_start = int(item.GetStart())
    tl_end = int(item.GetEnd())
    tl_duration = int(item.GetDuration())

    # Source offsets (how much extra media is available beyond the trim)
    left_offset = int(item.GetLeftOffset())
    right_offset = int(item.GetRightOffset())

    # Associated media pool item
    mpi = item.GetMediaPoolItem()

    clip_name = ""
    file_path = ""
    proxy_path = ""
    analysis_path = ""
    media_fps = timeline_fps
    source_start_tc = ""
    source_end_tc = ""
    source_start_frame = 0
    source_end_frame = 0
    source_duration_frames = 0
    clip_properties = {}

    if mpi is not None:
        clip_name = mpi.GetClipProperty("Clip Name") or ""
        file_path = mpi.GetClipProperty("File Path") or ""
        proxy_path = mpi.GetClipProperty("Proxy Media Path") or ""

        # Get media FPS (may differ from timeline FPS)
        fps_str = mpi.GetClipProperty("FPS")
        if fps_str:
            try:
                media_fps = float(fps_str)
            except (ValueError, TypeError):
                media_fps = timeline_fps

        # Source in/out from the media pool item
        source_start_tc = mpi.GetClipProperty("Start") or ""
        source_end_tc = mpi.GetClipProperty("End") or ""

        # Calculate the source frame range that this clip uses
        # MediaPoolItem "Start" is the very beginning of the source file
        # The actual in-point on the timeline = source start + left_offset
        media_start_frame = _timecode_to_frames(source_start_tc, media_fps)
        source_start_frame = media_start_frame + left_offset
        source_end_frame = source_start_frame + tl_duration
        source_duration_frames = tl_duration

        # Full source file duration
        source_total_duration = mpi.GetClipProperty("Duration") or ""

        # Choose which file to use for analysis
        if prefer_proxy and proxy_path:
            analysis_path = proxy_path
        elif file_path:
            analysis_path = file_path
        else:
            analysis_path = ""

        # Collect useful metadata
        clip_properties = {
            "duration_tc": source_total_duration,
            "resolution": mpi.GetClipProperty("Resolution") or "",
            "video_codec": mpi.GetClipProperty("Video Codec") or "",
            "audio_codec": mpi.GetClipProperty("Audio Codec") or "",
            "audio_channels": mpi.GetClipProperty("Audio Ch") or "",
        }

        # Track unique media files for the summary
        if file_path and file_path not in seen_media:
            seen_media[file_path] = {
                "file_path": file_path,
                "proxy_path": proxy_path,
                "fps": media_fps,
                "duration_tc": source_total_duration,
                "resolution": clip_properties.get("resolution", ""),
            }

    return {
        "clip_name": clip_name,
        "timeline_start_frame": tl_start,
        "timeline_end_frame": tl_end,
        "timeline_duration_frames": tl_duration,
        "source_start_frame": source_start_frame,
        "source_end_frame": source_end_frame,
        "source_duration_frames": source_duration_frames,
        "left_offset": left_offset,
        "right_offset": right_offset,
        "file_path": file_path,
        "proxy_path": proxy_path,
        "analysis_path": analysis_path,
        "media_fps": media_fps,
        "source_start_tc": source_start_tc,
        "source_end_tc": source_end_tc,
        "properties": clip_properties,
    }


def main() -> int:
    """CLI entry point for timeline export."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Export DaVinci Resolve timeline structure to JSON "
        "for offline basketball analysis",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <timeline_name>_timeline.json)",
    )
    parser.add_argument(
        "--prefer-proxy",
        action="store_true",
        help="Use proxy media paths for analysis when available",
    )
    parser.add_argument(
        "--track",
        type=int,
        default=None,
        help="Export only this video track number (1-based)",
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

    result = export_timeline(
        output_path=None,  # we'll write after getting the name
        prefer_proxy=args.prefer_proxy,
        track_index=args.track,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    # Determine output path
    output_path = args.output
    if output_path is None:
        tl_name = result["timeline"]["name"].replace(" ", "_")
        output_path = Path(f"{tl_name}_timeline.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    # Print summary
    tl = result["timeline"]
    tracks = result["tracks"]
    total_clips = sum(len(t["clips"]) for t in tracks)
    media_count = len(result["media_files"])
    proxy_count = sum(1 for m in result["media_files"] if m.get("proxy_path"))

    print(f"Timeline: '{tl['name']}' ({tl['fps']} fps)")
    print(f"  Tracks exported: {len(tracks)}")
    print(f"  Total clips: {total_clips}")
    print(f"  Unique media files: {media_count}")
    print(f"  Media with proxies: {proxy_count}")
    print()
    print(f"Exported to: {output_path}")
    print()
    print("Next step — analyze the timeline:")
    print(f"  basketball-analyze --timeline {output_path} -o analysis.json")

    return 0


if __name__ == "__main__":
    sys.exit(main())
