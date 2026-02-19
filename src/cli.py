"""Command-line interface for standalone basketball video analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.video_analyzer import analyze_timeline, analyze_video, save_results
from src.config import AnalysisConfig, AudioConfig, EventConfig, VideoConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="basketball-analyze",
        description="Analyze basketball game video to detect interesting plays. "
        "Produces a JSON file that can be imported into DaVinci Resolve "
        "to annotate the timeline with markers.\n\n"
        "Two input modes:\n"
        "  1. Single video:  basketball-analyze game.mp4\n"
        "  2. Timeline JSON: basketball-analyze --timeline timeline.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        type=Path,
        nargs="?",
        default=None,
        help="Path to basketball game video file (single-video mode)",
    )
    parser.add_argument(
        "--timeline",
        type=Path,
        default=None,
        help="Path to a Resolve timeline export JSON (timeline mode). "
        "Generate this with: python -m src.resolve.export",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: <video_name>_analysis.json)",
    )
    parser.add_argument(
        "--yolo-model",
        default="yolov8m.pt",
        help="YOLO model file or name (default: yolov8m.pt)",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.5,
        help="YOLO detection confidence threshold (default: 0.5)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Analyze every Nth frame (default: 2)",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum event confidence to include in output (default: 0.7)",
    )
    parser.add_argument(
        "--crowd-threshold",
        type=float,
        default=0.6,
        help="Crowd excitement detection threshold (default: 0.6)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Validate input — need exactly one of video or --timeline
    if args.video and args.timeline:
        print("Error: provide either a video file or --timeline, not both", file=sys.stderr)
        return 1
    if not args.video and not args.timeline:
        print("Error: provide a video file or --timeline <json>", file=sys.stderr)
        parser.print_usage(sys.stderr)
        return 1

    input_path = args.timeline or args.video
    if not input_path.exists():
        print(f"Error: file not found: {input_path}", file=sys.stderr)
        return 1

    is_timeline_mode = args.timeline is not None

    # Build config from CLI args
    config = AnalysisConfig(
        video=VideoConfig(
            yolo_model=args.yolo_model,
            yolo_confidence=args.yolo_confidence,
            frame_skip=args.frame_skip,
        ),
        audio=AudioConfig(
            excitement_threshold=args.crowd_threshold,
        ),
        events=EventConfig(
            min_confidence=args.min_confidence,
        ),
        device=args.device,
    )

    # Determine output path
    output_path = args.output
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_analysis.json"

    # Run analysis
    if is_timeline_mode:
        print(f"Analyzing timeline: {input_path}")
    else:
        print(f"Analyzing video: {input_path}")
    print(f"Output: {output_path}")
    print()

    if is_timeline_mode:
        results = analyze_timeline(input_path, config)
    else:
        results = analyze_video(input_path, config)

    save_results(results, output_path)

    # Print summary
    summary = results.get("summary", {})
    print()
    print("=== Analysis Complete ===")
    if is_timeline_mode:
        clips = results.get("clips_summary", {})
        print(f"  Clips analyzed: {clips.get('clips_analyzed', 0)}")
        print(f"  Clips skipped:  {clips.get('clips_skipped', 0)}")
    print(f"  Total events: {summary.get('total_events', 0)}")
    print(f"  Total scenes: {summary.get('total_scenes', 0)}")
    counts = summary.get("event_counts", {})
    if counts:
        print("  Event breakdown:")
        for event_type, count in sorted(counts.items()):
            print(f"    {event_type}: {count}")
    print()
    print(f"Results saved to: {output_path}")
    print()
    print("To import into DaVinci Resolve, run:")
    print(f"  python -m src.resolve.markers {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
