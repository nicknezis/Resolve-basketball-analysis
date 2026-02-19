"""Command-line interface for standalone basketball video analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.video_analyzer import analyze_video, save_results
from src.config import AnalysisConfig, AudioConfig, EventConfig, VideoConfig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="basketball-analyze",
        description="Analyze basketball game video to detect interesting plays. "
        "Produces a JSON file that can be imported into DaVinci Resolve "
        "to annotate the timeline with markers.",
    )
    parser.add_argument(
        "video",
        type=Path,
        help="Path to basketball game video file",
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

    # Validate input
    if not args.video.exists():
        print(f"Error: video file not found: {args.video}", file=sys.stderr)
        return 1

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
        output_path = args.video.parent / f"{args.video.stem}_analysis.json"

    # Run analysis
    print(f"Analyzing: {args.video}")
    print(f"Output:    {output_path}")
    print()

    results = analyze_video(args.video, config)
    save_results(results, output_path)

    # Print summary
    summary = results.get("summary", {})
    print()
    print("=== Analysis Complete ===")
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
