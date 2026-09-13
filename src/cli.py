"""Command-line interface for standalone basketball video analysis."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.analysis.video_analyzer import analyze_timeline, analyze_video, save_results
from src.config import AnalysisConfig, AudioConfig, EventConfig, TrackingConfig, VideoConfig


def _parse_clip_indices(clip_str: str) -> list[int]:
    """Parse a clip index specification into a sorted deduplicated list.

    Supports single indices, comma-separated lists, and ranges:
      "3"       -> [3]
      "0,2,5"   -> [0, 2, 5]
      "1-4"     -> [1, 2, 3, 4]
      "0,3-5,8" -> [0, 3, 4, 5, 8]
    """
    indices: set[int] = set()
    for part in clip_str.split(","):
        part = part.strip()
        if "-" in part:
            start_str, end_str = part.split("-", 1)
            start, end = int(start_str.strip()), int(end_str.strip())
            if end < start:
                raise ValueError(f"Invalid range: {part}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(part))
    return sorted(indices)


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
        default="yolo11m.pt",
        help="YOLO model file or name (default: yolo11m.pt)",
    )
    parser.add_argument(
        "--yolo-confidence",
        type=float,
        default=0.5,
        help="Detection confidence threshold for ball/player boxes (default: 0.5)",
    )
    parser.add_argument(
        "--hoop-confidence",
        type=float,
        default=0.3,
        help="Confidence threshold for hoop/rim boxes, applied to every backend "
        "(default: 0.3 — rims are small and benefit from a looser floor)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="YOLO inference resolution on the long edge (default: 1280). "
        "Ultralytics' own default of 640 shrinks the ball to ~10 px.",
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
        help="Minimum fused (video+audio) confidence to include an event (default: 0.7)",
    )
    parser.add_argument(
        "--min-video-confidence",
        type=float,
        default=0.5,
        help="Minimum video-only confidence for a shot to be considered (default: 0.5)",
    )
    parser.add_argument(
        "--crowd-threshold",
        type=float,
        default=0.6,
        help="Crowd excitement detection threshold (default: 0.6)",
    )
    parser.add_argument(
        "--clip",
        type=str,
        default=None,
        metavar="SPEC",
        help="In timeline mode, process only the specified clips (0-based). "
        "Supports: single (3), list (0,2,5), range (1-4), mixed (0,3-5,8).",
    )
    parser.add_argument(
        "--input-lut",
        type=Path,
        default=None,
        metavar="PATH",
        help="Path to a .cube 3D LUT file (or .zip/.lut archive containing one) "
        "for converting log footage to Rec.709. Example: s-gamut3.cine-slog3.lut",
    )
    parser.add_argument(
        "--roboflow-model",
        default=None,
        metavar="MODEL_ID",
        help="Roboflow model ID (e.g. basketball-player-detection-3-ycjdo/6). "
        "With --detector roboflow it is the primary detector; with any other "
        "backend it supplements hoop/ball detection. Runs locally via the "
        "'inference' package; requires ROBOFLOW_API_KEY.",
    )
    parser.add_argument(
        "--roboflow-confidence",
        type=float,
        default=0.4,
        help="Confidence threshold for Roboflow ball/player boxes (default: 0.4)",
    )
    parser.add_argument(
        "--detector",
        choices=["yolo", "rfdetr", "roboflow"],
        default="yolo",
        help="Detection model backend (default: yolo). 'roboflow' needs --roboflow-model.",
    )
    parser.add_argument(
        "--rfdetr-size",
        choices=["nano", "small", "medium", "large", "xlarge", "2xlarge"],
        default="small",
        help="RF-DETR model size (default: small). Only used with --detector rfdetr.",
    )
    parser.add_argument(
        "--rfdetr-weights",
        type=str,
        default=None,
        help="Path to fine-tuned RF-DETR checkpoint (.pth file)",
    )
    parser.add_argument(
        "--rfdetr-classes",
        type=str,
        default=None,
        help="Comma-separated class names for fine-tuned RF-DETR model "
        "(e.g., 'ball,ball-in-basket,number,player,...,referee,rim')",
    )
    parser.add_argument(
        "--rfdetr-resolution",
        type=int,
        default=None,
        help="RF-DETR input resolution (must be divisible by 56)",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Enable NMS deduplication (requires supervision library)",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for NMS deduplication (default: 0.5)",
    )
    parser.add_argument(
        "--consensus",
        type=int,
        default=3,
        help="Multi-frame consensus: min detections to confirm ball tracking "
        "(1=disabled; default: 3)",
    )
    parser.add_argument(
        "--polygon-zone",
        action="store_true",
        help="Also accept a made shot when the descent enters a trapezoidal net "
        "zone below the rim (requires supervision)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Compute device (default: auto)",
    )
    parser.add_argument(
        "--no-players",
        action="store_true",
        help="Disable player detection and tracking (faster analysis)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show live detection preview during analysis (per-frame)",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Replay each clip with full analysis results after processing",
    )
    parser.add_argument(
        "--review-export",
        type=Path,
        default=None,
        metavar="DIR",
        help="Save annotated review videos to DIR without opening a window. "
        "Files are named clip_<N>.mp4 (timeline mode) or review.mp4 (single video).",
    )
    parser.add_argument(
        "--review-codec",
        choices=["hevc", "h264", "mp4v"],
        default="hevc",
        help="Codec for --review-export (default: hevc). hevc/h264 use ffmpeg "
        "(VideoToolbox on macOS when available); mp4v is the OpenCV fallback.",
    )
    parser.add_argument(
        "--review-quality",
        type=int,
        default=None,
        metavar="N",
        help="Encoder quality for --review-export: CRF for libx264/libx265 "
        "(default 18/20), kbit/s for hardware encoders (default ~12000/7000 "
        "per megapixel for h264/hevc).",
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

    # Parse clip indices
    clip_indices = None
    if args.clip is not None:
        if not is_timeline_mode:
            print("Error: --clip can only be used with --timeline", file=sys.stderr)
            return 1
        try:
            clip_indices = _parse_clip_indices(args.clip)
        except ValueError as e:
            print(f"Error: invalid --clip value: {e}", file=sys.stderr)
            return 1

    # Build config from CLI args
    rfdetr_classes = (
        [c.strip() for c in args.rfdetr_classes.split(",") if c.strip()]
        if args.rfdetr_classes
        else None
    ) or None
    if args.detector == "roboflow" and not args.roboflow_model:
        print("Error: --detector roboflow requires --roboflow-model", file=sys.stderr)
        return 1
    if args.polygon_zone:
        try:
            import supervision  # noqa: F401
        except ImportError:
            print("Error: --polygon-zone requires the 'supervision' package", file=sys.stderr)
            return 1

    config = AnalysisConfig(
        video=VideoConfig(
            yolo_model=args.yolo_model,
            yolo_confidence=args.yolo_confidence,
            imgsz=args.imgsz,
            hoop_confidence=args.hoop_confidence,
            frame_skip=args.frame_skip,
            input_lut=args.input_lut,
            roboflow_model_id=args.roboflow_model,
            roboflow_confidence=args.roboflow_confidence,
            detector_backend=args.detector,
            rfdetr_model_size=args.rfdetr_size,
            rfdetr_weights=args.rfdetr_weights,
            rfdetr_class_names=rfdetr_classes,
            rfdetr_num_classes=len(rfdetr_classes) if rfdetr_classes else None,
            rfdetr_resolution=args.rfdetr_resolution,
            nms_enabled=args.nms,
            nms_threshold=args.nms_threshold,
            review_codec=args.review_codec,
            review_quality=args.review_quality,
        ),
        audio=AudioConfig(
            excitement_threshold=args.crowd_threshold,
        ),
        tracking=TrackingConfig(
            enable_player_tracking=not args.no_players,
            consensus_required=args.consensus,
            use_polygon_zone=args.polygon_zone,
        ),
        events=EventConfig(
            min_confidence=args.min_confidence,
            min_video_confidence=args.min_video_confidence,
        ),
        device=args.device,
        preview=args.preview,
        review=args.review,
        review_export=args.review_export,
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
        results = analyze_timeline(
            input_path, config, clip_indices=clip_indices, output_path=output_path,
        )
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
