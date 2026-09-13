"""Bake-off: how well do candidate detectors find the hoop and ball in *our* footage?

Samples frames from timeline clips (LUT applied, downscaled exactly as the
analyzer does) and runs each candidate model over them, reporting the share
of frames with a hoop / ball detection, mean confidences, and speed.

Usage:
    python scripts/bench_hoop_models.py --timeline huskies_260912_timeline.json \
        --clip 0-4 --input-lut lut/FX30.cube --out docs/model-bakeoff.md

Candidates are Roboflow ``workspace/project`` slugs (latest trained version is
resolved through the Roboflow API) or explicit ``project/version`` model IDs,
plus the stock YOLO baseline at two inference sizes.  Requires
``ROBOFLOW_API_KEY`` for the Roboflow entries.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.color import FrameLUT  # noqa: E402
from src.analysis.object_detector import (  # noqa: E402
    ROLE_BALL, ROLE_BALL_IN_BASKET, ROLE_HOOP, ROLE_PLAYER, ROLE_REFEREE,
    ObjectDetector, class_role,
)
from src.cli import _parse_clip_indices  # noqa: E402
from src.config import VideoConfig  # noqa: E402

logger = logging.getLogger("bench")

DEFAULT_CANDIDATES = [
    "roboflow-jvuqo/basketball-player-detection-3-ycjdo",
    "basketballcomputervision/basketball-computer-vision",
    "basketball-hoop-tsdku/basketball-and-rim",
    "computer-vision-d5fjh/basketball-detection-dn6fg",
    "roboflow-universe-projects/basketball-players-fy4c2",
]


@dataclass
class Tally:
    name: str
    frames: int = 0
    hoop_frames: int = 0
    ball_frames: int = 0
    bib_frames: int = 0
    player_boxes: int = 0
    referee_boxes: int = 0
    hoop_conf: list[float] = field(default_factory=list)
    ball_conf: list[float] = field(default_factory=list)
    classes_seen: dict[str, int] = field(default_factory=dict)
    ms_per_frame: float = 0.0
    error: str | None = None

    def row(self) -> str:
        if self.error:
            return f"| `{self.name}` | ERROR: {self.error[:80]} | | | | | | |"
        f = max(self.frames, 1)
        mh = np.mean(self.hoop_conf) if self.hoop_conf else 0.0
        mb = np.mean(self.ball_conf) if self.ball_conf else 0.0
        return (
            f"| `{self.name}` | {100 * self.hoop_frames / f:.0f}% ({mh:.2f}) "
            f"| {100 * self.ball_frames / f:.0f}% ({mb:.2f}) | {self.bib_frames} "
            f"| {self.player_boxes / f:.1f} | {self.referee_boxes / f:.1f} "
            f"| {self.ms_per_frame:.0f} ms | {', '.join(sorted(self.classes_seen))} |"
        )


def _sample_frames(timeline: dict, clip_indices: list[int] | None, per_clip: int,
                   lut: FrameLUT, max_resolution: int) -> list[tuple[str, int, np.ndarray]]:
    clips = [c for t in timeline["tracks"] for c in t["clips"]]
    if clip_indices is not None:
        clips = [clips[i] for i in clip_indices if 0 <= i < len(clips)]

    samples = []
    for clip in clips:
        path = clip.get("analysis_path") or clip["file_path"]
        start, end = clip["source_start_frame"], clip["source_end_frame"]
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            logger.warning("cannot open %s", path)
            continue
        for fi in np.linspace(start, max(start, end - 1), per_clip).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok:
                continue
            frame = lut.apply(frame)
            h, w = frame.shape[:2]
            if max(h, w) > max_resolution:
                s = max_resolution / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            samples.append((clip.get("clip_name", path), int(fi), frame))
        cap.release()
    return samples


def _resolve_roboflow_version(slug: str, api_key: str) -> str | None:
    """Turn ``workspace/project`` into ``project/<latest trained version>``."""
    import requests

    if slug.count("/") == 1 and slug.split("/")[1].isdigit():
        return slug  # already project/version
    try:
        resp = requests.get(f"https://api.roboflow.com/{slug}", params={"api_key": api_key}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not resolve %s: %s", slug, exc)
        return None
    versions = data.get("versions", [])
    trained = [v for v in versions if v.get("model")]
    if not trained:
        logger.warning("%s has no trained versions (%d versions total)", slug, len(versions))
        return None
    # ids look like "workspace/project/7"
    best = max(trained, key=lambda v: int(str(v["id"]).rsplit("/", 1)[-1]))
    _, project, version = str(best["id"]).split("/")
    model = best.get("model", {})
    logger.info("%s → %s/%s (type=%s, mAP=%s)", slug, project, version,
                model.get("type") or model.get("modelType"), model.get("map"))
    return f"{project}/{version}"


def _run_roboflow(model_id: str, samples, confidence: float, api_key: str) -> Tally:
    from inference import get_model

    tally = Tally(name=model_id)
    try:
        model = get_model(model_id=model_id, api_key=api_key)
    except Exception as exc:  # noqa: BLE001
        tally.error = f"{type(exc).__name__}: {exc}"
        return tally

    times = []
    for _clip, _fi, frame in samples:
        t0 = time.perf_counter()
        try:
            results = model.infer(frame, confidence=confidence)
        except Exception as exc:  # noqa: BLE001
            tally.error = f"{type(exc).__name__}: {exc}"
            return tally
        times.append(time.perf_counter() - t0)
        preds = []
        for r in results if isinstance(results, list) else [results]:
            preds.extend(getattr(r, "predictions", None) or (r.get("predictions", []) if isinstance(r, dict) else []))
        _tally_predictions(tally, [
            (getattr(p, "class_name", None) or (p.get("class") if isinstance(p, dict) else ""),
             float(getattr(p, "confidence", None) or (p.get("confidence") if isinstance(p, dict) else 0.0)))
            for p in preds
        ])
    tally.ms_per_frame = 1000 * float(np.mean(times)) if times else 0.0
    return tally


def _run_yolo(model_name: str, imgsz: int, samples, confidence: float) -> Tally:
    tally = Tally(name=f"yolo:{model_name}@{imgsz}")
    try:
        det = ObjectDetector(VideoConfig(yolo_model=model_name, imgsz=imgsz, yolo_confidence=confidence,
                                         hoop_confidence=confidence))
    except Exception as exc:  # noqa: BLE001
        tally.error = f"{type(exc).__name__}: {exc}"
        return tally
    times = []
    for _clip, fi, frame in samples:
        t0 = time.perf_counter()
        fd = det.detect_frame(frame, fi)
        times.append(time.perf_counter() - t0)
        preds = [(d.class_name, d.confidence) for d in fd.balls + fd.hoops + fd.players + fd.referees]
        _tally_predictions(tally, preds)
    tally.ms_per_frame = 1000 * float(np.mean(times)) if times else 0.0
    return tally


def _tally_predictions(tally: Tally, preds: list[tuple[str, float]]) -> None:
    tally.frames += 1
    hoop = ball = bib = False
    for name, conf in preds:
        tally.classes_seen[name] = tally.classes_seen.get(name, 0) + 1
        role = class_role(name)
        if role == ROLE_HOOP:
            hoop = True
            tally.hoop_conf.append(conf)
        elif role == ROLE_BALL:
            ball = True
            tally.ball_conf.append(conf)
        elif role == ROLE_BALL_IN_BASKET:
            ball = bib = True
            tally.ball_conf.append(conf)
        elif role == ROLE_PLAYER:
            tally.player_boxes += 1
        elif role == ROLE_REFEREE:
            tally.referee_boxes += 1
    tally.hoop_frames += hoop
    tally.ball_frames += ball
    tally.bib_frames += bib


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--clip", type=str, default=None, help="clip spec, e.g. 0-4")
    ap.add_argument("--input-lut", type=Path, default=None)
    ap.add_argument("--frames-per-clip", type=int, default=24)
    ap.add_argument("--confidence", type=float, default=0.3)
    ap.add_argument("--max-resolution", type=int, default=1920)
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_CANDIDATES),
                    help="comma-separated Roboflow workspace/project slugs or project/version IDs")
    ap.add_argument("--no-yolo-baseline", action="store_true")
    ap.add_argument("--out", type=Path, default=None, help="write the markdown report here")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")

    timeline = json.loads(args.timeline.read_text())
    clip_indices = _parse_clip_indices(args.clip) if args.clip else None
    lut = FrameLUT(args.input_lut)
    samples = _sample_frames(timeline, clip_indices, args.frames_per_clip, lut, args.max_resolution)
    if not samples:
        print("no frames sampled", file=sys.stderr)
        return 1
    h, w = samples[0][2].shape[:2]
    logger.info("Sampled %d frames at %dx%d", len(samples), w, h)

    tallies: list[Tally] = []
    if not args.no_yolo_baseline:
        for imgsz in (640, 1280):
            tallies.append(_run_yolo("yolo11m.pt", imgsz, samples, args.confidence))
            logger.info(tallies[-1].row())

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    slugs = [s.strip() for s in args.models.split(",") if s.strip()]
    if slugs and not api_key:
        logger.warning("ROBOFLOW_API_KEY not set — skipping Roboflow candidates")
        slugs = []
    for slug in slugs:
        model_id = _resolve_roboflow_version(slug, api_key)
        if model_id is None:
            tallies.append(Tally(name=slug, error="no trained version / lookup failed"))
            continue
        tallies.append(_run_roboflow(model_id, samples, args.confidence, api_key))
        logger.info(tallies[-1].row())

    header = (
        f"# Hoop/ball detector bake-off\n\n"
        f"{len(samples)} frames from `{args.timeline.name}` (clips {args.clip or 'all'}), "
        f"{w}x{h}, confidence ≥ {args.confidence}, LUT={args.input_lut}\n\n"
        "| Model | Hoop frames (mean conf) | Ball frames (mean conf) | ball-in-basket frames "
        "| Players/frame | Refs/frame | Speed | Classes seen |\n"
        "|---|---|---|---|---|---|---|---|\n"
    )
    report = header + "\n".join(t.row() for t in tallies) + "\n"
    print("\n" + report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report)
        print(f"written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
