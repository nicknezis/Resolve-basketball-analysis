"""Score an analysis JSON against hand-labelled shots.

Ground truth is a CSV with columns ``timeline_frame,clip,outcome,notes`` where
``outcome`` is ``made`` or ``miss`` and ``timeline_frame`` is the frame (on the
Resolve timeline) where the ball leaves the shooter's hands.  Lines starting
with ``#`` are ignored.

Usage:
    python scripts/eval_shots.py my_game_analysis.json ground_truth/my_game.csv

(``ground_truth/`` is gitignored — labels are per-game data, not source.)

Reports attempt precision/recall, made/miss accuracy on matched shots, the
unmatched ground-truth shots, false positives, and any event longer than
``--max-event-sec`` (a symptom of arc-finding bugs).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SHOT_TYPES = {"made_shot", "shot_attempt"}


@dataclass
class GtShot:
    frame: int
    sec: float
    clip: str
    outcome: str
    notes: str


def load_ground_truth(path: Path, fps: float) -> list[GtShot]:
    shots: list[GtShot] = []
    with open(path, newline="") as f:
        rows = [r for r in f if r.strip() and not r.lstrip().startswith("#")]
    for row in csv.DictReader(rows):
        frame = int(row["timeline_frame"])
        outcome = row.get("outcome", "").strip().lower()
        if outcome not in ("made", "miss"):
            raise ValueError(f"outcome must be made|miss, got {outcome!r} (frame {frame})")
        shots.append(GtShot(frame, frame / fps, row.get("clip", ""), outcome, row.get("notes", "")))
    return sorted(shots, key=lambda s: s.frame)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("analysis_json", type=Path)
    ap.add_argument("ground_truth_csv", type=Path)
    ap.add_argument("--tolerance-sec", type=float, default=1.5,
                    help="a GT shot matches an event if it falls within [start-tol, end+tol]")
    ap.add_argument("--max-event-sec", type=float, default=4.0)
    args = ap.parse_args(argv)

    data = json.loads(args.analysis_json.read_text())
    fps = data.get("timeline", {}).get("fps") or data.get("video_info", {}).get("fps")
    if not fps:
        print("cannot determine fps from analysis JSON", file=sys.stderr)
        return 1

    gt = load_ground_truth(args.ground_truth_csv, fps)
    events = [e for e in data["events"] if e["type"] in SHOT_TYPES]
    events.sort(key=lambda e: e["start_sec"])

    # Greedy one-to-one matching, nearest event start to each GT release
    matched: list[tuple[GtShot, dict]] = []
    unmatched_gt: list[GtShot] = []
    used: set[int] = set()
    tol = args.tolerance_sec
    for shot in gt:
        best_i, best_d = None, float("inf")
        for i, ev in enumerate(events):
            if i in used:
                continue
            if ev["start_sec"] - tol <= shot.sec <= ev["end_sec"] + tol:
                d = abs(ev["start_sec"] - shot.sec)
                if d < best_d:
                    best_i, best_d = i, d
        if best_i is None:
            unmatched_gt.append(shot)
        else:
            used.add(best_i)
            matched.append((shot, events[best_i]))
    false_pos = [ev for i, ev in enumerate(events) if i not in used]

    tp = len(matched)
    precision = tp / len(events) if events else 0.0
    recall = tp / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    outcome_correct = outcome_wrong = outcome_unknown = 0
    for shot, ev in matched:
        pred = ev.get("details", {}).get("made")
        if pred is None:
            outcome_unknown += 1
        elif (pred is True) == (shot.outcome == "made"):
            outcome_correct += 1
        else:
            outcome_wrong += 1

    long_events = [e for e in events if e["end_sec"] - e["start_sec"] > args.max_event_sec]

    print(f"Analysis: {args.analysis_json}  (fps {fps:.2f})")
    print(f"Ground truth: {len(gt)} shots   Predicted: {len(events)} shot events\n")
    print(f"Attempt detection:  precision {precision:.2f}  recall {recall:.2f}  F1 {f1:.2f}")
    judged = outcome_correct + outcome_wrong
    acc = outcome_correct / judged if judged else 0.0
    print(f"Made/miss on matched: {outcome_correct}/{judged} correct ({acc:.0%}), "
          f"{outcome_unknown} unknown (no hoop evidence)\n")

    if matched:
        print("Matched:")
        for shot, ev in matched:
            pred = ev.get("details", {}).get("made")
            pred_s = {True: "made", False: "miss", None: "unknown"}[pred]
            flag = "" if pred is None or (pred is True) == (shot.outcome == "made") else "  <-- WRONG"
            print(f"  GT {shot.frame:>6} {shot.outcome:<4} | ev {ev['start_frame']:>6}-{ev['end_frame']:<6} "
                  f"{ev['type']:<12} pred={pred_s:<7} conf={ev['confidence']:.2f} "
                  f"({ev['end_sec'] - ev['start_sec']:.1f}s){flag}")
    if unmatched_gt:
        print("\nMissed ground-truth shots:")
        for shot in unmatched_gt:
            print(f"  frame {shot.frame} ({shot.sec:.1f}s) {shot.outcome} {shot.clip} {shot.notes}")
    if false_pos:
        print("\nFalse positives:")
        for ev in false_pos:
            print(f"  {ev['start_frame']}-{ev['end_frame']} {ev['type']} conf={ev['confidence']:.2f} "
                  f"({ev['end_sec'] - ev['start_sec']:.1f}s) {ev.get('details', {}).get('source_clip', '')}")
    if long_events:
        print(f"\nEvents longer than {args.max_event_sec:.0f}s ({len(long_events)}):")
        for ev in long_events:
            print(f"  {ev['start_frame']}-{ev['end_frame']} {ev['type']} {ev['end_sec'] - ev['start_sec']:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
