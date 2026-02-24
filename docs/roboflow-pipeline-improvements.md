# Improvement Suggestions Based on Roboflow Basketball CV Pipeline

## Context

This document compares the current Resolve Basketball Analysis pipeline against the
state-of-the-art techniques presented in the Roboflow video
([YouTube](https://youtu.be/yGQb9KkvQ1Q)) and accompanying
[blog post](https://blog.roboflow.com/identify-basketball-players/). The video
demonstrates a full basketball CV pipeline using RF-DETR, SAM2, SigLIP, SmolVLM2,
and homography-based court mapping. Below is a gap analysis with suggestions
categorized by **importance** and **ease of implementation**.

---

## Current Pipeline vs. Video Recommendations

| Capability | Current Implementation | Video Recommendation |
|---|---|---|
| **Object Detection** | YOLOv11m (COCO pretrained) — detects generic `person` + `sports_ball` | RF-DETR fine-tuned on basketball-specific dataset (players, ball, hoop, jersey numbers) |
| **Player Tracking** | DeepSORT (bounding-box level) | SAM2 (pixel-level segmentation tracking) |
| **Team Classification** | HSV color extraction from upper 40% of bbox → k-means (k=2) | SigLIP embeddings + UMAP dimensionality reduction + K-means |
| **Jersey Number Reading** | Not implemented | SmolVLM2 (fine-tuned) or ResNet classifier |
| **Court Mapping** | Not implemented | Homography via YOLOv11 keypoint model on court landmarks |
| **Shot Detection** | Heuristic arc analysis (Kalman-tracked ball trajectory) | Dedicated make-or-miss jump shot classifier |
| **Audio Analysis** | Mel-spectrogram crowd excitement + whistle detection | Not covered (your pipeline is ahead here) |

---

## Suggestions

### Tier 1: High Importance, Easier to Implement

These deliver the biggest bang for the buck and build on infrastructure you already have.

#### 1. Fine-Tune Object Detection on a Basketball-Specific Dataset
- **Gap:** Your YOLO model uses generic COCO weights. It detects `person` (not `player`) and `sports_ball` (not `basketball` specifically). Hoop detection relies on optional Roboflow API calls.
- **Recommendation:** Fine-tune YOLOv11 (or switch to RF-DETR) on Roboflow's [Basketball Player Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo). This gives you dedicated classes for `player`, `basketball`, `hoop`, and `jersey_number` bounding boxes — all in one model pass.
- **Impact:** Dramatically improves detection accuracy for basketball-specific objects. Eliminates reliance on the external Roboflow API for supplemental hoop/ball detections. Reduces false positives from non-player persons (refs, coaches, crowd).
- **Effort:** Medium — Roboflow provides ready-made datasets and [fine-tuning notebooks](https://colab.research.google.com/github/roboflow/rf-detr). Your `ObjectDetector` class (`src/analysis/object_detector.py`) already supports custom model class names.
- **Files affected:** `src/analysis/object_detector.py`, `src/config.py` (new default model path)

#### 2. Upgrade Team Classification: SigLIP Embeddings + UMAP + K-means
- **Gap:** Current HSV-based team clustering (`player_tracker.py:classify_teams`) extracts median HSV color from the upper 40% of player bounding boxes. This is fragile — it fails when teams have similar jersey hue, when lighting changes, when players are partially occluded, or when broadcast color grading shifts tones.
- **Recommendation:** Replace HSV extraction with SigLIP vision embeddings. Crop the central torso region of each player, run through SigLIP to get a rich feature vector, reduce dimensionality with UMAP, then cluster with K-means (k=2). This captures texture, pattern, and color simultaneously.
- **Impact:** Much more robust team separation. Works even with similar-colored jerseys (e.g., white vs. light gray) because it encodes visual features beyond just color.
- **Effort:** Medium — requires adding `transformers` (for SigLIP) and `umap-learn` as dependencies. The clustering logic in `PlayerTracker.classify_teams()` can be swapped in-place.
- **Files affected:** `src/analysis/player_tracker.py`, `requirements.txt`

#### 3. Improve Shot Detection with a Dedicated Make/Miss Classifier
- **Gap:** Your current shot detection (`ball_tracker.py:find_shots`) uses geometric heuristics: it looks for parabolic arcs in ball trajectory, then checks if the ball passes within 80px of the hoop center. This works for clean arcs but struggles with: bounced-in shots, blocked shots, bank shots, tip-ins, and camera angle changes.
- **Recommendation:** Train or use a dedicated make-or-miss classifier as shown in the video's [Make or Miss Jumpshot Detection notebook](https://colab.research.google.com/github/roboflow/notebooks). This uses a short clip around the shot event and classifies the outcome directly, rather than relying on ball-hoop proximity.
- **Impact:** Significantly more accurate made/missed classification. The current `hoop_proximity_px = 80` threshold is a blunt instrument that can't handle edge cases.
- **Effort:** Medium — Could be implemented as a secondary validation step after your existing arc detection identifies a candidate shot. Your `EventClassifier` already has a multi-modal fusion framework that could incorporate this as an additional signal.
- **Files affected:** `src/analysis/ball_tracker.py`, `src/analysis/event_classifier.py`, `src/config.py`

---

### Tier 2: High Importance, Harder to Implement

These are transformative but require more architectural work.

#### 4. Add Court Mapping via Homography
- **Gap:** Your pipeline has no spatial awareness of where events happen on the court. You can't distinguish a 3-pointer from a mid-range shot, can't detect paint violations, and can't generate top-down tactical visualizations.
- **Recommendation:** Use a YOLOv11 keypoint detection model trained on Roboflow's [Basketball Court Detection 2 Dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) to identify court landmarks (paint corners, 3-point arc, half-court line, etc.). Compute a homography matrix to map image coordinates to a canonical court template. Then project player positions to court coordinates each frame.
- **Impact:** Enables an entirely new class of analysis: shot charts, player heatmaps, spacing metrics, 3-point vs. 2-point classification, zone-based stats, and top-down court visualization. This is the single most impactful feature for tactical analysis.
- **Effort:** Hard — requires a new module for keypoint detection, homography computation (OpenCV `findHomography` + `perspectiveTransform`), a court template image, and integration with the player tracking output. Also needs handling of camera angle changes across scenes.
- **Files affected:** New module `src/analysis/court_mapper.py`, `src/analysis/video_analyzer.py` (pipeline integration), `src/analysis/preview.py` (visualization), `src/config.py`

#### 5. Upgrade Player Tracking: SAM2 Segmentation Tracking
- **Gap:** DeepSORT tracks players at the bounding-box level. It loses tracks during occlusion, struggles with player crossings, and provides no pixel-level segmentation for downstream tasks (like isolating jersey regions cleanly).
- **Recommendation:** Replace or augment DeepSORT with SAM2. Initialize SAM2 with bounding boxes from your detector, and it will track + segment players at the pixel level across frames. This provides more stable identity tracking and cleaner crops for team classification and jersey reading.
- **Impact:** More stable player identity across frames, fewer track ID switches, pixel-level masks enable cleaner jersey crop extraction for team classification and number OCR.
- **Effort:** Hard — SAM2 is significantly heavier than DeepSORT (the video notes ~1-2 FPS on a T4 GPU). Requires `sam2` dependency, substantial changes to `PlayerTracker`, and careful GPU memory management since YOLO is already running. May need to process in two passes.
- **Files affected:** `src/analysis/player_tracker.py`, `src/config.py`, `requirements.txt`
- **Caveat:** The video itself acknowledges SAM2 is the biggest performance bottleneck. For a Resolve-integrated tool where processing speed matters, this may be better as an optional "high-accuracy" mode.

---

### Tier 3: Medium Importance, Medium Effort

#### 6. Add Jersey Number Recognition
- **Gap:** Your pipeline identifies players only by track ID and team color. There's no way to associate detections with specific named players.
- **Recommendation:** Add jersey number OCR using either: (a) a fine-tuned SmolVLM2 (86% accuracy per the video), or (b) a ResNet-based number classifier (93% accuracy, faster). Use a voting/consensus system — require 3 identical readings across consecutive frames before assigning a number to a track, since visibility varies with camera angle.
- **Impact:** Enables per-player statistics (shot charts per player, minutes tracking, matchup analysis). Transforms the output from anonymous team blobs into identified player data.
- **Effort:** Medium — the detection model (from suggestion #1) can provide jersey number bounding boxes. The OCR/classification step is a new module. The consensus voting system adds complexity but is straightforward.
- **Files affected:** New module `src/analysis/jersey_reader.py`, `src/analysis/player_tracker.py` (integration), `src/config.py`

#### 7. Use Roboflow Supervision Library for Annotation & Utilities
- **Gap:** Your `preview.py` handles visualization manually with raw OpenCV drawing calls. Bounding box rendering, label placement, trail drawing, and zone visualization are all hand-coded.
- **Recommendation:** Adopt the [Roboflow Supervision library](https://github.com/roboflow/supervision) which provides production-quality annotators (`BoxAnnotator`, `LabelAnnotator`, `TraceAnnotator`, `HeatMapAnnotator`), zone-based counting (`PolygonZone`), and tracking utilities — all designed for sports analytics.
- **Impact:** Cleaner visualizations with less code. Access to heatmap, trace, and zone annotators out of the box. The library also provides utilities for detection filtering, NMS, and format conversion.
- **Effort:** Low-Medium — mostly replacing existing drawing code in `preview.py` with Supervision calls. The library is actively maintained and well-documented.
- **Files affected:** `src/analysis/preview.py`, `requirements.txt`

#### 8. Leverage Roboflow Sports Library for Court Templates
- **Gap:** No court visualization exists currently.
- **Recommendation:** The [Roboflow Sports library](https://github.com/roboflow/sports) provides pre-built basketball court templates, coordinate systems, and rendering utilities specifically designed for sports analytics overlays.
- **Impact:** Accelerates court mapping implementation (suggestion #4) by providing ready-made court templates and projection utilities rather than building from scratch.
- **Effort:** Low (as an add-on to suggestion #4) — it's a lightweight library that handles the court rendering side.
- **Files affected:** Would be used by the new `court_mapper.py` module

---

### Tier 4: Lower Importance / Nice-to-Have

#### 9. Multi-Frame Consensus for Ball Detection
- **Gap:** Your ball tracker uses a single-frame detection + Kalman prediction. The video pipeline's approach of validating detections across multiple frames (similar to their jersey number consensus) could reduce false ball detections.
- **Recommendation:** Add a short-window consensus filter: require N out of M consecutive frames to confirm a ball detection before starting arc tracking. Your Kalman filter partially addresses this, but an explicit consensus gate would reduce phantom arcs.
- **Effort:** Low — small addition to `BallTracker.update()`.
- **Files affected:** `src/analysis/ball_tracker.py`

#### 10. Switch from YOLO to RF-DETR as Primary Detector
- **Gap:** YOLO is fast but RF-DETR achieves SOTA accuracy on COCO and is specifically designed for fine-tuning on custom datasets.
- **Recommendation:** Evaluate RF-DETR as a drop-in replacement for YOLO. Your `ObjectDetector` already abstracts the detection interface, so this is a model swap.
- **Impact:** Potentially better detection accuracy, especially after fine-tuning.
- **Effort:** Low-Medium — RF-DETR has a different API than Ultralytics YOLO, so the `detect_frame()` method needs adaptation. Fine-tuning notebooks are provided.
- **Files affected:** `src/analysis/object_detector.py`, `requirements.txt`
- **Caveat:** RF-DETR may be slower than YOLO for real-time needs. Benchmark before committing.

---

## Summary Matrix

| # | Suggestion | Importance | Ease | Priority Score |
|---|---|---|---|---|
| 1 | Fine-tune detection on basketball dataset | High | Medium | **A** |
| 2 | SigLIP-based team classification | High | Medium | **A** |
| 3 | Dedicated make/miss shot classifier | High | Medium | **A** |
| 4 | Court mapping via homography | High | Hard | **B** |
| 5 | SAM2 player tracking | High | Hard | **B** |
| 6 | Jersey number recognition | Medium | Medium | **B** |
| 7 | Supervision library for visualization | Medium | Low-Med | **B** |
| 8 | Sports library for court templates | Medium | Low | **C** |
| 9 | Multi-frame ball detection consensus | Low-Med | Low | **C** |
| 10 | Evaluate RF-DETR as detector | Low-Med | Low-Med | **C** |

**Priority A** = Implement first (highest ROI)
**Priority B** = Implement second (high value, more effort)
**Priority C** = Implement as time permits (incremental improvements)

---

## What Your Pipeline Already Does Better

Worth noting: your pipeline has capabilities the video does **not** cover:
- **Audio analysis** (crowd excitement, whistle detection) — a unique multi-modal advantage
- **DaVinci Resolve integration** — the video is research-only; yours creates actionable edit markers
- **Scene detection** — automatic shot boundary detection for timeline-aware analysis
- **Multi-modal event fusion** — combining video + audio confidence with configurable weights

These are genuine differentiators that should be preserved and enhanced, not replaced.

---

## Resources (from the video)

### Datasets
- [Basketball Player Detection Dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo) — players, ball, hoop, jersey numbers
- [Basketball Court Detection 2 (Keypoints)](https://universe.roboflow.com/roboflow-jvuqo/basketball-court-detection-2) — court landmark keypoints for homography
- [Basketball Jersey Numbers OCR Dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-jersey-numbers-ocr) — 3,615 jersey number crops for fine-tuning

### Notebooks
- [How to Detect, Track, and Identify Basketball Players](https://colab.research.google.com/github/roboflow/notebooks) — full pipeline notebook
- [Make or Miss - Jumpshot Detection](https://colab.research.google.com/github/roboflow/notebooks) — shot classification notebook
- [Fine-tune RF-DETR on Custom Dataset](https://colab.research.google.com/github/roboflow/rf-detr) — RF-DETR fine-tuning
- [Segment Video with SAM 2](https://colab.research.google.com/github/roboflow/notebooks) — SAM2 video segmentation

### Libraries & Tools
- [RF-DETR GitHub](https://github.com/roboflow/rf-detr) — SOTA object detection model (ICLR 2026)
- [Supervision GitHub](https://github.com/roboflow/supervision) — CV annotation, tracking, and zone utilities
- [Sports GitHub](https://github.com/roboflow/sports) — Court templates and sports-specific CV utilities

### Blog Posts & Video
- [How to Detect, Track, and Identify Basketball Players (Blog)](https://blog.roboflow.com/identify-basketball-players/)
- [Detect NBA 3-Second Violations with AI (Court Mapping)](https://blog.roboflow.com/detect-3-second-violation-ai-basketball/)
