# Analysis Techniques

Technical reference for all video and audio analysis methods implemented in the basketball analysis engine.

## Overview

The engine uses a six-phase pipeline to detect basketball game events from video:

1. **Scene Detection** -- Identify shot/scene boundaries using frame-to-frame HSV comparison
2. **Audio Analysis** -- Extract crowd excitement peaks and referee whistle events from the audio track
3. **Object Detection** -- Run YOLO inference to locate the basketball, hoop, and players in each frame
4. **Ball Tracking** -- Link per-frame ball candidates into tracklets offline, pick the ball's chain, detect shot arcs
5. **Player Tracking** -- Maintain player identity across frames with ByteTrack and classify teams by jersey appearance
6. **Event Classification** -- Fuse video and audio signals into final `GameEvent` objects with confidence scores

Each phase is implemented as an independent module under `src/analysis/`. The pipeline orchestrator (`video_analyzer.py`) chains them together and supports two modes: single-video analysis and timeline-aware multi-clip analysis for DaVinci Resolve integration.

---

## Object Detection

**Module:** `src/analysis/object_detector.py`

### Model

Three backends, selected with `--detector`:

| Backend | Model | Hoop class? |
|---------|-------|-------------|
| `yolo` (default) | Ultralytics `yolo11m.pt` (auto-downloads) or any YOLO weights via `--yolo-model` | Only with a basketball fine-tune — stock COCO has none |
| `rfdetr` | RF-DETR (`pip install rfdetr`), COCO or `--rfdetr-weights` + `--rfdetr-classes` | Only with a fine-tuned checkpoint |
| `roboflow` | A Roboflow Universe/workspace model via `--roboflow-model ID`, run locally on ONNX by the `inference` package (needs `ROBOFLOW_API_KEY`) | Yes, if the model has a `rim`/`hoop`/`basket` class |

`--roboflow-model` can also be paired with the `yolo`/`rfdetr` backends as a *supplemental* hoop/ball source; its boxes are merged into the same `FrameDetections`.

### Class roles

Every model's class names are normalised through `CLASS_ROLE_MAP` (lower-cased, `_` → `-`):

| Role | Class names | Bucket |
|------|-------------|--------|
| ball | `ball`, `basketball`, `sports ball` | `fd.balls` |
| ball-in-basket | `ball-in-basket`, `made` | `fd.balls` **and** `fd.balls_in_basket` (direct made-shot evidence) |
| hoop | `hoop`, `rim`, `basket`, `basketball-hoop`, `hoop-rim` | `fd.hoops` |
| player | `player`, `person`, `shooter`, `player-in-possession`, `player-jump-shot`, `player-layup-dunk`, `player-shot-block` | `fd.players` |
| referee | `referee`, `ref` | `fd.referees` (never fed to team classification) |
| ignore | `number`, `backboard`, `net`, `people`, `shoot` | dropped |

A YOLO model counts as basketball-specific when any of its class names maps to the hoop role. When no configured model can produce hoops, the detector logs a warning at start-up: made/miss classification is then impossible and every shot is reported with `made: null`.

### Thresholds and inference size

- `--yolo-confidence` / `--roboflow-confidence` gate ball and player boxes; `--hoop-confidence` (default 0.3) is a separate, looser floor for hoop boxes on every backend — rims are small and benefit from it.
- `--imgsz` (default 1280) sets YOLO's inference size on the long edge. Ultralytics' own default of 640 shrinks a 720p ball to ~10 px.
- With stock COCO weights only the `person` and `sports ball` classes are requested from YOLO.

Use `scripts/bench_hoop_models.py` to compare candidate models on your own footage; results for the reference game are in `docs/model-bakeoff.md`.

### Frame Preprocessing

- **Frame skip:** Only every Nth frame is analyzed (default N=2) to trade accuracy for speed.
- **Downscaling:** Frames larger than `max_resolution` (default 1920px on the longest edge) are scaled down proportionally before inference.

### Device Selection

Automatic priority: CUDA > MPS (Apple Silicon) > CPU. Controlled by the `device` field on `AnalysisConfig` (`"auto"`, `"cuda"`, `"mps"`, or `"cpu"`). The device string is passed directly to YOLO's `device` parameter.

### Output

Each analyzed frame produces a `FrameDetections` object containing categorized lists of `Detection` objects (balls, hoops, players, referees, balls_in_basket). Each detection stores:
- Raw model class name, confidence score, bounding box `(x1, y1, x2, y2)`, frame index
- Derived properties: `center` (bbox midpoint), `area` (bbox pixel area)

---

## Ball Tracking

**Module:** `src/analysis/ball_tracker.py`

### Offline Tracklet Linking

The analyser works on finished clips, so ball tracking is **not causal**: every ball candidate in the clip is known before any tracking decision is made. (The previous design — an online Kalman filter that committed to one detection per frame and dropped its distance gate after five missed frames — snapped onto heads and light fixtures as soon as the detector offered more than one candidate per frame.)

1. **Record.** During the detection pass, `BallTracker.update()` stores every ball candidate per analysed frame (centre, size, confidence) plus the player centres and the rim centre. It returns the top candidate only as a provisional position for the live preview.
2. **Camera-motion compensation.** A per-frame image shift is estimated from the rim (a perfect static anchor when visible) or the median displacement of matched player boxes, and accumulated. Linking and motion scoring run in these compensated coordinates, so a pan neither breaks a track nor makes a fixture look like it is moving. Disable with `compensate_camera_motion=False`.
3. **Link.** Candidates are joined into *tracklets* greedily by distance to each tracklet's predicted position (last detection + damped velocity × elapsed frames). The gate is `link_slack_px + max_ball_speed_px_per_frame × frames_since_last_detection`, capped at `max_link_jump_px` — it widens while the ball is missing but can never reach across the frame; longer hops are the chain's job (step 4). Consecutive boxes must be within `tracklet_size_ratio_max` in size. A tracklet closes after `max_ball_gap_frames` without a detection; tracklets shorter than `min_tracklet_detections` are dropped, so a single stray box never becomes a track. Candidates centred within `edge_margin_px` of the frame border are never recorded (clipped boxes, ceiling fixtures).
4. **Select.** Each tracklet is scored `detections × mean confidence × (0.25 + 0.75 × motion)`, where `motion` is the tracklet's camera-compensated *extent* relative to `motion_full_span_px` — extent, not per-step displacement, because detection jitter on a tiny fixture is a few px every step yet never goes anywhere. Tracklets whose extent stays under `static_span_px` for at least `static_clutter_sec` are discarded as clutter. A dynamic programme then picks the chain of time-disjoint tracklets with the highest total score, charging `teleport_penalty_per_px` for any jump between consecutive tracklets that exceeds what the ball could travel in the gap.
5. **Fill.** The chosen tracklets become `BallPosition` lists; gaps inside a tracklet are linearly interpolated at the analysis cadence and flagged `predicted=True`. Each tracklet is a separate segment — shot detection never runs across a break.

Linking a handful of boxes per frame costs microseconds; the detector dominates runtime.

### Shot Arc Detection

Shot detection runs as a second pass over the full tracked trajectory. The algorithm applies multiple validation gates:

1. Scan positions with a sliding window looking for upward motion (decreasing y in image coordinates)
2. Track the peak (minimum y value) of the arc
3. When a **detected** (not interpolated) position descends past the peak by at least `shot_arc_end_descent_px` (default 50px), record the arc candidate. Predicted positions never terminate an arc — extrapolation must not manufacture a descent.
4. Compute `arc_height = start_y - peak_y`; only accept if it exceeds `shot_min_arc_height_px` (default 50px)
5. **Maximum arc duration gate**: Reject arcs whose source-frame span exceeds `shot_max_arc_sec` (default 3.0 s). Real basketball shots take 1-2 seconds; longer arcs indicate tracking noise or continuous ball movement.
6. **Minimum descent ratio gate**: The ball must descend at least `shot_min_descent_ratio` (default 0.15) of its ascent height after the peak. This is set low to accommodate layups, where the player carries the ball upward (inflating the measured ascent) but the actual descent through the hoop is short. Jump shots produce ratios near 1.0; layups typically produce 0.15-0.35.
7. **Hoop-directed descent gate**: When a hoop position is known, the median x-coordinate of the ball during the descent phase must be within `shot_hoop_x_range_ratio` (default 0.35) of the analysed frame width from the hoop's x-coordinate. This distinguishes shots (aimed at the hoop) from passes (aimed at teammates elsewhere on the court).
8. **Peak-above-rim gate** (`shot_require_peak_above_rim`, default on): when a rim observation is available for the arc, the arc's peak must be at or above the rim top (minus `shot_peak_rim_margin_px`, default 20). Passes, dribbles and hand-offs also trace up-and-down arcs, but they stay below rim height; a ball that enters the rim must have been above it. This gate removed about half of the false attempts on the reference footage.

**Everything is measured in source frames, never list indices.** The tracked position list is sparse — nothing is appended while the ball is lost — so two positions adjacent in the list can be many seconds apart in the video. A gap larger than `max_ball_gap_frames` between consecutive positions terminates the current arc: a ball that was lost and re-acquired elsewhere is not one trajectory. (Before this rule a single "shot" could span 11 s of footage and swallow several real shots.)

The event time window is also tightened: instead of spanning from where the sliding window first detected upward motion, the shot event starts at most `shot_pre_peak_sec` (default 0.5 s) before the arc peak. The arc height is still calculated from the original start for correctness, but the reported event window covers only the shot flight.

For made-shot detection, the search window extends `shot_post_arc_sec` (default 0.35 s) past the arc end, but never across a track break. This captures backboard shots where the ball's descent triggers the arc end at the backboard bounce, but the actual through-hoop transition happens a few frames later. If a made shot is found in the extended window, the event window is expanded to include those positions.

### Shot Quality Metrics

During arc validation, two quality metrics are computed and stored on each `ShotEvent`:

- `hoop_x_distance`: Pixel distance between the descent median x-coordinate and the hoop x-coordinate. Lower values indicate the ball was aimed directly at the hoop.
- `descent_ratio`: The ratio of descent height to ascent height. Higher values indicate a cleaner parabolic arc.

These metrics are used downstream by the event classifier to adjust confidence scores.

### Near-Rim Re-Detection

Made/miss is decided in the few frames where the ball meets the rim, which is also where the main pass sees it worst: the ball is small, half-hidden by net and backboard, and the pass only samples every Nth frame. After the first round of shot detection, `rim_roi.RimRoiRedetector` re-reads **every source frame** of each shot window (`rim_roi_pre_sec` before the peak to `rim_roi_post_sec` after the arc end), crops `rim_roi_width_factor` rim widths around the rim (`rim_roi_above_factor` heights above, `rim_roi_below_factor` below, covering the net), upscales the crop by `rim_roi_scale` and runs the same detector. The extra ball/hoop boxes are mapped back, merged into the tracker (`BallTracker.add_frames`) and the detection list (so the review overlay shows them), and shot detection runs again on the denser trajectory. Cost ≈ 1 s of detector time per shot. Disable with `rim_roi_redetect=False`.

### Made Shot Detection

`ShotEvent.made` is **tri-state**: `True`, `False`, or `None` when no usable hoop observation exists for the shot. "No rim in view" is reported as unknown, never as a miss, and the classifier gives it a lower base confidence.

1. **Rim selection.** Per-frame hoop observations (bbox + centre) are collected during detection. For a given descent, only observations within `hoop_obs_max_age_frames` (default 30) of the descent window are eligible — with a panning camera, a rim seen 20 s earlier says nothing about where the rim is now. When both baskets are in view, the observation horizontally nearest the descent (within two rim-widths) wins, then the one nearest in time. Each position is then judged against the observation of *that* rim nearest in time, so a pan during the flight is followed.
2. **Rim phases.** Every detected position after the arc peak is classified against the rim box: **approach** (above the rim top by `hoop_entry_y_margin_px`, horizontally within the rim span ± `hoop_x_tolerance_ratio`), **rim** (inside the band from the rim top − margin to the bottom + ½ rim height), **post_net** (below the band, within the rim span ± `net_x_tolerance_ratio`) or **post_off** (below the band but beside the net). The verdict comes from the sequence:

   | Sequence | Verdict | `made_via` |
   |---|---|---|
   | approach → post_net, ball slowed below freefall | made | `through_net` |
   | approach → post_net at freefall speed | miss — it passed in front of / behind the rim | `passed_rim` |
   | approach → post_off, or rim band then back up | miss | `rim_out` |
   | approach → rim band, then the track ends | look-ahead (below) | `rim_entry` / `rim_out` |
   | never approached within the rim span | miss | `short` |

3. **Freefall speed check.** A parabola `y = a·t² + b·t + c` is fitted to the detected flight; it is trusted only if `a > 0` and the residual is under 6% of the flight's vertical span (real flights fit to a few px at 720p). The measured descent speed over the first three `post_net` points is compared with the fitted speed at that time; `speed_ratio` above `post_rim_speed_made_max` (default 0.85) means the net did not slow the ball. Without a trustworthy fit or enough post-rim points the ratio is `None` and the geometric verdict stands.
4. **Rim entry** (`made_via="rim_entry"` / `"rim_out"`). The detector often loses the ball as it enters the net. If the last *detected* position after a peak lies inside the rim's footprint and the track then ends or breaks, the arc is accepted without the descent-ratio gate. If the ball is re-detected within `rim_entry_lookahead_sec` *above* the rim bottom it rimmed out; if it reappears below the rim, or not at all, it went in.
5. **Ball-in-basket** (`made_via="ball_in_basket"`). If the detector emits a `ball-in-basket` class between the arc peak and the end of the post-arc window, the shot is a make regardless of geometry.
6. **Polygon zone** (`--polygon-zone`) and **centre proximity** (`set_hoop_positions`) remain as opt-in / legacy fallbacks.

Each shot carries `speed_ratio`, `fit_rmse`, `rim_impact` (if heard), `shot_zone`, `shot_distance_ft` and `shot_type` in its details for auditing. Made shots from the three zone are emitted as `three_pointer` events.

---

## Player Tracking

**Module:** `src/analysis/player_tracker.py`

### ByteTrack Multi-Object Tracking

Player identity is maintained across frames with ByteTrack (`trackers.ByteTrackTracker`, Roboflow's tracker library), which associates boxes by predicted-position IoU only. At 60 fps players move a few pixels between analysed frames, so an appearance embedder adds cost without information — DeepSORT's per-box re-ID network was ~40% of the per-frame budget.

- **Activation:** a detection needs `track_activation_threshold` (default 0.5) confidence to start a new track; boxes above `track_high_conf_threshold` (0.6) are matched first, the rest in ByteTrack's second low-confidence pass.
- **Lost buffer:** tracks survive `track_lost_buffer_frames` (default 30 analysed frames) without a match.
- **Confirmation:** a track is reported after `track_min_hits` (default 3) consecutive matches.
- **Matching:** IoU ≥ `track_min_iou` (default 0.1).

The tracker receives the same (downscaled) frame the boxes were computed on.

### Jersey Crops

For every confirmed track, up to `jersey_crops_per_track` torso crops are stored, at most one every `jersey_sample_every_frames` source frames: the central 15–55 % of the box height with 20 % trimmed from each side, so heads, legs and the neighbour's shirt stay out. Tracks with fewer than `jersey_min_crops` crops are never assigned a team.

### Team Classification

`team_method="siglip"` (default) embeds every crop with SigLIP (`team_embed_model`, batched on MPS/CUDA — 64 crops in ~0.45 s), averages per track, reduces with PCA and runs k-means (k=2). Tracks farther than mean + `team_outlier_std` × std from both centres stay unassigned — that is where referees' stripes and spectators land, so they are not forced into a team. People the court homography places off the court (see below) are excluded before clustering. `team_method="hsv"` is the previous median-colour k-means and the automatic fallback if the model cannot be loaded. The review export colours tracked boxes by team (blue / green, gray = unassigned).

---

## Court Geometry (experimental, opt-in)

**Module:** `src/analysis/court.py` — enable with `--court-preset nfhs|nba|fiba`.

**Status.** On the reference footage this works on one end of the court and not the other: with the camera looking at the far basket the projected lines sit on the painted lines and shooter zones look right; looking at the near basket the accepted keyframes still produce a mapping that is visibly wrong, and zones assigned to the same shot change between keyframes. The keypoint model returns 14–20 landmarks per frame, but only 6–9 of them agree with the template within 60 cm, so RANSAC picks a small self-consistent subset that may not be the right one. Until a keypoint model trained on this kind of footage exists (or a court-specific calibration), treat zones and distances as hints and leave the feature off for production runs.

A Roboflow court-keypoint model (`CourtConfig.model_id`, default `basketball-court-detection-2/22`, 17 landmarks) runs every `every_sec` seconds of a clip. Keypoints above `keypoint_confidence` are matched by name to a template of 33 court vertices (vendored from roboflow/sports, MIT) computed from a dimension preset — `nfhs` (US high school: 84 ft court, 12 ft key, 19'9" arc that runs straight to the baseline once level with the basket), `nba` or `fiba` — and a RANSAC homography image → court (cm) is fitted; keyframes with fewer than `min_keypoints` inliers, a mean re-projection above `max_reproj_px`, or inliers that don't span an area (`min_spread_ratio`) are discarded. Between keyframes the mapping is carried by the camera-shift estimate from the ball tracker, which is enough for a pan.

A homography fitted to landmarks at one end of the court is only trustworthy *near those landmarks*: on the reference footage the model returns 14–20 keypoints per frame but typically 6–9 agree with the template, and the resulting mappings extrapolate the far corners thousands of pixels off-frame. So every court query is answered only for points inside the keyframe's inlier hull plus `calibrated_buffer_cm`; elsewhere the answer is "unknown" (no zone, no off-court flag), never a guess.

Uses:
- **Off-court people.** Each tracked person's foot point (bottom-centre of the box) is projected at every frame; a majority outside the lines by more than `court_margin_cm` marks the track `on_court=False` — excluded from team classification and from shooter matching.
- **Shot zone and distance.** The shooter is the on-court track whose box holds the ball at the start of the shot window; their foot point is projected and classified as `paint`, `two` or `three` (`three_point_margin_cm` tolerance, corner rule respected), with the distance to the basket centre in feet. A made shot from the three zone is emitted as `three_pointer`; `details.shot_type` is `three`, `layup` (≤ `layup_max_ft`) or `jumper`.
- **Overlay.** Review exports draw the projected court lines (`overlay`), so a wrong homography is obvious at a glance.

Disable with `--court-preset off`; `--no-players` also disables it (no shooter to place).

---

## Audio Analysis

**Module:** `src/analysis/audio_analyzer.py`

Audio is extracted from the video file using FFmpeg (`pcm_s16le`, mono, 22050 Hz sample rate) into a temporary WAV file.

### Crowd Excitement Detection

Detects peaks of crowd noise energy in a specific frequency band.

**Method:**

1. Load audio at `sample_rate` (default 22050 Hz)
2. Compute a **mel-spectrogram** with:
   - `n_fft` = `window_sec * sr` (default 2.0s window = 44100 samples)
   - `hop_length` = `hop_sec * sr` (default 0.5s hop = 11025 samples)
   - `n_mels` = 128 mel filter banks
3. Convert to dB scale (`power_to_db`, referenced to max)
4. Map the crowd frequency band (`crowd_freq_low_hz` to `crowd_freq_high_hz`, default 500-4000 Hz) to mel bin indices using `librosa.mel_frequencies`
5. Compute **mean energy** across the selected mel bins for each time window
6. **Min-max normalize** the energy vector to [0, 1]
7. Threshold at `excitement_threshold` (default 0.6)
8. Extract contiguous regions above the threshold as `AudioEvent` objects, with the peak score within each region as the event score

### Rim / Backboard Impact Transients

`detect_rim_impacts()` looks for short broadband bursts: band-limited spectral flux (positive log-magnitude change summed over the 1–6 kHz STFT bins) at 5 ms resolution (`impact_hop_sec`), scored by prominence over a rolling median/MAD baseline (`impact_local_window_sec`) so a loud gym neither hides nor manufactures them; bursts longer than `impact_max_duration_sec` (crowd, whistles) are dropped. Each surviving onset becomes a `rim_impact` audio event with `score` = prominence / (3 × `impact_min_prominence`), capped at 1. The classifier looks for one within `impact_window_before_sec` … `impact_window_after_sec` of the frame the ball first reaches the rim band (`ShotEvent.rim_frame`) and, if found, adds +0.03 video confidence and `details.rim_impact`. On the reference footage the rim contact of a known make produces a clear transient ~0.1 s after the visual contact (sound travel + shutter), but dribbles and shoe squeaks produce about one transient per second too — treat this as weak corroboration, not evidence on its own. A clean swish is too quiet to detect.

### Game-Wide Crowd Normalisation

Crowd energy used to be min-max normalised per clip, which made the loudest two seconds of *every* clip score 1.0. With `crowd_norm="game"` (default) the timeline analysis runs an audio pre-pass over all selected clips first (`crowd_band_energy`, absolute dB), takes the `crowd_norm_low_pct` / `crowd_norm_high_pct` percentiles of all windows as the 0 / 1 anchors (`crowd_norm_from_energies`), and scores every clip against them. The pre-pass extracts each clip's audio once and hands the trimmed WAV to the clip analysis, so nothing is decoded twice.

### Whistle Detection

Detects referee whistles via spectral peak analysis in a narrow high-frequency band.

**Method:**

1. Load audio at `sample_rate` (default 22050 Hz)
2. Apply an **FFT-based bandpass filter**:
   - Compute `rfft` of the full signal
   - Zero out all frequency bins outside `whistle_freq_low_hz` to `whistle_freq_high_hz` (default 2000-4500 Hz)
   - Inverse FFT back to time domain
3. Compute **onset strength** via `librosa.onset.onset_strength` with a 10ms hop (`hop_length = 0.01 * sr = 220 samples`)
4. Normalize the onset envelope to [0, 1]
5. Threshold at `whistle_energy_threshold` (default 0.7)
6. Extract contiguous regions above threshold
7. **Filter out events shorter than 0.1 seconds** (whistles have a sustained tone; very short peaks are noise)

---

## Scene Detection

**Module:** `src/analysis/scene_detector.py`

Uses PySceneDetect's `ContentDetector` to find shot/scene boundaries.

- **Algorithm:** Frame-to-frame comparison in HSV color space. When the difference score exceeds the threshold, a scene cut is registered.
- **Threshold:** `content_threshold` (default 27.0)
- **Minimum scene length:** `min_scene_len_sec` (default 0.5s), converted to frames using the video's FPS

Each detected scene stores start/end frame numbers and start/end timestamps in seconds.

---

## Event Classification

**Module:** `src/analysis/event_classifier.py`

### Multi-Modal Fusion

Video is the primary signal; audio can only raise a shot's confidence into the remaining headroom:

```
fused_confidence = video_conf + audio_weight * audio_conf * (1 - video_conf)
```

- `audio_weight` = 0.4 (default)

Audio is therefore never *required*: a clean made shot in a silent gym still passes. (The previous `0.6·video + 0.4·audio` formula made it mathematically impossible for any shot to be reported without crowd noise, because the best video-only score was 0.57 against a 0.7 threshold.)

### Shot Event Classification

For each `ShotEvent` from the ball tracker:

1. **Base video confidence** by hoop evidence:
   - `made is True` (ball seen through the rim): **0.85**
   - `made is False` (rim in view, ball did not go through): **0.70**
   - `made is None` (no rim to judge against): **0.55**
   - Bonus +0.05 for arcs taller than 150px
   - Bonus +0.05 if `hoop_x_distance < 50px` (descent lands near the hoop)
   - Bonus +0.05 if `descent_ratio > 0.7` (clean parabolic descent)
   - capped at 1.0
2. **Audio correlation:** Search for audio events in a window from **1 second before** to **4 seconds after** the shot (crowd reaction lags the play). Crowd excitement scores are taken directly; whistle scores are scaled by 0.8.
3. **Fusion:** Apply the formula above
4. **Filter:** `video_confidence >= min_video_confidence` (default 0.5) **and** `confidence >= min_confidence` (default 0.7). With the defaults, shots with hoop evidence pass on video alone; hoop-less arcs still need corroborating audio.

Each shot event carries `details.made` (`true`/`false`/`null`), `details.made_via`, `details.hoop_x_distance` and the arc height.

### Standalone Crowd Events

High crowd excitement peaks not explained by any shot event (not overlapping within a 2-second padding) are emitted as `"crowd_excitement"` events. These capture plays the video analysis might miss (steals, blocks, fast breaks without shots). They carry `video_confidence = 0.0` and `audio_confidence` equal to their score.

### Event Merging

After classification, events are sorted by time and adjacent **crowd-excitement** events with a gap smaller than `merge_gap_sec` (default 2.0s) are merged. The merged event spans from the earliest start to the latest end, and takes the maximum confidence across the merged events. Shot events are never merged — two arcs 1.5 s apart are two shots.

---

## Pipeline Orchestration

**Module:** `src/analysis/video_analyzer.py`

### Single-Video Mode (`analyze_video`)

Processes one video file through the six-phase pipeline in order:

1. Scene detection
2. Audio analysis (extract + crowd excitement + whistle detection)
3. Object detection (YOLO on every Nth frame, optionally supplemented by Roboflow)
4. Ball tracking (offline tracklet linking + shot arc detection)
5. Player tracking (ByteTrack + team classification) — skipped when `--no-players` is set
6. Event classification (multi-modal fusion)

Output events use frame numbers and timestamps relative to the video file.

### Timeline Mode (`analyze_timeline`)

Processes a DaVinci Resolve timeline export JSON. For each clip in the timeline:

1. Open the clip's media file (proxy if available, else full-res)
2. Seek to the source in-point and analyze only the used frame range
3. Run the full analysis pipeline on the source range
4. Map detected events from source-local frames to timeline frames

**Frame mapping math:**

```
fps_ratio = timeline_fps / media_fps
timeline_frame = timeline_start + (source_frame - source_start) * fps_ratio
timeline_sec = timeline_frame / timeline_fps
```

This handles FPS differences between source media and timeline (e.g., 59.94fps media on a 23.976fps timeline).

After all clips are processed, events are sorted by timeline position and merged across clip boundaries using the same merge logic.

The `--clip SPEC` flag limits analysis to specific clips (0-based). Supports single indices (`3`), comma-separated lists (`0,2,5`), ranges (`1-4`), and mixed (`0,3-5,8`).

---

## Configuration Reference

All parameters are defined as dataclasses in `src/config.py`. The top-level `AnalysisConfig` composes all sub-configs.

### VideoConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `yolo_model` | `str` | `"yolo11m.pt"` | YOLO model file (auto-downloads) |
| `yolo_confidence` | `float` | `0.5` | Minimum confidence for ball/player boxes |
| `imgsz` | `int` | `1280` | YOLO inference size on the long edge (`--imgsz`) |
| `hoop_confidence` | `float` | `0.3` | Minimum confidence for hoop/rim boxes, all backends (`--hoop-confidence`) |
| `frame_skip` | `int` | `2` | Analyze every Nth frame |
| `max_resolution` | `int` | `1920` | Downscale frames larger than this |
| `input_lut` | `Path \| None` | `None` | Path to `.cube` 3D LUT file (or `.zip`/`.lut` archive) for log footage |
| `roboflow_model_id` | `str \| None` | `None` | Roboflow model ID — primary with `detector_backend="roboflow"`, supplemental otherwise |
| `roboflow_confidence` | `float` | `0.4` | Confidence threshold for Roboflow ball/player boxes |
| `detector_backend` | `str` | `"yolo"` | `"yolo"`, `"rfdetr"` or `"roboflow"` |
| `rfdetr_model_size` | `str` | `"small"` | RF-DETR size (`nano` … `2xlarge`) |
| `rfdetr_weights` | `str \| None` | `None` | Fine-tuned RF-DETR checkpoint (`.pth`) |
| `rfdetr_num_classes` | `int \| None` | `None` | Head size of the checkpoint (derived from `--rfdetr-classes`) |
| `rfdetr_class_names` | `list[str] \| None` | `None` | Class names in dataset-index order |
| `rfdetr_resolution` | `int \| None` | `None` | RF-DETR input resolution (multiple of 56) |
| `nms_enabled` | `bool` | `False` | Per-role NMS via `supervision` (`--nms`) |
| `nms_threshold` | `float` | `0.5` | IoU threshold for NMS |
| `review_codec` | `str` | `"hevc"` | `--review-export` codec: `hevc`, `h264`, or `mp4v` (OpenCV fallback) |
| `review_quality` | `int \| None` | `None` | CRF (software encoders) or kbit/s (hardware encoders) |

### AudioConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `sample_rate` | `int` | `22050` | Audio sample rate (Hz) |
| `window_sec` | `float` | `2.0` | Mel-spectrogram window size (seconds) |
| `hop_sec` | `float` | `0.5` | Hop between analysis windows (seconds) |
| `crowd_freq_low_hz` | `int` | `500` | Lower bound of crowd noise band (Hz) |
| `crowd_freq_high_hz` | `int` | `4000` | Upper bound of crowd noise band (Hz) |
| `excitement_threshold` | `float` | `0.6` | Normalized excitement score threshold |
| `n_mels` | `int` | `128` | Number of mel filter banks |
| `whistle_freq_low_hz` | `int` | `2000` | Lower bound of whistle band (Hz) |
| `whistle_freq_high_hz` | `int` | `4500` | Upper bound of whistle band (Hz) |
| `whistle_energy_threshold` | `float` | `0.7` | Whistle onset strength threshold |
| `crowd_norm` | `str` | `"game"` | `game`: one crowd-energy scale across all clips (audio pre-pass); `clip`: per-clip min/max |
| `crowd_norm_low_pct` | `float` | `5.0` | Percentile of game-wide band energy that scores 0 |
| `crowd_norm_high_pct` | `float` | `99.5` | Percentile that scores 1 |
| `impact_freq_low_hz` / `impact_freq_high_hz` | `int` | `1000` / `6000` | Band for rim/backboard transients |
| `impact_hop_sec` | `float` | `0.005` | Onset resolution |
| `impact_min_prominence` | `float` | `10.0` | Onset prominence over the local baseline (MAD units) |
| `impact_max_duration_sec` | `float` | `0.25` | Longer bursts are not impacts |
| `impact_local_window_sec` | `float` | `1.5` | Baseline window |

### TrackingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_ball_gap_frames` | `int` | `18` | Source frames a tracklet may bridge without a detection |
| `max_ball_speed_px_per_frame` | `float` | `25.0` | Link gate grows by this per missed source frame |
| `link_slack_px` | `int` | `24` | Base link gate (detection jitter) |
| `max_link_jump_px` | `int` | `160` | Cap on the link gate; longer hops are left to the chain selection |
| `edge_margin_px` | `int` | `8` | Ball candidates centred this close to the frame border are ignored |
| `min_tracklet_detections` | `int` | `3` | Shorter tracklets are dropped (`--min-track-detections`) |
| `tracklet_size_ratio_max` | `float` | `2.5` | Max size ratio between consecutive linked boxes |
| `compensate_camera_motion` | `bool` | `True` | Link in rim/player-anchored coordinates |
| `static_span_px` | `int` | `40` | Tracklets whose compensated extent stays under this… |
| `static_clutter_sec` | `float` | `0.5` | …for at least this long are discarded as fixtures |
| `motion_full_span_px` | `int` | `200` | Compensated extent that earns full motion credit in scoring |
| `teleport_penalty_per_px` | `float` | `0.05` | Chain penalty per px of implausible jump between tracklets |
| `shot_min_arc_height_px` | `int` | `50` | Minimum arc height (pixels) for a shot |
| `shot_arc_end_descent_px` | `int` | `50` | Descent below the peak (by a detected position) that ends an arc |
| `hoop_proximity_px` | `int` | `80` | Distance (pixels) to count as through hoop (legacy centre-only fallback) |
| `hoop_x_tolerance_ratio` | `float` | `0.3` | Horizontal tolerance as fraction of hoop bbox width |
| `hoop_entry_y_margin_px` | `int` | `30` | Vertical margin above/below hoop top for entry detection |
| `hoop_obs_max_age_frames` | `int` | `30` | Hoop observations further than this from a shot's descent are ignored (→ `made=None`) |
| `shot_hoop_x_range_ratio` | `float` | `0.35` | Max horizontal distance from hoop (as fraction of frame width) for arc validation |
| `shot_require_peak_above_rim` | `bool` | `True` | Reject arcs that never rise above the rim they are judged against |
| `shot_peak_rim_margin_px` | `int` | `20` | Slack below the rim top still accepted by the peak gate |
| `shot_min_descent_ratio` | `float` | `0.15` | Ball must descend at least this fraction of ascent height (low for layups) |
| `shot_max_arc_sec` | `float` | `3.0` | Max duration of a single arc in source seconds |
| `shot_pre_peak_sec` | `float` | `0.5` | Max time before peak to include in shot event window |
| `shot_post_arc_sec` | `float` | `0.6` | Extra time past arc_end to check for made shot (backboard bounces) |
| `rim_entry_enabled` | `bool` | `True` | A descending ball that vanishes inside the rim footprint is a shot |
| `rim_entry_lookahead_sec` | `float` | `0.6` | …and a make unless re-detected above the rim within this time |
| `rim_roi_redetect` | `bool` | `True` | Second detector pass on rim crops during shot windows |
| `rim_roi_pre_sec` / `rim_roi_post_sec` | `float` | `0.15` / `1.0` | Window around each shot for the ROI pass |
| `rim_roi_scale` | `float` | `2.0` | Upscale factor for the rim crop |
| `rim_roi_width_factor` | `float` | `3.0` | Crop width in rim widths |
| `rim_roi_above_factor` / `rim_roi_below_factor` | `float` | `1.5` / `3.0` | Crop extent above / below the rim in rim heights |
| `net_x_tolerance_ratio` | `float` | `0.3` | Post-rim x must stay within the rim span ± this fraction of rim width to count as through the net |
| `post_rim_speed_made_max` | `float` | `0.85` | Measured/expected freefall speed below the rim above which the ball is judged to have passed the rim |
| `parabola_min_points` | `int` | `4` | Detected flight points needed for the projectile fit |
| `use_polygon_zone` | `bool` | `False` | Net-zone made-shot fallback (`--polygon-zone`, needs `supervision`) |
| `require_free_flight` | `bool` | `True` | A shot must leave every player box during the arc |
| `min_free_flight_sec` | `float` | `0.2` | Minimum unobstructed flight for a shot |
| `free_flight_box_pad` | `float` | `0.1` | Player-box padding for the containment test |
| `track_activation_threshold` | `float` | `0.5` | ByteTrack: detection confidence to start a track |
| `track_high_conf_threshold` | `float` | `0.6` | ByteTrack: high-confidence association split |
| `track_lost_buffer_frames` | `int` | `30` | ByteTrack: analysed frames a lost track is kept |
| `track_min_iou` | `float` | `0.1` | ByteTrack: minimum IoU for matching |
| `track_min_hits` | `int` | `3` | ByteTrack: matches before a track is reported |
| `jersey_sample_every_frames` | `int` | `20` | Source frames between stored torso crops per track |
| `jersey_crops_per_track` | `int` | `8` | Crops kept per track for team clustering |
| `jersey_min_crops` | `int` | `3` | Tracks with fewer crops stay unclassified |
| `team_method` | `str` | `"siglip"` | `siglip` embeddings + PCA + k-means, or `hsv` |
| `team_embed_model` | `str` | `"google/siglip-base-patch16-224"` | Embedding model |
| `team_outlier_std` | `float` | `2.0` | Tracks farther than this from both centres stay unassigned |
| `enable_player_tracking` | `bool` | `True` | Set `False` to skip player tracking (`--no-players`) |

### EventConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence` | `float` | `0.7` | Minimum fused confidence to report an event |
| `min_video_confidence` | `float` | `0.5` | Hard floor on a shot's video-only confidence (`--min-video-confidence`) |
| `audio_weight` | `float` | `0.4` | Share of the remaining headroom audio may add; never gates |
| `highlight_pre_pad_sec` | `float` | `3.0` | Seconds before event for highlight clip |
| `highlight_post_pad_sec` | `float` | `2.0` | Seconds after event for highlight clip |
| `merge_gap_sec` | `float` | `2.0` | Merge events closer than this (seconds) |
| `impact_window_before_sec` / `impact_window_after_sec` | `float` | `0.08` / `0.3` | Window around the rim-contact frame for an impact transient |

### CourtConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | `bool` | `False` | Run court keypoints / homography (experimental) |
| `preset` | `str` | `"nfhs"` | Court dimensions: `nfhs`, `nba`, `fiba` (`--court-preset`) |
| `model_id` | `str` | `"basketball-court-detection-2/22"` | Roboflow keypoint model (`--court-model`) |
| `every_sec` | `float` | `3.0` | Seconds between keypoint detections (`--court-every`) |
| `detection_confidence` / `keypoint_confidence` | `float` | `0.3` / `0.5` | Model and per-keypoint thresholds |
| `min_keypoints` | `int` | `7` | Inliers needed for a keyframe |
| `ransac_reproj_cm` / `max_reproj_px` | `float` | `60` / `15` | RANSAC threshold; keyframe rejection threshold |
| `min_spread_ratio` | `float` | `0.25` | Inlier landmarks must span an area, not a line |
| `calibrated_buffer_cm` | `float` | `300` | Court questions are only answered within the inlier hull + this |
| `max_keyframe_age_frames` | `int` | `600` | Don't use a keyframe farther away than this |
| `court_margin_cm` | `float` | `30` | Beyond the lines by more than this = off court (benches sit ~1 m outside) |
| `three_point_margin_cm` | `float` | `15` | Tolerance on the arc |
| `layup_max_ft` | `float` | `5.0` | Release distance below which a shot is a layup |
| `overlay` | `bool` | `True` | Draw court lines in review exports |

### SceneConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `content_threshold` | `float` | `27.0` | PySceneDetect ContentDetector threshold |
| `min_scene_len_sec` | `float` | `0.5` | Minimum scene length (seconds) |

### AnalysisConfig (top-level)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video` | `VideoConfig` | *(defaults)* | Video analysis settings |
| `audio` | `AudioConfig` | *(defaults)* | Audio analysis settings |
| `tracking` | `TrackingConfig` | *(defaults)* | Ball and player tracking settings |
| `events` | `EventConfig` | *(defaults)* | Event classification settings |
| `scene` | `SceneConfig` | *(defaults)* | Scene detection settings |
| `output_dir` | `Path` | `"output"` | Output directory for results |
| `device` | `str` | `"auto"` | Compute device: `"auto"`, `"cuda"`, `"mps"`, `"cpu"` |
| `preview` | `bool` | `False` | Show per-frame live detection preview during analysis |
| `review` | `bool` | `False` | Interactive replay with full overlays after each clip |
| `review_export` | `Path \| None` | `None` | Directory to save review replay videos (e.g. `clip_0.mp4`) |
