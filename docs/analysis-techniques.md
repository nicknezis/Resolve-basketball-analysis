# Analysis Techniques

Technical reference for all video and audio analysis methods implemented in the basketball analysis engine.

## Overview

The engine uses a six-phase pipeline to detect basketball game events from video:

1. **Scene Detection** -- Identify shot/scene boundaries using frame-to-frame HSV comparison
2. **Audio Analysis** -- Extract crowd excitement peaks and referee whistle events from the audio track
3. **Object Detection** -- Run YOLO inference to locate the basketball, hoop, and players in each frame
4. **Ball Tracking** -- Smooth ball trajectory with a Kalman filter and detect shot arcs
5. **Player Tracking** -- Maintain player identity across frames with DeepSORT and classify teams by jersey color
6. **Event Classification** -- Fuse video and audio signals into final `GameEvent` objects with confidence scores

Each phase is implemented as an independent module under `src/analysis/`. The pipeline orchestrator (`video_analyzer.py`) chains them together and supports two modes: single-video analysis and timeline-aware multi-clip analysis for DaVinci Resolve integration.

---

## Object Detection

**Module:** `src/analysis/object_detector.py`

### Model

YOLO Medium (`yolo11m.pt`) via the Ultralytics library. The model auto-downloads on first run.

Two model modes are supported:

| Mode | Detected classes | Source |
|------|-----------------|--------|
| **COCO (stock)** | `person` (class 0), `sports ball` (class 32) | Pre-trained YOLO |
| **Custom** | `basketball`, `hoop`, `player` | User-trained model |

The detector auto-detects which mode to use by checking if the loaded model's class names contain `"basketball"`. COCO models cannot detect the hoop -- shot detection relies on arc geometry alone unless a Roboflow model is also used (see below).

### Roboflow Supplemental Detection

An optional Roboflow model (`--roboflow-model MODEL_ID`) can supplement YOLO detection. It runs on every analyzed frame after YOLO and appends its results to the same `FrameDetections` object.

Supported Roboflow class mappings:

| Roboflow class | Mapped to | Appended to |
|----------------|-----------|-------------|
| `hoop`, `rim`, `basket` | `"hoop"` | `fd.hoops` |
| `basketball`, `ball` | `"basketball"` | `fd.balls` |

This is useful in two scenarios:
- **Hoop detection with COCO models:** Stock YOLO has no hoop class, so Roboflow provides the hoop detections needed for made-shot classification.
- **Supplemental ball detection:** Roboflow basketball detections can catch balls that YOLO's generic `sports ball` class misses.

Requires the `inference` pip package and a `ROBOFLOW_API_KEY` environment variable. Confidence threshold is controlled by `--roboflow-confidence` (default 0.4).

### Frame Preprocessing

- **Frame skip:** Only every Nth frame is analyzed (default N=2) to trade accuracy for speed.
- **Downscaling:** Frames larger than `max_resolution` (default 1920px on the longest edge) are scaled down proportionally before inference.

### Device Selection

Automatic priority: CUDA > MPS (Apple Silicon) > CPU. Controlled by the `device` field on `AnalysisConfig` (`"auto"`, `"cuda"`, `"mps"`, or `"cpu"`). The device string is passed directly to YOLO's `device` parameter.

### Output

Each analyzed frame produces a `FrameDetections` object containing categorized lists of `Detection` objects (balls, hoops, players). Each detection stores:
- Class name, confidence score, bounding box `(x1, y1, x2, y2)`, frame index
- Derived properties: `center` (bbox midpoint), `area` (bbox pixel area)

---

## Ball Tracking

**Module:** `src/analysis/ball_tracker.py`

### Kalman Filter

A 4-dimensional linear Kalman filter (from `filterpy`) tracks the ball's position and velocity:

- **State vector:** `[x, y, vx, vy]` -- 2D position + 2D velocity
- **Measurement vector:** `[x, y]` -- observed ball center from detection
- **State transition:** Constant-velocity model with `dt = 1.0` (frame-based time steps)
- **Process noise (Q):** Scaled by `kalman_process_noise` (default 0.03)
- **Measurement noise (R):** Diagonal matrix scaled by `kalman_measurement_noise` (default 0.1)
- **Initial covariance (P):** Scaled by 10.0

### Spatially-Gated Detection

Rather than simply selecting the highest-confidence ball detection per frame, the tracker uses the Kalman filter's predicted position to gate incoming detections. Each candidate is scored by a weighted blend of detection confidence and proximity to the predicted position:

```
score = (1 - ball_gate_weight) * confidence + ball_gate_weight * proximity
```

where `proximity = max(0, 1 - distance / max_ball_jump_px)`. Detections beyond `max_ball_jump_px` (default 200px) from the predicted position are rejected outright. This prevents false-positive ball detections on the far side of the frame from hijacking the trajectory.

After an extended tracking gap (more than `reacquire_after_gap_frames` frames with no accepted detection), the distance gate is disabled to allow the tracker to re-lock onto any detection.

When all detections in a frame are rejected, the tracker falls back to Kalman prediction if still within `max_ball_gap_frames` (default 10), producing an interpolated position marked as `predicted=True`.

### Shot Arc Detection

Shot detection runs as a second pass over the full tracked trajectory. The algorithm applies multiple validation gates:

1. Scan positions with a sliding window looking for upward motion (decreasing y in image coordinates)
2. Track the peak (minimum y value) of the arc
3. When the ball descends past the peak by at least `shot_min_arc_height_px` (default 50px), record the arc candidate
4. Compute `arc_height = start_y - peak_y`; only accept if it exceeds the minimum threshold
5. **Maximum arc duration gate**: Reject arcs spanning more than `shot_max_arc_frames` (default 90 positions, ~3 seconds at 30fps with frame_skip=2). Real basketball shots take 1-2 seconds; longer arcs indicate tracking noise or continuous ball movement.
6. **Minimum descent ratio gate**: The ball must descend at least `shot_min_descent_ratio` (default 0.15) of its ascent height after the peak. This is set low to accommodate layups, where the player carries the ball upward (inflating the measured ascent) but the actual descent through the hoop is short. Jump shots produce ratios near 1.0; layups typically produce 0.15-0.35.
7. **Hoop-directed descent gate**: When a hoop position is known, the median x-coordinate of the ball during the descent phase must be within `shot_hoop_x_range_ratio` (default 0.5) of the frame width from the hoop's x-coordinate. This distinguishes shots (aimed at the hoop) from passes (aimed at teammates elsewhere on the court).

The event time window is also tightened: instead of spanning from where the sliding window first detected upward motion, the shot event starts at most `shot_pre_peak_frames` (default 15, ~0.5s) before the arc peak. The arc height is still calculated from the original start for correctness, but the reported event window covers only the shot flight.

### Shot Quality Metrics

During arc validation, two quality metrics are computed and stored on each `ShotEvent`:

- `hoop_x_distance`: Pixel distance between the descent median x-coordinate and the hoop x-coordinate. Lower values indicate the ball was aimed directly at the hoop.
- `descent_ratio`: The ratio of descent height to ascent height. Higher values indicate a cleaner parabolic arc.

These metrics are used downstream by the event classifier to adjust confidence scores.

### Made Shot Detection

If hoop detections exist, the median hoop position across all frames is computed. A shot is classified as "made" if any ball position during the arc passes within `hoop_proximity_px` (default 80px) Euclidean distance of the median hoop center.

---

## Player Tracking

**Module:** `src/analysis/player_tracker.py`

### DeepSORT Multi-Object Tracking

Player identity is maintained across frames using `deep-sort-realtime`:

- **Max age:** Tracks are dropped after `deepsort_max_age` (default 30) consecutive frames without a matching detection
- **N-init:** A track must be confirmed by `deepsort_n_init` (default 3) consecutive detections before it is reported
- **Embedder GPU:** DeepSORT's appearance embedder uses CUDA if available; MPS is not supported by the library, so on Apple Silicon the embedder always runs on CPU

Detections are fed to DeepSORT in `[x1, y1, width, height]` format with confidence and class label.

### Jersey Color Sampling

For each confirmed track in each frame, the dominant jersey color is extracted:

1. Crop the **upper 40%** of the player bounding box (approximates the torso/jersey area)
2. Convert the crop from BGR to HSV color space
3. Compute the **median HSV value** across all pixels in the crop
4. Store the sample; each player accumulates samples over their tracked lifetime

### Team Classification

After all frames are processed, players are classified into two teams:

1. For each player with at least 3 color samples, compute the **median HSV** across all their samples
2. Stack all player median colors into a matrix
3. Run **OpenCV k-means** (k=2, up to 100 iterations, epsilon 0.2, 10 random restarts with `KMEANS_PP_CENTERS`)
4. Assign each player to `"team_a"` or `"team_b"` based on their cluster label

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

Video and audio signals are combined using a weighted linear fusion:

```
fused_confidence = video_weight * video_conf + audio_weight * audio_conf
```

- `video_weight` = 0.6 (default)
- `audio_weight` = 0.4 (default)

### Shot Event Classification

For each `ShotEvent` from the ball tracker:

1. **Base video confidence:**
   - Made shot: 0.7
   - Missed shot: 0.5
   - Bonus +0.1 (capped at 1.0) for arcs taller than 150px
   - Bonus +0.1 if `hoop_x_distance < 50px` (descent lands near the hoop)
   - Bonus +0.05 if `descent_ratio > 0.7` (clean parabolic descent)
2. **Audio correlation:** Search for audio events in a window from **1 second before** to **4 seconds after** the shot (crowd reaction lags the play). Crowd excitement scores are taken directly; whistle scores are scaled by 0.8.
3. **Fusion:** Apply the weighted formula above
4. **Filter:** Only events with `confidence >= min_confidence` (default 0.7) are kept

### Standalone Crowd Events

High crowd excitement peaks not explained by any shot event (not overlapping within a 2-second padding) are emitted as `"crowd_excitement"` events. These capture plays the video analysis might miss (steals, blocks, fast breaks without shots). They carry `video_confidence = 0.0` and `audio_confidence` equal to their score.

### Event Merging

After classification, events are sorted by time and adjacent events of the **same type** with a gap smaller than `merge_gap_sec` (default 2.0s) are merged. The merged event spans from the earliest start to the latest end, and takes the maximum confidence across the merged events.

---

## Pipeline Orchestration

**Module:** `src/analysis/video_analyzer.py`

### Single-Video Mode (`analyze_video`)

Processes one video file through the six-phase pipeline in order:

1. Scene detection
2. Audio analysis (extract + crowd excitement + whistle detection)
3. Object detection (YOLO on every Nth frame, optionally supplemented by Roboflow)
4. Ball tracking (Kalman filter + shot arc detection)
5. Player tracking (DeepSORT + team classification) — skipped when `--no-players` is set
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
| `yolo_confidence` | `float` | `0.5` | Minimum detection confidence |
| `frame_skip` | `int` | `2` | Analyze every Nth frame |
| `max_resolution` | `int` | `1920` | Downscale frames larger than this |
| `input_lut` | `Path \| None` | `None` | Path to `.cube` 3D LUT file (or `.zip`/`.lut` archive) for log footage |
| `roboflow_model_id` | `str \| None` | `None` | Roboflow model ID for supplemental detection (e.g. `"basketball-detection/1"`) |
| `roboflow_confidence` | `float` | `0.5` | Confidence threshold for Roboflow model |

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

### TrackingConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `kalman_process_noise` | `float` | `0.03` | Kalman filter process noise scale |
| `kalman_measurement_noise` | `float` | `0.1` | Kalman filter measurement noise scale |
| `max_ball_gap_frames` | `int` | `10` | Max frames to interpolate missing ball |
| `shot_min_arc_height_px` | `int` | `50` | Minimum arc height (pixels) for a shot |
| `hoop_proximity_px` | `int` | `80` | Distance (pixels) to count as through hoop |
| `hoop_x_tolerance_ratio` | `float` | `0.3` | Horizontal tolerance as fraction of hoop bbox width |
| `hoop_entry_y_margin_px` | `int` | `30` | Vertical margin above/below hoop top for entry detection |
| `max_ball_jump_px` | `int` | `200` | Max pixel distance from predicted position to accept a detection |
| `ball_gate_weight` | `float` | `0.5` | Blend factor for gated detection: 0=pure confidence, 1=pure proximity |
| `reacquire_after_gap_frames` | `int` | `5` | After this many missed frames, disable distance gate for re-acquisition |
| `shot_hoop_x_range_ratio` | `float` | `0.5` | Max horizontal distance from hoop (as fraction of frame width) for arc validation |
| `shot_min_descent_ratio` | `float` | `0.15` | Ball must descend at least this fraction of ascent height (low for layups) |
| `shot_max_arc_frames` | `int` | `90` | Max tracked positions in a single arc (~3s at 30fps/skip-2) |
| `shot_pre_peak_frames` | `int` | `15` | Max frames before peak to include in shot event window |
| `deepsort_max_age` | `int` | `30` | Frames before dropping unmatched track |
| `deepsort_n_init` | `int` | `3` | Detections needed to confirm a track |
| `enable_player_tracking` | `bool` | `True` | Set `False` to skip DeepSORT player tracking (`--no-players`) |

### EventConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `min_confidence` | `float` | `0.7` | Minimum confidence to report an event |
| `video_weight` | `float` | `0.6` | Video signal weight in fusion |
| `audio_weight` | `float` | `0.4` | Audio signal weight in fusion |
| `highlight_pre_pad_sec` | `float` | `3.0` | Seconds before event for highlight clip |
| `highlight_post_pad_sec` | `float` | `2.0` | Seconds after event for highlight clip |
| `merge_gap_sec` | `float` | `2.0` | Merge events closer than this (seconds) |

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
