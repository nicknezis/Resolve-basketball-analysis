# Basketball Game Analyzer for DaVinci Resolve

Automatically detect interesting plays in basketball game footage — made shots, dunks, three-pointers, fast breaks, crowd reactions — and mark them on your DaVinci Resolve timeline.

The tool works in two steps:

1. **Analyze** — A standalone Python engine processes your video and audio outside of Resolve, detecting events using computer vision (YOLO object detection, Kalman filter ball tracking) and audio analysis (crowd excitement scoring, whistle detection). It produces a JSON file.
2. **Import** — A small Resolve script reads that JSON and drops color-coded markers onto your timeline, right where the action happens.

Because the analysis runs independently, you can process footage on any machine — even a headless GPU server — and bring the results back to your editing workstation.

---

## Quick Start (I have a timeline in Resolve, what do I do?)

This walkthrough assumes you have DaVinci Resolve Studio open with a basketball game project loaded and a timeline active.

### 1. Install

```bash
# Clone the repository
git clone https://github.com/nicknezis/Resolve-basketball-analysis.git
cd Resolve-basketball-analysis

# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate    # macOS / Linux
# .venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

`requirements.txt` pins `setuptools<81` because `deep-sort-realtime` still
imports `pkg_resources`. The `inference` package declares `setuptools>=83`, so
`pip check` reports a conflict — it is harmless. Use Python 3.12: newer
interpreters don't have torch wheels yet.

You also need **FFmpeg** installed for audio extraction:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt install ffmpeg

# Windows — download from https://ffmpeg.org/download.html and add to PATH
```

### 2. Export your timeline from Resolve

With DaVinci Resolve running and your timeline open, run:

```bash
python -m src.resolve.export --prefer-proxy -o my_game_timeline.json
```

This connects to Resolve via its scripting API and captures:
- Every clip on every video track
- Each clip's position on the timeline (start/end frame)
- Each clip's source in/out points in the original media file
- Full-resolution and proxy file paths

The `--prefer-proxy` flag tells the analyzer to use your smaller proxy files instead of full-res media, which is much faster for analysis. If a clip has no proxy, it falls back to the full-resolution file.

You should see output like:

```
Timeline: 'Game vs Central High' (29.97 fps)
  Tracks exported: 2
  Total clips: 14
  Unique media files: 6
  Media with proxies: 6

Exported to: my_game_timeline.json
```

### 3. Run the analysis

Now analyze the footage. This step does **not** need Resolve running — you can do it on any machine that has access to the media files.

```bash
basketball-analyze --timeline my_game_timeline.json -o my_game_analysis.json
```

Or if you installed with `pip install -e .`:

```bash
python -m src.cli --timeline my_game_timeline.json -o my_game_analysis.json
```

That runs stock YOLO, which **cannot see the hoop** — every shot comes back as an
attempt with `made: null`. For made/miss detection use a basketball model
(see [Detection Backends](#detection-backends--model-weights)); the
recommended invocation for gym footage is:

```bash
python -m src.cli --timeline my_game_timeline.json \
  --detector roboflow --roboflow-model basketball-detection-dn6fg/4 \
  --input-lut lut/FX30.cube --nms \
  --review-export review \
  -o my_game_analysis.json
```

(`--input-lut` only if the footage is log; `--review-export` writes annotated
HEVC clips you can scrub to check the detections — `--review-codec h264` if
you need wider compatibility.)

The analyzer processes each clip's used portion, running:
- **YOLO object detection** — finds the basketball, hoop, and players in each frame
- **Ball tracking** — follows the ball across frames with a spatially-gated Kalman filter (rejects false detections far from predicted position)
- **Shot detection** — validates arcs against multiple gates (duration, descent quality, hoop direction) and determines made vs. miss
- **Player tracking** — tracks players with DeepSORT, classifies teams by jersey color
- **Crowd excitement** — analyzes audio mel-spectrograms in the 500–4000 Hz band for roar peaks
- **Whistle detection** — finds referee whistles via spectral peaks in the 2000–4500 Hz band
- **Event classification** — fuses video and audio signals into final events with confidence scores

Events are mapped back to **timeline frame positions**, so they'll land exactly where you expect.

You should see output like:

```
=== Analysis Complete ===
  Clips analyzed: 14
  Clips skipped:  0
  Total events: 23
  Total scenes: 47
  Event breakdown:
    crowd_excitement: 8
    made_shot: 9
    shot_attempt: 6

Results saved to: my_game_analysis.json
```

### 4. Import markers into Resolve

Back at your Resolve workstation, with the same timeline open:

```bash
python -m src.resolve.markers my_game_analysis.json
```

This creates color-coded markers on your timeline:

| Marker Color | Event Type |
|-------------|------------|
| Blue | Made 2-point shot |
| Green | Made 3-pointer |
| Red | Dunk |
| Yellow | Fast break |
| Purple | Block / steal |
| Pink | Buzzer beater |
| Cyan | Crowd reaction (audio-only detection) |
| Cream | Shot attempt (missed) |

Each marker includes a note with the confidence score and detection details. The full event data is stored in the marker's `customData` field for programmatic access.

Options:

```bash
# Clear old markers before importing
python -m src.resolve.markers --clear my_game_analysis.json

# Only import high-confidence events
python -m src.resolve.markers --min-confidence 0.8 my_game_analysis.json
```

---

## Alternative: Analyze a Single Video File

If you just have a video file and don't need timeline integration:

```bash
basketball-analyze game_footage.mp4
```

This produces `game_footage_analysis.json` in the same directory. The events use frame numbers relative to the video file. You can still import this into Resolve — the marker script will place them based on absolute frame position.

---

## CLI Reference

### `basketball-analyze` — Run analysis

```
basketball-analyze [video] [--timeline JSON] [options]

Positional:
  video                    Path to a video file (single-video mode)

Input:
  --timeline PATH          Path to a Resolve timeline export JSON (timeline mode)
  --clip SPEC              Clips to analyze: single (3), list (0,2,5), range (1-4),
                           or mixed (0,3-5,8). Timeline mode only.
  --input-lut PATH         3D LUT (.cube or .zip/.lut archive) for log footage

Detection:
  --detector {yolo,rfdetr,roboflow}
                           Detection model backend (default: yolo)
  --yolo-model NAME        YOLO model name or path (default: yolo11m.pt)
  --yolo-confidence FLOAT  Confidence threshold for ball/player boxes (default: 0.5)
  --hoop-confidence FLOAT  Confidence threshold for hoop/rim boxes (default: 0.3)
  --imgsz N                YOLO inference size on the long edge (default: 1280;
                           Ultralytics' own 640 default shrinks the ball to ~10 px)
  --roboflow-model ID      Roboflow model (e.g. basketball-player-detection-3-ycjdo/6).
                           Primary detector with --detector roboflow, otherwise a
                           supplemental hoop/ball source. Runs locally via the
                           `inference` package; requires ROBOFLOW_API_KEY.
  --roboflow-confidence F  Roboflow ball/player confidence threshold (default: 0.4)
  --frame-skip N           Analyze every Nth frame (default: 2)
  --no-players             Disable player detection and tracking (faster)

RF-DETR backend (only used with --detector rfdetr; see "Detection Backends"):
  --rfdetr-size SIZE       nano|small|medium|large|xlarge|2xlarge (default: small)
  --rfdetr-weights PATH    Path to an RF-DETR checkpoint (.pth). Omit to
                           auto-download the base COCO weights for the size.
  --rfdetr-classes NAMES   Comma-separated class names for a fine-tuned model,
                           in dataset index order (e.g. ball,hoop,player,...)
  --rfdetr-resolution N    Input resolution, must be divisible by 56

Detection post-processing (all backends):
  --nms                    Deduplicate overlapping boxes via supervision NMS
  --nms-threshold FLOAT    IoU threshold for NMS (default: 0.5)
  --consensus N            Require N consistent ball detections before starting
                           a track (1=disabled; default: 3)
  --polygon-zone           Also count a made shot when the descent enters a
                           trapezoidal net zone below the rim (needs supervision)

Classification:
  --min-confidence FLOAT   Minimum fused (video+audio) confidence (default: 0.7)
  --min-video-confidence F Minimum video-only confidence for a shot (default: 0.5)
  --crowd-threshold FLOAT  Crowd excitement threshold (default: 0.6)

Output:
  -o, --output PATH        Output JSON path (default: <input>_analysis.json)
  --review-export DIR      Save annotated review videos to DIR (clip_<N>.mp4),
                           no window is opened
  --review-codec CODEC     hevc (default) | h264 | mp4v. hevc/h264 are encoded by
                           ffmpeg (VideoToolbox on macOS when available) with AAC
                           audio and the hvc1 tag; mp4v is the OpenCV fallback
  --review-quality N       CRF for libx264/libx265 (default 18/20) or kbit/s for
                           hardware encoders (default ~12 / ~7 Mbit/s per
                           megapixel for h264 / hevc)

Display:
  --preview                Show live detection preview during analysis
  --review                 Interactive replay with overlays after analysis
  --device {auto,cuda,cpu} Compute device (default: auto)
  -v, --verbose            Enable debug logging
```

### `python -m src.resolve.export` — Export timeline from Resolve

```
Options:
  -o, --output PATH        Output JSON path (default: <timeline_name>_timeline.json)
  --prefer-proxy           Use proxy media paths for analysis when available
  --track N                Export only this video track number (1-based)
  -v, --verbose            Enable debug logging
```

### `python -m src.resolve.markers` — Import markers into Resolve

```
Positional:
  json_file                Path to analysis results JSON

Options:
  --clear                  Remove all existing markers before importing
  --min-confidence FLOAT   Only import events above this confidence (default: 0.0)
  -v, --verbose            Enable debug logging
```

---

## Detection Backends & Model Weights

The analyzer supports three detection backends, selected with `--detector`:

| Backend | Flag | Weights | Hoop class? |
|---------|------|---------|-------------|
| **YOLO** (default) | `--detector yolo` | `yolo11m.pt` (or `--yolo-model`) — Ultralytics downloads on first run | **No** with stock COCO weights; yes with a basketball fine-tune |
| **RF-DETR** | `--detector rfdetr` | base COCO checkpoint per `--rfdetr-size`, or `--rfdetr-weights PATH` | Only with a fine-tuned checkpoint + `--rfdetr-classes` |
| **Roboflow** | `--detector roboflow --roboflow-model ID` | downloaded once by the `inference` package (needs `ROBOFLOW_API_KEY`), then runs locally on ONNX | Yes — pick a model with a `rim`/`hoop` class |

> **Made/miss needs a hoop.** Stock COCO weights have no hoop/rim class, so with
> plain `yolo11m.pt` every shot is reported as `shot_attempt` with `details.made: null`
> and the detector logs a warning at start-up. Use `--detector roboflow`, add
> `--roboflow-model` as a supplement, or load a basketball fine-tune.

All backends can be combined with `--nms` (deduplicate overlapping boxes),
`--consensus N` (require N consistent ball detections before starting a track),
and `--roboflow-model` (supplemental hoop/ball detection for the YOLO/RF-DETR
backends). Class names from every model are normalised to roles (`ball`,
`ball-in-basket`, `hoop`/`rim`, `player`, `referee`) in
`src/analysis/object_detector.py::CLASS_ROLE_MAP`; referees are kept out of
team classification, and a `ball-in-basket` box during a shot's descent counts
as a make.

### Choosing a Roboflow model

`scripts/bench_hoop_models.py` samples frames from your own timeline and scores
candidate Universe models on hoop / ball detection rate and speed:

```bash
python scripts/bench_hoop_models.py --timeline timeline.json --clip 0-4 \
  --input-lut lut/FX30.cube --out docs/model-bakeoff.md
```

Results for the reference footage are in [`docs/model-bakeoff.md`](docs/model-bakeoff.md).
Public Universe models are trained mostly on broadcast footage; the durable fix
is fine-tuning on frames from your own gym (see `docs/roboflow-pipeline-improvements.md`).

### RF-DETR weights

RF-DETR requires the optional dependency: `pip install rfdetr supervision`
(or `pip install -e ".[rfdetr]"`).

- **Base COCO weights (auto-downloaded).** Running `--detector rfdetr` *without*
  `--rfdetr-weights` constructs the model for the chosen `--rfdetr-size`, and the
  `rfdetr` package downloads that size's COCO-pretrained checkpoint into its own
  cache — the same way YOLO auto-downloads `yolo11m.pt`. Nothing in this repo
  downloads weights; it only passes a path through to the library.
- **Explicit checkpoint.** `--rfdetr-weights ./rf-detr-small.pth` loads a specific
  `.pth` file. This repo does **not** ship or fetch that file — you place it
  yourself (e.g. a copy of the library's base checkpoint, or a checkpoint you
  trained).

> **COCO weights ≠ basketball classes.** A COCO-pretrained RF-DETR checkpoint
> detects the 80 generic COCO classes (`person`, `sports ball`) — the same
> categories YOLO already provides. To get dedicated `ball` / `hoop` / `rim` /
> `player` classes in one pass you need a checkpoint **fine-tuned on a basketball
> dataset**, loaded with `--rfdetr-weights` *and* described with `--rfdetr-classes`
> (names in dataset-index order). Roboflow's
> [Basketball Player Detection dataset](https://universe.roboflow.com/roboflow-jvuqo/basketball-player-detection-3-ycjdo)
> and [RF-DETR fine-tuning notebooks](https://colab.research.google.com/github/roboflow/rf-detr)
> are the starting point for producing such a checkpoint. Without `--rfdetr-classes`,
> class IDs are interpreted as COCO indices.

Example — fine-tuned RF-DETR with the 10-class basketball dataset:

```bash
basketball-analyze --timeline timeline.json --clip 0 \
  --detector rfdetr --rfdetr-weights ./basketball-rfdetr.pth \
  --rfdetr-classes ball,ball-in-basket,number,player,player-in-possession,player-jump-shot,player-layup-dunk,player-shot-block,referee,rim \
  --nms --consensus 3 -o analysis.json
```

`.pt` and `.pth` weight files are gitignored — keep them out of the repo.

---

## How It Works

### Analysis Pipeline

```
Video file ──┬──► Scene Detection (PySceneDetect)
             │
             ├──► Object Detection (YOLO) ──► Ball Tracking (spatially-gated Kalman filter)
             │                                       │
             │                                       ├──► Shot Detection
             │                                       │    (arc validation + hoop-directed descent)
             │                                       │
             ├──► Player Tracking (DeepSORT) ──► Team Classification
             │                                   (jersey color k-means)
             │
Audio track ─┼──► Crowd Excitement Scoring (mel-spectrogram energy)
             │
             └──► Whistle Detection (spectral peak analysis)

                          │
                          ▼
                  Event Classifier
                  (fuses video + audio confidence)
                          │
                          ▼
                  events.json ──► Resolve markers
```

### Timeline-Aware Processing

When you export a timeline, clips often use only a portion of the underlying media file. The analyzer handles this correctly:

- Each clip's **source in/out points** are extracted from Resolve (via `GetLeftOffset` / `GetRightOffset`)
- The analyzer seeks to the source in-point and processes only the used frames
- Audio is extracted and trimmed to match the same source range
- Detected events are mapped from **source-local frame numbers** to **timeline frame numbers**, accounting for FPS differences between the media and the timeline

This means a clip trimmed from a 2-hour recording to 45 seconds only analyzes those 45 seconds, and the resulting markers land at the correct position on your timeline.

### Proxy Support

If your project uses proxy media (smaller, lower-res files for faster editing), pass `--prefer-proxy` during export:

```bash
python -m src.resolve.export --prefer-proxy -o timeline.json
```

The analyzer will use the proxy files for detection, which is significantly faster. Since YOLO downscales frames to 640px for inference anyway, using proxies rarely affects detection accuracy.

---

## Project Structure

```
Resolve-basketball-analysis/
├── src/
│   ├── analysis/                    # Standalone analysis engine
│   │   ├── video_analyzer.py        # Pipeline orchestrator (single + timeline modes)
│   │   ├── audio_analyzer.py        # Crowd excitement + whistle detection
│   │   ├── object_detector.py       # YOLO / RF-DETR + Roboflow detection (ball, hoop, players)
│   │   ├── ball_tracker.py          # Kalman filter tracking + shot detection
│   │   ├── player_tracker.py        # DeepSORT tracking + team color classification
│   │   ├── scene_detector.py        # PySceneDetect wrapper
│   │   ├── event_classifier.py      # Multi-modal event fusion
│   │   ├── preview.py               # Live preview + review replay + video export
│   │   └── color.py                 # 3D LUT parsing + per-frame color conversion
│   │
│   ├── resolve/                     # DaVinci Resolve integration scripts
│   │   ├── export.py                # Export timeline structure to JSON
│   │   ├── markers.py               # Import analysis results as markers
│   │   └── highlights.py            # Auto-generate highlight sub-timeline
│   │
│   ├── config.py                    # All configuration and thresholds
│   └── cli.py                       # Command-line entry point
│
├── models/                          # Model weights (not checked in)
├── tests/                           # Test suite
├── PLAN.md                          # Technical architecture plan
├── pyproject.toml                   # Python project configuration
└── requirements.txt                 # pip dependencies
```

---

## Requirements

- **Python** 3.10+
- **FFmpeg** (for audio extraction)
- **DaVinci Resolve Studio** (for the export/import scripts — the scripting API requires the paid version)
- **NVIDIA GPU** with CUDA is recommended for YOLO inference but not required (falls back to CPU)

### Resolve Scripting API Setup

DaVinci Resolve's Python scripting API must be accessible. Resolve Studio installs a `DaVinciResolveScript.py` module that the export and import scripts load automatically. The typical locations are:

| OS | Path |
|----|------|
| Linux | `/opt/resolve/Developer/Scripting/Modules/` |
| macOS | `/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting/Modules/` |
| Windows | `C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules\` |

If the scripts can't find the module, you can add the path manually:

```bash
export PYTHONPATH="/opt/resolve/Developer/Scripting/Modules:$PYTHONPATH"
```

Resolve must be running when you execute the export or import scripts.

---

## Configuration

All detection thresholds live in `src/config.py` and can be tuned without modifying analysis code. The most useful ones are exposed as CLI flags:

| Setting | CLI Flag | Default | What It Controls |
|---------|----------|---------|-----------------|
| YOLO model | `--yolo-model` | `yolo11m.pt` | Model size: `n` (fast) to `x` (accurate) |
| Detection threshold | `--yolo-confidence` | `0.5` | How confident YOLO must be to report a detection |
| Frame skip | `--frame-skip` | `2` | Analyze every Nth frame (higher = faster, less accurate) |
| Min event confidence | `--min-confidence` | `0.7` | Events below this are excluded from output |
| Crowd threshold | `--crowd-threshold` | `0.6` | Sensitivity of crowd excitement detection |

---

## Troubleshooting

**"Cannot connect to DaVinci Resolve"**
- Make sure Resolve is running before executing export/import scripts
- Ensure you have Resolve Studio (not the free version) — the scripting API requires Studio

**"Media file not found" during analysis**
- The timeline export JSON contains absolute file paths from the machine where the export ran
- If you're analyzing on a different machine, the paths won't match. Edit the JSON or ensure the media is mounted at the same paths.

**Analysis is slow**
- Use `--prefer-proxy` during export to analyze proxy files instead of full-res
- Increase `--frame-skip` (e.g., `--frame-skip 5`) to analyze fewer frames
- Use a smaller YOLO model: `--yolo-model yolov8n.pt` (fastest) or `yolov8s.pt`
- Ensure CUDA is available: the analyzer auto-detects GPU by default

**Markers appear at wrong positions**
- Check that you're importing into the same timeline that was exported
- If you re-edited the timeline after exporting, re-export and re-analyze

**YOLO downloads a model on first run**
- This is normal. Ultralytics auto-downloads the model weights the first time. Subsequent runs use the cached file.

**RF-DETR: `ImportError: The 'rfdetr' package is required`**
- The RF-DETR backend is optional. Install it with `pip install rfdetr supervision` (or `pip install -e ".[rfdetr]"`).

**RF-DETR detects `person`/`sports ball` instead of `ball`/`hoop`/`player`**
- You're running COCO-pretrained weights, which only know the 80 COCO classes. Load a basketball fine-tuned checkpoint with `--rfdetr-weights` and pass the class names via `--rfdetr-classes` (in dataset-index order). See [Detection Backends & Model Weights](#detection-backends--model-weights).
