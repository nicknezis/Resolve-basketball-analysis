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

The analyzer processes each clip's used portion, running:
- **YOLO object detection** — finds the basketball, hoop, and players in each frame
- **Ball tracking** — follows the ball across frames with a Kalman filter, detects shot arcs
- **Shot detection** — determines if the ball passes through the hoop (made shot vs. miss)
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

Options:
  --timeline PATH          Path to a Resolve timeline export JSON (timeline mode)
  -o, --output PATH        Output JSON path (default: <input>_analysis.json)
  --yolo-model NAME        YOLO model name or path (default: yolov8m.pt)
  --yolo-confidence FLOAT  Detection confidence threshold (default: 0.5)
  --frame-skip N           Analyze every Nth frame (default: 2)
  --min-confidence FLOAT   Minimum event confidence to report (default: 0.7)
  --crowd-threshold FLOAT  Crowd excitement threshold (default: 0.6)
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

## How It Works

### Analysis Pipeline

```
Video file ──┬──► Scene Detection (PySceneDetect)
             │
             ├──► Object Detection (YOLOv8) ──► Ball Tracking (Kalman filter)
             │                                       │
             │                                       ├──► Shot Detection
             │                                       │    (arc trajectory + hoop proximity)
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
│   │   ├── object_detector.py       # YOLO-based ball/hoop/player detection
│   │   ├── ball_tracker.py          # Kalman filter tracking + shot detection
│   │   ├── player_tracker.py        # DeepSORT tracking + team color classification
│   │   ├── scene_detector.py        # PySceneDetect wrapper
│   │   └── event_classifier.py      # Multi-modal event fusion
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
| YOLO model | `--yolo-model` | `yolov8m.pt` | Model size: `n` (fast) to `x` (accurate) |
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
