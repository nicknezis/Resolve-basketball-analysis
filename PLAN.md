# DaVinci Resolve Basketball Analysis Plugin — Technical Plan

## Executive Summary

This document plans a DaVinci Resolve plugin system that analyzes basketball game footage to detect interesting plays. The recommended architecture is a **hybrid approach**: a Python-based analysis engine for video/audio processing combined with Resolve's scripting API for timeline integration. An optional OpenFX (OFX) plugin can provide real-time visual overlays.

---

## 1. Plugin Framework Evaluation

### 1.1 OpenFX (OFX)

OpenFX is an open-source C API standard (BSD 3-Clause, Academy Software Foundation) for 2D visual effects plugins. It has been stable since 2004, with version 1.5 released September 2024 adding GPU support via CUDA, OpenCL, and Metal.

**How it works:**
- Plugins compile as `.ofx.bundle` shared libraries installed in standard OS paths
- The host provides C function-pointer "suites" (Property, Image Effect, Parameter, OpenGL/GPU)
- The plugin implements entry points; the host calls "actions" (describe, create instance, render, parameter changed)
- Direct in-memory access to video frame pixel data in the render pipeline

**Relevance to this project:**
- Gives real-time per-frame pixel access — useful for overlay visualization
- Cannot access audio data (image-only API)
- Cannot manipulate timelines, markers, or project structure
- Requires C++ development with CUDA for Resolve GPU acceleration
- Requires DaVinci Resolve Studio (paid version)
- Embedding ML models (PyTorch/YOLO) in C++ is complex

**Verdict:** Useful as an optional visualization layer, not suitable as the primary analysis architecture.

### 1.2 ResolveFX

ResolveFX is **not a separate API**. It is Blackmagic's brand name for their built-in proprietary effects shipped with Resolve Studio. They use OpenFX internally but are deeply optimized for Resolve's pipeline. Third-party developers cannot create ResolveFX — only standard OFX plugins.

### 1.3 DaVinci Resolve Scripting API (Python/Lua)

A scripting interface for automation and workflow control. Available via Python 3.10+ and Lua (built-in LuaJIT).

**Can do:**
- Project management (create, open, close projects)
- Media pool operations (import, organize, set metadata)
- Timeline manipulation (create timelines, add/remove clips, set in/out points)
- Marker management (add/query/delete markers with color, name, note, duration, custom data)
- Render automation (set format/codec, queue and start renders)
- Run headless via `-nogui` for batch processing

**Cannot do:**
- No in-memory frame access (must render to disk first)
- No direct Neural Engine access
- Limited UI (no custom parameter panels)
- Documentation is sparse and sometimes inaccurate

**Key marker API for this project:**
```python
timeline.AddMarker(frameId, color, name, note, duration, customData)
timeline.GetMarkers()  # returns {frameId: {color, duration, note, name, customData}}
timeline.DeleteMarkerAtFrame(frameId)
```

**Verdict:** Essential for the integration layer — exports media, imports analysis results as markers, and manipulates timelines.

### 1.4 Other Frameworks

| Framework | Suitable? | Why / Why Not |
|-----------|-----------|---------------|
| **DCTL** (DaVinci Color Transform Language) | No | Per-pixel color math only; no multi-frame data, tracking, or external libraries |
| **Fuses** (Lua Fusion plugins) | No | CPU-only, no access to external C/ML libraries |
| **Workflow Integration Plugins** (Electron) | Partial | Custom UI inside Resolve, but Windows/macOS only (no Linux), adds Electron complexity |
| **Encoder/Codec Plugins** | No | Encoding/decoding only |

### 1.5 DaVinci Resolve Neural Engine

The Neural Engine is Blackmagic's proprietary AI/ML inference engine. It powers features like Magic Mask, Face Recognition, Scene Cut Detection, Speed Warp, Super Scale, Object Removal, and Smart Reframing.

**Critical finding: The Neural Engine is NOT accessible to third-party plugins or scripts.**

- No public API to invoke Neural Engine features programmatically
- No way to load custom ML models into it
- No documented interface for developers
- OFX plugins and the Neural Engine can compete for GPU memory

We must run our own ML models externally.

---

## 2. Recommended Architecture: Hybrid Approach

```
┌─────────────────────────────────────────────────────────┐
│                   DaVinci Resolve                        │
│                                                          │
│  ┌──────────────┐    ┌─────────────────────────────┐    │
│  │ Scripting API │    │ OFX Plugin (optional)       │    │
│  │ (Python)      │    │ Visual overlays on timeline │    │
│  │               │    │ (tracking boxes, arcs, etc.)│    │
│  │ • Export media│    └─────────────────────────────┘    │
│  │ • Add markers │                                       │
│  │ • Edit timeline│                                      │
│  └──────┬───────┘                                        │
│         │                                                │
└─────────┼────────────────────────────────────────────────┘
          │  file I/O + API calls
          │
┌─────────▼────────────────────────────────────────────────┐
│            Analysis Engine (Python)                        │
│                                                            │
│  ┌────────────┐ ┌────────────┐ ┌───────────────────────┐ │
│  │ Video       │ │ Audio      │ │ Event Classification  │ │
│  │ Analysis    │ │ Analysis   │ │                       │ │
│  │             │ │            │ │ • Made shots          │ │
│  │ • YOLO      │ │ • librosa  │ │ • Fast breaks        │ │
│  │ • OpenCV    │ │ • mel-spec │ │ • Dunks / alley-oops │ │
│  │ • MediaPipe │ │ • MFCC     │ │ • Blocks             │ │
│  │ • PyScene   │ │ • crowd    │ │ • 3-pointers         │ │
│  │   Detect    │ │   scoring  │ │ • Buzzer beaters     │ │
│  └────────────┘ └────────────┘ └───────────────────────┘ │
│                                                            │
│  Output: JSON with timestamped events + confidence scores  │
└────────────────────────────────────────────────────────────┘
```

### Why Hybrid?

| Concern | Hybrid Approach |
|---------|-----------------|
| Frame access | Render to disk, then process with full Python ML stack |
| Audio access | Extract audio track, analyze with librosa |
| ML models | Full PyTorch/YOLO/MediaPipe ecosystem available |
| Timeline integration | Scripting API adds markers, creates sub-timelines |
| Real-time overlays | Optional OFX plugin reads analysis JSON, draws overlays |
| Development speed | Python for analysis; C++ only if overlays needed |
| Cross-platform | Python + Resolve scripting works on Linux, macOS, Windows |

---

## 3. Analysis Engine Design

### 3.1 Video Analysis Pipeline

#### Scene/Shot Change Detection
- **Tool:** PySceneDetect (content-aware mode)
- **Purpose:** Segment continuous footage into discrete shots/scenes before deeper analysis
- **Method:** Frame histogram differences with adaptive thresholding
- **Output:** List of (start_frame, end_frame) for each scene

#### Object Detection (Ball, Hoop, Players)
- **Tool:** YOLOv8 or YOLOv11 (Ultralytics)
- **Purpose:** Detect basketball, hoop/rim, and players per frame
- **Method:** Fine-tune on basketball-specific datasets (Roboflow has ~3,000+ annotated basketball images)
- **Output:** Per-frame bounding boxes with class labels and confidence scores

#### Ball Tracking and Shot Detection
- **Tool:** YOLO detections + Kalman filter
- **Purpose:** Track ball trajectory, detect made/missed shots
- **Method:**
  1. Detect ball position per frame with YOLO
  2. Smooth trajectory with Kalman filter
  3. Fit linear regression to ball arc
  4. Detect scoring: ball moves downward through hoop bounding box across consecutive frames
- **Reference:** [AI-Basketball-Shot-Detection-Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker)

#### Player Tracking
- **Tool:** YOLO + DeepSORT or ByteTrack
- **Purpose:** Maintain player identity across frames, classify teams
- **Method:**
  1. YOLO detects player bounding boxes
  2. DeepSORT/ByteTrack assigns persistent IDs using appearance + motion features
  3. HSV color analysis on jersey regions for team classification
- **Optional:** SAM2 for precise segmentation, SmolVLM2 for jersey number OCR

#### Pose Estimation (Optional, for shooting form)
- **Tool:** MediaPipe Pose
- **Purpose:** Detect shooting motion, analyze form
- **Method:** Detect body joints, measure elbow/wrist/shoulder angles, identify shooting posture

### 3.2 Audio Analysis Pipeline

#### Audio Extraction
- Extract audio track from video using FFmpeg
- Resample to consistent rate (22,050 Hz standard for librosa)

#### Crowd Excitement Detection
- **Tool:** librosa
- **Method:**
  1. Compute mel-spectrograms over sliding windows (e.g., 2-second windows, 0.5s hop)
  2. Calculate RMS energy in frequency bands where crowd noise is prominent (500 Hz – 4 kHz)
  3. Compute "excitement score" = normalized energy relative to baseline
  4. Threshold to identify excitement peaks
- **Validation:** Warner Bros. Discovery uses this approach in production — 24/25 match events detected correctly

#### Whistle/Buzzer Detection
- **Method:** Detect sharp tonal peaks at specific frequencies
- **Tool:** librosa onset detection + spectral centroid analysis

#### Commentary Excitement (Optional)
- **Method:** MFCC-based classification to detect raised voice pitch/energy in commentary audio
- **Tool:** pyAudioAnalysis or custom CNN classifier on MFCCs

### 3.3 Event Classification

Combine video and audio signals to classify basketball events:

| Event Type | Video Signals | Audio Signals | Confidence Boost |
|------------|--------------|---------------|-----------------|
| **Made shot** | Ball trajectory through hoop | Crowd roar, whistle | High if both present |
| **3-pointer** | Shot origin beyond arc + made shot | Louder crowd reaction | Court position data |
| **Dunk** | Player near hoop + ball velocity spike | Crowd roar peak | Pose estimation (arm extension) |
| **Fast break** | Rapid player movement direction change | Building crowd noise | Multiple players tracked |
| **Block** | Two players converging + ball trajectory reversal | Crowd reaction | Pose estimation |
| **Steal/turnover** | Ball possession change between teams | Mixed crowd reaction | Team color tracking |
| **Buzzer beater** | Made shot + game clock near zero | Maximum crowd roar | Audio buzzer detection |
| **Free throw** | Static player positions + single shooter | Quiet → reaction | Scene pattern matching |

#### Multi-Modal Fusion
- Assign independent confidence scores from video and audio pipelines
- Weight and combine: `final_score = w_video * video_conf + w_audio * audio_conf`
- A January 2025 paper (arXiv 2501.16100v2) demonstrated that combining audio + video significantly improves highlight detection
- Crowd noise "roar" maintains high prediction scores, bridging nearby exciting events into single highlights

### 3.4 Output Format

```json
{
  "analysis_version": "1.0",
  "source_file": "game_footage.mp4",
  "fps": 29.97,
  "events": [
    {
      "type": "made_shot",
      "subtype": "3_pointer",
      "start_frame": 14523,
      "end_frame": 14610,
      "start_timecode": "00:08:04:15",
      "end_timecode": "00:08:07:12",
      "confidence": 0.92,
      "video_confidence": 0.88,
      "audio_confidence": 0.96,
      "details": {
        "shooter_team": "home",
        "court_position": [28.5, 12.3],
        "crowd_excitement_score": 0.87
      }
    }
  ],
  "scenes": [
    {
      "start_frame": 0,
      "end_frame": 450,
      "scene_type": "gameplay"
    }
  ]
}
```

---

## 4. Resolve Integration Design

### 4.1 Workflow

1. **User opens project** in DaVinci Resolve with basketball footage on timeline
2. **User runs analysis script** (from Resolve's Workspace > Scripts menu or external CLI)
3. **Script exports media:**
   - Renders timeline or selected clips to intermediate format (e.g., ProRes or H.264)
   - Extracts audio to WAV via FFmpeg
4. **Analysis engine processes** video and audio (can run on GPU for YOLO acceleration)
5. **Script reads results JSON** and adds color-coded markers to timeline:

| Marker Color | Event Type |
|-------------|------------|
| Blue | Made 2-point shot |
| Green | Made 3-pointer |
| Red | Dunk |
| Yellow | Fast break |
| Purple | Block / steal |
| Pink | Buzzer beater |
| Cyan | High crowd excitement (unclassified) |

6. **Optionally auto-generates** a "Highlights" sub-timeline containing only marked events with configurable padding

### 4.2 Script Structure

```
resolve_integration/
├── basketball_analyzer.py      # Main entry point / orchestrator
├── resolve_export.py           # Media export via Resolve scripting API
├── resolve_markers.py          # Marker creation and timeline manipulation
├── config.py                   # User-configurable thresholds and settings
└── cli.py                      # Command-line interface for standalone use
```

### 4.3 Configuration Options

```python
# config.py — user-adjustable parameters
ANALYSIS_CONFIG = {
    # Detection thresholds
    "min_event_confidence": 0.7,        # Minimum confidence to create a marker
    "crowd_excitement_threshold": 0.6,   # Normalized excitement score threshold

    # Event padding (frames before/after event for highlight clips)
    "highlight_pre_padding_sec": 3.0,
    "highlight_post_padding_sec": 2.0,

    # Audio analysis
    "audio_window_sec": 2.0,            # Mel-spectrogram window size
    "audio_hop_sec": 0.5,               # Hop between windows
    "crowd_freq_low_hz": 500,           # Lower bound of crowd noise band
    "crowd_freq_high_hz": 4000,         # Upper bound of crowd noise band

    # Video analysis
    "yolo_model": "yolov8m",            # YOLO model size (n/s/m/l/x)
    "yolo_confidence": 0.5,             # YOLO detection threshold
    "frame_skip": 2,                    # Analyze every Nth frame for speed

    # Output
    "create_highlight_timeline": True,
    "marker_duration_frames": 1,
    "export_format": "mp4",
}
```

---

## 5. Optional OFX Overlay Plugin

If real-time visual feedback is desired inside Resolve's viewer:

### Purpose
- Draw bounding boxes around detected players and ball
- Show ball trajectory arcs
- Display event labels and confidence scores
- Color-code players by team

### Implementation
- C++ OFX plugin reading pre-computed analysis JSON
- GPU rendering via CUDA/OpenCL for overlay compositing
- Parameters exposed in Resolve UI: toggle layers, adjust opacity, select event types to display

### Scope
This is a secondary deliverable. The core analysis workflow functions entirely through the Python scripting path.

---

## 6. Technology Stack

### Core Dependencies

| Component | Library/Tool | Version | Purpose |
|-----------|-------------|---------|---------|
| Object detection | ultralytics (YOLOv8) | >=8.0 | Ball, hoop, player detection |
| Video processing | OpenCV (cv2) | >=4.8 | Frame I/O, color analysis, homography |
| Scene detection | PySceneDetect | >=0.6 | Shot boundary detection |
| Audio analysis | librosa | >=0.10 | Mel-spectrograms, MFCCs, onset detection |
| Audio extraction | FFmpeg | >=5.0 | Extract audio from video files |
| Tracking | deep-sort-realtime | >=1.3 | Multi-object tracking |
| Pose estimation | mediapipe | >=0.10 | Shooting form analysis (optional) |
| ML framework | PyTorch | >=2.0 | YOLO backend, custom models |
| Data handling | numpy, pandas | latest | Array ops, event data management |
| Resolve API | DaVinciResolveScript | bundled | Timeline/marker/render control |

### Optional Dependencies

| Component | Library/Tool | Purpose |
|-----------|-------------|---------|
| Segmentation | SAM2 | Precise player/ball segmentation |
| OCR | SmolVLM2 | Jersey number reading |
| Audio classification | pyAudioAnalysis | Pre-built audio classifiers |
| OFX SDK | OpenFX 1.5 | Overlay plugin (if building) |
| CUDA Toolkit | >=11.8 | GPU acceleration for YOLO + OFX |

### Hardware Recommendations

- **GPU:** NVIDIA with CUDA support (RTX 3060+ recommended for real-time YOLO inference)
- **RAM:** 16 GB minimum (32 GB recommended for HD footage analysis)
- **Storage:** SSD recommended for intermediate file I/O during export/analysis

---

## 7. Project Structure

```
Resolve-basketball-analysis/
├── PLAN.md                          # This document
├── README.md                        # User-facing setup and usage guide
├── pyproject.toml                   # Python project config and dependencies
├── requirements.txt                 # Pip dependencies
│
├── src/
│   ├── analysis/                    # Core analysis engine
│   │   ├── __init__.py
│   │   ├── video_analyzer.py        # Orchestrates video analysis pipeline
│   │   ├── audio_analyzer.py        # Crowd excitement + whistle detection
│   │   ├── object_detector.py       # YOLO-based detection (ball, hoop, players)
│   │   ├── ball_tracker.py          # Kalman filter ball tracking + shot detection
│   │   ├── player_tracker.py        # DeepSORT player tracking + team classification
│   │   ├── scene_detector.py        # PySceneDetect wrapper
│   │   ├── event_classifier.py      # Multi-modal event classification logic
│   │   └── pose_analyzer.py         # MediaPipe pose estimation (optional)
│   │
│   ├── resolve/                     # DaVinci Resolve integration
│   │   ├── __init__.py
│   │   ├── export.py                # Export media from Resolve timeline
│   │   ├── markers.py               # Add/manage timeline markers
│   │   ├── highlights.py            # Auto-generate highlight timelines
│   │   └── utils.py                 # Resolve API helpers
│   │
│   ├── config.py                    # Configuration and thresholds
│   └── cli.py                       # Command-line entry point
│
├── models/                          # Pre-trained and fine-tuned model weights
│   └── .gitkeep
│
├── scripts/                         # Utility scripts
│   ├── install_resolve_script.py    # Copy integration script to Resolve's script dir
│   └── download_models.py           # Download required model weights
│
├── tests/                           # Test suite
│   ├── test_audio_analyzer.py
│   ├── test_object_detector.py
│   ├── test_ball_tracker.py
│   ├── test_event_classifier.py
│   └── fixtures/                    # Test media samples
│
└── ofx_plugin/                      # Optional OFX overlay plugin
    ├── CMakeLists.txt
    ├── src/
    │   ├── BasketballOverlay.cpp     # Main OFX plugin
    │   ├── BasketballOverlay.h
    │   └── kernels/
    │       ├── overlay.cu            # CUDA overlay rendering
    │       └── overlay.cl            # OpenCL fallback
    └── README.md
```

---

## 8. Development Phases

### Phase 1: Foundation
- Set up Python project structure, dependencies, and configuration
- Implement scene/shot detection with PySceneDetect
- Implement audio extraction and basic crowd excitement scoring with librosa
- Build Resolve scripting integration: export media, add markers
- End-to-end test: analyze sample footage → markers appear on Resolve timeline

### Phase 2: Object Detection and Tracking
- Integrate YOLOv8 for ball, hoop, and player detection
- Implement Kalman filter ball tracking
- Implement made-shot detection (ball through hoop)
- Add DeepSORT player tracking with team classification via jersey color
- Test with real basketball footage

### Phase 3: Event Classification
- Build multi-modal event classifier combining video + audio signals
- Implement detection for: made shots, 3-pointers, dunks, fast breaks, blocks
- Tune confidence thresholds on sample footage
- Add auto-highlight timeline generation

### Phase 4: Polish and Optional Features
- Pose estimation for shooting form analysis
- Buzzer beater detection (audio buzzer + game clock heuristics)
- OFX overlay plugin for real-time visualization (if desired)
- Performance optimization (GPU batching, frame skipping strategies)
- Documentation and user guide

---

## 9. Key Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Resolve scripting API is buggy/underdocumented | Integration issues | Use community resources (X-Raym docs, forums); build robust error handling; test against multiple Resolve versions |
| YOLO accuracy on broadcast basketball footage | Missed or false detections | Fine-tune on basketball-specific datasets from Roboflow; use ensemble of detection + tracking |
| Audio analysis unreliable with commentary overlay | Crowd detection false positives | Use frequency-band filtering to isolate crowd noise from commentary; train classifier to distinguish |
| GPU memory contention (YOLO + Resolve) | Crashes or slowdowns | Run analysis engine separately from Resolve; process exported files, not live timeline |
| No Neural Engine access | Cannot leverage Resolve's built-in AI | Use open-source alternatives (YOLO, MediaPipe) which are equally or more capable for this specific task |
| Large video files slow to process | Poor user experience | Implement frame skipping, GPU acceleration, progress reporting, and ability to analyze selected ranges |

---

## 10. References

- [OpenFX 1.5 Documentation](https://openfx.readthedocs.io/en/main/)
- [DaVinci Resolve Scripting API (X-Raym formatted docs)](https://extremraym.com/cloud/resolve-scripting-doc/)
- [DaVinci Resolve OpenFX Samples](https://github.com/illusionyy/DaVinci-Resolve-OpenFX-Samples)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect)
- [librosa Audio Analysis](https://librosa.org/)
- [MediaPipe Pose](https://mediapipe.dev/)
- [AI Basketball Shot Detection Tracker](https://github.com/avishah3/AI-Basketball-Shot-Detection-Tracker)
- [BasketTracking](https://github.com/Basket-Analytics/BasketTracking)
- [Automated Sport Highlights from Audio/Video (arXiv 2501.16100)](https://arxiv.org/html/2501.16100v2)
- [Warner Bros. Discovery Audio Analysis (AWS)](https://aws.amazon.com/blogs/media/how-warner-bros-discovery-uses-audio-analysis-to-improve-data-accuracy-and-enrich-the-fan-experience/)
