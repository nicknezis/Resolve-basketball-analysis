"""Configuration and thresholds for basketball video analysis."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoConfig:
    """Settings for video analysis pipeline."""

    yolo_model: str = "yolo11m.pt"
    yolo_confidence: float = 0.5
    imgsz: int = 1280  # YOLO inference resolution (long edge); 640 loses the ball
    hoop_confidence: float = 0.3  # separate, looser threshold for hoop/rim boxes
    frame_skip: int = 2  # analyze every Nth frame for speed
    max_resolution: int = 1920  # downscale larger frames for faster inference
    input_lut: Path | None = None  # path to .cube 3D LUT file (or .zip/.lut containing one)
    roboflow_model_id: str | None = None  # Roboflow model ID (e.g. "basketball-detection/1")
    roboflow_confidence: float = 0.4  # confidence threshold for Roboflow model
    detector_backend: str = "yolo"  # "yolo", "rfdetr", or "roboflow"
    rfdetr_model_size: str = "small"  # "nano", "small", "medium", "large", "xlarge", "2xlarge"
    rfdetr_weights: str | None = None  # path to fine-tuned .pth checkpoint
    rfdetr_num_classes: int | None = None  # number of classes (None = COCO default)
    rfdetr_class_names: list[str] | None = None  # class names matching dataset indices
    rfdetr_resolution: int | None = None  # input resolution (must be divisible by 56)
    nms_enabled: bool = False  # apply supervision NMS deduplication
    nms_threshold: float = 0.5  # IoU threshold for NMS
    review_codec: str = "hevc"  # "hevc", "h264", or "mp4v" (OpenCV fallback)
    review_quality: int | None = None  # CRF (software) or kbit/s (hardware); None = default


@dataclass
class AudioConfig:
    """Settings for audio analysis pipeline."""

    sample_rate: int = 22050
    window_sec: float = 2.0  # mel-spectrogram window size
    hop_sec: float = 0.5  # hop between windows
    crowd_freq_low_hz: int = 500  # lower bound of crowd noise band
    crowd_freq_high_hz: int = 4000  # upper bound of crowd noise band
    excitement_threshold: float = 0.6  # normalized excitement score threshold
    n_mels: int = 128
    whistle_freq_low_hz: int = 2000
    whistle_freq_high_hz: int = 4500
    whistle_energy_threshold: float = 0.7


@dataclass
class TrackingConfig:
    """Settings for ball and player tracking."""

    kalman_process_noise: float = 0.03
    kalman_measurement_noise: float = 0.1
    max_ball_gap_frames: int = 10  # max frames to interpolate missing ball
    shot_min_arc_height_px: int = 50  # minimum arc height to count as a shot attempt
    shot_arc_end_descent_px: int = 50  # descent below peak that terminates an arc
    hoop_proximity_px: int = 80  # pixels from hoop center to count as "through hoop"
    hoop_x_tolerance_ratio: float = 0.3  # horizontal tolerance as fraction of hoop bbox width
    hoop_entry_y_margin_px: int = 30  # vertical margin above/below hoop top for entry detection
    hoop_obs_max_age_frames: int = 30  # ignore hoop observations further than this from a shot's descent
    max_ball_jump_px: int = 200  # max pixel distance from predicted position to accept a detection
    ball_gate_weight: float = 0.5  # blend factor: 0=pure confidence, 1=pure proximity-to-prediction
    reacquire_after_gap_frames: int = 5  # after this many missed frames, accept any detection
    shot_hoop_x_range_ratio: float = 0.35  # max horizontal distance from hoop (as fraction of frame width)
    shot_require_peak_above_rim: bool = True  # when a rim is in view, the arc must peak at/above it
    shot_peak_rim_margin_px: int = 20  # slack below the rim top still accepted as "above"
    shot_min_descent_ratio: float = 0.15  # ball must descend at least this fraction of ascent height (low to allow layups)
    shot_max_arc_sec: float = 3.0  # max wall-clock duration of a single arc (source frames / fps)
    shot_pre_peak_sec: float = 0.5  # max time before peak to include in shot event window
    shot_post_arc_sec: float = 0.35  # extra time past arc_end to check for made shot (backboard bounces)
    consensus_required: int = 3  # min detections in window to confirm ball (1=disabled)
    consensus_window: int = 5  # rolling window size for consensus check
    consensus_max_spread_px: int = 100  # max spatial spread for consensus candidates
    use_polygon_zone: bool = False  # use supervision PolygonZone for made-shot fallback
    deepsort_max_age: int = 30
    deepsort_n_init: int = 3
    enable_player_tracking: bool = True  # set False to skip DeepSORT player tracking


@dataclass
class EventConfig:
    """Settings for event classification."""

    min_confidence: float = 0.7  # minimum fused confidence to report an event
    min_video_confidence: float = 0.5  # hard floor on the video-only confidence of a shot
    audio_weight: float = 0.4  # how much of the remaining headroom audio can add (never gates)
    highlight_pre_pad_sec: float = 3.0  # seconds before event for highlight clip
    highlight_post_pad_sec: float = 2.0  # seconds after event for highlight clip
    merge_gap_sec: float = 2.0  # merge events closer than this into one highlight


@dataclass
class SceneConfig:
    """Settings for scene/shot detection."""

    content_threshold: float = 27.0  # PySceneDetect ContentDetector threshold
    min_scene_len_sec: float = 0.5  # minimum scene length


@dataclass
class AnalysisConfig:
    """Top-level configuration combining all sub-configs."""

    video: VideoConfig = field(default_factory=VideoConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    events: EventConfig = field(default_factory=EventConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    output_dir: Path = field(default_factory=lambda: Path("output"))
    device: str = "auto"  # "auto", "cuda", "cpu"
    preview: bool = False  # per-frame live preview during analysis
    review: bool = False  # per-clip replay with full results after analysis
    review_export: Path | None = None  # directory to save review replay videos


# Marker color mapping used by the Resolve import script
MARKER_COLORS = {
    "made_shot": "Blue",
    "three_pointer": "Green",
    "dunk": "Red",
    "fast_break": "Yellow",
    "block": "Purple",
    "steal": "Purple",
    "buzzer_beater": "Pink",
    "crowd_excitement": "Cyan",
    "shot_attempt": "Cream",
}

EVENT_LABELS = {
    "made_shot": "Made Shot",
    "three_pointer": "3-Pointer",
    "dunk": "Dunk",
    "fast_break": "Fast Break",
    "block": "Block",
    "steal": "Steal",
    "buzzer_beater": "Buzzer Beater",
    "crowd_excitement": "Crowd Reaction",
    "shot_attempt": "Shot Attempt",
}
