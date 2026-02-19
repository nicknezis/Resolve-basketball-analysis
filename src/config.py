"""Configuration and thresholds for basketball video analysis."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class VideoConfig:
    """Settings for video analysis pipeline."""

    yolo_model: str = "yolov8m.pt"
    yolo_confidence: float = 0.5
    frame_skip: int = 2  # analyze every Nth frame for speed
    max_resolution: int = 1920  # downscale larger frames for faster inference


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
    hoop_proximity_px: int = 80  # pixels from hoop center to count as "through hoop"
    deepsort_max_age: int = 30
    deepsort_n_init: int = 3


@dataclass
class EventConfig:
    """Settings for event classification."""

    min_confidence: float = 0.7  # minimum confidence to report an event
    video_weight: float = 0.6  # weight for video confidence in fusion
    audio_weight: float = 0.4  # weight for audio confidence in fusion
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
