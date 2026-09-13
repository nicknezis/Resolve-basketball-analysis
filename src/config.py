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
    # Rim/backboard impact transients (short broadband bursts)
    impact_freq_low_hz: int = 1000
    impact_freq_high_hz: int = 6000
    impact_hop_sec: float = 0.005  # 5 ms onset resolution
    impact_min_prominence: float = 10.0  # band flux above the local median, in MAD units (gym audio is full of dribbles)
    impact_max_duration_sec: float = 0.25  # longer bursts are crowd/whistle, not an impact
    impact_local_window_sec: float = 1.5  # window for the local median/MAD baseline
    crowd_norm: str = "game"  # "game": normalise crowd energy across all clips; "clip": per clip (legacy)
    crowd_norm_low_pct: float = 5.0  # percentile of band energy that maps to score 0
    crowd_norm_high_pct: float = 99.5  # percentile that maps to score 1


@dataclass
class TrackingConfig:
    """Settings for ball and player tracking."""

    # --- offline ball-track linking (see ball_tracker.py) ---
    max_ball_gap_frames: int = 18  # a tracklet may bridge this many source frames without a detection
    max_ball_speed_px_per_frame: float = 25.0  # link gate grows by this per missed source frame (~1.2 frame-widths/s at 720p)
    link_slack_px: int = 24  # base link gate, absorbs detection jitter
    max_link_jump_px: int = 160  # gate cap: longer hops are left to the tracklet chain (with its teleport penalty)
    edge_margin_px: int = 8  # ball candidates centred this close to the frame border are dropped (partial boxes, fixtures)
    min_tracklet_detections: int = 3  # shorter tracklets are noise
    tracklet_size_ratio_max: float = 2.5  # consecutive ball boxes must be within this size ratio
    compensate_camera_motion: bool = True  # link in rim/player-anchored coordinates so pans don't break tracks
    static_span_px: int = 40  # tracklets whose compensated extent stays under this...
    static_clutter_sec: float = 0.5  # ...for at least this long are fixtures, not the ball
    motion_full_span_px: int = 200  # compensated extent at which a tracklet gets full motion credit in scoring
    teleport_penalty_per_px: float = 0.05  # chain penalty for jumps the ball could not have made
    shot_min_arc_height_px: int = 50  # minimum arc height to count as a shot attempt
    shot_arc_end_descent_px: int = 50  # descent below peak that terminates an arc
    hoop_proximity_px: int = 80  # pixels from hoop center to count as "through hoop"
    hoop_x_tolerance_ratio: float = 0.3  # horizontal tolerance as fraction of hoop bbox width
    hoop_entry_y_margin_px: int = 30  # vertical margin above/below hoop top for entry detection
    hoop_obs_max_age_frames: int = 30  # ignore hoop observations further than this from a shot's descent
    shot_hoop_x_range_ratio: float = 0.35  # max horizontal distance from hoop (as fraction of frame width)
    shot_require_peak_above_rim: bool = True  # when a rim is in view, the arc must peak at/above it
    shot_peak_rim_margin_px: int = 20  # slack below the rim top still accepted as "above"
    shot_min_descent_ratio: float = 0.15  # ball must descend at least this fraction of ascent height (low to allow layups)
    shot_max_arc_sec: float = 3.0  # max wall-clock duration of a single arc (source frames / fps)
    shot_pre_peak_sec: float = 0.5  # max time before peak to include in shot event window
    shot_post_arc_sec: float = 0.6  # extra time past arc_end to check for made shot (a rattle can take 0.5 s)
    rim_entry_enabled: bool = True  # a descending ball that vanishes inside the rim footprint counts as a shot
    rim_entry_lookahead_sec: float = 0.6  # ...and a make, unless it is re-detected above the rim within this time
    # --- near-rim re-detection (second detector pass on shot windows) ---
    rim_roi_redetect: bool = True
    rim_roi_pre_sec: float = 0.15  # window starts this long before the arc peak
    rim_roi_post_sec: float = 1.0  # ...and ends this long after the arc end (covers a rattle)
    rim_roi_scale: float = 2.0  # upscale factor for the rim crop
    rim_roi_width_factor: float = 3.0  # crop width in rim widths
    rim_roi_above_factor: float = 1.5  # crop extends this many rim heights above the rim top
    rim_roi_below_factor: float = 3.0  # ...and this many below the rim bottom (the net and beyond)
    # --- phase-based made/miss ---
    net_x_tolerance_ratio: float = 0.3  # post-rim x must stay within the rim span ± this fraction of rim width
    post_rim_speed_made_max: float = 0.85  # measured/expected freefall speed below the rim; slower = net friction = made
    parabola_min_points: int = 4  # detected flight points needed to fit gravity
    require_free_flight: bool = True  # the ball must leave every player box during a shot arc
    min_free_flight_sec: float = 0.2  # ...for at least this long (contested hand-offs never do)
    free_flight_box_pad: float = 0.1  # expand player boxes by this fraction when testing containment
    free_flight_head_fraction: float = 0.15  # ball above this fraction of a box's height (head level) counts as free
    use_polygon_zone: bool = False  # use supervision PolygonZone for made-shot fallback
    # --- player tracking (ByteTrack) ---
    track_activation_threshold: float = 0.5  # detection confidence needed to start a new track
    track_high_conf_threshold: float = 0.6  # ByteTrack first-pass (high-confidence) association split
    track_lost_buffer_frames: int = 30  # analysed frames a lost track is kept before it is dropped
    track_min_iou: float = 0.1  # minimum IoU for matching a box to a track
    track_min_hits: int = 3  # consecutive matches before a track is reported
    jersey_sample_every_frames: int = 20  # source frames between stored torso crops per track
    jersey_crops_per_track: int = 8
    jersey_min_crops: int = 3  # tracks with fewer crops are not assigned a team
    team_method: str = "siglip"  # "siglip" (embeddings + PCA + k-means) or "hsv" (median colour k-means)
    team_embed_model: str = "google/siglip-base-patch16-224"
    team_outlier_std: float = 2.0  # tracks farther than mean + this*std from both centres stay unassigned
    enable_player_tracking: bool = True  # set False to skip player tracking


@dataclass
class EventConfig:
    """Settings for event classification."""

    min_confidence: float = 0.7  # minimum fused confidence to report an event
    min_video_confidence: float = 0.5  # hard floor on the video-only confidence of a shot
    audio_weight: float = 0.4  # how much of the remaining headroom audio can add (never gates)
    highlight_pre_pad_sec: float = 3.0  # seconds before event for highlight clip
    highlight_post_pad_sec: float = 2.0  # seconds after event for highlight clip
    merge_gap_sec: float = 2.0  # merge events closer than this into one highlight
    impact_window_before_sec: float = 0.08  # sound can only lead the tracked rim contact by frame-timing error
    impact_window_after_sec: float = 0.3  # ...or lag it (ball reaches the band before the clang)


@dataclass
class CourtConfig:
    """Court keypoints → homography → on-court filtering and shot zones.

    Experimental and off by default: on the reference footage the public
    keypoint model agrees with the court template on only 6–9 landmarks per
    frame and the fitted homography is only reliable near them (see
    docs/analysis-techniques.md, "Court Geometry").
    """

    enabled: bool = False
    preset: str = "nfhs"  # "nfhs" (US high school), "nba", "fiba"
    model_id: str = "basketball-court-detection-2/22"  # Roboflow keypoint model (17 landmarks, RF-DETR)
    every_sec: float = 3.0  # run the keypoint model this often; camera shift carries the mapping between runs
    detection_confidence: float = 0.3
    keypoint_confidence: float = 0.5
    min_keypoints: int = 7
    ransac_reproj_cm: float = 60.0  # RANSAC inlier threshold in court cm
    max_reproj_px: float = 15.0  # reject keyframes whose inliers re-project worse than this
    min_spread_ratio: float = 0.25  # inlier landmarks must span an area (minor/major extent), not a line
    calibrated_buffer_cm: float = 300.0  # answer court questions only within the inlier hull + this buffer
    max_keyframe_age_frames: int = 600  # don't trust a keyframe more than 10 s (at 60 fps) away
    court_margin_cm: float = 30.0  # further outside the lines than this = bench/spectators (benches sit ~1 m out)
    three_point_margin_cm: float = 15.0  # tolerance on the arc when calling a three
    layup_max_ft: float = 5.0  # shots released closer than this to the basket are layups
    overlay: bool = True  # draw projected court lines in review exports


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
    court: CourtConfig = field(default_factory=CourtConfig)
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
