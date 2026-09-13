"""Court geometry: keypoints → homography → on-court test, shot zones, distances.

The keypoint model (``basketball-court-detection-2`` on Roboflow Universe)
labels up to 33 court landmarks with the two-digit names below.  Their
template coordinates follow the layout in roboflow/sports (MIT), re-derived
here from a small dimension preset so that a high-school court (NFHS: 84 ft
long, 12 ft key, 19'9" arc) is handled as well as NBA / FIBA.

Coordinates: x along the court length (left baseline = 0), y across the
width (top sideline = 0), centimetres.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CourtConfig

logger = logging.getLogger(__name__)

# Keypoint class names emitted by the model, in template-vertex order.
LABELS = [
    "01", "02", "04", "05", "07", "08", "09", "10", "11", "12", "13", "14",
    "15", "16", "17", "19", "21", "23", "25", "26", "27", "28", "29", "30",
    "31", "32", "33", "34", "35", "37", "38", "40", "41",
]
LABEL_INDEX = {name: i for i, name in enumerate(LABELS)}

# Dimension presets in centimetres
PRESETS: dict[str, dict[str, float]] = {
    "nba": dict(court_width=1524, court_length=2865, three_point_arc_radius=724,
                straight_section_three_point_line=424, sideline_to_three_point_line=91,
                paint_width=488, paint_length=579, baseline_to_rim_center=160,
                baseline_to_throw_line=835),
    "fiba": dict(court_width=1500, court_length=2800, three_point_arc_radius=675,
                 straight_section_three_point_line=330, sideline_to_three_point_line=90,
                 paint_width=490, paint_length=580, baseline_to_rim_center=157,
                 baseline_to_throw_line=830),
    # NFHS (US high school): 84 x 50 ft, 19'9" arc from the basket centre that
    # runs straight to the baseline once level with the basket, 12 ft key,
    # free-throw line 19 ft from the baseline, 28 ft hash marks.
    "nfhs": dict(court_width=1524, court_length=2560, three_point_arc_radius=602,
                 straight_section_three_point_line=160, sideline_to_three_point_line=160,
                 paint_width=366, paint_length=579, baseline_to_rim_center=160,
                 baseline_to_throw_line=853),
}

FT_PER_CM = 1.0 / 30.48


def court_vertices(preset: str) -> np.ndarray:
    """The 33 template vertices (cm) for a preset, in :data:`LABELS` order."""
    d = PRESETS[preset]
    W, L = d["court_width"], d["court_length"]
    R3, S3, D3 = d["three_point_arc_radius"], d["straight_section_three_point_line"], d["sideline_to_three_point_line"]
    PW, PL, BR, BT = d["paint_width"], d["paint_length"], d["baseline_to_rim_center"], d["baseline_to_throw_line"]
    ps = (W - PW) / 2  # paint start (y)
    mid = W / 2
    v = [
        (0, 0), (0, D3), (0, ps), (0, ps + PW), (0, W - D3), (0, W),           # 00-05
        (BR, mid),                                                            # 06 left basket
        (S3, D3), (S3, W - D3),                                               # 07-08
        (PL, ps), (PL, ps + PW / 2), (PL, ps + PW),                           # 09-11 free-throw line
        (BT, 0), (BR + R3, mid), (BT, W),                                     # 12-14
        (L / 2, 0), (L / 2, mid), (L / 2, W),                                 # 15-17 centre line
        (L - BT, 0), (L - BR - R3, mid), (L - BT, W),                         # 18-20
        (L - PL, ps), (L - PL, ps + PW / 2), (L - PL, ps + PW),               # 21-23
        (L - S3, D3), (L - S3, W - D3),                                       # 24-25
        (L - BR, mid),                                                        # 26 right basket
        (L, 0), (L, D3), (L, ps), (L, ps + PW), (L, W - D3), (L, W),          # 27-32
    ]
    return np.array(v, dtype=np.float64)


# Court line segments (vertex index pairs) worth drawing for verification
EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5),            # left baseline
    (2, 9), (11, 3), (9, 10), (10, 11),                # left paint
    (1, 7), (4, 8),                                    # left 3pt straight sections
    (0, 12), (12, 15), (15, 18), (18, 27),             # top sideline
    (5, 14), (14, 17), (17, 20), (20, 32),             # bottom sideline
    (15, 16), (16, 17),                                # centre line
    (27, 28), (28, 29), (29, 30), (30, 31), (31, 32),  # right baseline
    (29, 21), (21, 22), (22, 23), (23, 30),            # right paint
    (28, 24), (31, 25),                                # right 3pt straight sections
]


@dataclass
class CourtKeyframe:
    frame_idx: int
    H: np.ndarray  # image (analysis px) → court (cm), 3x3
    H_inv: np.ndarray
    n_points: int
    reproj_px: float
    hull: np.ndarray  # convex hull (court cm) of the inlier landmarks: the calibrated area
    spread: float  # ratio of the inliers' minor/major spatial extent (0 = collinear)


class CourtModel:
    """Court geometry for one preset."""

    def __init__(self, preset: str = "nfhs") -> None:
        if preset not in PRESETS:
            raise ValueError(f"Unknown court preset {preset!r}; choose from {sorted(PRESETS)}")
        self.preset = preset
        self.dims = PRESETS[preset]
        self.vertices = court_vertices(preset)
        d = self.dims
        self.length, self.width = d["court_length"], d["court_width"]
        self.baskets = [
            (d["baseline_to_rim_center"], self.width / 2),
            (self.length - d["baseline_to_rim_center"], self.width / 2),
        ]

    def nearest_basket(self, x: float, y: float) -> tuple[tuple[float, float], bool]:
        """(basket_xy, is_left)"""
        return (self.baskets[0], True) if x < self.length / 2 else (self.baskets[1], False)

    def distance_cm(self, x: float, y: float) -> float:
        (bx, by), _ = self.nearest_basket(x, y)
        return math.hypot(x - bx, y - by)

    def on_court(self, x: float, y: float, margin_cm: float = 0.0) -> bool:
        return -margin_cm <= x <= self.length + margin_cm and -margin_cm <= y <= self.width + margin_cm

    def zone(self, x: float, y: float, three_margin_cm: float = 0.0) -> str:
        """'paint', 'two' or 'three' for a shooter standing at (x, y)."""
        d = self.dims
        (bx, by), left = self.nearest_basket(x, y)
        dist = math.hypot(x - bx, y - by)
        # Corner: inside the straight section the line is parallel to the sideline
        from_baseline = x if left else self.length - x
        if from_baseline <= d["straight_section_three_point_line"]:
            if abs(y - self.width / 2) > (self.width / 2 - d["sideline_to_three_point_line"]) + three_margin_cm:
                return "three"
        elif dist > d["three_point_arc_radius"] + three_margin_cm:
            return "three"
        if from_baseline <= d["paint_length"] and abs(y - self.width / 2) <= d["paint_width"] / 2:
            return "paint"
        return "two"


class CourtTracker:
    """Maintains image→court homographies over a clip from sparse keypoint detections.

    Keyframes come from the keypoint model every few seconds; between
    keyframes the camera-shift estimate (from the ball tracker) translates
    the mapping, which is enough for a pan.
    """

    def __init__(self, court: CourtModel, config: CourtConfig, model=None) -> None:
        self.court = court
        self.config = config
        self.model = model
        self.keyframes: list[CourtKeyframe] = []
        self.shift_at = lambda frame_idx: (0.0, 0.0)  # set by the caller once shifts are known

    # -- detection -------------------------------------------------------

    def observe(self, frame_bgr: np.ndarray, frame_idx: int) -> CourtKeyframe | None:
        """Run the keypoint model on a frame and, if enough points fit, add a keyframe."""
        if self.model is None:
            return None
        try:
            res = self.model.infer(frame_bgr, confidence=self.config.detection_confidence)
        except Exception as exc:  # noqa: BLE001
            logger.debug("court keypoint inference failed on frame %d: %s", frame_idx, exc)
            return None
        r = res[0] if isinstance(res, list) else res
        preds = getattr(r, "predictions", None) or (r.get("predictions", []) if isinstance(r, dict) else [])
        if not preds:
            return None
        best = max(preds, key=lambda p: float(getattr(p, "confidence", 0.0)))
        pts_img, pts_court = [], []
        for kp in getattr(best, "keypoints", []) or []:
            name = str(getattr(kp, "class_name", ""))
            conf = float(getattr(kp, "confidence", 0.0))
            if conf < self.config.keypoint_confidence or name not in LABEL_INDEX:
                continue
            pts_img.append((float(kp.x), float(kp.y)))
            pts_court.append(self.court.vertices[LABEL_INDEX[name]])
        kf = self.fit(np.array(pts_img), np.array(pts_court), frame_idx)
        if kf is not None:
            self.keyframes.append(kf)
        return kf

    def fit(self, pts_img: np.ndarray, pts_court: np.ndarray, frame_idx: int) -> CourtKeyframe | None:
        """RANSAC homography; rejected if too few points or poor reprojection."""
        if len(pts_img) < self.config.min_keypoints:
            return None
        H, mask = cv2.findHomography(
            pts_img.astype(np.float64), pts_court.astype(np.float64),
            cv2.RANSAC, self.config.ransac_reproj_cm,
        )
        if H is None or mask is None or int(mask.sum()) < self.config.min_keypoints:
            return None
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return None
        inl = mask.ravel().astype(bool)
        back = cv2.perspectiveTransform(pts_court[inl].reshape(-1, 1, 2), H_inv).reshape(-1, 2)
        reproj = float(np.mean(np.linalg.norm(back - pts_img[inl], axis=1)))
        if reproj > self.config.max_reproj_px:
            logger.debug("court keyframe %d rejected: reproj %.1f px", frame_idx, reproj)
            return None
        # A homography fitted to a small or nearly collinear patch of landmarks
        # re-projects those landmarks perfectly and everything else nowhere
        # near; require the inliers to span an area.
        centred = pts_court[inl] - pts_court[inl].mean(axis=0)
        sv = np.linalg.svd(centred, compute_uv=False)
        spread = float(sv[1] / sv[0]) if sv[0] > 0 else 0.0
        if spread < self.config.min_spread_ratio:
            logger.debug("court keyframe %d rejected: landmark spread %.2f", frame_idx, spread)
            return None
        hull = cv2.convexHull(pts_court[inl].astype(np.float32)).reshape(-1, 2)
        return CourtKeyframe(
            frame_idx=frame_idx, H=H, H_inv=H_inv, n_points=int(inl.sum()),
            reproj_px=reproj, hull=hull, spread=spread,
        )

    # -- queries ---------------------------------------------------------

    @property
    def available(self) -> bool:
        return bool(self.keyframes)

    def _keyframe(self, frame_idx: int) -> CourtKeyframe | None:
        if not self.keyframes:
            return None
        kf = min(self.keyframes, key=lambda k: abs(k.frame_idx - frame_idx))
        if abs(kf.frame_idx - frame_idx) > self.config.max_keyframe_age_frames:
            return None
        return kf

    def to_court(self, points_img: np.ndarray, frame_idx: int, calibrated_only: bool = True) -> np.ndarray | None:
        """Map analysis-pixel points at ``frame_idx`` to court cm, or None if no keyframe.

        With ``calibrated_only`` (default) points that land outside the
        keyframe's calibrated area — the inlier landmarks' hull plus
        ``calibrated_buffer_cm`` — come back as NaN: the mapping is only an
        interpolation there, an extrapolation everywhere else.
        """
        kf = self._keyframe(frame_idx)
        if kf is None:
            return None
        sx0, sy0 = self.shift_at(kf.frame_idx)
        sx1, sy1 = self.shift_at(frame_idx)
        pts = np.asarray(points_img, dtype=np.float64).reshape(-1, 2) - np.array([sx1 - sx0, sy1 - sy0])
        out = cv2.perspectiveTransform(pts.reshape(-1, 1, 2), kf.H).reshape(-1, 2)
        if calibrated_only:
            buf = self.config.calibrated_buffer_cm
            hull = kf.hull.astype(np.float32)
            for i, (x, y) in enumerate(out):
                if cv2.pointPolygonTest(hull, (float(x), float(y)), True) < -buf:
                    out[i] = np.nan
        return out

    def to_image(self, points_court: np.ndarray, frame_idx: int) -> np.ndarray | None:
        kf = self._keyframe(frame_idx)
        if kf is None:
            return None
        sx0, sy0 = self.shift_at(kf.frame_idx)
        sx1, sy1 = self.shift_at(frame_idx)
        pts = np.asarray(points_court, dtype=np.float64).reshape(-1, 1, 2)
        out = cv2.perspectiveTransform(pts, kf.H_inv).reshape(-1, 2)
        return out + np.array([sx1 - sx0, sy1 - sy0])

    def court_lines(self, frame_idx: int) -> list[np.ndarray]:
        """Court edges (and both 3-pt arcs) as image-space polylines for overlays."""
        kf = self._keyframe(frame_idx)
        if kf is None:
            return []
        lines: list[np.ndarray] = []
        v = self.court.vertices
        for a, b in EDGES:
            seg = self.to_image(np.array([v[a], v[b]]), frame_idx)
            if seg is not None:
                lines.append(seg)
        d = self.court.dims
        r, s = d["three_point_arc_radius"], d["straight_section_three_point_line"]
        for (bx, by), left in ((self.court.baskets[0], True), (self.court.baskets[1], False)):
            # Arc from the end of one straight section to the other
            dx = s - bx if left else bx - (self.court.length - s)
            cos_lim = max(-1.0, min(1.0, dx / r))
            lim = math.acos(cos_lim)
            angles = np.linspace(-(math.pi - lim), (math.pi - lim), 40) if left else np.linspace(-lim, lim, 40)
            arc = np.array([(bx + r * math.cos(t), by + r * math.sin(t)) for t in angles])
            if not left:
                arc = np.array([(bx - r * math.cos(t), by + r * math.sin(t)) for t in angles])
            pts = self.to_image(arc, frame_idx)
            if pts is not None:
                lines.append(pts)
        return lines


def load_court_model(model_id: str):
    """Load the Roboflow court-keypoint model via ``inference`` (needs ROBOFLOW_API_KEY)."""
    import os

    from inference import get_model

    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError("ROBOFLOW_API_KEY is required for the court keypoint model")
    model = get_model(model_id=model_id, api_key=api_key)
    logger.info("Loaded court keypoint model: %s", model_id)
    return model
