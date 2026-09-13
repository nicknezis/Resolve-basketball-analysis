"""Court template, homography and zone logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis.court import EDGES, LABELS, CourtModel, CourtTracker, court_vertices
from src.config import CourtConfig


def test_template_has_33_vertices_for_every_preset():
    for preset in ("nba", "fiba", "nfhs"):
        v = court_vertices(preset)
        assert v.shape == (33, 2)
        assert len(LABELS) == 33
    assert max(max(a, b) for a, b in EDGES) < 33


def test_nfhs_zones():
    c = CourtModel("nfhs")
    bx, by = c.baskets[0]
    assert c.zone(bx + 100, by) == "paint"            # right under the basket
    assert c.zone(bx + 500, by) == "two"              # 16 ft straight out
    assert c.zone(bx + 650, by) == "three"            # beyond 19'9" (602 cm)
    assert c.zone(60, 30) == "three"                  # corner, inside the straight section
    assert c.zone(60, 300) == "two"                   # baseline but inside the line
    assert c.zone(c.length - bx - 650, by) == "three"  # other basket
    assert c.on_court(-50, 100, margin_cm=120) and not c.on_court(-200, 100, margin_cm=120)


def test_homography_roundtrip_with_synthetic_camera():
    c = CourtModel("nfhs")
    cfg = CourtConfig()
    tr = CourtTracker(c, cfg, model=None)
    # A plausible perspective mapping court(cm) -> image(px)
    H_true = np.array([[0.35, -0.12, 200.0], [0.02, 0.18, 120.0], [0.0, 0.0002, 1.0]])
    idx = [1, 2, 3, 4, 6, 9, 10, 11, 13, 7, 8]
    court_pts = c.vertices[idx]
    img = np.array([(H_true @ np.array([x, y, 1.0])) for x, y in court_pts])
    img = img[:, :2] / img[:, 2:3]
    img += np.random.default_rng(0).normal(0, 0.8, img.shape)  # a bit of keypoint noise
    kf = tr.fit(img, court_pts, frame_idx=100)
    assert kf is not None and kf.n_points >= 8 and kf.reproj_px < 5
    tr.keyframes.append(kf)
    back = tr.to_court(img[:3], frame_idx=100)
    assert np.allclose(back, court_pts[:3], atol=25)   # within 25 cm
    # A point far outside the landmarks' hull is an extrapolation → NaN
    far = tr.to_image(np.array([[c.length - 100.0, 50.0]]), frame_idx=100)
    assert np.isnan(tr.to_court(far, frame_idx=100)[0][0])
    assert not np.isnan(tr.to_court(far, frame_idx=100, calibrated_only=False)[0][0])

    # Camera pans 40 px to the right by frame 130: the same court point is now 40 px further right
    tr.shift_at = lambda f: (40.0, 0.0) if f >= 130 else (0.0, 0.0)
    moved = img[:3] + np.array([40.0, 0.0])
    assert np.allclose(tr.to_court(moved, frame_idx=130), court_pts[:3], atol=25)
    assert tr.court_lines(100)  # overlay polylines exist


def test_fit_rejects_too_few_points():
    tr = CourtTracker(CourtModel("nba"), CourtConfig(), model=None)
    assert tr.fit(np.zeros((3, 2)), np.zeros((3, 2)), 0) is None


def test_unknown_preset():
    with pytest.raises(ValueError):
        CourtModel("wnba")
