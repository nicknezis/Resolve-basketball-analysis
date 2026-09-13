from src.analysis.rim_roi import RimRoiRedetector
from src.config import TrackingConfig


def test_crop_box_is_clamped_and_covers_net():
    r = RimRoiRedetector(detector=None, lut=None, max_resolution=1920, config=TrackingConfig())
    x1, y1, x2, y2 = r.crop_box((600, 100, 700, 130), 1280, 720)
    assert x1 == 500 and x2 == 800          # 3 rim widths centred on the rim
    assert y1 == 55 and y2 == 220           # 1.5 heights above, 3 below
    x1, y1, x2, y2 = r.crop_box((10, 5, 110, 35), 1280, 720)
    assert x1 == 0 and y1 == 0              # clamped to the frame
