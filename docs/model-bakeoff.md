# Hoop/ball detector bake-off

120 frames from `huskies_260912_timeline.json` (clips 0-4), 1280x720, confidence ≥ 0.3, LUT=lut/FX30.cube

| Model | Hoop frames (mean conf) | Ball frames (mean conf) | ball-in-basket frames | Players/frame | Refs/frame | Speed | Classes seen |
|---|---|---|---|---|---|---|---|
| `yolo:yolo11m.pt@640` | 0% (0.00) | 23% (0.64) | 0 | 11.4 | 0.0 | 31 ms | person, sports ball |
| `yolo:yolo11m.pt@1280` | 0% (0.00) | 25% (0.62) | 0 | 13.2 | 0.0 | 45 ms | person, sports ball |
| `basketball-player-detection-3-ycjdo/18` | ERROR: RoboflowAPINotNotFoundError: Could not find requested Roboflow resource. Check t | | | | | | |
| `basketball-computer-vision/13` | 5% (0.46) | 31% (0.75) | 0 | 6.6 | 0.2 | 16 ms | ball, player, referee, rim |
| `basketball-hoop-tsdku/basketball-and-rim` | ERROR: no trained version / lookup failed | | | | | | |
| `basketball-detection-dn6fg/4` | 75% (0.79) | 54% (0.64) | 0 | 4.5 | 0.0 | 18 ms | ball, basket, person |
| `basketball-players-fy4c2/25` | 64% (0.75) | 66% (0.76) | 0 | 9.2 | 1.1 | 273 ms | Ball, Hoop, Player, Ref |

Additional runs (8 frames/clip, 40 frames):

| Model | Hoop frames (mean conf) | Ball frames (mean conf) | ball-in-basket frames | Players/frame | Refs/frame | Speed | Classes seen |
|---|---|---|---|---|---|---|---|
| `basketball-player-detection-3-ycjdo/6` | 38% (0.57) | 58% (0.68) | 0 | 12.8 | 0.6 | 324 ms | ball, number, player, player-in-possession, player-layup-dunk, player-shot-block, referee, rim |
| `basketball-player-detection-3-ycjdo/12` | 12% (0.51) | 35% (0.67) | 0 | 5.2 | 0.8 | 26 ms | ball, number, player, player-in-possession, player-jump-shot, player-shot-block, referee, rim |

## Notes

- Footage: handheld/panned 720p Sony proxies, S-Log3 corrected with `lut/FX30.cube`, both baskets appear. Speeds are CPU ONNX on an Apple-silicon Mac (Roboflow models) and MPS (YOLO).
- **Winner for now: `basketball-detection-dn6fg/4`** (`computer-vision-d5fjh/basketball-detection-dn6fg` on Universe) — best hoop recall by a wide margin, 2× the ball recall of stock YOLO, and fast. Classes `ball`, `basket`, `person`; no referee class.
- `basketball-players-fy4c2/25` is the runner-up and has a `Ref` class, but at ~270 ms/frame it is 15× slower.
- Roboflow's own 10-class `basketball-player-detection-3-ycjdo`: v18 (mAP 88) exists but `inference` cannot fetch it locally; v6 loads but finds the rim in only 38% of frames and is slow; v12 is fast but weak. Its `ball-in-basket` class never fired on these frames.
- Stock `yolo11m.pt` never sees a hoop (COCO has no class for it); raising `imgsz` 640→1280 barely helps on 720p proxies because the source is already small — it matters when analysing 1920-px+ frames.
- All public models are trained mostly on broadcast/NBA footage. The durable fix is a fine-tune on frames from this gym; the review exports are a convenient labelling source.
