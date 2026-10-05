from pathlib import Path
from boxmot import Boxmot

ROOT = Path(__file__).resolve().parents[2]

MODEL = ROOT / "models" / "pytorch" / "yolo11m.pt"
VIDEO = ROOT / "test_videos" / "video1.mp4"

tracker = Boxmot(
    detector=str(MODEL),
    tracker="bytetrack",
    project=str(ROOT / "outputs" / "trackers")
)

print("=" * 70)
print("BoxMOT TEST")
print("=" * 70)
print("Model   :", MODEL)
print("Video   :", VIDEO)
print("Tracker :", tracker.tracker)
print("ReID    :", tracker.reid)
print("=" * 70)

result = tracker.track(
    source=str(VIDEO),
    imgsz=640,
    conf=0.25,
    device="cpu",
    half=False,
    save=True,
    save_txt=True,
    show=False,
    verbose=True,
)

print()
print("=" * 70)
print("TEST COMPLETE")
print("=" * 70)
print("Result:", result)
print("=" * 70)