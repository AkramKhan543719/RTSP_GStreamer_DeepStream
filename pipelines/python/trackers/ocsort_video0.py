import csv
import time
from pathlib import Path

import cv2
from boxmot import Boxmot


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

VIDEO = ROOT / "test_videos" / "video0.avi"
MODEL = ROOT / "models" / "pytorch" / "yolo11m.pt"

OUTPUT_DIR = ROOT / "outputs" / "trackers" / "ocsort"
OUTPUT_VIDEO = OUTPUT_DIR / "video0_yolo11m_ocsort.mp4"
OUTPUT_CSV = OUTPUT_DIR / "video0_yolo11m_ocsort_tracks.csv"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"


# ============================================================
# VIDEO INFORMATION
# ============================================================

cap = cv2.VideoCapture(str(VIDEO))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO}")

input_fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

cap.release()


print("=" * 70)
print("YOLO11m + OC-SORT")
print("=" * 70)

print(f"Video       : {VIDEO}")
print(f"Model       : {MODEL}")
print(f"Input FPS   : {input_fps:.2f}")
print(f"Resolution  : {width}x{height}")
print(f"Frames      : {total_frames}")
print(f"Confidence  : {CONFIDENCE}")
print(f"Image size  : {IMAGE_SIZE}")
print(f"Device      : {DEVICE}")
print(f"Output      : {OUTPUT_VIDEO}")
print()


# ============================================================
# BOXMOT
# ============================================================

print("Initializing BoxMOT...")

tracker = Boxmot(
    detector=str(MODEL),
    tracker="ocsort",
    project=str(OUTPUT_DIR),
)

print("BoxMOT initialized successfully.")
print(f"Tracker : {tracker.tracker}")
print(f"Detector: {tracker.detector}")
print()


# ============================================================
# RUN TRACKING
# ============================================================

print("=" * 70)
print("STARTING OC-SORT")
print("=" * 70)

start_time = time.perf_counter()

result = tracker.track(
    source=str(VIDEO),
    imgsz=IMAGE_SIZE,
    conf=CONFIDENCE,
    device=DEVICE,
    save=True,
    save_txt=True,
    show=False,
    verbose=True,
)

elapsed = time.perf_counter() - start_time

print()
print("=" * 70)
print("OC-SORT TRACKING COMPLETE")
print("=" * 70)
print(f"Processing time : {elapsed:.3f} sec")
print(f"Processing FPS  : {total_frames / elapsed:.3f}")
print(f"Result object   : {type(result)}")
print("=" * 70)