from pathlib import Path
import time
from boxmot import Boxmot


ROOT = Path(__file__).resolve().parents[3]

VIDEO = ROOT / "test_videos" / "video0.avi"
MODEL = ROOT / "models" / "pytorch" / "yolo11m.pt"
OUTPUT = ROOT / "outputs" / "trackers"

TRACKER_NAME = "mcbyte"


def main():

    print("=" * 70)
    print("McByte - VIDEO0")
    print("=" * 70)

    print(f"Video   : {VIDEO}")
    print(f"Model   : {MODEL}")
    print(f"Tracker : {TRACKER_NAME}")
    print(f"Output  : {OUTPUT}")
    print()

    if not VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO}")

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    tracker = Boxmot(
        detector=str(MODEL),
        tracker=TRACKER_NAME,
        project=str(OUTPUT),
    )

    print("BoxMOT configured successfully.")
    print(f"Detector : {tracker.detector}")
    print(f"Tracker  : {tracker.tracker}")
    print()

    start = time.perf_counter()

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

    elapsed = time.perf_counter() - start

    print()
    print("=" * 70)
    print("MCBYTE TRACKING COMPLETE")
    print("=" * 70)
    print(f"Processing time : {elapsed:.3f} sec")
    print(f"Processing FPS  : {998 / elapsed:.3f}")
    print(f"Result object   : {type(result)}")
    print("=" * 70)


if __name__ == "__main__":
    main()