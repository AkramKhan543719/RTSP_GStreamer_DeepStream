import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "pytorch" / "yolo11m.pt"
VIDEO_DIR = ROOT / "test_videos"
OUTPUT_DIR = ROOT / "outputs" / "trackers"
RESULTS_DIR = ROOT / "06_Trackers" / "results"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 640

VIDEOS = [
    "video1.mp4",
    "video2.mp4",
    "video3.mp4",
    "video4.mp4",
]


# ============================================================
# CSV
# ============================================================

CSV_PATH = RESULTS_DIR / "tracker_runtime_results.csv"


def create_csv():

    if CSV_PATH.exists():
        return

    with open(CSV_PATH, "w", newline="") as f:

        writer = csv.writer(f)

        writer.writerow([
            "tracker",
            "video",
            "frames",
            "processing_time_sec",
            "processing_fps",
            "detections",
            "tracks",
            "ID_switches",
            "IDF1",
            "HOTA",
            "MOTA",
        ])


# ============================================================
# VIDEO PROCESSOR
# ============================================================

def process_video(model, video_path, tracker_name):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():

        print(f"ERROR: Cannot open {video_path}")

        return None

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_name = (
        f"{video_path.stem}_{tracker_name}.mp4"
    )

    output_path = OUTPUT_DIR / output_name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    frame_count = 0
    detection_count = 0
    track_count = 0

    start_time = time.perf_counter()

    print()
    print("=" * 70)
    print(f"TRACKER : {tracker_name}")
    print(f"VIDEO   : {video_path.name}")
    print("=" * 70)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model(
            frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            verbose=False,
            device="cpu"
        )

        result = results[0]

        boxes = result.boxes

        if boxes is not None:

            detection_count += len(boxes)

            for box in boxes:

                xyxy = box.xyxy[0].cpu().numpy()

                x1, y1, x2, y2 = map(int, xyxy)

                confidence = float(
                    box.conf[0].cpu().numpy()
                )

                cls = int(
                    box.cls[0].cpu().numpy()
                )

                label = model.names[cls]

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"{label} {confidence:.2f}",
                    (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )

        writer.write(frame)

        if frame_count % 50 == 0:

            elapsed = time.perf_counter() - start_time

            current_fps = frame_count / elapsed

            print(
                f"Frames: {frame_count}/{total_frames} | "
                f"FPS: {current_fps:.2f}"
            )

    elapsed = time.perf_counter() - start_time

    processing_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0
    )

    cap.release()
    writer.release()

    print()
    print("-" * 70)
    print("PROCESSING COMPLETE")
    print("-" * 70)

    print(f"Frames processed : {frame_count}")
    print(f"Detections       : {detection_count}")
    print(f"Processing time  : {elapsed:.3f} sec")
    print(f"Processing FPS   : {processing_fps:.3f}")
    print(f"Output           : {output_path}")
    print("-" * 70)

    return {
        "tracker": tracker_name,
        "video": video_path.name,
        "frames": frame_count,
        "processing_time_sec": round(elapsed, 3),
        "processing_fps": round(processing_fps, 3),
        "detections": detection_count,
        "tracks": "",
        "ID_switches": "",
        "IDF1": "",
        "HOTA": "",
        "MOTA": "",
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11m TRACKER EXPERIMENT FRAMEWORK")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    create_csv()

    print()
    print("Loading YOLO11m...")

    model = YOLO(str(MODEL_PATH))

    print("YOLO11m loaded successfully.")

    all_results = []

    tracker_name = "BASELINE_DETECTION"

    for video_name in VIDEOS:

        video_path = VIDEO_DIR / video_name

        if not video_path.exists():

            print(
                f"WARNING: {video_path} does not exist."
            )

            continue

        result = process_video(
            model,
            video_path,
            tracker_name
        )

        if result:
            all_results.append(result)

    if all_results:

        with open(
            CSV_PATH,
            "a",
            newline=""
        ) as f:

            writer = csv.writer(f)

            for result in all_results:

                writer.writerow([
                    result["tracker"],
                    result["video"],
                    result["frames"],
                    result["processing_time_sec"],
                    result["processing_fps"],
                    result["detections"],
                    result["tracks"],
                    result["ID_switches"],
                    result["IDF1"],
                    result["HOTA"],
                    result["MOTA"],
                ])

    print()
    print("=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)

    print(f"Results: {CSV_PATH}")


if __name__ == "__main__":
    main()