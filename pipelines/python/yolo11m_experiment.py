import csv
import time
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "pytorch" / "yolo11m.pt"
VIDEO_DIR = ROOT / "test_videos"
OUTPUT_DIR = ROOT / "outputs" / "yolo11m_baseline"

RESULTS_DIR = ROOT / "experiments"
RESULTS_FILE = RESULTS_DIR / "yolo11m_baseline_results.csv"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 640

# Start with CPU-safe settings.
# We will check GPU/TensorRT/ONNX separately later.
DEVICE = "cpu"


# ============================================================
# DIRECTORY SETUP
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(model, video_path):

    video_name = video_path.stem

    output_path = OUTPUT_DIR / f"{video_name}_yolo11m.mp4"

    print()
    print("=" * 70)
    print(f"PROCESSING: {video_path.name}")
    print("=" * 70)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        fps = 30.0

    print(f"Input FPS      : {fps:.2f}")
    print(f"Resolution     : {width}x{height}")
    print(f"Total frames   : {total_frames}")
    print(f"Confidence     : {CONFIDENCE}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Device         : {DEVICE}")
    print(f"Output         : {output_path}")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():
        print("ERROR: Could not create output video.")
        cap.release()
        return None

    processed_frames = 0
    total_detections = 0

    start_time = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # ----------------------------------------------------
        # YOLO11m inference
        # ----------------------------------------------------

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # ----------------------------------------------------
        # Detection count
        # ----------------------------------------------------

        if result.boxes is not None:
            detection_count = len(result.boxes)
        else:
            detection_count = 0

        total_detections += detection_count

        # ----------------------------------------------------
        # Draw detections
        # ----------------------------------------------------

        annotated_frame = result.plot()

        writer.write(annotated_frame)

        processed_frames += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if processed_frames % 25 == 0:

            elapsed = time.perf_counter() - start_time

            processing_fps = (
                processed_frames / elapsed
                if elapsed > 0
                else 0
            )

            percentage = (
                processed_frames / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Frames: {processed_frames}/{total_frames} "
                f"({percentage:.1f}%) | "
                f"FPS: {processing_fps:.2f}",
                end="\r"
            )

    # ========================================================
    # FINISH
    # ========================================================

    elapsed_time = time.perf_counter() - start_time

    processing_fps = (
        processed_frames / elapsed_time
        if elapsed_time > 0
        else 0
    )

    cap.release()
    writer.release()

    print()
    print()
    print("-" * 70)
    print("PROCESSING COMPLETE")
    print("-" * 70)

    print(f"Frames processed : {processed_frames}")
    print(f"Detections       : {total_detections}")
    print(f"Processing time  : {elapsed_time:.2f} sec")
    print(f"Processing FPS   : {processing_fps:.2f}")
    print(f"Output           : {output_path}")
    print("-" * 70)

    return {
        "tracker": "Baseline",
        "video": video_name,
        "frames": processed_frames,
        "processing_time_sec": round(elapsed_time, 3),
        "processing_fps": round(processing_fps, 3),
        "detections": total_detections,
        "tracks": "",
        "IDF1": "",
        "HOTA": "",
        "MOTA": "",
        "ID_switches": "",
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):

    fieldnames = [
        "tracker",
        "video",
        "frames",
        "processing_time_sec",
        "processing_fps",
        "detections",
        "tracks",
        "IDF1",
        "HOTA",
        "MOTA",
        "ID_switches",
    ]

    with open(
        RESULTS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11m BASELINE EXPERIMENT")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if not MODEL_PATH.exists():
        print()
        print("ERROR: YOLO11m model not found.")
        print(MODEL_PATH)
        return

    # --------------------------------------------------------
    # LOAD MODEL ONLY ONCE
    # --------------------------------------------------------

    print()
    print("Loading YOLO11m...")

    model = YOLO(str(MODEL_PATH))

    print("YOLO11m loaded successfully.")

    # --------------------------------------------------------
    # FIND VIDEOS
    # --------------------------------------------------------

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print("ERROR: No MP4 videos found.")
        return

    print()
    print(f"Videos found: {len(videos)}")

    for video in videos:
        print(f"  - {video.name}")

    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    results = []

    overall_start = time.perf_counter()

    for video_path in videos:

        result = process_video(
            model,
            video_path
        )

        if result is not None:
            results.append(result)

    overall_time = time.perf_counter() - overall_start

    # --------------------------------------------------------
    # SAVE RESULTS
    # --------------------------------------------------------

    save_results(results)

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL VIDEOS PROCESSED")
    print("=" * 70)

    print(f"Videos processed : {len(results)}")
    print(f"Total time       : {overall_time:.2f} sec")
    print(f"Results CSV      : {RESULTS_FILE}")

    print("=" * 70)


if __name__ == "__main__":
    main()