import cv2
import csv
import time
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "onnx" / "yolo11m.onnx"
VIDEO_PATH = ROOT / "test_videos" / "video0.avi"

OUTPUT_DIR = ROOT / "outputs" / "video0_onnx"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "video0_yolo11m_onnx.mp4"

RESULTS_DIR = ROOT / "experiments"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "video0_onnx_results.csv"


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 640
DEVICE = "cpu"


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11m ONNX - VIDEO0 BASELINE")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Video : {VIDEO_PATH}")
    print(f"Output: {OUTPUT_VIDEO}")
    print()

    # --------------------------------------------------------
    # Load ONNX model
    # --------------------------------------------------------

    print("Loading YOLO11m ONNX model...")

    model = YOLO(str(MODEL_PATH))

    print("YOLO11m ONNX loaded successfully.")
    print()

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    input_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("=" * 70)
    print("VIDEO INFORMATION")
    print("=" * 70)

    print(f"Input FPS      : {input_fps:.2f}")
    print(f"Resolution     : {width}x{height}")
    print(f"Total frames   : {total_frames}")
    print(f"Confidence     : {CONFIDENCE}")
    print(f"Image size     : {IMAGE_SIZE}")
    print(f"Device         : {DEVICE}")
    print()

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        input_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not create output video.")

    # --------------------------------------------------------
    # Processing
    # --------------------------------------------------------

    frame_count = 0
    detection_count = 0

    start_time = time.perf_counter()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        results = model.predict(
            source=frame,
            conf=CONFIDENCE,
            imgsz=IMAGE_SIZE,
            device=DEVICE,
            verbose=False
        )

        result = results[0]

        # Count detections
        if result.boxes is not None:
            num_detections = len(result.boxes)
            detection_count += num_detections

        # Draw detections
        annotated_frame = result.plot()

        writer.write(annotated_frame)

        # Progress
        if frame_count % 50 == 0 or frame_count == total_frames:

            elapsed = time.perf_counter() - start_time

            current_fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            progress = (
                frame_count / total_frames * 100
                if total_frames > 0
                else 0
            )

            print(
                f"Frames: {frame_count}/{total_frames} "
                f"({progress:.1f}%) | "
                f"FPS: {current_fps:.2f}"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    elapsed_time = time.perf_counter() - start_time

    cap.release()
    writer.release()

    processing_fps = (
        frame_count / elapsed_time
        if elapsed_time > 0
        else 0
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        csv_writer = csv.writer(f)

        csv_writer.writerow([
            "pipeline",
            "model",
            "video",
            "frames",
            "input_fps",
            "width",
            "height",
            "confidence",
            "image_size",
            "device",
            "processing_time_sec",
            "processing_fps",
            "detections",
            "output"
        ])

        csv_writer.writerow([
            "ONNX_CPU",
            "YOLO11m",
            VIDEO_PATH.name,
            frame_count,
            f"{input_fps:.3f}",
            width,
            height,
            CONFIDENCE,
            IMAGE_SIZE,
            DEVICE,
            f"{elapsed_time:.3f}",
            f"{processing_fps:.3f}",
            detection_count,
            str(OUTPUT_VIDEO)
        ])

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PROCESSING COMPLETE")
    print("=" * 70)

    print(f"Frames processed : {frame_count}")
    print(f"Detections       : {detection_count}")
    print(f"Processing time  : {elapsed_time:.3f} sec")
    print(f"Processing FPS   : {processing_fps:.3f}")
    print(f"Output           : {OUTPUT_VIDEO}")
    print(f"Results CSV      : {RESULTS_CSV}")

    print("=" * 70)


if __name__ == "__main__":
    main()