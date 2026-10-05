from pathlib import Path
from ultralytics import YOLO
import cv2
import time


# ============================================================
# YOLO11m PYTORCH BASELINE
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "pytorch" / "yolo11m.pt"
INPUT_DIR = ROOT / "test_videos"
OUTPUT_DIR = ROOT / "outputs" / "yolo11m_baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_video(model, input_path, output_path):
    print("\n" + "=" * 60)
    print(f"INPUT  : {input_path.name}")
    print(f"OUTPUT : {output_path}")
    print("=" * 60)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print(f"ERROR: Could not open {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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
        return

    frame_count = 0
    detection_count = 0

    start_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(
            frame,
            verbose=False
        )

        result = results[0]

        annotated = result.plot()

        boxes = result.boxes

        if boxes is not None:
            detection_count += len(boxes)

        writer.write(annotated)

        frame_count += 1

        if frame_count % 30 == 0:

            elapsed = time.time() - start_time

            current_fps = frame_count / elapsed if elapsed > 0 else 0

            if total_frames > 0:
                progress = (frame_count / total_frames) * 100
                print(
                    f"\rFrames: {frame_count}/{total_frames} "
                    f"({progress:.1f}%) | "
                    f"FPS: {current_fps:.2f}",
                    end=""
                )
            else:
                print(
                    f"\rFrames: {frame_count} | "
                    f"FPS: {current_fps:.2f}",
                    end=""
                )

    cap.release()
    writer.release()

    elapsed = time.time() - start_time

    processing_fps = frame_count / elapsed if elapsed > 0 else 0

    print("\n")
    print("-" * 60)
    print("PROCESSING COMPLETE")
    print("-" * 60)
    print(f"Frames processed : {frame_count}")
    print(f"Detections       : {detection_count}")
    print(f"Processing time  : {elapsed:.2f} sec")
    print(f"Processing FPS   : {processing_fps:.2f}")
    print(f"Output           : {output_path}")
    print("-" * 60)


def main():

    print("=" * 60)
    print("          YOLO11m PYTORCH BASELINE")
    print("=" * 60)

    print(f"Model : {MODEL_PATH}")
    print(f"Input : {INPUT_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    if not MODEL_PATH.exists():
        print("\nERROR: YOLO11m model not found.")
        return

    print("\nLoading YOLO11m...")

    model = YOLO(str(MODEL_PATH))

    print("YOLO11m loaded successfully.")
    print(f"Task: {model.task}")

    videos = sorted(INPUT_DIR.glob("*.mp4"))

    if not videos:
        print("\nERROR: No MP4 videos found.")
        return

    print(f"\nFound {len(videos)} test videos:")

    for video in videos:
        print(f"  - {video.name}")

    for video in videos:

        output_path = OUTPUT_DIR / f"{video.stem}_yolo11m.mp4"

        process_video(
            model,
            video,
            output_path
        )

    print("\n" + "=" * 60)
    print("ALL VIDEOS PROCESSED")
    print("=" * 60)


if __name__ == "__main__":
    main()