from pathlib import Path
import time
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


ROOT = Path(__file__).resolve().parents[3]

VIDEO = ROOT / "test_videos" / "video0.avi"
MODEL = ROOT / "models" / "pytorch" / "yolo11m.pt"

OUTPUT_DIR = ROOT / "outputs" / "trackers" / "deepsort"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_VIDEO = OUTPUT_DIR / "video0_deepsort.mp4"
OUTPUT_TRACKS = OUTPUT_DIR / "video0_deepsort_tracks.txt"


CONF = 0.25
IMAGE_SIZE = 640


def main():

    print("=" * 70)
    print("DEEP SORT - VIDEO0")
    print("=" * 70)

    print(f"Video   : {VIDEO}")
    print(f"Model   : {MODEL}")
    print(f"Output  : {OUTPUT_DIR}")
    print()

    if not VIDEO.exists():
        raise FileNotFoundError(f"Video not found: {VIDEO}")

    if not MODEL.exists():
        raise FileNotFoundError(f"Model not found: {MODEL}")

    # ------------------------------------------------------------
    # Video
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError("Could not open video.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Resolution : {width}x{height}")
    print(f"Input FPS  : {input_fps:.3f}")
    print(f"Frames     : {total_frames}")
    print()

    # ------------------------------------------------------------
    # YOLO
    # ------------------------------------------------------------

    model = YOLO(str(MODEL))

    # ------------------------------------------------------------
    # Deep SORT
    # ------------------------------------------------------------

    tracker = DeepSort(
        max_iou_distance=0.7,
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=None,
        embedder="mobilenet",
        half=False,
        bgr=True,
        embedder_gpu=False,
    )

    print("YOLO model loaded.")
    print("Deep SORT initialized.")
    print("Embedder : mobilenet")
    print("Device   : CPU")
    print()

    # ------------------------------------------------------------
    # Output video
    # ------------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(OUTPUT_VIDEO),
        fourcc,
        input_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError("Could not create output video.")

    # ------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------

    frame_count = 0
    detections_total = 0
    track_rows = 0

    unique_ids = set()

    processing_start = time.perf_counter()

    # ------------------------------------------------------------
    # Tracking
    # ------------------------------------------------------------

    with open(
        OUTPUT_TRACKS,
        "w",
        encoding="utf-8"
    ) as track_file:

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            # ----------------------------------------------------
            # YOLO inference
            # ----------------------------------------------------

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONF,
                device="cpu",
                verbose=False,
            )

            result = results[0]

            detections = []

            if result.boxes is not None:

                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()

                detections_total += len(boxes)

                for box, score, cls in zip(
                    boxes,
                    scores,
                    classes
                ):

                    x1, y1, x2, y2 = box

                    w = x2 - x1
                    h = y2 - y1

                    detections.append(
                        (
                            [
                                float(x1),
                                float(y1),
                                float(w),
                                float(h),
                            ],
                            float(score),
                            int(cls),
                        )
                    )

            # ----------------------------------------------------
            # Deep SORT
            # ----------------------------------------------------

            tracks = tracker.update_tracks(
                detections,
                frame=frame,
            )

            # ----------------------------------------------------
            # Draw tracks
            # ----------------------------------------------------

            for track in tracks:

                if not track.is_confirmed():
                    continue

                track_id = track.track_id

                ltrb = track.to_ltrb()

                if ltrb is None:
                    continue

                x1, y1, x2, y2 = map(int, ltrb)

                unique_ids.add(track_id)
                track_rows += 1

                # ----------------------------------------------
                # Track file
                # ----------------------------------------------

                track_file.write(
                    f"{frame_count},"
                    f"{track_id},"
                    f"{x1},"
                    f"{y1},"
                    f"{x2 - x1},"
                    f"{y2 - y1}\n"
                )

                # ----------------------------------------------
                # Draw
                # ----------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2,
                )

                cv2.putText(
                    frame,
                    f"ID {track_id}",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

            writer.write(frame)

            # ----------------------------------------------------
            # Progress
            # ----------------------------------------------------

            if frame_count % 50 == 0:

                elapsed = time.perf_counter() - processing_start
                fps = frame_count / elapsed if elapsed > 0 else 0

                print(
                    f"Frame {frame_count}/{total_frames} | "
                    f"FPS: {fps:.2f} | "
                    f"Detections: {detections_total} | "
                    f"IDs: {len(unique_ids)}"
                )

    # ------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------

    cap.release()
    writer.release()

    elapsed = time.perf_counter() - processing_start

    processing_fps = (
        frame_count / elapsed
        if elapsed > 0
        else 0
    )

    # ------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------

    print()
    print("=" * 70)
    print("DEEP SORT TRACKING COMPLETE")
    print("=" * 70)

    print(f"Frames          : {frame_count}")
    print(f"Detections      : {detections_total}")
    print(f"Track rows      : {track_rows}")
    print(f"Unique IDs      : {len(unique_ids)}")
    print(f"Processing time : {elapsed:.3f} sec")
    print(f"Processing FPS  : {processing_fps:.3f}")
    print()
    print(f"Output video    : {OUTPUT_VIDEO}")
    print(f"Track file      : {OUTPUT_TRACKS}")

    print("=" * 70)


if __name__ == "__main__":
    main()