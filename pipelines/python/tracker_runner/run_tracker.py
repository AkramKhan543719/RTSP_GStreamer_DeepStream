"""
============================================================
YOLO11m + BoxMOT 18.0.0 Universal Tracker Runner
============================================================

Run:

    python run_tracker.py

The user selects the tracker at runtime.

Supported BoxMOT 18.0.0 trackers:

    1. ByteTrack
    2. OC-SORT
    3. SFSORT
    4. BoT-SORT
    5. StrongSORT
    6. Deep OC-SORT
    7. HybridSORT
    8. BoostTrack
    9. OccluBoost
============================================================
"""

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from tracker_registry import (
    TRACKERS,
    get_tracker,
    print_trackers,
)

from tracker_factory import (
    create_selected_tracker,
)


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "pytorch"
    / "yolo11m.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "test_videos"
    / "video0.avi"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "runtime_trackers"
)


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640
DEVICE = "cpu"
HALF = False


# ============================================================
# HEADER
# ============================================================

def print_header():

    print()
    print("=" * 72)
    print("          YOLO11m + BoxMOT 18.0.0")
    print("             UNIVERSAL TRACKER")
    print("=" * 72)

    print(f"Model      : {MODEL_PATH}")
    print(f"Input      : {VIDEO_PATH}")
    print(f"Confidence : {CONFIDENCE}")
    print(f"IoU        : {IOU_THRESHOLD}")
    print(f"Image size : {IMAGE_SIZE}")
    print(f"Device     : {DEVICE}")

    print("=" * 72)


# ============================================================
# PATH VALIDATION
# ============================================================

def validate_paths():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# TRACKER SELECTION
# ============================================================

def select_tracker():

    while True:

        print_trackers()

        value = input(
            "Enter tracker number [1-9] "
            "or Q to quit: "
        ).strip()

        if value.lower() == "q":

            print("Exiting...")
            sys.exit(0)

        try:

            number = int(value)

        except ValueError:

            print(
                "Invalid input. "
                "Please enter a number."
            )

            continue

        tracker = get_tracker(number)

        if tracker is None:

            print(
                "Invalid tracker number."
            )

            continue

        return tracker


# ============================================================
# VIDEO INFORMATION
# ============================================================

def get_video_info():

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Cannot open video:\n{VIDEO_PATH}"
        )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    frame_count = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    cap.release()

    return (
        width,
        height,
        fps,
        frame_count,
    )


# ============================================================
# VIDEO WRITER
# ============================================================

def create_video_writer(
    output_path,
    width,
    height,
    fps,
):

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():

        raise RuntimeError(
            f"Unable to create output video:\n"
            f"{output_path}"
        )

    return writer


# ============================================================
# DETECTIONS
# ============================================================

def get_detections(result):

    if result.boxes is None:

        return np.empty(
            (0, 6),
            dtype=np.float32,
        )

    if len(result.boxes) == 0:

        return np.empty(
            (0, 6),
            dtype=np.float32,
        )

    boxes = (
        result.boxes.xyxy
        .cpu()
        .numpy()
    )

    confidence = (
        result.boxes.conf
        .cpu()
        .numpy()
    )

    classes = (
        result.boxes.cls
        .cpu()
        .numpy()
    )

    detections = np.column_stack(
        (
            boxes,
            confidence,
            classes,
        )
    )

    return detections.astype(
        np.float32
    )


# ============================================================
# TRACK DRAWING
# ============================================================

def draw_tracks(
    frame,
    tracks,
):

    if tracks is None:

        return frame

    for track in tracks:

        if len(track) < 5:

            continue

        x1 = int(track[0])
        y1 = int(track[1])
        x2 = int(track[2])
        y2 = int(track[3])

        track_id = int(track[4])

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2,
        )

        label = (
            f"ID: {track_id}"
        )

        cv2.putText(
            frame,
            label,
            (x1, max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame


# ============================================================
# TRACKER LABEL
# ============================================================

def draw_tracker_label(
    frame,
    tracker_name,
    frame_number,
    total_frames,
    fps,
):

    height, width = frame.shape[:2]

    # -----------------------------------------
    # Tracker label - top right
    # -----------------------------------------

    label = (
        f"TRACKER: {tracker_name}"
    )

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = 0.8
    thickness = 2

    text_size = cv2.getTextSize(
        label,
        font,
        scale,
        thickness,
    )[0]

    text_width = text_size[0]

    x = (
        width
        - text_width
        - 25
    )

    y = 40

    cv2.rectangle(
        frame,
        (
            x - 10,
            8,
        ),
        (
            width - 10,
            55,
        ),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # -----------------------------------------
    # Frame/FPS - top left
    # -----------------------------------------

    info = (
        f"Frame: "
        f"{frame_number}/"
        f"{total_frames} | "
        f"FPS: {fps:.2f}"
    )

    cv2.rectangle(
        frame,
        (10, 8),
        (350, 55),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        info,
        (20, 40),
        font,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


# ============================================================
# MAIN TRACKING
# ============================================================

def run_tracker(tracker_info):

    print()
    print("=" * 72)
    print(
        f"SELECTED TRACKER : "
        f"{tracker_info.name}"
    )
    print("=" * 72)

    # -----------------------------------------
    # Video information
    # -----------------------------------------

    (
        width,
        height,
        video_fps,
        total_frames,
    ) = get_video_info()

    print(
        f"Video : "
        f"{width}x{height} @ "
        f"{video_fps:.2f} FPS"
    )

    # -----------------------------------------
    # Output paths
    # -----------------------------------------

    output_directory = (
        OUTPUT_ROOT
        / tracker_info.backend
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_video = (
        output_directory
        / f"video0_{tracker_info.backend}.mp4"
    )

    metrics_path = (
        output_directory
        / f"video0_{tracker_info.backend}_metrics.json"
    )

    log_path = (
        output_directory
        / f"video0_{tracker_info.backend}.log"
    )

    print(
        f"Output video : {output_video}"
    )

    # -----------------------------------------
    # YOLO
    # -----------------------------------------

    print()
    print("Loading YOLO11m...")

    model = YOLO(
        str(MODEL_PATH)
    )

    # -----------------------------------------
    # BoxMOT
    # -----------------------------------------

    print(
        f"Loading BoxMOT tracker: "
        f"{tracker_info.backend}"
    )

    tracker = create_selected_tracker(
        tracker_info.backend,
        device=DEVICE,
        half=HALF,
        per_class=False,
    )

    print("Tracker loaded.")

    # -----------------------------------------
    # Video
    # -----------------------------------------

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    writer = create_video_writer(
        output_video,
        width,
        height,
        video_fps,
    )

    # -----------------------------------------
    # Statistics
    # -----------------------------------------

    frame_number = 0
    total_detections = 0
    unique_track_ids = set()

    start_time = time.perf_counter()

    # -----------------------------------------
    # Processing
    # -----------------------------------------

    try:

        while True:

            success, frame = cap.read()

            if not success:

                break

            frame_number += 1

            # ---------------------------------
            # YOLO detection
            # ---------------------------------

            result = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                verbose=False,
            )[0]

            detections = get_detections(
                result
            )

            total_detections += (
                len(detections)
            )

            # ---------------------------------
            # BoxMOT
            # ---------------------------------

            if len(detections) > 0:

                tracks = tracker.update(
                    detections,
                    frame,
                )

            else:

                tracks = []

            # ---------------------------------
            # Track IDs
            # ---------------------------------

            if tracks is not None:

                for track in tracks:

                    if len(track) >= 5:

                        try:

                            unique_track_ids.add(
                                int(track[4])
                            )

                        except Exception:

                            pass

            # ---------------------------------
            # Draw
            # ---------------------------------

            frame = draw_tracks(
                frame,
                tracks,
            )

            elapsed = (
                time.perf_counter()
                - start_time
            )

            current_fps = (
                frame_number / elapsed
                if elapsed > 0
                else 0
            )

            frame = draw_tracker_label(
                frame,
                tracker_info.name,
                frame_number,
                total_frames,
                current_fps,
            )

            writer.write(frame)

            # ---------------------------------
            # Console progress
            # ---------------------------------

            if frame_number % 25 == 0:

                print(
                    f"\rProcessing: "
                    f"{frame_number}/"
                    f"{total_frames} "
                    f"({frame_number / total_frames * 100:.1f}%)",
                    end="",
                    flush=True,
                )

    finally:

        cap.release()
        writer.release()

    # -----------------------------------------
    # Final statistics
    # -----------------------------------------

    total_time = (
        time.perf_counter()
        - start_time
    )

    average_fps = (
        frame_number / total_time
        if total_time > 0
        else 0
    )

    average_latency = (
        total_time
        / frame_number
        * 1000
        if frame_number > 0
        else 0
    )

    # -----------------------------------------
    # JSON
    # -----------------------------------------

    metrics = {

        "project": {
            "name":
                "YOLO11m + BoxMOT Object Tracking",

            "tracker":
                tracker_info.name,

            "boxmot_version":
                "18.0.0",
        },

        "model": {
            "name":
                "YOLO11m",

            "format":
                "pt",

            "path":
                str(MODEL_PATH),
        },

        "configuration": {

            "confidence_threshold":
                CONFIDENCE,

            "iou_threshold":
                IOU_THRESHOLD,

            "image_size":
                IMAGE_SIZE,

            "device":
                DEVICE,

            "tracker_backend":
                tracker_info.backend,
        },

        "video": {

            "input":
                str(VIDEO_PATH),

            "width":
                width,

            "height":
                height,

            "fps":
                video_fps,

            "total_frames":
                total_frames,
        },

        "tracking_statistics": {

            "frame_count":
                frame_number,

            "total_detections":
                total_detections,

            "unique_track_ids":
                len(unique_track_ids),

            "average_fps":
                average_fps,

            "average_latency_ms":
                average_latency,

            "total_processing_time_seconds":
                total_time,
        },

        "outputs": {

            "video":
                str(output_video),

            "json":
                str(metrics_path),

            "log":
                str(log_path),
        },
    }

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # -----------------------------------------
    # Log
    # -----------------------------------------

    with open(
        log_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Tracker: "
            f"{tracker_info.name}\n"
        )

        file.write(
            f"BoxMOT: 18.0.0\n"
        )

        file.write(
            f"Frames: "
            f"{frame_number}\n"
        )

        file.write(
            f"Detections: "
            f"{total_detections}\n"
        )

        file.write(
            f"Unique IDs: "
            f"{len(unique_track_ids)}\n"
        )

        file.write(
            f"Processing time: "
            f"{total_time:.3f} sec\n"
        )

        file.write(
            f"Average FPS: "
            f"{average_fps:.3f}\n"
        )

    # -----------------------------------------
    # Result
    # -----------------------------------------

    print()
    print()
    print("=" * 72)
    print("                    COMPLETE")
    print("=" * 72)

    print(
        f"Tracker          : "
        f"{tracker_info.name}"
    )

    print(
        f"Frames           : "
        f"{frame_number}"
    )

    print(
        f"Detections       : "
        f"{total_detections}"
    )

    print(
        f"Unique Track IDs : "
        f"{len(unique_track_ids)}"
    )

    print(
        f"Processing time  : "
        f"{total_time:.3f} sec"
    )

    print(
        f"Average FPS      : "
        f"{average_fps:.3f}"
    )

    print(
        f"Output video     : "
        f"{output_video}"
    )

    print(
        f"Metrics JSON     : "
        f"{metrics_path}"
    )

    print(
        f"Log              : "
        f"{log_path}"
    )

    print("=" * 72)


# ============================================================
# MAIN
# ============================================================

def main():

    print_header()

    print(
        "\nChecking BoxMOT..."
    )

    import boxmot

    print(
        "BoxMOT version:",
        getattr(
            boxmot,
            "__version__",
            "unknown",
        ),
    )

    validate_paths()

    tracker = select_tracker()

    run_tracker(tracker)


if __name__ == "__main__":

    main()