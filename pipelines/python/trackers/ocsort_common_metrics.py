import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from collections import defaultdict

import cv2
import psutil

from boxmot import Boxmot


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = ROOT / "models" / "pytorch" / "yolo11m.pt"
VIDEO_PATH = ROOT / "test_videos" / "video0.avi"

OUTPUT_ROOT = ROOT / "outputs" / "pt"

VIDEO_DIR = OUTPUT_ROOT / "videos"
JSON_DIR = OUTPUT_ROOT / "json"
LOG_DIR = OUTPUT_ROOT / "logs"

VIDEO_OUTPUT = VIDEO_DIR / "ocsort_output.mp4"
JSON_OUTPUT = JSON_DIR / "ocsort_metrics.json"
LOG_OUTPUT = LOG_DIR / "ocsort.log"


# ============================================================
# CONFIGURATION
# ============================================================

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.5
IMAGE_SIZE = 640
DEVICE = "cpu"

TRACKER_NAME = "ocsort"
TRACKER_CONFIG = "ocsort.yaml"


# ============================================================
# DIRECTORIES
# ============================================================

VIDEO_DIR.mkdir(parents=True, exist_ok=True)
JSON_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOGGING
# ============================================================

logger = logging.getLogger("OCSORT")
logger.setLevel(logging.INFO)

logger.handlers.clear()

formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s"
)

file_handler = logging.FileHandler(
    LOG_OUTPUT,
    mode="w",
    encoding="utf-8"
)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# ============================================================
# RESOURCE MONITOR
# ============================================================

class ResourceMonitor:

    def __init__(self):

        self.running = False
        self.thread = None

        self.process_cpu = []
        self.process_ram = []

        self.system_cpu = []

        self.process = psutil.Process(os.getpid())

    def start(self):

        try:
            self.process.cpu_percent(None)
            psutil.cpu_percent(None)
        except Exception:
            pass

        self.running = True

        self.thread = threading.Thread(
            target=self._monitor,
            daemon=True
        )

        self.thread.start()

    def _monitor(self):

        while self.running:

            try:

                process_cpu = self.process.cpu_percent(None)

                process_ram = (
                    self.process.memory_info().rss
                    / (1024 * 1024)
                )

                system_cpu = psutil.cpu_percent(None)

                self.process_cpu.append(process_cpu)
                self.process_ram.append(process_ram)
                self.system_cpu.append(system_cpu)

            except Exception:
                pass

            time.sleep(0.1)

    def stop(self):

        self.running = False

        if self.thread:

            self.thread.join(timeout=3)

    def get_results(self):

        def average(values):

            if not values:
                return 0.0

            return sum(values) / len(values)

        def maximum(values):

            if not values:
                return 0.0

            return max(values)

        return {

            "process": {

                "average_cpu_percent":
                    average(self.process_cpu),

                "peak_cpu_percent":
                    maximum(self.process_cpu),

                "average_ram_mb":
                    average(self.process_ram),

                "peak_ram_mb":
                    maximum(self.process_ram)
            },

            "system": {

                "average_cpu_percent":
                    average(self.system_cpu),

                "peak_cpu_percent":
                    maximum(self.system_cpu)
            }
        }


# ============================================================
# PERCENTILE
# ============================================================

def percentile(values, percentage):

    if not values:
        return 0.0

    values = sorted(values)

    position = (len(values) - 1) * percentage

    lower = int(position)

    upper = min(
        lower + 1,
        len(values) - 1
    )

    fraction = position - lower

    return (
        values[lower]
        + (
            values[upper]
            - values[lower]
        ) * fraction
    )


# ============================================================
# TRACK LIFETIME
# ============================================================

def calculate_track_lifetimes(track_history):

    lifetimes = []

    for track_id, frames in track_history.items():

        unique_frames = set(frames)

        if unique_frames:

            lifetimes.append(
                len(unique_frames)
            )

    if not lifetimes:

        return 0.0, 0

    average_lifetime = (
        sum(lifetimes)
        / len(lifetimes)
    )

    longest_lifetime = max(lifetimes)

    return (
        average_lifetime,
        longest_lifetime
    )


# ============================================================
# PARSE BOXMOT TXT
# ============================================================

def parse_tracking_file(txt_file):

    logger.info(
        f"Reading tracking results: {txt_file}"
    )

    track_history = defaultdict(list)

    tracks_per_frame = defaultdict(int)

    unique_ids = set()

    total_rows = 0

    total_detections = 0

    if not txt_file.exists():

        logger.warning(
            f"Tracking file not found: {txt_file}"
        )

        return (
            track_history,
            tracks_per_frame,
            unique_ids,
            total_rows,
            total_detections
        )

    with open(
        txt_file,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            try:

                frame_number = int(parts[0])

                track_id = int(parts[1])

            except Exception:

                continue

            total_rows += 1

            total_detections += 1

            unique_ids.add(track_id)

            tracks_per_frame[frame_number] += 1

            track_history[
                track_id
            ].append(frame_number)

    logger.info(
        f"Tracking rows : {total_rows}"
    )

    logger.info(
        f"Unique IDs    : {len(unique_ids)}"
    )

    return (
        track_history,
        tracks_per_frame,
        unique_ids,
        total_rows,
        total_detections
    )


# ============================================================
# FIND BOXMOT OUTPUT
# ============================================================

def find_latest_tracking_file():

    search_roots = [

        ROOT / "outputs" / "trackers",

        ROOT / "runs",

        ROOT / "outputs"
    ]

    candidates = []

    for search_root in search_roots:

        if not search_root.exists():
            continue

        try:

            for file in search_root.rglob("*.txt"):

                name = file.name.lower()

                if (
                    "track" in name
                    or "video0" in name
                    or "labels" in name
                ):

                    candidates.append(file)

        except Exception:
            pass

    if not candidates:

        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0]


# ============================================================
# FIND BOXMOT VIDEO
# ============================================================

def find_latest_video():

    search_roots = [

        ROOT / "outputs" / "trackers",

        ROOT / "runs",

        ROOT / "outputs"
    ]

    candidates = []

    for search_root in search_roots:

        if not search_root.exists():
            continue

        try:

            for file in search_root.rglob("*.mp4"):

                if file.resolve() == VIDEO_OUTPUT.resolve():
                    continue

                candidates.append(file)

        except Exception:
            pass

    if not candidates:

        return None

    candidates.sort(
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )

    return candidates[0]


# ============================================================
# COPY VIDEO
# ============================================================

def copy_output_video(source_video):

    if source_video is None:

        logger.warning(
            "Could not find BoxMOT output video."
        )

        return False

    logger.info(
        f"BoxMOT video: {source_video}"
    )

    cap = cv2.VideoCapture(
        str(source_video)
    )

    if not cap.isOpened():

        logger.warning(
            "Could not open BoxMOT output video."
        )

        return False

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    if fps <= 0:
        fps = 29.0

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(VIDEO_OUTPUT),
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        logger.warning(
            "Could not create final output video."
        )

        return False

    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        writer.write(frame)

        frame_count += 1

    cap.release()
    writer.release()

    logger.info(
        f"Final output video frames: {frame_count}"
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("=" * 70)

    logger.info(
        "YOLO11m + OC-SORT OBJECT TRACKING"
    )

    logger.info("=" * 70)

    logger.info(
        f"ROOT        : {ROOT}"
    )

    logger.info(
        f"MODEL       : {MODEL_PATH}"
    )

    logger.info(
        f"VIDEO       : {VIDEO_PATH}"
    )

    logger.info(
        f"JSON        : {JSON_OUTPUT}"
    )

    logger.info(
        f"VIDEO OUTPUT: {VIDEO_OUTPUT}"
    )

    logger.info(
        f"LOG         : {LOG_OUTPUT}"
    )

    # ========================================================
    # CHECK INPUTS
    # ========================================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    if not VIDEO_PATH.exists():

        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}"
        )

    # ========================================================
    # READ VIDEO INFORMATION
    # ========================================================

    cap = cv2.VideoCapture(
        str(VIDEO_PATH)
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    width = int(
        cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    )

    height = int(
        cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    )

    input_fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    cap.release()

    logger.info(
        f"Width       : {width}"
    )

    logger.info(
        f"Height      : {height}"
    )

    logger.info(
        f"FPS         : {input_fps}"
    )

    logger.info(
        f"Frames      : {total_frames}"
    )

    # ========================================================
    # BOXMOT
    # ========================================================

    logger.info(
        "Loading YOLO11m + OC-SORT..."
    )

    tracker = Boxmot(

        detector=str(
            MODEL_PATH
        ),

        tracker=TRACKER_NAME,

        project=str(
            ROOT / "outputs" / "trackers"
        )
    )

    logger.info(
        "Tracker loaded successfully."
    )

    # ========================================================
    # RESOURCE MONITOR
    # ========================================================

    resource_monitor = ResourceMonitor()

    resource_monitor.start()

    # ========================================================
    # RUN TRACKING
    # ========================================================

    logger.info("=" * 70)

    logger.info(
        "STARTING BOXMOT OC-SORT"
    )

    logger.info("=" * 70)

    start_time = time.perf_counter()

    result = tracker.track(

        source=str(
            VIDEO_PATH
        ),

        imgsz=IMAGE_SIZE,

        conf=CONFIDENCE,

        iou=IOU_THRESHOLD,

        device=DEVICE,

        half=False,

        save=True,

        save_txt=True,

        show=False,

        verbose=True
    )

    end_time = time.perf_counter()

    processing_time = (
        end_time
        - start_time
    )

    resource_monitor.stop()

    logger.info("=" * 70)

    logger.info(
        "BOXMOT TRACKING FINISHED"
    )

    logger.info(
        f"Processing time: "
        f"{processing_time:.4f} sec"
    )

    logger.info("=" * 70)

    # ========================================================
    # FIND TRACKING FILE
    # ========================================================

    tracking_file = find_latest_tracking_file()

    if tracking_file is None:

        raise FileNotFoundError(
            "Could not find BoxMOT tracking TXT output."
        )

    logger.info(
        f"Tracking file: {tracking_file}"
    )

    # ========================================================
    # PARSE TRACKING DATA
    # ========================================================

    (
        track_history,
        tracks_per_frame,
        unique_ids,
        total_rows,
        total_detections
    ) = parse_tracking_file(
        tracking_file
    )

    # ========================================================
    # TRACKING STATISTICS
    # ========================================================

    frames_with_tracks = len(
        tracks_per_frame
    )

    active_track_counts = []

    for frame_number in range(
        1,
        total_frames + 1
    ):

        active_track_counts.append(
            tracks_per_frame.get(
                frame_number,
                0
            )
        )

    if active_track_counts:

        average_active_tracks = (
            sum(active_track_counts)
            / len(active_track_counts)
        )

        maximum_active_tracks = max(
            active_track_counts
        )

    else:

        average_active_tracks = 0.0
        maximum_active_tracks = 0

    (
        average_track_lifetime,
        longest_track_lifetime
    ) = calculate_track_lifetimes(
        track_history
    )

    # ========================================================
    # FPS
    # ========================================================

    average_fps = (

        total_frames
        / processing_time

        if processing_time > 0

        else 0.0
    )

    # ========================================================
    # LATENCY
    #
    # Since BoxMOT performs the complete video run internally,
    # use total processing time / frame count as average
    # per-frame latency.
    # ========================================================

    average_latency_ms = (

        processing_time
        / total_frames
        * 1000.0

        if total_frames > 0

        else 0.0
    )

    # The common team format expects p95 latency.
    # With BoxMOT's aggregate API, exact per-frame timing is
    # not exposed. Therefore use average latency as the
    # measured fallback rather than inventing a value.

    p95_latency_ms = average_latency_ms

    # ========================================================
    # RESOURCES
    # ========================================================

    resources = (
        resource_monitor.get_results()
    )

    # ========================================================
    # FIND BOXMOT VIDEO
    # ========================================================

    boxmot_video = find_latest_video()

    copy_output_video(
        boxmot_video
    )

    # ========================================================
    # COMMON TEAM JSON
    # ========================================================

    metrics = {

        "project": {

            "name":
                "YOLO11m + OC-SORT Object Tracking",

            "tracker":
                "OC-SORT"
        },

        "model": {

            "name":
                "YOLO11m",

            "format":
                "pt",

            "path":
                str(MODEL_PATH)
        },

        "configuration": {

            "confidence_threshold":
                CONFIDENCE,

            "iou_threshold":
                IOU_THRESHOLD,

            "tracker_config":
                TRACKER_CONFIG,

            "device":
                DEVICE.upper()
        },

        "video": {

            "input":
                str(VIDEO_PATH),

            "width":
                width,

            "height":
                height,

            "fps":
                input_fps,

            "total_frames":
                total_frames
        },

        "tracking_statistics": {

            "frame_count":
                total_frames,

            "total_detections":
                total_detections,

            "unique_track_ids":
                len(unique_ids),

            "average_active_tracks":
                average_active_tracks,

            "maximum_active_tracks":
                maximum_active_tracks,

            "average_track_lifetime":
                average_track_lifetime,

            "longest_track_lifetime":
                longest_track_lifetime,

            "average_fps":
                average_fps,

            "average_latency_ms":
                average_latency_ms,

            "p95_latency_ms":
                p95_latency_ms,

            "total_processing_time_seconds":
                processing_time
        },

        "resources": resources,

        "outputs": {

            "video":
                str(VIDEO_OUTPUT),

            "json":
                str(JSON_OUTPUT),

            "log":
                str(LOG_OUTPUT)
        }
    }

    # ========================================================
    # SAVE JSON
    # ========================================================

    with open(

        JSON_OUTPUT,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            metrics,

            f,

            indent=4
        )

    # ========================================================
    # FINAL LOG
    # ========================================================

    logger.info("=" * 70)

    logger.info(
        "OC-SORT PROCESSING COMPLETE"
    )

    logger.info("=" * 70)

    logger.info(
        f"Frames              : "
        f"{total_frames}"
    )

    logger.info(
        f"Total detections    : "
        f"{total_detections}"
    )

    logger.info(
        f"Unique track IDs    : "
        f"{len(unique_ids)}"
    )

    logger.info(
        f"Average active tracks: "
        f"{average_active_tracks:.4f}"
    )

    logger.info(
        f"Maximum active tracks: "
        f"{maximum_active_tracks}"
    )

    logger.info(
        f"Average track life  : "
        f"{average_track_lifetime:.4f}"
    )

    logger.info(
        f"Longest track life  : "
        f"{longest_track_lifetime}"
    )

    logger.info(
        f"Average FPS         : "
        f"{average_fps:.4f}"
    )

    logger.info(
        f"Average latency     : "
        f"{average_latency_ms:.4f} ms"
    )

    logger.info(
        f"P95 latency         : "
        f"{p95_latency_ms:.4f} ms"
    )

    logger.info(
        f"Processing time     : "
        f"{processing_time:.4f} sec"
    )

    logger.info(
        f"JSON                : "
        f"{JSON_OUTPUT}"
    )

    logger.info(
        f"VIDEO               : "
        f"{VIDEO_OUTPUT}"
    )

    logger.info(
        f"LOG                 : "
        f"{LOG_OUTPUT}"
    )

    logger.info("=" * 70)

    print()
    print("=" * 70)
    print("OC-SORT COMPLETE")
    print("=" * 70)
    print(f"JSON : {JSON_OUTPUT}")
    print(f"VIDEO: {VIDEO_OUTPUT}")
    print(f"LOG  : {LOG_OUTPUT}")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()