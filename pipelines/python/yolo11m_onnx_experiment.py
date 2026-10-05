import csv
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "onnx" / "yolo11m.onnx"
VIDEO_DIR = ROOT / "test_videos"

OUTPUT_DIR = ROOT / "outputs" / "yolo11m_onnx"
RESULTS_DIR = ROOT / "experiments"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_CSV = RESULTS_DIR / "yolo11m_onnx_results.csv"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = 640
CONFIDENCE = 0.25
IOU_THRESHOLD = 0.45


# ============================================================
# COCO CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase",
    "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table",
    "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush"
]


# ============================================================
# LOAD ONNX MODEL
# ============================================================

def load_model():
    print("Loading YOLO11m ONNX...")

    providers = ort.get_available_providers()

    print("ONNX Runtime providers:")
    for provider in providers:
        print(f"  - {provider}")

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"]
    )

    input_info = session.get_inputs()[0]

    print()
    print("ONNX model loaded successfully.")
    print(f"Input name : {input_info.name}")
    print(f"Input shape: {input_info.shape}")
    print(f"Input type : {input_info.type}")
    print()

    return session


# ============================================================
# LETTERBOX
# ============================================================

def letterbox(image, new_size=640):
    height, width = image.shape[:2]

    scale = min(new_size / width, new_size / height)

    new_width = int(round(width * scale))
    new_height = int(round(height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full(
        (new_size, new_size, 3),
        114,
        dtype=np.uint8
    )

    pad_x = (new_size - new_width) // 2
    pad_y = (new_size - new_height) // 2

    canvas[
        pad_y:pad_y + new_height,
        pad_x:pad_x + new_width
    ] = resized

    return canvas, scale, pad_x, pad_y


# ============================================================
# PREPROCESS
# ============================================================

def preprocess(frame):
    image, scale, pad_x, pad_y = letterbox(
        frame,
        IMG_SIZE
    )

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    image = np.ascontiguousarray(image)

    return image, scale, pad_x, pad_y


# ============================================================
# NMS
# ============================================================

def nms(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (
        np.maximum(0, x2 - x1)
        * np.maximum(0, y2 - y1)
    )

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:
        i = order[0]

        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])

        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(
            0,
            xx2 - xx1
        )

        height = np.maximum(
            0,
            yy2 - yy1
        )

        intersection = width * height

        union = (
            areas[i]
            + areas[order[1:]]
            - intersection
        )

        iou = np.zeros_like(intersection)

        valid = union > 0

        iou[valid] = (
            intersection[valid]
            / union[valid]
        )

        remaining = np.where(
            iou <= iou_threshold
        )[0]

        order = order[
            remaining + 1
        ]

    return keep


# ============================================================
# YOLO11 POSTPROCESSING
# ============================================================

def postprocess(
    output,
    original_shape,
    scale,
    pad_x,
    pad_y
):
    """
    Handles YOLO11 ONNX output.

    Typical YOLO11 output:

        (1, 84, 8400)

    or

        (1, 8400, 84)

    First 4 values:
        x, y, w, h

    Remaining values:
        class confidence scores
    """

    output = np.squeeze(output)

    if output.ndim != 2:
        return []

    # Convert to (number_of_predictions, attributes)
    if output.shape[0] < output.shape[1]:
        output = output.T

    boxes = []
    scores = []
    class_ids = []

    for detection in output:
        if len(detection) < 5:
            continue

        x = float(detection[0])
        y = float(detection[1])
        w = float(detection[2])
        h = float(detection[3])

        class_scores = detection[4:]

        class_id = int(
            np.argmax(class_scores)
        )

        confidence = float(
            class_scores[class_id]
        )

        if confidence < CONFIDENCE:
            continue

        # Convert xywh -> xyxy
        x1 = x - w / 2
        y1 = y - h / 2
        x2 = x + w / 2
        y2 = y + h / 2

        # Undo letterbox
        x1 = (x1 - pad_x) / scale
        y1 = (y1 - pad_y) / scale
        x2 = (x2 - pad_x) / scale
        y2 = (y2 - pad_y) / scale

        original_height, original_width = original_shape[:2]

        x1 = max(
            0,
            min(x1, original_width - 1)
        )

        y1 = max(
            0,
            min(y1, original_height - 1)
        )

        x2 = max(
            0,
            min(x2, original_width - 1)
        )

        y2 = max(
            0,
            min(y2, original_height - 1)
        )

        boxes.append(
            [x1, y1, x2, y2]
        )

        scores.append(confidence)
        class_ids.append(class_id)

    keep = nms(
        boxes,
        scores,
        IOU_THRESHOLD
    )

    detections = []

    for index in keep:
        detections.append(
            {
                "box": boxes[index],
                "confidence": scores[index],
                "class_id": class_ids[index]
            }
        )

    return detections


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(frame, detections):
    for detection in detections:
        x1, y1, x2, y2 = map(
            int,
            detection["box"]
        )

        confidence = detection["confidence"]
        class_id = detection["class_id"]

        if class_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[class_id]
        else:
            class_name = str(class_id)

        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2
        )

    return frame


# ============================================================
# PROCESS ONE VIDEO
# ============================================================

def process_video(session, video_path):
    print()
    print("=" * 70)
    print(f"PROCESSING: {video_path.name}")
    print("=" * 70)

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():
        print(
            f"ERROR: Could not open {video_path}"
        )
        return None

    input_fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    output_path = (
        OUTPUT_DIR
        / f"{video_path.stem}_yolo11m_onnx.mp4"
    )

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        input_fps,
        (width, height)
    )

    print(f"Input FPS      : {input_fps:.2f}")
    print(
        f"Resolution     : "
        f"{width}x{height}"
    )
    print(f"Total frames   : {total_frames}")
    print(f"Confidence     : {CONFIDENCE}")
    print(f"Image size     : {IMG_SIZE}")
    print("Device         : CPU")
    print(f"Output         : {output_path}")

    frame_count = 0
    detection_count = 0

    start_time = time.perf_counter()

    input_name = session.get_inputs()[0].name

    while True:
        success, frame = capture.read()

        if not success:
            break

        input_tensor, scale, pad_x, pad_y = preprocess(
            frame
        )

        outputs = session.run(
            None,
            {
                input_name: input_tensor
            }
        )

        detections = postprocess(
            outputs[0],
            frame.shape,
            scale,
            pad_x,
            pad_y
        )

        detection_count += len(
            detections
        )

        frame = draw_detections(
            frame,
            detections
        )

        writer.write(frame)

        frame_count += 1

        if (
            frame_count % 25 == 0
            or frame_count == total_frames
        ):
            elapsed = (
                time.perf_counter()
                - start_time
            )

            processing_fps = (
                frame_count / elapsed
                if elapsed > 0
                else 0
            )

            percentage = (
                frame_count
                / total_frames
                * 100
                if total_frames > 0
                else 0
            )

            print(
                f"\rFrames: "
                f"{frame_count}/{total_frames} "
                f"({percentage:.1f}%) | "
                f"FPS: {processing_fps:.2f}",
                end="",
                flush=True
            )

    capture.release()
    writer.release()

    elapsed_time = (
        time.perf_counter()
        - start_time
    )

    processing_fps = (
        frame_count / elapsed_time
        if elapsed_time > 0
        else 0
    )

    print()
    print()
    print("-" * 70)
    print("PROCESSING COMPLETE")
    print("-" * 70)
    print(
        f"Frames processed : {frame_count}"
    )
    print(
        f"Detections       : {detection_count}"
    )
    print(
        f"Processing time  : "
        f"{elapsed_time:.3f} sec"
    )
    print(
        f"Processing FPS   : "
        f"{processing_fps:.3f}"
    )
    print(
        f"Output           : "
        f"{output_path}"
    )
    print("-" * 70)

    return {
        "video": video_path.name,
        "frames": frame_count,
        "processing_time_sec": round(
            elapsed_time,
            3
        ),
        "processing_fps": round(
            processing_fps,
            3
        ),
        "detections": detection_count,
        "output": str(output_path)
    }


# ============================================================
# SAVE CSV
# ============================================================

def save_results(results):
    with open(
        RESULTS_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow(
            [
                "pipeline",
                "model",
                "video",
                "frames",
                "processing_time_sec",
                "processing_fps",
                "detections"
            ]
        )

        for result in results:
            writer.writerow(
                [
                    "ONNX_CPU",
                    "YOLO11m",
                    result["video"],
                    result["frames"],
                    result["processing_time_sec"],
                    result["processing_fps"],
                    result["detections"]
                ]
            )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("YOLO11m ONNX CPU EXPERIMENT")
    print("=" * 70)

    print(f"Model : {MODEL_PATH}")
    print(f"Videos: {VIDEO_DIR}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    if not MODEL_PATH.exists():
        print("ERROR: ONNX model not found.")
        print(MODEL_PATH)
        return

    if not VIDEO_DIR.exists():
        print("ERROR: test_videos folder not found.")
        print(VIDEO_DIR)
        return

    videos = sorted(
        VIDEO_DIR.glob("*.mp4")
    )

    if not videos:
        print("ERROR: No MP4 videos found.")
        return

    print(
        f"Videos found: {len(videos)}"
    )

    for video in videos:
        print(f"  - {video.name}")

    print()

    session = load_model()

    results = []

    total_start = time.perf_counter()

    for video_path in videos:

        result = process_video(
            session,
            video_path
        )

        if result is not None:
            results.append(result)

    total_time = (
        time.perf_counter()
        - total_start
    )

    save_results(results)

    print()
    print("=" * 70)
    print("ALL ONNX VIDEOS PROCESSED")
    print("=" * 70)

    print(
        f"Videos processed : {len(results)}"
    )

    print(
        f"Total time       : "
        f"{total_time:.2f} sec"
    )

    print(
        f"Results CSV      : "
        f"{RESULTS_CSV}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()