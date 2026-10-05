import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PT_CSV = os.path.join(
    BASE_DIR,
    "experiments",
    "video0_pt_results.csv"
)

ONNX_CSV = os.path.join(
    BASE_DIR,
    "experiments",
    "video0_onnx_results.csv"
)

OUTPUT_CSV = os.path.join(
    BASE_DIR,
    "experiments",
    "video0_pt_vs_onnx_comparison.csv"
)


def read_csv(path):
    with open(path, "r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def to_float(value):
    try:
        return float(value)
    except:
        return 0.0


def to_int(value):
    try:
        return int(float(value))
    except:
        return 0


def main():

    print("=" * 70)
    print("VIDEO0 - PT vs ONNX COMPARISON")
    print("=" * 70)

    if not os.path.exists(PT_CSV):
        print("ERROR: PT result CSV not found:")
        print(PT_CSV)
        return

    if not os.path.exists(ONNX_CSV):
        print("ERROR: ONNX result CSV not found:")
        print(ONNX_CSV)
        return

    pt_rows = read_csv(PT_CSV)
    onnx_rows = read_csv(ONNX_CSV)

    if not pt_rows or not onnx_rows:
        print("ERROR: One or both CSV files are empty.")
        return

    pt = pt_rows[0]
    onnx = onnx_rows[0]

    # ------------------------------------------------------------
    # Extract values
    # ------------------------------------------------------------

    frames_pt = to_int(pt["frames"])
    frames_onnx = to_int(onnx["frames"])

    fps_input_pt = to_float(pt["input_fps"])
    fps_input_onnx = to_float(onnx["input_fps"])

    width_pt = to_int(pt["width"])
    width_onnx = to_int(onnx["width"])

    height_pt = to_int(pt["height"])
    height_onnx = to_int(onnx["height"])

    confidence_pt = to_float(pt["confidence"])
    confidence_onnx = to_float(onnx["confidence"])

    image_size_pt = to_int(pt["image_size"])
    image_size_onnx = to_int(onnx["image_size"])

    processing_time_pt = to_float(pt["processing_time_sec"])
    processing_time_onnx = to_float(onnx["processing_time_sec"])

    processing_fps_pt = to_float(pt["processing_fps"])
    processing_fps_onnx = to_float(onnx["processing_fps"])

    detections_pt = to_int(pt["detections"])
    detections_onnx = to_int(onnx["detections"])

    # ------------------------------------------------------------
    # Differences
    # ------------------------------------------------------------

    time_difference = processing_time_onnx - processing_time_pt

    fps_difference = processing_fps_onnx - processing_fps_pt

    detection_difference = detections_onnx - detections_pt

    if processing_time_pt > 0:
        time_change_percent = (
            (processing_time_onnx - processing_time_pt)
            / processing_time_pt
        ) * 100
    else:
        time_change_percent = 0.0

    if processing_fps_pt > 0:
        fps_change_percent = (
            (processing_fps_onnx - processing_fps_pt)
            / processing_fps_pt
        ) * 100
    else:
        fps_change_percent = 0.0

    if detections_pt > 0:
        detection_change_percent = (
            (detections_onnx - detections_pt)
            / detections_pt
        ) * 100
    else:
        detection_change_percent = 0.0

    # ------------------------------------------------------------
    # Print comparison
    # ------------------------------------------------------------

    print()
    print("INPUT VIDEO")
    print("-" * 70)

    print(f"Video              : {pt['video']}")
    print(f"Frames - PT        : {frames_pt}")
    print(f"Frames - ONNX      : {frames_onnx}")
    print(f"Input FPS - PT     : {fps_input_pt:.3f}")
    print(f"Input FPS - ONNX   : {fps_input_onnx:.3f}")
    print(f"Resolution - PT    : {width_pt}x{height_pt}")
    print(f"Resolution - ONNX  : {width_onnx}x{height_onnx}")

    print()
    print("MODEL CONFIGURATION")
    print("-" * 70)

    print(f"Model              : {pt['model']}")
    print(f"Confidence - PT    : {confidence_pt}")
    print(f"Confidence - ONNX  : {confidence_onnx}")
    print(f"Image Size - PT    : {image_size_pt}")
    print(f"Image Size - ONNX  : {image_size_onnx}")
    print(f"Device - PT        : {pt['device']}")
    print(f"Device - ONNX      : {onnx['device']}")

    print()
    print("PERFORMANCE")
    print("-" * 70)

    print(f"PT processing time   : {processing_time_pt:.3f} sec")
    print(f"ONNX processing time : {processing_time_onnx:.3f} sec")

    print(f"Time difference      : {time_difference:+.3f} sec")
    print(f"Time change          : {time_change_percent:+.2f}%")

    print()

    print(f"PT processing FPS    : {processing_fps_pt:.3f}")
    print(f"ONNX processing FPS  : {processing_fps_onnx:.3f}")

    print(f"FPS difference       : {fps_difference:+.3f}")
    print(f"FPS change           : {fps_change_percent:+.2f}%")

    print()
    print("DETECTIONS")
    print("-" * 70)

    print(f"PT detections        : {detections_pt}")
    print(f"ONNX detections      : {detections_onnx}")

    print(f"Detection difference : {detection_difference:+d}")
    print(f"Detection change     : {detection_change_percent:+.2f}%")

    # ------------------------------------------------------------
    # Output file size
    # ------------------------------------------------------------

    pt_output = pt["output"]
    onnx_output = onnx["output"]

    pt_size = 0
    onnx_size = 0

    if os.path.exists(pt_output):
        pt_size = os.path.getsize(pt_output)

    if os.path.exists(onnx_output):
        onnx_size = os.path.getsize(onnx_output)

    print()
    print("OUTPUT FILES")
    print("-" * 70)

    print(f"PT output   : {pt_output}")
    print(f"ONNX output : {onnx_output}")

    if pt_size > 0:
        print(f"PT size     : {pt_size / (1024 * 1024):.2f} MB")
    else:
        print("PT size     : Not found")

    if onnx_size > 0:
        print(f"ONNX size   : {onnx_size / (1024 * 1024):.2f} MB")
    else:
        print("ONNX size   : Not found")

    if pt_size > 0 and onnx_size > 0:
        size_difference = onnx_size - pt_size
        size_percent = (size_difference / pt_size) * 100

        print(
            f"Size difference : "
            f"{size_difference / (1024 * 1024):+.2f} MB"
        )

        print(f"Size change     : {size_percent:+.2f}%")

    # ------------------------------------------------------------
    # Basic consistency checks
    # ------------------------------------------------------------

    print()
    print("CONSISTENCY CHECKS")
    print("-" * 70)

    print(
        "Frame count       : "
        + ("MATCH" if frames_pt == frames_onnx else "DIFFERENT")
    )

    print(
        "Input FPS         : "
        + ("MATCH" if abs(fps_input_pt - fps_input_onnx) < 0.01 else "DIFFERENT")
    )

    print(
        "Resolution        : "
        + (
            "MATCH"
            if width_pt == width_onnx and height_pt == height_onnx
            else "DIFFERENT"
        )
    )

    print(
        "Confidence        : "
        + (
            "MATCH"
            if abs(confidence_pt - confidence_onnx) < 0.0001
            else "DIFFERENT"
        )
    )

    print(
        "Image size        : "
        + (
            "MATCH"
            if image_size_pt == image_size_onnx
            else "DIFFERENT"
        )
    )

    # ------------------------------------------------------------
    # Save comparison CSV
    # ------------------------------------------------------------

    fields = [
        "video",
        "model",
        "frames_pt",
        "frames_onnx",
        "input_fps",
        "resolution",
        "confidence",
        "image_size",
        "pt_device",
        "onnx_device",
        "pt_processing_time_sec",
        "onnx_processing_time_sec",
        "time_difference_sec",
        "time_change_percent",
        "pt_processing_fps",
        "onnx_processing_fps",
        "fps_difference",
        "fps_change_percent",
        "pt_detections",
        "onnx_detections",
        "detection_difference",
        "detection_change_percent",
        "pt_output",
        "onnx_output",
    ]

    row = {
        "video": pt["video"],
        "model": pt["model"],
        "frames_pt": frames_pt,
        "frames_onnx": frames_onnx,
        "input_fps": fps_input_pt,
        "resolution": f"{width_pt}x{height_pt}",
        "confidence": confidence_pt,
        "image_size": image_size_pt,
        "pt_device": pt["device"],
        "onnx_device": onnx["device"],
        "pt_processing_time_sec": round(processing_time_pt, 3),
        "onnx_processing_time_sec": round(processing_time_onnx, 3),
        "time_difference_sec": round(time_difference, 3),
        "time_change_percent": round(time_change_percent, 3),
        "pt_processing_fps": round(processing_fps_pt, 3),
        "onnx_processing_fps": round(processing_fps_onnx, 3),
        "fps_difference": round(fps_difference, 3),
        "fps_change_percent": round(fps_change_percent, 3),
        "pt_detections": detections_pt,
        "onnx_detections": detections_onnx,
        "detection_difference": detection_difference,
        "detection_change_percent": round(detection_change_percent, 3),
        "pt_output": pt_output,
        "onnx_output": onnx_output,
    }

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()
        writer.writerow(row)

    print()
    print("=" * 70)
    print("COMPARISON COMPLETE")
    print("=" * 70)
    print(f"Comparison CSV:")
    print(OUTPUT_CSV)
    print("=" * 70)


if __name__ == "__main__":
    main()