import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODELS = {
    "pt": ROOT / "models/pytorch/yolo11m.pt",
    "onnx": ROOT / "models/onnx/yolo11m.onnx",
    "trt": ROOT / "outputs/gpu_benchmark/trt/yolo11m_fp16.engine",
}

OUTPUT_ROOT = ROOT / "outputs/gpu_benchmark"
REPORT = ROOT / "experiments/gpu_pt_onnx_trt_comparison.csv"


def benchmark(name, model_path, video_path, imgsz, conf, device, pt_half):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"\n{'=' * 60}\nBACKEND: {name.upper()}\nMODEL: {model_path}\n{'=' * 60}")

    out_dir = OUTPUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_video = out_dir / f"{video_path.stem}_{name}.mp4"

    model = YOLO(str(model_path), task="detect")
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        source_fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create output video: {output_video}")

    frames = 0
    detections = 0
    inference_seconds = 0.0
    start_total = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if device == "cuda":
                torch.cuda.synchronize()

            start_inference = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=imgsz,
                conf=conf,
                device=device,
                half=(pt_half if name == "pt" else name == "trt"),
                verbose=False,
            )

            if device == "cuda":
                torch.cuda.synchronize()

            inference_seconds += time.perf_counter() - start_inference

            result = results[0]
            if result.boxes is not None:
                detections += len(result.boxes)

            writer.write(result.plot())
            frames += 1

            if frames % 50 == 0:
                print(f"Processed {frames} frames...")

    finally:
        cap.release()
        writer.release()

    total_seconds = time.perf_counter() - start_total

    row = {
        "backend": name,
        "frames": frames,
        "detections": detections,
        "inference_seconds": round(inference_seconds, 4),
        "average_inference_ms": round(
            inference_seconds * 1000 / max(frames, 1), 3
        ),
        "inference_fps": round(frames / max(inference_seconds, 1e-9), 2),
        "total_seconds": round(total_seconds, 4),
        "end_to_end_fps": round(frames / max(total_seconds, 1e-9), 2),
        "output_video": str(output_video.relative_to(ROOT)),
    }

    with (out_dir / "results.json").open("w") as f:
        json.dump(row, f, indent=2)

    print(json.dumps(row, indent=2))
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video",
        default=str(ROOT / "test_videos/Video0.avi"),
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--pt-half", action="store_true")
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["pt", "onnx", "trt"],
        default=["pt", "onnx", "trt"],
    )
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in PyTorch.")

    device = "cpu" if args.device == "cpu" else "cuda"
    rows = []

    for name in args.backends:
        rows.append(
            benchmark(
                name,
                MODELS[name],
                video_path,
                args.imgsz,
                args.conf,
                device,
                args.pt_half,
            )
        )

    REPORT.parent.mkdir(parents=True, exist_ok=True)
    with REPORT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nComparison CSV saved to: {REPORT}")


if __name__ == "__main__":
    main()
