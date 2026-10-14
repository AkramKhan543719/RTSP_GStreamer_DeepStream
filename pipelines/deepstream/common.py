from pathlib import Path
import os

PROJECT_ROOT = Path(
    os.environ.get(
        "RTSP_DS_PROJECT",
        Path.home() / "Documents/interns/Akram/RTSP_GStreamer_DeepStream"
    )
).expanduser().resolve()

OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "deepstream"
TASK_ROOT = PROJECT_ROOT / "tasks" / "deepstream"
PIPELINE_ROOT = PROJECT_ROOT / "pipelines" / "deepstream"

MODEL_PT = PROJECT_ROOT / "models" / "pytorch" / "yolo11m.pt"
MODEL_ONNX = PROJECT_ROOT / "models" / "onnx" / "yolo11m.onnx"
ENGINE = PROJECT_ROOT / "outputs" / "gpu_benchmark" / "trt" / "yolo11m_fp16.engine"
VIDEO = PROJECT_ROOT / "test_videos" / "Video0.avi"


def task_output(task_name: str) -> Path:
    path = OUTPUT_ROOT / task_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def require_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty file: {path}")
