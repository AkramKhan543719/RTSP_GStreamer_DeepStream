cat > pipelines/gpu_benchmark/export_trt.py <<'PY'
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
ONNX_MODEL = ROOT / "models" / "onnx" / "yolo11m.onnx"
OUTPUT_DIR = ROOT / "outputs" / "gpu_benchmark" / "trt"


def main():
    if not ONNX_MODEL.exists():
        raise FileNotFoundError(f"Model not found: {ONNX_MODEL}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(ONNX_MODEL), task="detect")
    exported = model.export(
        format="engine",
        imgsz=640,
        batch=1,
        device=0,
        half=True,
        workspace=4,
    )

    print(f"TensorRT export result: {exported}")


if __name__ == "__main__":
    main()
PY