from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "pytorch" / "yolo11m.pt"
ONNX_DIR = ROOT / "models" / "onnx"

ONNX_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 70)
    print("YOLO11m -> ONNX EXPORT")
    print("=" * 70)

    print(f"PyTorch model : {MODEL_PATH}")
    print(f"ONNX directory: {ONNX_DIR}")

    if not MODEL_PATH.exists():
        print("ERROR: Model not found.")
        return

    model = YOLO(str(MODEL_PATH))

    print()
    print("Exporting YOLO11m to ONNX...")

    exported = model.export(
        format="onnx",
        imgsz=640,
        opset=17,
        simplify=True,
        dynamic=False,
    )

    exported = Path(exported)

    destination = ONNX_DIR / "yolo11m.onnx"

    if exported.resolve() != destination.resolve():
        if destination.exists():
            destination.unlink()

        exported.replace(destination)

    print()
    print("=" * 70)
    print("ONNX EXPORT COMPLETE")
    print("=" * 70)
    print(f"ONNX model: {destination}")
    print(f"Size      : {destination.stat().st_size / (1024 * 1024):.2f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()