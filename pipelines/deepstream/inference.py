from common import ENGINE, MODEL_ONNX, VIDEO, task_output, require_file
import subprocess
import sys

out = task_output("task04_inference")
report = out / "inference_preflight.txt"

checks = []

for label, path in [
    ("ONNX model", MODEL_ONNX),
    ("TensorRT engine", ENGINE),
    ("Input video", VIDEO),
]:
    try:
        require_file(path)
        message = f"PASS: {label}: {path} ({path.stat().st_size:,} bytes)"
    except FileNotFoundError as exc:
        message = f"FAIL: {exc}"

    print(message)
    checks.append(message)

try:
    result = subprocess.run(
        ["gst-inspect-1.0", "nvinfer"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    plugin_ok = result.returncode == 0
except (OSError, subprocess.TimeoutExpired):
    plugin_ok = False

checks.append(
    "PASS: DeepStream nvinfer plugin available"
    if plugin_ok
    else "FAIL: DeepStream nvinfer plugin unavailable"
)

checks.append(
    "PENDING: Verify a YOLO11-compatible DeepStream output parser. "
    "The existence of a TensorRT engine does not establish parser compatibility."
)

report.write_text("\n".join(checks) + "\n", encoding="utf-8")

print(f"\nReport: {report}")

if any(line.startswith("FAIL:") for line in checks):
    sys.exit(1)

print("Preflight completed. This does not yet execute YOLO inference.")
