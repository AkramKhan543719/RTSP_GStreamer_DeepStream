import subprocess
from pathlib import Path

ROOT = Path.home() / "Documents/interns/Akram/RTSP_GStreamer_DeepStream"
OUT = ROOT / "outputs/deepstream/task02_gstreamer"
OUT.mkdir(parents=True, exist_ok=True)

plugins = [
    "nvinfer",
    "nvstreammux",
    "nvvideoconvert",
    "nvdsosd",
    "nvv4l2decoder",
]

results = []
for plugin in plugins:
    result = subprocess.run(
        ["gst-inspect-1.0", plugin],
        capture_output=True,
        text=True
    )
    status = "PASS" if result.returncode == 0 else "FAIL"
    results.append(f"{status}: {plugin}")
    print(results[-1])

report = OUT / "gstreamer_report.txt"
report.write_text("\n".join(results) + "\n", encoding="utf-8")

if any(line.startswith("FAIL") for line in results):
    raise SystemExit("Some GStreamer plugins are missing. Check the report.")

print(f"\nReport saved: {report}")
