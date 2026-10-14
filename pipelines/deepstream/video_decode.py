import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "Documents/interns/Akram/RTSP_GStreamer_DeepStream"
VIDEO = ROOT / "test_videos/Video0.avi"
OUT = ROOT / "outputs/deepstream/task03_video_decode"
OUT.mkdir(parents=True, exist_ok=True)

LOG = OUT / "decode.log"

if not VIDEO.is_file():
    sys.exit(f"Video not found: {VIDEO}")

cmd = [
    "gst-launch-1.0", "-e",
    "filesrc", f"location={VIDEO}",
    "!", "decodebin",
    "!", "videoconvert",
    "!", "fakesink",
]

print(f"Testing video: {VIDEO}")
print("Running GStreamer decode pipeline...")

with LOG.open("w", encoding="utf-8") as file:
    result = subprocess.run(
        cmd,
        stdout=file,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=300,
    )

if result.returncode != 0:
    print(f"FAILED. Check log: {LOG}")
    sys.exit(result.returncode)

(OUT / "result.txt").write_text(
    f"Status: PASS\nVideo: {VIDEO}\n"
    "Pipeline: filesrc -> decodebin -> videoconvert -> fakesink\n",
    encoding="utf-8",
)

print("Status: PASS")
print(f"Log: {LOG}")
print(f"Result: {OUT / 'result.txt'}")
