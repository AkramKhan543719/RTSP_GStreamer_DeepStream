import platform
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path.home() / "Documents/interns/Akram/RTSP_GStreamer_DeepStream"
OUTPUT = PROJECT / "outputs" / "deepstream" / "task01_environment"
OUTPUT.mkdir(parents=True, exist_ok=True)

report = OUTPUT / "environment_report.txt"
lines = [
    "DEEPSTREAM ENVIRONMENT REPORT",
    f"Python: {sys.version}",
    f"Platform: {platform.platform()}",
    "",
]

checks = [
    ("NVIDIA GPU", ["nvidia-smi"]),
    ("DeepStream", ["deepstream-app", "--version-all"]),
    ("GStreamer", ["gst-inspect-1.0", "--version"]),
]

for label, command in checks:
    lines.extend([f"========== {label} =========="])
    executable = shutil.which(command[0])

    if executable is None:
        lines.append("FAIL: command not found")
        continue

    try:
        result = subprocess.run(
            [executable, *command[1:]],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout + result.stderr
        lines.append(output.strip())
        lines.append(f"Exit code: {result.returncode}")
    except Exception as exc:
        lines.append(f"ERROR: {exc}")

trtexec = Path("/usr/src/tensorrt/bin/trtexec")
lines.extend(["", "========== TENSORRT =========="])

if trtexec.exists():
    try:
        result = subprocess.run(
            [str(trtexec), "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        lines.append((result.stdout + result.stderr).strip())
        lines.append(f"Exit code: {result.returncode}")
    except Exception as exc:
        lines.append(f"ERROR: {exc}")
else:
    lines.append("FAIL: trtexec not found")

report.write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
print(f"\nReport saved: {report}")
