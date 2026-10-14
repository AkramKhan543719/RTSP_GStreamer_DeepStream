import shutil
import subprocess
import sys

from common import ENGINE, task_output, require_file

out = task_output("task06_performance")
log = out / "tensorrt_benchmark.log"
timings = out / "tensorrt_times.json"

try:
    require_file(ENGINE)
except FileNotFoundError as exc:
    sys.exit(str(exc))

trtexec = "/usr/src/tensorrt/bin/trtexec"

if not shutil.which(trtexec):
    sys.exit(f"TensorRT executable not found: {trtexec}")

command = [
    trtexec,
    f"--loadEngine={ENGINE}",
    "--warmUp=500",
    "--duration=10",
    f"--exportTimes={timings}",
]

print("Running TensorRT engine benchmark:")
