import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

FFMPEG = Path(
    r"C:\Users\PathanMohammedAkramK\Downloads\ffmpeg-9.0-essentials_build"
    r"\ffmpeg-9.0-essentials_build\bin\ffmpeg.exe"
)

CAMERA = "HP HD Camera"

WIDTH = 640
HEIGHT = 480
FPS = 30

RTSP_URL = "rtsp://localhost:8554/webcam"


def main():

    if not FFMPEG.exists():
        print("ERROR: FFmpeg 9.0 not found:")
        print(FFMPEG)
        sys.exit(1)

    print("\n============================================================")
    print("                 WEBCAM → RTSP")
    print("============================================================")
    print(f"Camera     : {CAMERA}")
    print(f"Resolution : {WIDTH}x{HEIGHT}")
    print(f"FPS        : {FPS}")
    print(f"RTSP       : {RTSP_URL}")
    print("============================================================")
    print("\nPress Ctrl+C to stop the webcam.\n")

    command = [
        str(FFMPEG),
        "-hide_banner",

        "-f",
        "dshow",

        "-video_size",
        f"{WIDTH}x{HEIGHT}",

        "-framerate",
        str(FPS),

        "-i",
        f"video={CAMERA}",

        "-an",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-tune",
        "zerolatency",

        "-pix_fmt",
        "yuv420p",

        "-f",
        "rtsp",

        "-rtsp_transport",
        "tcp",

        RTSP_URL,
    ]

    process = None

    try:
        process = subprocess.Popen(command)
        process.wait()

    except KeyboardInterrupt:

        print("\nStopping webcam...")

        if process:
            process.terminate()

            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()

    print("\nWebcam RTSP stopped.")


if __name__ == "__main__":
    main()