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

RTSP_URL = "rtsp://localhost:8554/video"

PROJECT_DIR = Path(__file__).resolve().parents[1]
VIDEO_DIR = PROJECT_DIR / "video"


def find_video():
    videos = list(VIDEO_DIR.glob("*.mp4"))

    if not videos:
        print("No MP4 video found in:")
        print(VIDEO_DIR)
        sys.exit(1)

    print("\nAvailable videos:\n")

    for i, video in enumerate(videos, 1):
        print(f"{i}. {video.name}")

    while True:
        try:
            choice = int(input("\nSelect video number: "))

            if 1 <= choice <= len(videos):
                return videos[choice - 1]

        except ValueError:
            pass

        print("Invalid selection.")


def main():

    if not FFMPEG.exists():
        print("ERROR: FFmpeg 9.0 not found:")
        print(FFMPEG)
        sys.exit(1)

    video = find_video()

    print("\n============================================================")
    print("             VIDEO → RTSP STREAM")
    print("============================================================")
    print(f"Video : {video}")
    print(f"RTSP  : {RTSP_URL}")
    print("============================================================")
    print("\nPress Ctrl+C to stop.\n")

    command = [
        str(FFMPEG),
        "-hide_banner",
        "-re",
        "-i",
        str(video),
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

    try:
        subprocess.run(command)

    except KeyboardInterrupt:
        print("\nStopping video stream...")

    print("\nVideo stream stopped.")


if __name__ == "__main__":
    main()