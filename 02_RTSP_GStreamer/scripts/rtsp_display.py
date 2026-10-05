import subprocess
import shutil
import sys


RTSP_URL = "rtsp://localhost:8554/video"


def main():

    gst = shutil.which("gst-launch-1.0")

    if not gst:
        print("ERROR: gst-launch-1.0 was not found in PATH.")
        sys.exit(1)

    print("\n============================================================")
    print("                 RTSP DISPLAY")
    print("============================================================")
    print(f"RTSP URL : {RTSP_URL}")
    print("============================================================")
    print("\nStarting GStreamer...")
    print("Press Ctrl+C to stop.\n")

    command = [
        gst,
        "-v",
        "rtspsrc",
        f"location={RTSP_URL}",
        "latency=100",
        "protocols=tcp",
        "!",
        "rtph264depay",
        "!",
        "h264parse",
        "!",
        "avdec_h264",
        "!",
        "videoconvert",
        "!",
        "autovideosink",
    ]

    try:
        subprocess.run(command)

    except KeyboardInterrupt:
        print("\nStopping GStreamer...")

    print("\nDisplay stopped.")


if __name__ == "__main__":
    main()