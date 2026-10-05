import subprocess
import sys
import time
from datetime import datetime
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

PROJECT_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = PROJECT_DIR / "outputs" / "webcam"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():

    if not FFMPEG.exists():
        print("ERROR: FFmpeg 9.0 was not found.")
        print(FFMPEG)
        sys.exit(1)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_file = OUTPUT_DIR / f"webcam_{timestamp}.mp4"

    print("\n============================================================")
    print("              WEBCAM RTSP + RECORDING")
    print("============================================================")
    print(f"Camera     : {CAMERA}")
    print(f"Resolution : {WIDTH}x{HEIGHT}")
    print(f"FPS        : {FPS}")
    print(f"RTSP       : {RTSP_URL}")
    print(f"Recording  : {output_file}")
    print("============================================================")

    # --------------------------------------------------------
    # PROCESS 1
    # Webcam → MediaMTX
    # --------------------------------------------------------

    webcam_command = [
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

    webcam_process = None
    recorder_process = None

    try:

        print("\nStarting webcam → RTSP...")

        webcam_process = subprocess.Popen(webcam_command)

        # Give MediaMTX a moment to publish the stream
        time.sleep(2)

        # ----------------------------------------------------
        # PROCESS 2
        # RTSP → MP4
        # ----------------------------------------------------

        print("Starting RTSP recording...")

        recorder_command = [
            str(FFMPEG),

            "-hide_banner",

            "-rtsp_transport",
            "tcp",

            "-i",
            RTSP_URL,

            "-an",

            "-c:v",
            "copy",

            "-movflags",
            "+faststart",

            "-y",

            str(output_file),
        ]

        recorder_process = subprocess.Popen(recorder_command)

        print("\n============================================================")
        print("WEBCAM IS LIVE")
        print("============================================================")
        print(f"RTSP      : {RTSP_URL}")
        print(f"Recording : {output_file}")
        print("\nPress Ctrl+C to STOP.")
        print("The MP4 will be finalized automatically.")
        print("============================================================\n")

        recorder_process.wait()

    except KeyboardInterrupt:

        print("\n\nCtrl+C received.")
        print("Stopping recording...")

        if recorder_process:

            recorder_process.terminate()

            try:
                recorder_process.wait(timeout=10)

            except subprocess.TimeoutExpired:

                print("Recorder did not stop normally. Killing it.")

                recorder_process.kill()

                recorder_process.wait()

        print("Stopping webcam...")

        if webcam_process:

            webcam_process.terminate()

            try:
                webcam_process.wait(timeout=5)

            except subprocess.TimeoutExpired:

                webcam_process.kill()

                webcam_process.wait()

    finally:

        # Make sure both processes are gone
        if recorder_process and recorder_process.poll() is None:
            recorder_process.kill()

        if webcam_process and webcam_process.poll() is None:
            webcam_process.kill()

    print("\n============================================================")
    print("             RECORDING SESSION FINISHED")
    print("============================================================")

    if output_file.exists():

        size_mb = output_file.stat().st_size / (1024 * 1024)

        if size_mb > 0:

            print(f"Output : {output_file}")
            print(f"Size   : {size_mb:.2f} MB")

        else:

            print("WARNING: Recording file is empty.")

    else:

        print("WARNING: Recording file was not created.")

    print("============================================================")


if __name__ == "__main__":
    main()