from pathlib import Path
import cv2
import json


ROOT = Path(__file__).resolve().parents[2]
VIDEO_DIR = ROOT / "test_videos"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def inspect_video(video_path):

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        return {
            "video": video_path.name,
            "error": "Could not open video"
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc_value = int(cap.get(cv2.CAP_PROP_FOURCC))

    fourcc = "".join(
        [
            chr(fourcc_value & 0xFF),
            chr((fourcc_value >> 8) & 0xFF),
            chr((fourcc_value >> 16) & 0xFF),
            chr((fourcc_value >> 24) & 0xFF),
        ]
    )

    duration = frames / fps if fps > 0 else 0

    cap.release()

    return {
        "video": video_path.name,
        "width": width,
        "height": height,
        "resolution": f"{width}x{height}",
        "fps": round(fps, 3),
        "frames": frames,
        "duration_seconds": round(duration, 3),
        "fourcc": fourcc,
        "size_mb": round(video_path.stat().st_size / (1024 * 1024), 3),
    }


def main():

    print("=" * 70)
    print("VIDEO INSPECTION")
    print("=" * 70)

    videos = sorted(VIDEO_DIR.glob("*.mp4"))

    results = []

    for video in videos:

        info = inspect_video(video)

        results.append(info)

        print("\n" + "-" * 70)

        for key, value in info.items():
            print(f"{key:20}: {value}")

    output_file = OUTPUT_DIR / "video_info.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print("\n" + "=" * 70)
    print(f"Saved: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()