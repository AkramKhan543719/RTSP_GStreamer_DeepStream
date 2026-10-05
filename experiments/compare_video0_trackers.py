from pathlib import Path
import csv
import cv2


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_DIR = ROOT / "outputs" / "trackers"
RESULTS_DIR = ROOT / "experiments"

VIDEO = ROOT / "test_videos" / "video0.avi"

OUTPUT_CSV = RESULTS_DIR / "video0_tracker_comparison.csv"


TRACKERS = {
    "ByteTrack": {
        "time": 378.788,
        "fps": 2.635,
        "tracks": OUTPUT_DIR / "track" / "video0" / "tracks.txt",
        "video": OUTPUT_DIR / "track" / "video0" / "tracks.mp4",
    },

    "OC-SORT": {
        "time": 379.657,
        "fps": 2.629,
        "tracks": OUTPUT_DIR / "ocsort" / "track" / "video0" / "tracks.txt",
        "video": OUTPUT_DIR / "ocsort" / "track" / "video0" / "tracks.mp4",
    },

    "BoT-SORT": {
        "time": 1101.219,
        "fps": 0.906,
        "tracks": OUTPUT_DIR / "botsort" / "track" / "video0" / "tracks.txt",
        "video": OUTPUT_DIR / "botsort" / "track" / "video0" / "tracks.mp4",
    },

    "Deep SORT": {
        "time": 1051.395,
        "fps": 0.949,
        "tracks": OUTPUT_DIR / "deepsort" / "video0_deepsort_tracks.txt",
        "video": OUTPUT_DIR / "deepsort" / "video0_deepsort.mp4",
    },
}


def count_track_rows(path):
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def analyze_track_file(path):
    rows = 0
    ids = set()
    frames = set()
    id_frame_pairs = set()

    with open(path, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            parts = line.split(",")

            if len(parts) < 2:
                continue

            try:
                frame_id = int(float(parts[0]))
                track_id = int(float(parts[1]))
            except ValueError:
                continue

            rows += 1
            ids.add(track_id)
            frames.add(frame_id)
            id_frame_pairs.add((track_id, frame_id))

    return {
        "track_rows": rows,
        "unique_ids": len(ids),
        "frames_with_tracks": len(frames),
        "id_frame_pairs": len(id_frame_pairs),
    }


def get_video_info(path):

    if not path.exists():
        return {
            "size_bytes": 0,
            "width": "",
            "height": "",
            "fps": "",
            "frames": "",
        }

    cap = cv2.VideoCapture(str(path))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()

    return {
        "size_bytes": path.stat().st_size,
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames,
    }


def main():

    # ------------------------------------------------------------
    # Input video information
    # ------------------------------------------------------------

    cap = cv2.VideoCapture(str(VIDEO))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open {VIDEO}")

    input_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    # ------------------------------------------------------------
    # Build rows
    # ------------------------------------------------------------

    rows = []

    for name, info in TRACKERS.items():

        track_file = info["tracks"]
        output_video = info["video"]

        if not track_file.exists():
            print(f"WARNING: missing track file: {track_file}")
            continue

        analysis = analyze_track_file(track_file)
        video_info = get_video_info(output_video)

        track_rows = analysis["track_rows"]
        unique_ids = analysis["unique_ids"]

        avg_tracks_per_frame = (
            track_rows / input_frames
            if input_frames
            else 0
        )

        ids_per_100_frames = (
            unique_ids / input_frames * 100
            if input_frames
            else 0
        )

        rows.append({
            "tracker": name,
            "video": "video0.avi",
            "frames": input_frames,
            "input_fps": round(input_fps, 3),
            "resolution": f"{input_width}x{input_height}",
            "detections": 13053,
            "track_rows": track_rows,
            "unique_ids": unique_ids,
            "frames_with_tracks": analysis["frames_with_tracks"],
            "avg_tracks_per_frame": round(
                avg_tracks_per_frame, 4
            ),
            "ids_per_100_frames": round(
                ids_per_100_frames, 4
            ),
            "processing_time_sec": info["time"],
            "processing_fps": info["fps"],
            "output_size_bytes": video_info["size_bytes"],
            "output_size_mb": round(
                video_info["size_bytes"] / (1024 * 1024),
                3
            ),
            "output_width": video_info["width"],
            "output_height": video_info["height"],
            "output_fps": round(video_info["fps"], 3)
            if video_info["fps"] != ""
            else "",
            "output_frames": video_info["frames"],
        })

    # ------------------------------------------------------------
    # Write CSV
    # ------------------------------------------------------------

    fields = [
        "tracker",
        "video",
        "frames",
        "input_fps",
        "resolution",
        "detections",
        "track_rows",
        "unique_ids",
        "frames_with_tracks",
        "avg_tracks_per_frame",
        "ids_per_100_frames",
        "processing_time_sec",
        "processing_fps",
        "output_size_bytes",
        "output_size_mb",
        "output_width",
        "output_height",
        "output_fps",
        "output_frames",
    ]

    with open(
        OUTPUT_CSV,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)

    # ------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------

    print()
    print("=" * 100)
    print("VIDEO0 TRACKER COMPARISON COMPLETE")
    print("=" * 100)

    print(f"Input video : {VIDEO}")
    print(f"Frames      : {input_frames}")
    print(f"Input FPS   : {input_fps:.3f}")
    print(f"Resolution  : {input_width}x{input_height}")
    print()

    print(
        f"{'Tracker':<15}"
        f"{'Tracks':>10}"
        f"{'IDs':>8}"
        f"{'Time(s)':>12}"
        f"{'FPS':>10}"
        f"{'Output(MB)':>14}"
    )

    print("-" * 100)

    for row in rows:

        print(
            f"{row['tracker']:<15}"
            f"{row['track_rows']:>10}"
            f"{row['unique_ids']:>8}"
            f"{row['processing_time_sec']:>12.3f}"
            f"{row['processing_fps']:>10.3f}"
            f"{row['output_size_mb']:>14.3f}"
        )

    print("-" * 100)

    print()
    print(f"CSV:")
    print(OUTPUT_CSV)

    print("=" * 100)


if __name__ == "__main__":
    main()