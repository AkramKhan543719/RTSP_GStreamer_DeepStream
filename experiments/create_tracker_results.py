from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = (
    ROOT
    / "06_Trackers"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRACKERS = [
    "McByte",
    "OC-SORT",
    "BoT-SORT",
    "ByteTrack_FastReID",
]


VIDEOS = [
    "video1",
    "video2",
    "video3",
    "video4",
]


FIELDS = [
    "tracker",
    "video",
    "frames",
    "processing_time_sec",
    "processing_fps",
    "detections",
    "tracks",
    "IDF1",
    "HOTA",
    "MOTA",
    "ID_switches",
]


def main():

    output = RESULTS_DIR / "tracker_comparison.csv"

    rows = []

    for tracker in TRACKERS:

        for video in VIDEOS:

            rows.append({
                "tracker": tracker,
                "video": video,
                "frames": "",
                "processing_time_sec": "",
                "processing_fps": "",
                "detections": "",
                "tracks": "",
                "IDF1": "",
                "HOTA": "",
                "MOTA": "",
                "ID_switches": "",
            })

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS
        )

        writer.writeheader()
        writer.writerows(rows)

    print("=" * 60)
    print("TRACKER RESULT TEMPLATE CREATED")
    print("=" * 60)
    print(output)


if __name__ == "__main__":
    main()