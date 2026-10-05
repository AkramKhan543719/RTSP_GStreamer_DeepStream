from pathlib import Path
import csv
import statistics
from collections import defaultdict


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRACKER_ROOT = ROOT / "outputs" / "trackers"
RESULTS_DIR = ROOT / "experiments" / "video0_tracker_analysis"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# TRACKER FILES
# ============================================================

TRACKERS = {
    "ByteTrack": TRACKER_ROOT / "track" / "video0" / "tracks.txt",

    "OC-SORT": TRACKER_ROOT / "ocsort" / "track" / "video0" / "tracks.txt",

    "BoT-SORT": TRACKER_ROOT / "botsort" / "track" / "video0" / "tracks.txt",

    "Deep SORT": TRACKER_ROOT / "deepsort" / "video0_deepsort_tracks.txt",
}


# ============================================================
# INPUT INFORMATION
# ============================================================

VIDEO_NAME = "video0.avi"
TOTAL_FRAMES = 998
INPUT_FPS = 29.0
WIDTH = 1920
HEIGHT = 1080


# ============================================================
# LOAD TRACK FILE
# ============================================================

def load_tracks(path):

    rows = []

    if not path.exists():
        raise FileNotFoundError(f"Track file not found:\n{path}")

    with open(path, "r", encoding="utf-8") as f:

        for line_number, line in enumerate(f, start=1):

            line = line.strip()

            if not line:
                continue

            parts = [x.strip() for x in line.split(",")]

            try:

                frame = int(float(parts[0]))
                track_id = int(float(parts[1]))

                x = float(parts[2])
                y = float(parts[3])
                w = float(parts[4])
                h = float(parts[5])

                confidence = None
                class_id = None
                detection_index = None

                # BoxMOT format:
                #
                # frame,id,x,y,w,h,conf,class,detection
                #
                # Deep SORT:
                #
                # frame,id,x,y,w,h

                if len(parts) >= 7:
                    try:
                        confidence = float(parts[6])
                    except ValueError:
                        confidence = None

                if len(parts) >= 8:
                    try:
                        class_id = int(float(parts[7]))
                    except ValueError:
                        class_id = None

                if len(parts) >= 9:
                    try:
                        detection_index = int(float(parts[8]))
                    except ValueError:
                        detection_index = None

                rows.append({
                    "frame": frame,
                    "id": track_id,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "confidence": confidence,
                    "class_id": class_id,
                    "detection_index": detection_index,
                })

            except (ValueError, IndexError):

                print(
                    f"WARNING: Could not parse line "
                    f"{line_number} in {path}"
                )

    return rows


# ============================================================
# BASIC STATISTICS
# ============================================================

def calculate_statistics(rows):

    if not rows:
        return {}

    frames = [r["frame"] for r in rows]
    ids = [r["id"] for r in rows]

    unique_ids = sorted(set(ids))
    unique_frames = sorted(set(frames))

    # --------------------------------------------------------
    # Track -> frames
    # --------------------------------------------------------

    id_frames = defaultdict(list)

    for row in rows:
        id_frames[row["id"]].append(row["frame"])

    track_lengths = []

    for track_id, frame_list in id_frames.items():

        unique_track_frames = sorted(set(frame_list))

        track_lengths.append(
            len(unique_track_frames)
        )

    # --------------------------------------------------------
    # Frame -> active IDs
    # --------------------------------------------------------

    frame_ids = defaultdict(set)

    for row in rows:
        frame_ids[row["frame"]].add(row["id"])

    active_counts = [
        len(frame_ids[f])
        for f in sorted(frame_ids)
    ]

    # --------------------------------------------------------
    # Track continuity
    # --------------------------------------------------------

    fragmentation_events = 0
    total_gaps = 0
    longest_gap = 0

    for track_id, frame_list in id_frames.items():

        unique_track_frames = sorted(set(frame_list))

        if len(unique_track_frames) < 2:
            continue

        for previous, current in zip(
            unique_track_frames,
            unique_track_frames[1:]
        ):

            gap = current - previous - 1

            if gap > 0:
                fragmentation_events += 1
                total_gaps += gap
                longest_gap = max(longest_gap, gap)

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence_values = [
        r["confidence"]
        for r in rows
        if r["confidence"] is not None
    ]

    # --------------------------------------------------------
    # Bounding boxes
    # --------------------------------------------------------

    areas = [
        r["w"] * r["h"]
        for r in rows
    ]

    widths = [r["w"] for r in rows]
    heights = [r["h"] for r in rows]

    # --------------------------------------------------------
    # Long / short tracks
    # --------------------------------------------------------

    single_frame_tracks = sum(
        1 for x in track_lengths if x == 1
    )

    tracks_10_plus = sum(
        1 for x in track_lengths if x >= 10
    )

    tracks_30_plus = sum(
        1 for x in track_lengths if x >= 30
    )

    tracks_100_plus = sum(
        1 for x in track_lengths if x >= 100
    )

    # --------------------------------------------------------
    # Detection rows vs frame coverage
    # --------------------------------------------------------

    total_rows = len(rows)

    avg_rows_per_frame = (
        total_rows / len(unique_frames)
        if unique_frames
        else 0
    )

    avg_active_tracks = (
        statistics.mean(active_counts)
        if active_counts
        else 0
    )

    max_active_tracks = (
        max(active_counts)
        if active_counts
        else 0
    )

    min_active_tracks = (
        min(active_counts)
        if active_counts
        else 0
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    return {

        "track_rows": total_rows,

        "unique_ids": len(unique_ids),

        "frames_with_tracks": len(unique_frames),

        "frame_coverage_percent": (
            len(unique_frames) / TOTAL_FRAMES * 100
        ),

        "avg_track_length": (
            statistics.mean(track_lengths)
            if track_lengths
            else 0
        ),

        "median_track_length": (
            statistics.median(track_lengths)
            if track_lengths
            else 0
        ),

        "min_track_length": (
            min(track_lengths)
            if track_lengths
            else 0
        ),

        "max_track_length": (
            max(track_lengths)
            if track_lengths
            else 0
        ),

        "single_frame_tracks": single_frame_tracks,

        "tracks_10_plus_frames": tracks_10_plus,

        "tracks_30_plus_frames": tracks_30_plus,

        "tracks_100_plus_frames": tracks_100_plus,

        "avg_active_tracks_per_frame": avg_active_tracks,

        "min_active_tracks": min_active_tracks,

        "max_active_tracks": max_active_tracks,

        "avg_rows_per_frame": avg_rows_per_frame,

        "fragmentation_events": fragmentation_events,

        "total_missing_frame_gaps": total_gaps,

        "longest_missing_gap": longest_gap,

        "avg_confidence": (
            statistics.mean(confidence_values)
            if confidence_values
            else None
        ),

        "min_confidence": (
            min(confidence_values)
            if confidence_values
            else None
        ),

        "max_confidence": (
            max(confidence_values)
            if confidence_values
            else None
        ),

        "avg_bbox_width": statistics.mean(widths),

        "avg_bbox_height": statistics.mean(heights),

        "avg_bbox_area": statistics.mean(areas),

        "max_bbox_area": max(areas),

        "min_bbox_area": min(areas),
    }


# ============================================================
# WRITE PER-ID CSV
# ============================================================

def write_id_statistics(tracker_name, rows):

    output = (
        RESULTS_DIR /
        f"{tracker_name.lower().replace(' ', '_').replace('-', '')}"
        "_per_id.csv"
    )

    id_data = defaultdict(list)

    for row in rows:
        id_data[row["id"]].append(row["frame"])

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "tracker",
            "track_id",
            "first_frame",
            "last_frame",
            "track_length",
            "expected_span",
            "missing_frames",
            "continuity_percent",
        ])

        for track_id in sorted(id_data):

            frames = sorted(set(id_data[track_id]))

            first_frame = frames[0]
            last_frame = frames[-1]

            track_length = len(frames)

            expected_span = (
                last_frame -
                first_frame +
                1
            )

            missing_frames = (
                expected_span -
                track_length
            )

            continuity = (
                track_length /
                expected_span *
                100
                if expected_span > 0
                else 0
            )

            writer.writerow([
                tracker_name,
                track_id,
                first_frame,
                last_frame,
                track_length,
                expected_span,
                missing_frames,
                round(continuity, 3),
            ])

    return output


# ============================================================
# WRITE PER-FRAME CSV
# ============================================================

def write_frame_statistics(tracker_name, rows):

    output = (
        RESULTS_DIR /
        f"{tracker_name.lower().replace(' ', '_').replace('-', '')}"
        "_per_frame.csv"
    )

    frame_data = defaultdict(set)

    for row in rows:
        frame_data[row["frame"]].add(row["id"])

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "tracker",
            "frame",
            "active_tracks",
        ])

        for frame in range(1, TOTAL_FRAMES + 1):

            active = len(frame_data.get(frame, set()))

            writer.writerow([
                tracker_name,
                frame,
                active,
            ])

    return output


# ============================================================
# WRITE MASTER CSV
# ============================================================

def write_master_csv(results):

    output = RESULTS_DIR / "video0_tracker_analysis.csv"

    fields = [
        "tracker",
        "video",
        "frames",
        "input_fps",
        "resolution",

        "track_rows",
        "unique_ids",

        "frames_with_tracks",
        "frame_coverage_percent",

        "avg_track_length",
        "median_track_length",
        "min_track_length",
        "max_track_length",

        "single_frame_tracks",
        "tracks_10_plus_frames",
        "tracks_30_plus_frames",
        "tracks_100_plus_frames",

        "avg_active_tracks_per_frame",
        "min_active_tracks",
        "max_active_tracks",

        "avg_rows_per_frame",

        "fragmentation_events",
        "total_missing_frame_gaps",
        "longest_missing_gap",

        "avg_confidence",
        "min_confidence",
        "max_confidence",

        "avg_bbox_width",
        "avg_bbox_height",
        "avg_bbox_area",
        "min_bbox_area",
        "max_bbox_area",
    ]

    with open(
        output,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writeheader()

        for tracker_name, stats in results.items():

            row = {
                "tracker": tracker_name,
                "video": VIDEO_NAME,
                "frames": TOTAL_FRAMES,
                "input_fps": INPUT_FPS,
                "resolution": f"{WIDTH}x{HEIGHT}",
            }

            row.update(stats)

            writer.writerow(row)

    return output


# ============================================================
# WRITE HUMAN REPORT
# ============================================================

def write_report(results):

    output = RESULTS_DIR / "video0_tracker_analysis_report.txt"

    with open(
        output,
        "w",
        encoding="utf-8"
    ) as f:

        f.write("=" * 100 + "\n")
        f.write("VIDEO0 TRACKER DETAILED ANALYSIS\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Video       : {VIDEO_NAME}\n")
        f.write(f"Frames      : {TOTAL_FRAMES}\n")
        f.write(f"Input FPS   : {INPUT_FPS}\n")
        f.write(f"Resolution  : {WIDTH}x{HEIGHT}\n\n")

        for tracker_name, s in results.items():

            f.write("=" * 100 + "\n")
            f.write(f"{tracker_name}\n")
            f.write("=" * 100 + "\n")

            f.write(
                f"Track rows                  : {s['track_rows']}\n"
            )

            f.write(
                f"Unique IDs                  : {s['unique_ids']}\n"
            )

            f.write(
                f"Frames with tracks          : {s['frames_with_tracks']}\n"
            )

            f.write(
                f"Frame coverage              : "
                f"{s['frame_coverage_percent']:.2f}%\n"
            )

            f.write(
                f"Average track length        : "
                f"{s['avg_track_length']:.2f} frames\n"
            )

            f.write(
                f"Median track length         : "
                f"{s['median_track_length']:.2f} frames\n"
            )

            f.write(
                f"Minimum track length        : "
                f"{s['min_track_length']} frames\n"
            )

            f.write(
                f"Maximum track length        : "
                f"{s['max_track_length']} frames\n"
            )

            f.write(
                f"Single-frame tracks         : "
                f"{s['single_frame_tracks']}\n"
            )

            f.write(
                f"Tracks >= 10 frames        : "
                f"{s['tracks_10_plus_frames']}\n"
            )

            f.write(
                f"Tracks >= 30 frames        : "
                f"{s['tracks_30_plus_frames']}\n"
            )

            f.write(
                f"Tracks >= 100 frames       : "
                f"{s['tracks_100_plus_frames']}\n"
            )

            f.write(
                f"Average active tracks/frame : "
                f"{s['avg_active_tracks_per_frame']:.2f}\n"
            )

            f.write(
                f"Minimum active tracks/frame : "
                f"{s['min_active_tracks']}\n"
            )

            f.write(
                f"Maximum active tracks/frame : "
                f"{s['max_active_tracks']}\n"
            )

            f.write(
                f"Average rows/frame          : "
                f"{s['avg_rows_per_frame']:.2f}\n"
            )

            f.write(
                f"Fragmentation events        : "
                f"{s['fragmentation_events']}\n"
            )

            f.write(
                f"Total missing frame gaps    : "
                f"{s['total_missing_frame_gaps']}\n"
            )

            f.write(
                f"Longest missing gap         : "
                f"{s['longest_missing_gap']} frames\n"
            )

            if s["avg_confidence"] is not None:

                f.write(
                    f"Average confidence          : "
                    f"{s['avg_confidence']:.4f}\n"
                )

                f.write(
                    f"Minimum confidence          : "
                    f"{s['min_confidence']:.4f}\n"
                )

                f.write(
                    f"Maximum confidence          : "
                    f"{s['max_confidence']:.4f}\n"
                )

            else:

                f.write(
                    "Confidence                   : "
                    "Not available in track file\n"
                )

            f.write(
                f"Average bbox width          : "
                f"{s['avg_bbox_width']:.2f}\n"
            )

            f.write(
                f"Average bbox height         : "
                f"{s['avg_bbox_height']:.2f}\n"
            )

            f.write(
                f"Average bbox area           : "
                f"{s['avg_bbox_area']:.2f}\n"
            )

            f.write("\n")

    return output


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 100)
    print("VIDEO0 TRACKER ANALYSIS")
    print("=" * 100)

    results = {}

    for tracker_name, track_file in TRACKERS.items():

        print()
        print("-" * 100)
        print(f"Processing: {tracker_name}")
        print(f"File      : {track_file}")
        print("-" * 100)

        if not track_file.exists():

            print("WARNING: File does not exist.")
            print("Skipping.")

            continue

        rows = load_tracks(track_file)

        print(f"Loaded rows: {len(rows)}")

        stats = calculate_statistics(rows)

        results[tracker_name] = stats

        id_csv = write_id_statistics(
            tracker_name,
            rows
        )

        frame_csv = write_frame_statistics(
            tracker_name,
            rows
        )

        print(
            f"Unique IDs          : "
            f"{stats['unique_ids']}"
        )

        print(
            f"Average track length: "
            f"{stats['avg_track_length']:.2f}"
        )

        print(
            f"Max track length    : "
            f"{stats['max_track_length']}"
        )

        print(
            f"Fragmentation events: "
            f"{stats['fragmentation_events']}"
        )

        print(f"Per-ID CSV           : {id_csv}")
        print(f"Per-frame CSV        : {frame_csv}")

    if not results:

        raise RuntimeError(
            "No tracker results were found."
        )

    master_csv = write_master_csv(results)

    report = write_report(results)

    print()
    print("=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)

    print(f"Master CSV : {master_csv}")
    print(f"Report     : {report}")

    print()
    print("TRACKER SUMMARY")
    print("-" * 100)

    print(
        f"{'Tracker':<18}"
        f"{'Rows':>10}"
        f"{'IDs':>10}"
        f"{'AvgLen':>12}"
        f"{'MaxLen':>12}"
        f"{'Frag':>12}"
        f"{'Coverage':>12}"
    )

    print("-" * 100)

    for tracker_name, s in results.items():

        print(
            f"{tracker_name:<18}"
            f"{s['track_rows']:>10}"
            f"{s['unique_ids']:>10}"
            f"{s['avg_track_length']:>12.2f}"
            f"{s['max_track_length']:>12}"
            f"{s['fragmentation_events']:>12}"
            f"{s['frame_coverage_percent']:>11.2f}%"
        )

    print("-" * 100)


if __name__ == "__main__":
    main()