from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "pytorch"
    / "yolo11m.pt"
)

VIDEO_DIR = ROOT / "test_videos"

TRACKER_DIR = ROOT / "06_Trackers"

OUTPUT_DIR = ROOT / "outputs" / "trackers"

RESULTS_DIR = (
    ROOT
    / "06_Trackers"
    / "results"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRACKERS = {
    "McByte": {
        "directory": TRACKER_DIR / "McByte",
        "enabled": False,
    },

    "OC-SORT": {
        "directory": TRACKER_DIR / "OC-SORT",
        "enabled": False,
    },

    "BoT-SORT": {
        "directory": TRACKER_DIR / "BoT-SORT",
        "enabled": False,
    },

    "ByteTrack_FastReID": {
        "directory": TRACKER_DIR / "ByteTrack_FastReID",
        "enabled": False,
    },
}