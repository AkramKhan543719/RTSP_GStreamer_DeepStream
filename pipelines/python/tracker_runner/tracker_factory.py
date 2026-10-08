"""
BoxMOT 18.0.0 tracker factory.

Supports:
    - ByteTrack
    - OC-SORT
    - SFSORT
    - BoT-SORT
    - StrongSORT
    - Deep OC-SORT
    - HybridSORT
    - BoostTrack
    - OccluBoost

ReID trackers use the project's shared:
    models/reid/lmbn_n_duke.pth
"""

from pathlib import Path

from boxmot.trackers.tracker_zoo import (
    create_tracker,
    TRACKER_MAPPING,
    REID_TRACKERS,
    get_tracker_config,
)


# ---------------------------------------------------------------------
# Project paths
# ---------------------------------------------------------------------

# tracker_factory.py
# -> tracker_runner
# -> python
# -> pipelines
# -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[3]

REID_WEIGHTS = (
    PROJECT_ROOT
    / "models"
    / "reid"
    / "lmbn_n_duke.pth"
)


# ---------------------------------------------------------------------
# Tracker display names
# ---------------------------------------------------------------------

TRACKER_DISPLAY_NAMES = {
    "bytetrack": "ByteTrack",
    "ocsort": "OC-SORT",
    "sfsort": "SFSORT",
    "botsort": "BoT-SORT",
    "strongsort": "StrongSORT",
    "deepocsort": "Deep OC-SORT",
    "hybridsort": "HybridSORT",
    "boosttrack": "BoostTrack",
    "occluboost": "OccluBoost",
}


# ---------------------------------------------------------------------
# All BoxMOT 18.0.0 trackers available in this installation
# ---------------------------------------------------------------------

TRACKER_LIST = [
    "bytetrack",
    "ocsort",
    "sfsort",
    "botsort",
    "strongsort",
    "deepocsort",
    "hybridsort",
    "boosttrack",
    "occluboost",
]


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def get_tracker_name(tracker_type):
    """Return the human-readable tracker name."""
    return TRACKER_DISPLAY_NAMES.get(
        tracker_type,
        tracker_type,
    )


def tracker_requires_reid(tracker_type):
    """Return True when the tracker requires a ReID model."""
    return tracker_type in REID_TRACKERS


def get_reid_weights():
    """
    Return the project's ReID weights path.

    Raises a clear error if the model is missing.
    """
    if not REID_WEIGHTS.exists():
        raise FileNotFoundError(
            "\n"
            "ReID model not found.\n"
            f"Expected path:\n{REID_WEIGHTS}\n"
            "\n"
            "Please make sure lmbn_n_duke.pth exists in:\n"
            "models\\reid\\lmbn_n_duke.pth\n"
        )

    if REID_WEIGHTS.stat().st_size == 0:
        raise RuntimeError(
            f"ReID model is empty:\n{REID_WEIGHTS}"
        )

    return REID_WEIGHTS


# ---------------------------------------------------------------------
# Tracker factory
# ---------------------------------------------------------------------

def create_selected_tracker(
    tracker_type,
    device="cpu",
    half=False,
    per_class=False,
):
    """
    Create a BoxMOT 18.0.0 tracker.

    ReID trackers automatically use:
        models/reid/lmbn_n_duke.pth
    """

    # -------------------------------------------------------------
    # Validate tracker
    # -------------------------------------------------------------

    if tracker_type not in TRACKER_MAPPING:
        raise ValueError(
            f"Unsupported tracker: {tracker_type}"
        )

    # -------------------------------------------------------------
    # Tracker configuration
    # -------------------------------------------------------------

    config_path = get_tracker_config(
        tracker_type
    )

    # -------------------------------------------------------------
    # Base arguments
    # -------------------------------------------------------------

    kwargs = {
        "tracker_type": tracker_type,
        "tracker_config": config_path,
        "device": device,
        "half": half,
        "per_class": per_class,
    }

    # -------------------------------------------------------------
    # ReID configuration
    # -------------------------------------------------------------

    if tracker_type in REID_TRACKERS:

        reid_weights = get_reid_weights()

        print()
        print("ReID configuration")
        print("-" * 60)
        print(f"ReID model : {reid_weights}")
        print("ReID device: {}".format(device))
        print("ReID half  : {}".format(half))
        print("-" * 60)

        kwargs["reid_weights"] = reid_weights

    # -------------------------------------------------------------
    # Create tracker using BoxMOT official factory
    # -------------------------------------------------------------

    return create_tracker(**kwargs)