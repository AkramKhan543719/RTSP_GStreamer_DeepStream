"""
Runtime tracker registry for BoxMOT 18.0.0.
"""

from dataclasses import dataclass


@dataclass
class TrackerInfo:
    number: int
    backend: str
    name: str
    description: str
    requires_reid: bool = False


TRACKERS = [
    TrackerInfo(
        1,
        "bytetrack",
        "ByteTrack",
        "Motion / IoU based tracker",
    ),

    TrackerInfo(
        2,
        "ocsort",
        "OC-SORT",
        "Observation-centric motion tracker",
    ),

    TrackerInfo(
        3,
        "sfsort",
        "SFSORT",
        "Motion / association based tracker",
    ),

    TrackerInfo(
        4,
        "botsort",
        "BoT-SORT",
        "Motion + camera-motion compensation",
        requires_reid=True,
    ),

    TrackerInfo(
        5,
        "strongsort",
        "StrongSORT",
        "Appearance / ReID based tracker",
        requires_reid=True,
    ),

    TrackerInfo(
        6,
        "deepocsort",
        "Deep OC-SORT",
        "OC-SORT + appearance information",
        requires_reid=True,
    ),

    TrackerInfo(
        7,
        "hybridsort",
        "HybridSORT",
        "Hybrid association tracker",
        requires_reid=True,
    ),

    TrackerInfo(
        8,
        "boosttrack",
        "BoostTrack",
        "Boosted association tracker",
        requires_reid=True,
    ),

    TrackerInfo(
        9,
        "occluboost",
        "OccluBoost",
        "Occlusion-aware tracker",
        requires_reid=True,
    ),
]


def get_tracker(number):

    for tracker in TRACKERS:

        if tracker.number == number:
            return tracker

    return None


def print_trackers():

    print()
    print("=" * 72)
    print("              BOXMOT 18.0.0 TRACKER SELECTION")
    print("=" * 72)

    for tracker in TRACKERS:

        reid_text = ""

        if tracker.requires_reid:
            reid_text = " [ReID required]"

        print(
            f"{tracker.number:2d}. "
            f"{tracker.name:<15} "
            f"- {tracker.description}"
            f"{reid_text}"
        )

    print("=" * 72)