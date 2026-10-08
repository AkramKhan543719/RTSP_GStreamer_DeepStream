"""
Video annotation utilities.

The tracker name is displayed in the upper-right corner.
"""

import cv2


def draw_tracker_label(
    frame,
    tracker_name,
    frame_number=None,
    total_frames=None,
    fps=None,
):
    height, width = frame.shape[:2]

    # Main tracker label
    label = f"TRACKER: {tracker_name}"

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2

    (text_width, text_height), baseline = cv2.getTextSize(
        label,
        font,
        font_scale,
        thickness,
    )

    x = width - text_width - 20
    y = 35

    # Background rectangle
    cv2.rectangle(
        frame,
        (x - 10, y - text_height - 10),
        (width - 10, y + baseline + 10),
        (0, 0, 0),
        -1,
    )

    cv2.putText(
        frame,
        label,
        (x, y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # Optional frame information
    info_parts = []

    if frame_number is not None and total_frames is not None:
        info_parts.append(
            f"Frame: {frame_number}/{total_frames}"
        )

    if fps is not None:
        info_parts.append(
            f"FPS: {fps:.2f}"
        )

    if info_parts:
        info = " | ".join(info_parts)

        cv2.putText(
            frame,
            info,
            (20, 35),
            font,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return frame