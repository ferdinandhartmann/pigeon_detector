from __future__ import annotations

import cv2
import numpy as np


def draw_debug_info(img: np.ndarray, frame_idx: int, proc_time: float) -> None:
    """Draws frame index and processing time on the top-left of the image."""
    cv2.putText(
        img,
        f"frame_idx: {frame_idx}",
        (10, 30),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"proc_time: {round(proc_time * 1000, 1)} ms",
        (10, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_detection(
    frame: np.ndarray,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    label: str,
    color: tuple = (0, 255, 0),
    text_color: tuple = (255, 255, 255),
) -> np.ndarray:
    """Draw a box and readable multiline detection/classification scores."""
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = label.splitlines()
    scale = 0.5
    width = max(cv2.getTextSize(line, font, scale, 1)[0][0] for line in lines)
    scale *= min(1.0, max(1, frame.shape[1] - 12) / max(1, width))
    sizes = [cv2.getTextSize(line, font, scale, 1) for line in lines]
    line_height = max(size[0][1] + size[1] for size in sizes) + 6
    width = max(size[0][0] for size in sizes) + 8
    height = line_height * len(lines) + 4
    left = max(0, min(x0, frame.shape[1] - width))
    top = max(0, min(y0, frame.shape[0] - height))
    cv2.rectangle(frame, (left, top), (left + width, top + height), (25, 25, 25), -1)
    for i, line in enumerate(lines):
        cv2.putText(
            frame, line, (left + 4, top + (i + 1) * line_height - 4),
            font, scale, text_color, 1, cv2.LINE_AA,
        )
    return frame
