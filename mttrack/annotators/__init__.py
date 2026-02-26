"""
Annotators for drawing tracking results on frames.
"""

import numpy as np
import cv2
from typing import Optional

from mttrack.service import TrackInfo


# Track colors
TRACK_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 0, 255), (0, 255, 128), (255, 128, 128), (128, 255, 128), (128, 128, 255),
    (255, 255, 128), (255, 128, 255), (128, 255, 255)
]


def get_track_color(track_id: int) -> tuple[int, int, int]:
    """Get color for a track ID."""
    return TRACK_COLORS[track_id % len(TRACK_COLORS)]


class TrackingAnnotator:
    """Annotator for drawing tracking results."""

    def __init__(
        self,
        thickness: int = 2,
        font_scale: float = 0.5,
        text_thickness: int = 1,
    ) -> None:
        """Initialize annotator.

        Args:
            thickness: Box thickness
            font_scale: Text font scale
            text_thickness: Text thickness
        """
        self.thickness = thickness
        self.font_scale = font_scale
        self.text_thickness = text_thickness

    def annotate(
        self,
        frame: np.ndarray,
        tracks: list[TrackInfo],
    ) -> np.ndarray:
        """Draw tracking annotations on frame.

        Args:
            frame: BGR frame
            tracks: List of TrackInfo

        Returns:
            Annotated frame
        """
        for track in tracks:
            bbox = track.bbox
            x1, y1, x2, y2 = map(int, bbox)

            # Get color
            color = get_track_color(track.track_id)

            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.thickness)

            # Prepare label text
            # Prefer VL label, fallback to YOLO class
            label_text = track.label if track.label else track.class_name

            # Add track ID
            label = f"ID:{track.track_id} {label_text}"

            # Draw background for text
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.text_thickness
            )
            cv2.rectangle(
                frame,
                (x1, y1 - text_h - baseline - 5),
                (x1 + text_w, y1),
                color,
                -1
            )

            # Draw text
            cv2.putText(
                frame,
                label,
                (x1, y1 - baseline - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale,
                (255, 255, 255),
                self.text_thickness,
                cv2.LINE_AA
            )

        return frame


def draw_track_id_only(
    frame: np.ndarray,
    track_id: int,
    bbox: tuple[float, float, float, float],
    class_name: Optional[str] = None,
    label: Optional[str] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a single track on frame.

    Args:
        frame: BGR frame
        track_id: Track ID
        bbox: Bounding box (x1, y1, x2, y2)
        class_name: YOLO class name
        label: VL label (preferred over class_name)
        thickness: Box thickness

    Returns:
        Annotated frame
    """
    x1, y1, x2, y2 = map(int, bbox)
    color = get_track_color(track_id)

    # Draw box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Use VL label if available, otherwise use class_name
    display_label = label if label else class_name

    if display_label:
        # Create label text
        label_text = f"ID:{track_id} {display_label}"
    else:
        label_text = f"ID:{track_id}"

    # Draw text background
    (text_w, text_h), baseline = cv2.getTextSize(
        label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - baseline - 5),
        (x1 + text_w, y1),
        color,
        -1
    )

    # Draw text
    cv2.putText(
        frame,
        label_text,
        (x1, y1 - baseline - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )

    return frame
