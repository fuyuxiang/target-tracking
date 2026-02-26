"""
Tracker service - orchestrates detection and tracking.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from mttrack.domain import BaseTracker, ByteTrackTracker, SORTTracker
from mttrack.infrastructure import BaseDetector, YoloDetector, DetectorResult


@dataclass
class TrackInfo:
    """Track information."""

    track_id: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    class_name: str
    class_id: int
    confidence: float
    label: Optional[str] = None
    label_confidence: float = 0.0


@dataclass
class FrameTracks:
    """Tracks for a single frame."""

    frame_id: int
    tracks: list[TrackInfo]


class TrackerService:
    """Service for object tracking."""

    def __init__(
        self,
        detector: BaseDetector,
        tracker_type: str = "bytetrack",
        tracker_kwargs: Optional[dict] = None,
    ) -> None:
        """Initialize tracker service.

        Args:
            detector: Object detector
            tracker_type: Type of tracker ("bytetrack" or "sort")
            tracker_kwargs: Additional tracker arguments
        """
        self.detector = detector
        self.tracker_kwargs = tracker_kwargs or {}
        self.tracker_type = tracker_type
        self._tracker: Optional[BaseTracker] = None
        self._frame_count = 0

        # Track info storage
        self._track_labels: dict[int, dict] = {}  # track_id -> {label, confidence}
        self._track_class_names: dict[int, str] = {}  # track_id -> class_name

    def _create_tracker(self) -> BaseTracker:
        """Create tracker instance."""
        if self.tracker_type == "bytetrack":
            return ByteTrackTracker(**self.tracker_kwargs)
        elif self.tracker_type == "sort":
            return SORTTracker(**self.tracker_kwargs)
        else:
            raise ValueError(f"Unknown tracker type: {self.tracker_type}")

    def process_frame(self, frame: np.ndarray) -> FrameTracks:
        """Process a single frame.

        Args:
            frame: BGR image

        Returns:
            FrameTracks with detected and tracked objects
        """
        self._frame_count += 1

        # Detect objects
        det_result = self.detector.detect(frame)

        if len(det_result.boxes) == 0:
            # No detections - still need to update tracker
            if self._tracker is None:
                self._tracker = self._create_tracker()
            self._tracker.update(
                np.array([]).reshape(0, 4),
                np.array([]),
                np.array([], dtype=int)
            )
            return FrameTracks(frame_id=self._frame_count, tracks=[])

        # Initialize tracker if needed
        if self._tracker is None:
            self._tracker = self._create_tracker()

        # Update tracker
        tracker_ids = self._tracker.update(
            det_result.boxes,
            det_result.confidences,
            det_result.class_ids
        )

        # Build result
        tracks = []
        for i, (box, conf, cls_id, cls_name, trk_id) in enumerate(zip(
            det_result.boxes,
            det_result.confidences,
            det_result.class_ids,
            det_result.class_names,
            tracker_ids
        )):
            if trk_id < 0:
                continue

            # Get cached label if available
            label_info = self._track_labels.get(trk_id, {})
            label = label_info.get("label")
            label_conf = label_info.get("confidence", 0.0)

            # Update class name from YOLO
            self._track_class_names[trk_id] = cls_name

            tracks.append(TrackInfo(
                track_id=int(trk_id),
                bbox=tuple(box.tolist()),
                class_name=cls_name,
                class_id=int(cls_id),
                confidence=float(conf),
                label=label,
                label_confidence=label_conf,
            ))

        return FrameTracks(frame_id=self._frame_count, tracks=tracks)

    def update_track_label(
        self,
        track_id: int,
        label: str,
        confidence: float
    ) -> None:
        """Update label for a track.

        Args:
            track_id: Track ID
            label: Class label
            confidence: Label confidence
        """
        self._track_labels[track_id] = {
            "label": label,
            "confidence": confidence
        }

    def get_track_label(self, track_id: int) -> Optional[dict]:
        """Get label for a track."""
        return self._track_labels.get(track_id)

    def reset(self) -> None:
        """Reset tracker state."""
        if self._tracker:
            self._tracker.reset()
        self._frame_count = 0
        self._track_labels.clear()
        self._track_class_names.clear()

    def warmup(self) -> None:
        """Warm up detector."""
        self.detector.warmup()
