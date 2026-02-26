"""
Domain models for multi-target tracking.
"""

from dataclasses import dataclass, field
from typing import Optional, List
import numpy as np


@dataclass
class Detection:
    """Single object detection."""

    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str = "unknown"


@dataclass
class Track:
    """Tracked object with ID and history."""

    track_id: int
    detections: List[Detection] = field(default_factory=list)
    label: Optional[str] = None  # VL model classification
    label_confidence: float = 0.0
    last_bbox: Optional[tuple[float, float, float, float]] = None
    age: int = 0
    hits: int = 0
    time_since_update: int = 0

    def update(self, detection: Detection) -> None:
        """Update track with new detection."""
        self.detections.append(detection)
        self.last_bbox = detection.bbox
        self.age += 1
        self.hits += 1
        self.time_since_update = 0

    def predict(self) -> None:
        """Mark track as predicted (no new detection)."""
        self.age += 1
        self.time_since_update += 1


@dataclass
class FrameResult:
    """Result for a single frame."""

    frame_id: int
    tracks: List[Track]
    image: np.ndarray


@dataclass
class LabelResult:
    """Labeling result from VL model."""

    track_id: int
    class_name: str
    confidence: float
    frame_id: int
