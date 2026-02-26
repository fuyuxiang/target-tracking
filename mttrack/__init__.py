"""
Multi-Target Tracking with VL Classification.

A modular tracking system that combines:
- YOLO for object detection
- ByteTrack/SORT for multi-object tracking
- Vision-Language model for classification
"""

from mttrack.domain import (
    Detection,
    Track,
    FrameResult,
    LabelResult,
    BaseTracker,
    KalmanBoxTracker,
    ByteTrackTracker,
    SORTTracker,
)

from mttrack.infrastructure import (
    BaseDetector,
    DetectorResult,
    YoloDetector,
    VllmClient,
    VLClassificationResult,
    VideoReader,
    VideoWriter,
    create_video_writer,
)

from mttrack.service import (
    TrackerService,
    TrackInfo,
    FrameTracks,
    LabelService,
    LabelCache,
)

from mttrack.annotators import (
    TrackingAnnotator,
    get_track_color,
    draw_track_id_only,
)

__version__ = "0.1.0"

__all__ = [
    # Domain
    "Detection",
    "Track",
    "FrameResult",
    "LabelResult",
    "BaseTracker",
    "KalmanBoxTracker",
    "ByteTrackTracker",
    "SORTTracker",
    # Infrastructure
    "BaseDetector",
    "DetectorResult",
    "YoloDetector",
    "VllmClient",
    "VLClassificationResult",
    "TARGET_CLASSES",
    "VideoReader",
    "VideoWriter",
    "create_video_writer",
    # Service
    "TrackerService",
    "TrackInfo",
    "FrameTracks",
    "LabelService",
    "LabelCache",
    # Annotators
    "TrackingAnnotator",
    "get_track_color",
    "draw_track_id_only",
]
