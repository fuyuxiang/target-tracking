"""
Infrastructure layer: external integrations.
"""

from mttrack.infrastructure.detector import BaseDetector, DetectorResult, YoloDetector
from mttrack.infrastructure.vllm_client import VllmClient, VLClassificationResult
from mttrack.infrastructure.video_io import VideoReader, VideoWriter, create_video_writer

__all__ = [
    "BaseDetector",
    "DetectorResult",
    "YoloDetector",
    "VllmClient",
    "VLClassificationResult",
    "VideoReader",
    "VideoWriter",
    "create_video_writer",
]
