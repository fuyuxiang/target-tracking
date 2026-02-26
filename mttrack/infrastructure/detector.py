"""
Detector interface and implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DetectorResult:
    """Detection result."""

    boxes: np.ndarray  # (N, 4) in xyxy format
    confidences: np.ndarray  # (N,)
    class_ids: np.ndarray  # (N,)
    class_names: list[str]  # (N,)


class BaseDetector(ABC):
    """Base detector interface."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> DetectorResult:
        """Detect objects in image.

        Args:
            image: BGR image

        Returns:
            DetectorResult with detections
        """
        pass

    @abstractmethod
    def warmup(self) -> None:
        """Warm up the detector."""
        pass


class YoloDetector(BaseDetector):
    """YOLO detector implementation."""

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        device: str = "cuda",
    ) -> None:
        """Initialize YOLO detector.

        Args:
            model_path: Path to YOLO model
            confidence_threshold: Confidence threshold
            device: Device to run on
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model = None
        self._class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
            'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
            'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
            'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
            'hair drier', 'toothbrush'
        ]

    def warmup(self) -> None:
        """Warm up the detector."""
        if self.model is None:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
            if self.device:
                self.model.to(self.device)

    def detect(self, image: np.ndarray) -> DetectorResult:
        """Detect objects in image.

        Args:
            image: BGR image

        Returns:
            DetectorResult with detections
        """
        if self.model is None:
            self.warmup()

        results = self.model(image, conf=self.confidence_threshold, verbose=False)

        if not results or len(results) == 0:
            return DetectorResult(
                boxes=np.array([], dtype=np.float32).reshape(0, 4),
                confidences=np.array([], dtype=np.float32),
                class_ids=np.array([], dtype=np.int32),
                class_names=[]
            )

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return DetectorResult(
                boxes=np.array([], dtype=np.float32).reshape(0, 4),
                confidences=np.array([], dtype=np.float32),
                class_ids=np.array([], dtype=np.int32),
                class_names=[]
            )

        # Get boxes in xyxy format
        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()
        class_ids = boxes.cls.cpu().numpy().astype(int)
        class_names = [self._class_names[cid] for cid in class_ids]

        return DetectorResult(
            boxes=xyxy,
            confidences=confidences,
            class_ids=class_ids,
            class_names=class_names
        )
