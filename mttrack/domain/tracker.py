"""
Base tracker interface.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np


@dataclass
class TrackerInfo:
    """Information about a registered tracker."""

    tracker_class: type
    parameters: dict[str, Any]


class BaseTracker(ABC):
    """Abstract base class for all trackers.

    Subclasses must define `tracker_id` class variable to be registered.
    """

    _registry: ClassVar[dict[str, TrackerInfo]] = {}
    tracker_id: ClassVar[str | None] = None

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Register subclass if it defines tracker_id."""
        super().__init_subclass__(**kwargs)
        tracker_id = getattr(cls, "tracker_id", None)
        if tracker_id is not None:
            BaseTracker._registry[tracker_id] = TrackerInfo(
                tracker_class=cls,
                parameters={}
            )

    @classmethod
    def get_registered_trackers(cls) -> list[str]:
        """Get list of registered tracker IDs."""
        return sorted(cls._registry.keys())

    @classmethod
    def create_tracker(cls, tracker_id: str, **kwargs) -> "BaseTracker":
        """Create a tracker by ID."""
        info = cls._registry.get(tracker_id)
        if info is None:
            raise ValueError(f"Unknown tracker: {tracker_id}")
        return info.tracker_class(**kwargs)

    @abstractmethod
    def update(self, detections: np.ndarray, confidences: np.ndarray = None,
               class_ids: np.ndarray = None) -> np.ndarray:
        """Update tracker with new detections.

        Args:
            detections: Array of shape (N, 4) with [x1, y1, x2, y2]
            confidences: Array of confidence scores (N,)
            class_ids: Array of class IDs (N,)

        Returns:
            Array of tracker IDs for each detection (N,)
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset tracker state."""
        pass

    @abstractmethod
    def get_active_tracks(self) -> list[dict]:
        """Get list of active tracks.

        Returns:
            List of dicts with track information
        """
        pass
