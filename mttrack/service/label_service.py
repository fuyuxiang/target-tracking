"""
Label service - VL model integration for track labeling.
"""

import time
from dataclasses import dataclass
from typing import Optional
import numpy as np

from mttrack.infrastructure import VllmClient, VLClassificationResult
from mttrack.service.tracker_service import TrackerService


@dataclass
class LabelRequest:
    """Request for VL labeling."""

    track_id: int
    crop: np.ndarray
    frame_id: int


@dataclass
class LabelCache:
    """Label cache entry."""

    class_name: str
    confidence: float
    timestamp: float


class LabelService:
    """Service for labeling tracks with VL model."""

    def __init__(
        self,
        vllm_client: Optional[VllmClient] = None,
        enabled: bool = True,
        label_interval: int = 30,  # Label every N frames
        cache_ttl: float = 60.0,  # Cache TTL in seconds
        max_concurrent: int = 1,
    ) -> None:
        """Initialize label service.

        Args:
            vllm_client: VLLM client
            enabled: Whether VL labeling is enabled
            label_interval: Label tracks every N frames
            cache_ttl: Cache time-to-live in seconds
            max_concurrent: Max concurrent VL requests
        """
        self.vllm_client = vllm_client
        self.enabled = enabled and vllm_client is not None
        self.label_interval = label_interval
        self.cache_ttl = cache_ttl

        # Cache: track_id -> LabelCache
        self._cache: dict[int, LabelCache] = {}

    def is_available(self) -> bool:
        """Check if VL service is available."""
        if not self.enabled or self.vllm_client is None:
            return False
        return self.vllm_client.is_available()

    def should_label(self, track_id: int, frame_id: int) -> bool:
        """Check if track should be labeled.

        Args:
            track_id: Track ID
            frame_id: Current frame ID

        Returns:
            True if should request VL classification
        """
        if not self.enabled:
            return False

        # Check cache
        cached = self._cache.get(track_id)
        if cached:
            # Check if cache is still valid
            if time.time() - cached.timestamp < self.cache_ttl:
                return False

        # Label every N frames
        return frame_id % self.label_interval == 0

    def label_track(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_id: int,
    ) -> Optional[VLClassificationResult]:
        """Label a track using VL model.

        Args:
            track_id: Track ID
            crop: Cropped image
            frame_id: Current frame ID

        Returns:
            VLClassificationResult or None
        """
        if not self.enabled or self.vllm_client is None:
            return None

        try:
            result = self.vllm_client.classify_crop(crop, track_id)

            # Cache result
            if result.class_name != "unknown":
                self._cache[track_id] = LabelCache(
                    class_name=result.class_name,
                    confidence=result.confidence,
                    timestamp=time.time()
                )

            return result

        except Exception as e:
            print(f"[LabelService] Failed to label track {track_id}: {e}")
            return None

    def get_cached_label(self, track_id: int) -> Optional[LabelCache]:
        """Get cached label for a track."""
        cached = self._cache.get(track_id)
        if cached and time.time() - cached.timestamp < self.cache_ttl:
            return cached
        return None

    def get_track_label(self, track_id: int) -> Optional[str]:
        """Get label for a track (from cache)."""
        cached = self.get_cached_label(track_id)
        return cached.class_name if cached else None

    def clear_cache(self) -> None:
        """Clear label cache."""
        self._cache.clear()

    def cleanup_old_tracks(self, active_track_ids: set[int]) -> None:
        """Remove cache entries for tracks that are no longer active."""
        stale_ids = set(self._cache.keys()) - active_track_ids
        for track_id in stale_ids:
            del self._cache[track_id]
