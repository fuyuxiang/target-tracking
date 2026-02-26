"""
Tests for service layer.
"""

import pytest
import numpy as np

from mttrack.service import (
    TrackInfo,
    FrameTracks,
    LabelCache,
)
from mttrack.service.label_service import LabelService


class TestModels:
    """Tests for service models."""

    def test_track_info(self):
        """Test TrackInfo model."""
        info = TrackInfo(
            track_id=1,
            bbox=(10, 20, 100, 80),
            class_name="person",
            class_id=0,
            confidence=0.95,
            label="person",
            label_confidence=0.9
        )

        assert info.track_id == 1
        assert info.label == "person"

    def test_frame_tracks(self):
        """Test FrameTracks model."""
        tracks = [
            TrackInfo(
                track_id=1,
                bbox=(10, 20, 100, 80),
                class_name="person",
                class_id=0,
                confidence=0.95
            )
        ]

        frame_tracks = FrameTracks(frame_id=1, tracks=tracks)
        assert frame_tracks.frame_id == 1
        assert len(frame_tracks.tracks) == 1

    def test_label_cache(self):
        """Test LabelCache model."""
        import time

        cache = LabelCache(
            class_name="car",
            confidence=0.85,
            timestamp=time.time()
        )

        assert cache.class_name == "car"
        assert cache.confidence == 0.85


class TestLabelService:
    """Tests for LabelService."""

    def test_init(self):
        """Test LabelService initialization."""
        service = LabelService(enabled=False)

        assert not service.enabled

    def test_should_label(self):
        """Test should_label logic."""
        service = LabelService(enabled=True, label_interval=30, vllm_client=None)

        # Should label every 30 frames (since no vllm_client, enabled becomes False)
        # Actually this returns False because vllm_client is None
        assert not service.enabled

        # Test with explicit enabled=True and proper initialization
        from unittest.mock import MagicMock
        mock_client = MagicMock()
        mock_client.is_available.return_value = True

        service2 = LabelService(vllm_client=mock_client, enabled=True, label_interval=30)
        assert service2.enabled
        assert service2.should_label(1, 30)

    def test_cache(self):
        """Test label cache."""
        import time

        service = LabelService(enabled=False)

        # Add to cache
        service._cache[1] = LabelCache(
            class_name="car",
            confidence=0.9,
            timestamp=time.time()
        )

        cached = service.get_cached_label(1)
        assert cached is not None
        assert cached.class_name == "car"

    def test_cleanup(self):
        """Test cache cleanup for inactive tracks."""
        service = LabelService(enabled=False)

        # Add stale entries
        service._cache[1] = LabelCache("car", 0.9, 0)
        service._cache[2] = LabelCache("person", 0.8, 0)

        # Only track 1 is active
        service.cleanup_old_tracks({1})

        assert 1 in service._cache
        assert 2 not in service._cache
