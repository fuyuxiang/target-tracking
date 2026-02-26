"""
Tests for domain layer.
"""

import pytest
import numpy as np

from mttrack.domain import (
    KalmanBoxTracker,
    ByteTrackTracker,
    SORTTracker,
    Detection,
    Track,
)


class TestKalmanBoxTracker:
    """Tests for KalmanBoxTracker."""

    def test_init(self):
        """Test Kalman filter initialization."""
        bbox = (10, 20, 100, 80)
        tracker = KalmanBoxTracker(bbox)

        assert tracker.tracker_id == -1
        assert tracker.number_of_successful_updates == 1
        assert tracker.time_since_update == 0

    def test_predict(self):
        """Test Kalman prediction."""
        bbox = (10, 20, 100, 80)
        tracker = KalmanBoxTracker(bbox)
        tracker.predict()

        assert tracker.time_since_update == 1

    def test_update(self):
        """Test Kalman update."""
        bbox = (10, 20, 100, 80)
        tracker = KalmanBoxTracker(bbox)
        tracker.update(bbox)

        assert tracker.time_since_update == 0
        assert tracker.number_of_successful_updates == 2

    def test_get_state_bbox(self):
        """Test getting state bounding box."""
        bbox = (10, 20, 100, 80)
        tracker = KalmanBoxTracker(bbox)
        state_bbox = tracker.get_state_bbox()

        assert state_bbox.shape == (4,)
        assert len(state_bbox) == 4


class TestByteTrackTracker:
    """Tests for ByteTrackTracker."""

    def test_init(self):
        """Test ByteTrack initialization."""
        tracker = ByteTrackTracker()

        assert tracker.tracker_id == "bytetrack"
        assert len(tracker.tracks) == 0

    def test_update_empty(self):
        """Test update with no detections."""
        tracker = ByteTrackTracker()
        boxes = np.array([]).reshape(0, 4)
        confs = np.array([])

        ids = tracker.update(boxes, confs)
        assert len(ids) == 0

    def test_update_with_detections(self):
        """Test update with detections."""
        tracker = ByteTrackTracker()

        # Single detection
        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])

        ids = tracker.update(boxes, confs)
        assert len(ids) == 1

    def test_reset(self):
        """Test tracker reset."""
        tracker = ByteTrackTracker()

        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])
        tracker.update(boxes, confs)

        tracker.reset()
        assert len(tracker.tracks) == 0


class TestSORTTracker:
    """Tests for SORTTracker."""

    def test_init(self):
        """Test SORT initialization."""
        tracker = SORTTracker()

        assert tracker.tracker_id == "sort"
        assert len(tracker.tracks) == 0

    def test_update_with_detections(self):
        """Test update with detections."""
        tracker = SORTTracker()

        boxes = np.array([[10, 10, 100, 100]], dtype=np.float32)
        confs = np.array([0.9])

        ids = tracker.update(boxes, confs)
        assert len(ids) == 1


class TestModels:
    """Tests for domain models."""

    def test_detection(self):
        """Test Detection model."""
        det = Detection(
            bbox=(10, 20, 100, 80),
            confidence=0.95,
            class_id=0,
            class_name="person"
        )

        assert det.bbox == (10, 20, 100, 80)
        assert det.confidence == 0.95

    def test_track(self):
        """Test Track model."""
        track = Track(track_id=1)

        assert track.track_id == 1
        assert len(track.detections) == 0

        det = Detection(
            bbox=(10, 20, 100, 80),
            confidence=0.95,
            class_id=0
        )
        track.update(det)

        assert len(track.detections) == 1
        assert track.age == 1
        assert track.hits == 1
