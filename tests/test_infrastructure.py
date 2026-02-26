"""
Tests for infrastructure layer.
"""

import pytest
import numpy as np

from mttrack.infrastructure import (
    DetectorResult,
    VideoReader,
    VideoWriter,
)


class TestVideoIO:
    """Tests for video I/O."""

    def test_video_writer_init(self, tmp_path):
        """Test VideoWriter initialization."""
        output_path = tmp_path / "test.mp4"

        # Need frame to initialize
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        writer = VideoWriter(output_path, fps=30.0)
        writer.write(frame)
        writer.close()

        assert output_path.exists()


class TestDetectorResult:
    """Tests for DetectorResult."""

    def test_empty_result(self):
        """Test empty detection result."""
        result = DetectorResult(
            boxes=np.array([]).reshape(0, 4),
            confidences=np.array([]),
            class_ids=np.array([], dtype=np.int32),
            class_names=[]
        )

        assert len(result.boxes) == 0
        assert len(result.confidences) == 0
        assert len(result.class_ids) == 0
        assert len(result.class_names) == 0

    def test_with_detections(self):
        """Test detection result with boxes."""
        result = DetectorResult(
            boxes=np.array([[10, 20, 100, 80]], dtype=np.float32),
            confidences=np.array([0.95]),
            class_ids=np.array([0]),
            class_names=["person"]
        )

        assert len(result.boxes) == 1
        assert len(result.confidences) == 1
        assert result.class_names[0] == "person"
