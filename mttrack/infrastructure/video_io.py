"""
Video reading and writing utilities.
"""

from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


class VideoReader:
    """Video reader with frame iteration."""

    def __init__(self, source: str | int | Path) -> None:
        """Initialize video reader.

        Args:
            source: Video file path, stream URL, or camera index
        """
        self.source = str(source) if isinstance(source, Path) else str(source)
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_count: int = 0
        self.fps: float = 0.0
        self.width: int = 0
        self.height: int = 0

    def __enter__(self) -> "VideoReader":
        """Open video capture."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Cannot open video: {self.source}")

        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return self

    def __exit__(self, *_) -> None:
        """Release video capture."""
        if self.cap:
            self.cap.release()

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames.

        Yields:
            (frame_id, frame) tuples
        """
        frame_id = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_id += 1
            yield frame_id, frame

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame.

        Returns:
            (success, frame) tuple
        """
        if self.cap is None:
            return False, None
        return self.cap.read()


class VideoWriter:
    """Video writer."""

    def __init__(
        self,
        output_path: str | Path,
        fps: float = 30.0,
        frame_size: Optional[tuple[int, int]] = None,
        codec: str = "mp4v",
    ) -> None:
        """Initialize video writer.

        Args:
            output_path: Output video path
            fps: Frames per second
            frame_size: (width, height) - required if not writing frames immediately
            codec: Video codec
        """
        self.output_path = Path(output_path)
        self.fps = fps
        self.frame_size = frame_size
        self.codec = codec
        self.writer: Optional[cv2.VideoWriter] = None
        self._is_open = False

    def __enter__(self) -> "VideoWriter":
        """Enter context manager."""
        return self

    def __exit__(self, *_) -> None:
        """Release video writer."""
        self.close()

    def write(self, frame: np.ndarray) -> None:
        """Write a frame.

        Args:
            frame: BGR frame
        """
        if not self._is_open:
            self._init_writer(frame)

        if self.writer is not None:
            self.writer.write(frame)

    def _init_writer(self, frame: np.ndarray) -> None:
        """Initialize writer with first frame."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        height, width = frame.shape[:2]
        if self.frame_size:
            width, height = self.frame_size

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height)
        )

        if not self.writer.isOpened():
            raise OSError(f"Failed to open video writer: {self.output_path}")

        self._is_open = True

    def close(self) -> None:
        """Close video writer."""
        if self.writer:
            self.writer.release()
            self.writer = None
        self._is_open = False


def create_video_writer(
    output_path: str | Path,
    fps: float = 30.0,
    frame_size: Optional[tuple[int, int]] = None,
) -> VideoWriter:
    """Create a video writer.

    Args:
        output_path: Output video path
        fps: Frames per second
        frame_size: (width, height)

    Returns:
        VideoWriter instance
    """
    return VideoWriter(output_path, fps, frame_size)
