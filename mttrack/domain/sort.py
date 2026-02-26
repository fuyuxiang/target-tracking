"""
SORT tracker implementation.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from mttrack.domain.kalman import KalmanBoxTracker
from mttrack.domain.tracker import BaseTracker
from mttrack.domain.bytetrack import get_iou_matrix, get_alive_trackers


class SORTTracker(BaseTracker):
    """SORT (Simple Online and Realtime Tracking) algorithm.

    Single-stage IoU association with Kalman filter.
    """

    tracker_id = "sort"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.25,
        minimum_consecutive_frames: int = 3,
        minimum_iou_threshold: float = 0.3,
    ) -> None:
        """Initialize SORT tracker.

        Args:
            lost_track_buffer: Frames to keep lost tracks
            frame_rate: Video frame rate
            track_activation_threshold: Min confidence for new tracks
            minimum_consecutive_frames: Frames to become mature
            minimum_iou_threshold: IoU threshold for matching
        """
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold

        self.tracks: list[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray,
               confidences: np.ndarray = None,
               class_ids: np.ndarray = None) -> np.ndarray:
        """Update tracker with detections.

        Args:
            detections: Array (N, 4) with [x1, y1, x2, y2]
            confidences: Array (N,) of confidence scores
            class_ids: Array (N,) of class IDs

        Returns:
            Array of tracker IDs for each detection
        """
        if len(self.tracks) == 0 and len(detections) == 0:
            return np.array([], dtype=int)

        # Predict
        for tracker in self.tracks:
            tracker.predict()

        # Match
        matched, unmatched_tracks, unmatched_dets = self._match(detections)

        # Update matched tracks
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(detections[det_idx])

        # Spawn new trackers for unmatched detections
        for det_idx in unmatched_dets:
            if confidences is None or confidences[det_idx] >= self.track_activation_threshold:
                new_tracker = KalmanBoxTracker(detections[det_idx])
                self.tracks.append(new_tracker)

        # Prune dead tracks
        self.tracks = get_alive_trackers(
            self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update
        )

        # Build result
        result = self._build_result(detections)
        return result

    def _match(self, detections: np.ndarray
               ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Match detections to tracks using IoU."""
        n_tracks = len(self.tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], set(), set(range(n_dets))
        if n_dets == 0:
            return [], set(range(n_tracks)), set()

        iou_matrix = get_iou_matrix(self.tracks, detections)

        matched = []
        unmatched_tracks = set(range(n_tracks))
        unmatched_dets = set(range(n_dets))

        row_idx, col_idx = linear_sum_assignment(iou_matrix, maximize=True)
        for r, c in zip(row_idx, col_idx):
            if iou_matrix[r, c] >= self.minimum_iou_threshold:
                matched.append((r, c))
                unmatched_tracks.discard(r)
                unmatched_dets.discard(c)

        return matched, unmatched_tracks, unmatched_dets

    def _build_result(self, detections: np.ndarray) -> np.ndarray:
        """Build final result with tracker IDs."""
        result = np.full(len(detections), -1, dtype=int)

        if len(self.tracks) == 0 or len(detections) == 0:
            return result

        iou_matrix = get_iou_matrix(self.tracks, detections)
        row_idx, col_idx = np.where(iou_matrix > self.minimum_iou_threshold)

        sorted_pairs = sorted(
            zip(row_idx, col_idx),
            key=lambda x: iou_matrix[x[0], x[1]],
            reverse=True
        )

        used_tracks = set()
        used_dets = set()
        for trk_idx, det_idx in sorted_pairs:
            if trk_idx in used_tracks or det_idx in used_dets:
                continue

            tracker = self.tracks[trk_idx]
            if tracker.number_of_successful_updates >= self.minimum_consecutive_frames:
                if tracker.tracker_id == -1:
                    tracker.tracker_id = KalmanBoxTracker.get_next_tracker_id()
                result[det_idx] = tracker.tracker_id
                used_tracks.add(trk_idx)
                used_dets.add(det_idx)

        return result

    def reset(self) -> None:
        """Reset tracker."""
        self.tracks = []
        KalmanBoxTracker.count_id = 0

    def get_active_tracks(self) -> list[dict]:
        """Get active tracks."""
        tracks = []
        for tracker in self.tracks:
            if tracker.time_since_update == 0:
                bbox = tracker.get_state_bbox()
                tracks.append({
                    'track_id': tracker.tracker_id,
                    'bbox': bbox.tolist(),
                    'age': tracker.age,
                    'hits': tracker.number_of_successful_updates
                })
        return tracks
