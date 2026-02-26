"""
ByteTrack tracker implementation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

import numpy as np
from scipy.optimize import linear_sum_assignment

from mttrack.domain.kalman import KalmanBoxTracker
from mttrack.domain.tracker import BaseTracker


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two boxes.

    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]

    Returns:
        IoU value
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2

    # Intersection
    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def get_iou_matrix(trackers: list[KalmanBoxTracker],
                   detections: np.ndarray) -> np.ndarray:
    """Build IoU matrix between trackers and detections.

    Args:
        trackers: List of KalmanBoxTracker
        detections: Array of shape (N, 4)

    Returns:
        IoU matrix of shape (len(trackers), N)
    """
    n_tracks = len(trackers)
    n_dets = len(detections)

    if n_tracks == 0 or n_dets == 0:
        return np.zeros((n_tracks, n_dets), dtype=np.float32)

    iou_matrix = np.zeros((n_tracks, n_dets), dtype=np.float32)
    for i, tracker in enumerate(trackers):
        pred_box = tracker.get_state_bbox()
        for j in range(n_dets):
            iou_matrix[i, j] = compute_iou(pred_box, detections[j])

    return iou_matrix


def get_alive_trackers(trackers: list[KalmanBoxTracker],
                       minimum_consecutive_frames: int,
                       maximum_frames_without_update: int) -> list[KalmanBoxTracker]:
    """Get alive trackers after pruning.

    Args:
        trackers: List of trackers
        minimum_consecutive_frames: Min frames to be mature
        maximum_frames_without_update: Max frames without update

    Returns:
        List of alive trackers
    """
    alive = []
    for tracker in trackers:
        is_mature = tracker.number_of_successful_updates >= minimum_consecutive_frames
        is_active = tracker.time_since_update == 0
        if tracker.time_since_update < maximum_frames_without_update and (is_mature or is_active):
            alive.append(tracker)
    return alive


class ByteTrackTracker(BaseTracker):
    """ByteTrack algorithm implementation.

    Two-stage association: first high confidence, then low confidence detections.
    """

    tracker_id = "bytetrack"

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold: float = 0.1,
        high_conf_det_threshold: float = 0.6,
    ) -> None:
        """Initialize ByteTrack tracker.

        Args:
            lost_track_buffer: Frames to keep lost tracks
            frame_rate: Video frame rate
            track_activation_threshold: Min confidence for new tracks
            minimum_consecutive_frames: Frames to become mature
            minimum_iou_threshold: IoU threshold for matching
            high_conf_det_threshold: Threshold separating high/low confidence
        """
        self.maximum_frames_without_update = int(frame_rate / 30.0 * lost_track_buffer)
        self.minimum_consecutive_frames = minimum_consecutive_frames
        self.minimum_iou_threshold = minimum_iou_threshold
        self.track_activation_threshold = track_activation_threshold
        self.high_conf_det_threshold = high_conf_det_threshold

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

        # Split into high and low confidence
        high_det, low_det, high_idx, low_idx = self._split_detections(
            detections, confidences
        )

        updated_tracker_ids = []

        # Step 1: Match high confidence detections
        matched, unmatched_tracks, unmatched_high = self._match(
            high_det, self.tracks
        )

        # Update matched tracks
        for trk_idx, det_idx in matched:
            self.tracks[trk_idx].update(high_det[det_idx])
            # Assign ID if mature
            if (self.tracks[trk_idx].number_of_successful_updates >= self.minimum_consecutive_frames
                    and self.tracks[trk_idx].tracker_id == -1):
                self.tracks[trk_idx].tracker_id = KalmanBoxTracker.get_next_tracker_id()

            updated_tracker_ids.append(self.tracks[trk_idx].tracker_id)

        remaining_tracks = [self.tracks[i] for i in unmatched_tracks]

        # Step 2: Match low confidence detections with remaining tracks
        matched2, unmatched_tracks2, unmatched_low = self._match(
            low_det, remaining_tracks
        )

        for trk_idx, det_idx in matched2:
            tracker = remaining_tracks[trk_idx]
            tracker.update(low_det[det_idx])
            updated_tracker_ids.append(tracker.tracker_id)

        # Handle unmatched high confidence - spawn new trackers
        for det_idx in unmatched_high:
            if confidences is not None and confidences[det_idx] >= self.track_activation_threshold:
                new_tracker = KalmanBoxTracker(detections[det_idx])
                self.tracks.append(new_tracker)

        # Update tracker IDs for unmatched detections
        # Need to map back to original indices
        # For simplicity, assign -1 for unmatched
        result_ids = np.full(len(detections), -1, dtype=int)

        # Rebuild results - this is simplified
        # In practice we'd need to track which detection maps to which

        # Prune dead tracks
        self.tracks = get_alive_trackers(
            self.tracks,
            self.minimum_consecutive_frames,
            self.maximum_frames_without_update
        )

        # Build final result
        final_ids = self._build_result(detections, confidences)
        return final_ids

    def _split_detections(self, detections: np.ndarray,
                          confidences: Optional[np.ndarray]
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split detections into high and low confidence."""
        if confidences is None:
            return detections, np.array([]).reshape(0, 4), np.arange(len(detections)), np.array([], dtype=int)

        high_mask = confidences >= self.high_conf_det_threshold
        low_mask = ~high_mask

        high_idx = np.where(high_mask)[0]
        low_idx = np.where(low_mask)[0]

        return detections[high_mask], detections[low_mask], high_idx, low_idx

    def _match(self, detections: np.ndarray,
                tracks: list[KalmanBoxTracker]
                ) -> tuple[list[tuple[int, int]], set[int], set[int]]:
        """Match detections to tracks using IoU."""
        n_tracks = len(tracks)
        n_dets = len(detections)

        if n_tracks == 0:
            return [], set(), set(range(n_dets))
        if n_dets == 0:
            return [], set(range(n_tracks)), set()

        iou_matrix = get_iou_matrix(tracks, detections)

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

    def _build_result(self, detections: np.ndarray,
                       confidences: Optional[np.ndarray]) -> np.ndarray:
        """Build final result with tracker IDs."""
        result = np.full(len(detections), -1, dtype=int)

        # Re-match for final result
        if len(self.tracks) == 0 or len(detections) == 0:
            return result

        iou_matrix = get_iou_matrix(self.tracks, detections)
        row_idx, col_idx = np.where(iou_matrix > self.minimum_iou_threshold)

        # Sort by IoU
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
