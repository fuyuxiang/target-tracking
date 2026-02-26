"""
Kalman Box Tracker implementation.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class KalmanBoxTracker:
    """Kalman filter for tracking a single bounding box.

    State vector: [cx, cy, w, h, vx, vy]
    - cx, cy: center coordinates
    - w, h: width and height
    - vx, vy: velocity
    """

    count_id: int = 0

    def __init__(self, bbox: tuple[float, float, float, float]) -> None:
        """Initialize tracker with bounding box.

        Args:
            bbox: Initial bounding box (x1, y1, x2, y2)
        """
        self.tracker_id: int = -1
        self.number_of_successful_updates: int = 1
        self.time_since_update: int = 0

        # State vector: [cx, cy, w, h, vx, vy]
        self.state: NDArray[np.float32] = np.zeros((6, 1), dtype=np.float32)
        self._init_state(bbox)

        # Kalman matrices
        self._init_kalman()

    def _init_state(self, bbox: tuple[float, float, float, float]) -> None:
        """Initialize state from bounding box."""
        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1
        self.state = np.array([[cx], [cy], [w], [h], [0.0], [0.0]], dtype=np.float32)

    def _init_kalman(self) -> None:
        """Initialize Kalman filter matrices."""
        n_state = 6  # [cx, cy, w, h, vx, vy]
        n_obs = 4    # [cx, cy, w, h]
        dt = 1.0

        # State transition matrix (F)
        self.F: NDArray[np.float32] = np.eye(n_state, dtype=np.float32)
        self.F[0, 4] = dt  # cx += vx * dt
        self.F[1, 5] = dt  # cy += vy * dt

        # Observation matrix (H)
        self.H: NDArray[np.float32] = np.zeros((n_obs, n_state), dtype=np.float32)
        for i in range(n_obs):
            self.H[i, i] = 1

        # State covariance (P)
        self.P: NDArray[np.float32] = np.eye(n_state, dtype=np.float32) * 10

        # Process noise (Q)
        self.Q: NDArray[np.float32] = np.eye(n_state, dtype=np.float32) * 0.1
        self.Q[4, 4] = 0.5  # vx noise
        self.Q[5, 5] = 0.5  # vy noise

        # Observation noise (R)
        self.R: NDArray[np.float32] = np.eye(n_obs, dtype=np.float32) * 1

    @classmethod
    def get_next_tracker_id(cls) -> int:
        """Get next unique tracker ID."""
        next_id = cls.count_id
        cls.count_id += 1
        return next_id

    def predict(self) -> None:
        """Predict next state."""
        # Predict state
        self.state = (self.F @ self.state).astype(np.float32)
        # Predict covariance
        self.P = (self.F @ self.P @ self.F.T + self.Q).astype(np.float32)
        # Update age
        self.time_since_update += 1

    def update(self, bbox: tuple[float, float, float, float]) -> None:
        """Update state with new detection.

        Args:
            bbox: New bounding box (x1, y1, x2, y2)
        """
        self.time_since_update = 0
        self.number_of_successful_updates += 1

        x1, y1, x2, y2 = bbox
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        w, h = x2 - x1, y2 - y1

        # Kalman update
        z = np.array([[cx], [cy], [w], [h]], dtype=np.float32)
        y_res = z - self.H @ self.state
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.state = (self.state + K @ y_res).astype(np.float32)
        self.P = ((np.eye(6) - K @ self.H) @ self.P).astype(np.float32)

    def get_state_bbox(self) -> np.ndarray:
        """Get current bounding box estimate.

        Returns:
            Bounding box [x1, y1, x2, y2]
        """
        cx = float(self.state[0, 0])
        cy = float(self.state[1, 0])
        w = float(self.state[2, 0])
        h = float(self.state[3, 0])

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        return np.array([x1, y1, x2, y2], dtype=np.float32)
