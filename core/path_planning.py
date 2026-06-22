"""
Path planning module.

Receives a processed BGR frame from FrameProcessor and extracts waypoints —
a list of (x, y) coordinate pairs that describe the detected path or object
positions in the scene.

This module is deliberately decoupled from streaming and threading.
It owns only the algorithm: frame in, coordinates out.

Swap PathPlanner._extract() for your own detection model (YOLO, homography,
lane detection, etc.) without touching any other module.
"""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["PathPlanner"]

class PathPlanner:
    """
    Extracts path waypoints from a processed video frame.

    Algorithm (default):
        1. Convert BGR → greyscale
        2. Gaussian blur to suppress noise
        3. Otsu adaptive threshold → binary mask
        4. Find external contours
        5. Filter by minimum pixel area
        6. Compute centroid of each qualifying contour
        7. Sort centroids left → right (ascending x)
        8. Return up to max_points pixel [x, y] pairs

    Replace _extract() to plug in any detection model without changing
    the interface or the rest of the pipeline.

    Args:
        min_area:   Minimum contour area in pixels. Smaller contours (noise)
                    are ignored. Tune to your camera resolution and target size.
        max_points: Hard cap on returned waypoints to keep API responses bounded.
    """

    def __init__(
        self,
        min_area:   float = 400.0,
        max_points: int   = 64,
    ) -> None:
        if min_area <= 0:
            raise ValueError(f"min_area must be > 0, got {min_area!r}")
        if max_points < 1:
            raise ValueError(f"max_points must be >= 1, got {max_points!r}")

        self._min_area   = min_area
        self._max_points = max_points

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def plan(self, frame: np.ndarray) -> list[list[float]]:
        """
        Extract path waypoints from a processed frame.

        Args:
            frame: BGR numpy array from FrameProcessor.process().
                   Must be non-empty.

        Returns:
            List of [x, y] pixel coordinate pairs.  Origin is top-left,
            x increases right, y increases downward.
            Empty list if no features found.
        """
        if frame is None or frame.size == 0:
            logger.warning("PathPlanner.plan() received an empty frame — skipping.")
            return []

        try:
            return self._extract(frame)
        except Exception as exc:
            logger.error(
                "PathPlanner._extract() raised %s: %s — returning [].",
                type(exc).__name__, exc, exc_info=True,
            )
            return []

    # ------------------------------------------------------------------
    # Override point — swap this method for your detection model
    # ------------------------------------------------------------------

    def _extract(self, frame: np.ndarray) -> list[list[float]]:
        """
        Core extraction algorithm.

        Returns pixel [x, y] waypoints sorted left → right.

        To integrate a different model::

            class YOLOPlanner(PathPlanner):
                def __init__(self):
                    super().__init__()
                    self._model = load_yolo("weights.pt")

                def _extract(self, frame):
                    detections = self._model(frame)
                    return [[det.cx, det.cy] for det in detections]
        """
        h, w = frame.shape[:2]

        # ── 1. Greyscale ────────────────────────────────────────────────
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── 2. Blur — 5×5 kernel removes single-pixel noise ─────────────
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # ── 3. Otsu threshold — automatic on bimodal histograms ──────────
        _, thresh = cv2.threshold(
            blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # ── 4. External contours only ────────────────────────────────────
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # ── 5–6. Filter by area; compute centroid ────────────────────────
        points: list[list[float]] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self._min_area:
                continue

            M = cv2.moments(c)
            if M["m00"] == 0:
                continue

            cx = round(M["m10"] / M["m00"], 1)
            cy = round(M["m01"] / M["m00"], 1)
            points.append([cx, cy])

        # ── 7. Sort left → right for a coherent path ordering ────────────
        points.sort(key=lambda p: p[0])

        # ── 8. Cap to max_points ──────────────────────────────────────────
        result = points[: self._max_points]

        logger.debug(
            "PathPlanner: %d contour(s) found, %d passed area filter, %d returned.",
            len(contours), len(points), len(result),
        )
        return result
