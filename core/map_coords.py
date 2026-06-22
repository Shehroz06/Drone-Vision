"""
Map coordinate transformation.

Converts raw normalised path waypoints from PathPlanner into smooth,
stable map coordinates ready for the API and frontend display.

Pipeline
--------
Raw [x, y] ∈ [0, 1]² from PathPlanner
        │
        ▼
1. Validate       remove NaN / inf; clamp to [0, 1]; drop extreme outliers
        │
        ▼
2. Spatial smooth  Gaussian blur along the path polyline
        │          Removes jaggedness caused by contour noise within one frame.
        ▼
3. Temporal smooth  EMA blend with the previous frame's output
        │           Removes jitter caused by detection variance across frames.
        ▼
4. Scale            multiply by map_width / map_height
        │
        ▼
Smooth [x, y] in map units

Thread-safety
-------------
MapCoordinateTransformer is not thread-safe.  One instance per worker
thread.  Call reset() when the stream reconnects so stale state is not
blended with the first frame of the new session.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["MapCoordinateTransformer", "generate_map_coordinates"]


class MapCoordinateTransformer:
    """
    Stateful coordinate transformer.  Maintains the previous frame's
    smoothed output for temporal EMA blending.

    Args:
        map_width:       Output x-range.  1.0 = keep normalised (Leaflet default).
        map_height:      Output y-range.  1.0 = keep normalised.
        spatial_window:  Half-width of the Gaussian kernel applied along the
                         path.  0 = disabled.  Larger → smoother but more lag.
        temporal_alpha:  EMA weight for the incoming frame.  Range (0, 1].
                         Higher = faster response, less smoothing.
                         Lower  = stronger smoothing, more lag.
                         Rule of thumb: 0.2–0.5 for 20–30 fps streams.
        min_points:      Return [] if fewer points than this arrive.
    """

    def __init__(
        self,
        map_width:       float = 1.0,
        map_height:      float = 1.0,
        spatial_window:  int   = 2,
        temporal_alpha:  float = 0.35,
        min_points:      int   = 1,
    ) -> None:
        if not (0.0 < temporal_alpha <= 1.0):
            raise ValueError(f"temporal_alpha must be in (0, 1], got {temporal_alpha!r}")
        if spatial_window < 0:
            raise ValueError(f"spatial_window must be >= 0, got {spatial_window!r}")

        self._map_w   = float(map_width)
        self._map_h   = float(map_height)
        self._sw      = spatial_window
        self._alpha   = temporal_alpha
        self._min_pts = min_points
        self._prev: Optional[np.ndarray] = None   # previous smoothed coords, [0,1] space

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_map_coordinates(
        self,
        points: list[list[float]],
    ) -> list[list[float]]:
        """
        Transform one frame's raw waypoints into smooth map coordinates.

        Args:
            points: Raw [[x, y], ...] pairs from PathPlanner.plan().
                    Values expected in [0.0, 1.0]; clamped if slightly outside.

        Returns:
            Smoothed [[x, y], ...] in map units ([0, map_width] × [0, map_height]).
            Empty list if points is empty or below min_points threshold.
        """
        if not points or len(points) < self._min_pts:
            self._prev = None          # reset temporal state on no-data frames
            return []

        arr = np.asarray(points, dtype=np.float64)

        arr = self._validate(arr)
        if arr is None:
            return []

        arr = self._spatial_smooth(arr)
        arr = self._temporal_smooth(arr)

        # Store un-scaled coords for next frame's EMA
        self._prev = arr.copy()

        return self._scale(arr)

    def reset(self) -> None:
        """
        Clear temporal state.

        Call whenever the stream reconnects or resumes after a gap.
        Without this, the first valid frame after a reconnect is blended
        with stale coordinates from the previous session.
        """
        self._prev = None
        logger.debug("MapCoordinateTransformer state reset.")

    # ------------------------------------------------------------------
    # Step 1 — Validate
    # ------------------------------------------------------------------

    def _validate(self, arr: np.ndarray) -> Optional[np.ndarray]:
        """
        Remove bad values and clamp to [0, 1].

        Drops points containing NaN or infinity (detection artefacts).
        Points slightly outside [0, 1] (floating-point rounding) are clamped.
        Points far outside [0, 1] (bad detections) trigger a warning.
        """
        # Drop rows with non-finite values
        finite = np.isfinite(arr).all(axis=1)
        if not finite.all():
            n_bad = (~finite).sum()
            logger.warning("Dropping %d point(s) with non-finite values.", n_bad)
            arr = arr[finite]

        if arr.size == 0:
            return None

        # Warn if values are clearly out of bounds (not just floating-point noise)
        _TOLERANCE = 0.05
        out = (~np.all((arr >= -_TOLERANCE) & (arr <= 1.0 + _TOLERANCE), axis=1)).sum()
        if out:
            logger.warning(
                "%d point(s) exceed [0,1] bounds by more than %.0f%% — "
                "check PathPlanner calibration.",
                out, _TOLERANCE * 100,
            )

        return np.clip(arr, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Step 2 — Spatial smooth
    # ------------------------------------------------------------------

    def _spatial_smooth(self, arr: np.ndarray) -> np.ndarray:
        """
        Apply a Gaussian-weighted moving average along the path polyline.

        Each waypoint is blended with its neighbours using a Gaussian kernel.
        Endpoints are padded with their own value (edge padding) so they are
        not pulled toward the interior of the path.

        Requires at least 3 points to be meaningful; skipped otherwise.
        """
        if self._sw <= 0 or len(arr) < 3:
            return arr

        w      = self._sw
        # Build 1-D Gaussian kernel of half-width w, sigma = w/2
        k      = np.arange(-w, w + 1, dtype=np.float64)
        sigma  = max(w / 2.0, 0.5)   # guard against sigma=0 when w=1
        kernel = np.exp(-0.5 * (k / sigma) ** 2)
        kernel /= kernel.sum()

        # Convolve each coordinate axis independently.
        # np.pad(..., mode='edge') replicates the boundary value, which prevents
        # the smoothed path from shrinking at its endpoints.
        smoothed = np.empty_like(arr)
        for dim in range(2):
            padded             = np.pad(arr[:, dim], w, mode='edge')
            smoothed[:, dim]   = np.convolve(padded, kernel, mode='valid')

        return smoothed

    # ------------------------------------------------------------------
    # Step 3 — Temporal smooth
    # ------------------------------------------------------------------

    def _temporal_smooth(self, arr: np.ndarray) -> np.ndarray:
        """
        Blend the current frame with the previous frame using EMA.

            S_t = alpha * X_t  +  (1 - alpha) * S_{t-1}

        alpha = 1.0  → no smoothing (pass-through)
        alpha = 0.2  → heavy smoothing (~14-frame effective window at 25 fps)
        alpha = 0.35 → moderate smoothing (default)

        When the point count changes between frames the history is invalid —
        snap immediately to the new path without blending, then resume EMA.
        This handles the case where the drone detects a different number of
        objects between frames, which is common during occlusion or re-entry.
        """
        if self._prev is None or arr.shape != self._prev.shape:
            # First frame, or count changed — no blending
            return arr

        return self._alpha * arr + (1.0 - self._alpha) * self._prev

    # ------------------------------------------------------------------
    # Step 4 — Scale
    # ------------------------------------------------------------------

    def _scale(self, arr: np.ndarray) -> list[list[float]]:
        """
        Multiply normalised [0, 1] coords by the target map dimensions.

        With map_width=1.0 and map_height=1.0 (defaults) the values remain
        in [0, 1], which is what Leaflet.CRS.Simple expects.

        With map_width=1280 and map_height=720 the output is pixel coordinates
        for a 1280×720 canvas.
        """
        scale  = np.array([self._map_w, self._map_h])
        scaled = arr * scale
        return [[round(float(x), 4), round(float(y), 4)] for x, y in scaled]


# ---------------------------------------------------------------------------
# Module-level default instance + standalone function (smoothing utility)
# ---------------------------------------------------------------------------

# One transformer shared across calls to the standalone function.
# Reset it (e.g. _default_transformer.reset()) when the stream reconnects
# if using the standalone API.
_default_transformer = MapCoordinateTransformer()


def generate_map_coordinates(
    points:          list[list[float]],
    map_width:       float = 1.0,
    map_height:      float = 1.0,
    spatial_window:  int   = 2,
    temporal_alpha:  float = 0.35,
) -> list[list[float]]:
    """
    Convert raw normalised path waypoints into smooth map coordinates.

    Convenience wrapper around MapCoordinateTransformer.  Uses a module-level
    default instance when called with default arguments (temporal smoothing
    persists across calls).  Passing non-default map_width / map_height creates
    a fresh transformer on every call, so temporal smoothing is disabled in
    that case — use MapCoordinateTransformer directly for stateful use.

    Args:
        points:          Raw [[x, y], ...] pairs from PathPlanner.plan().
        map_width:       Scale output x to this range.
        map_height:      Scale output y to this range.
        spatial_window:  Gaussian half-width along path (0 = disabled).
        temporal_alpha:  EMA weight for new data (0 < alpha ≤ 1).

    Returns:
        Smoothed [[x, y], ...] in map units.

    Examples::

        # Keep normalised (Leaflet CRS.Simple)
        coords = generate_map_coordinates(raw_points)

        # Scale to pixel canvas
        coords = generate_map_coordinates(raw_points, map_width=1280, map_height=720)

        # No smoothing
        coords = generate_map_coordinates(raw_points, spatial_window=0, temporal_alpha=1.0)
    """
    using_defaults = (
        map_width      == 1.0
        and map_height == 1.0
        and spatial_window == 2
        and temporal_alpha == 0.35
    )

    if using_defaults:
        return _default_transformer.generate_map_coordinates(points)

    # Non-default config — stateless one-shot transform
    return MapCoordinateTransformer(
        map_width      = map_width,
        map_height     = map_height,
        spatial_window = spatial_window,
        temporal_alpha = temporal_alpha,
    ).generate_map_coordinates(points)
