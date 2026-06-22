"""
Frame processing module.

Pure processing logic — no threads, no queue, no network I/O.
Receives one frame at a time via process(), runs it through a composable
pipeline of steps, and returns the result or None if the frame is dropped.

Threading and queue interaction belong in the caller (core/app.py).
This class is deliberately oblivious to both.

Fix in this version:
    - _should_skip() off-by-one corrected so the FIRST frame is always
      processed, matching the intuitive meaning of skip_interval.
"""

from __future__ import annotations

from typing import Callable, List, Optional

import numpy as np

from utils.frame_utils import is_duplicate_frame
from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["FrameProcessor"]

ProcessingStep = Callable[[np.ndarray], Optional[np.ndarray]]


class FrameProcessor:
    """
    Composable frame processor built around a sequential pipeline.

    Each frame passes through three layers in order:

        1. Gate checks  — skip_interval, duplicate detection (built-in)
        2. _apply()     — primary override point for subclasses
        3. Pipeline     — ordered steps registered via add_step()

    Any layer may return ``None`` to drop the frame; subsequent layers
    are not called.

    **Basic usage**::

        processor = FrameProcessor()
        result = processor.process(frame)          # pass-through

    **With registered steps**::

        processor = FrameProcessor(skip_interval=2)
        processor.add_step(resize_fn).add_step(detect_fn)
        result = processor.process(frame)

    **Via subclass** (AI inference example)::

        class YOLOProcessor(FrameProcessor):
            def __init__(self):
                super().__init__(skip_interval=3)
                self._model = load_model(...)

            def _apply(self, frame: np.ndarray) -> Optional[np.ndarray]:
                return self._model.infer(frame)

    Pipeline steps are callables ``(np.ndarray) -> Optional[np.ndarray]``.
    Returning ``None`` signals "drop this frame."
    """

    def __init__(self, skip_interval: int = 1) -> None:
        """
        Args:
            skip_interval:  Process 1 out of every N frames.  1 = every frame.
                            The FIRST frame received is always processed.
                            Frames 2, 3, … (N-1) are skipped; frame N is processed, etc.
        """
        if skip_interval < 1:
            raise ValueError(f"skip_interval must be >= 1, got {skip_interval!r}")

        self._skip_interval: int = skip_interval
        self._steps: List[ProcessingStep] = []
        self._last_frame: Optional[np.ndarray] = None
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Run a single frame through the full processing pipeline.

        Returns:
            np.ndarray  — processed frame ready for display or storage.
            None        — frame was dropped (skipped, duplicate, or a pipeline
                          step returned None).
        """
        self._frame_count += 1

        if self._should_skip():
            logger.debug(
                "Frame #%d skipped (interval=%d).", self._frame_count, self._skip_interval
            )
            return None

        if is_duplicate_frame(frame, self._last_frame):
            logger.debug("Frame #%d dropped — duplicate.", self._frame_count)
            return None

        # Store the raw input so duplicate detection compares source content,
        # not whatever an annotation step may have drawn on top of the frame.
        self._last_frame = frame

        return self._run_pipeline(frame)

    def add_step(self, step: ProcessingStep) -> "FrameProcessor":
        """
        Append a processing step to the end of the pipeline.

        Returns ``self`` to support chaining::

            processor.add_step(resize).add_step(detect).add_step(annotate)
        """
        self._steps.append(step)
        logger.debug(
            "Pipeline step registered: %s (total=%d)",
            getattr(step, "__name__", repr(step)),
            len(self._steps),
        )
        return self

    def reset(self) -> None:
        """
        Reset internal state: frame counter and last-frame reference.

        Call when the stream reconnects or resumes after a pause.
        Without this, the first frame back may be wrongly dropped as a
        duplicate if it matches the last stored frame from before the drop.
        """
        self._last_frame = None
        self._frame_count = 0
        logger.debug("FrameProcessor state reset.")

    @property
    def skip_interval(self) -> int:
        return self._skip_interval

    @skip_interval.setter
    def skip_interval(self, value: int) -> None:
        """Adjust skip interval at runtime without restarting the processor."""
        if value < 1:
            raise ValueError(f"skip_interval must be >= 1, got {value!r}")
        self._skip_interval = value
        logger.info("skip_interval updated to %d.", value)

    @property
    def step_count(self) -> int:
        return len(self._steps)

    @property
    def frames_seen(self) -> int:
        """Total frames passed to process(), including skipped and dropped."""
        return self._frame_count

    # ------------------------------------------------------------------
    # Subclass override point
    # ------------------------------------------------------------------

    def _apply(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Primary hook for subclass-owned processing logic.

        Called after gate checks and before registered pipeline steps.
        Default is a pass-through. Override to inject inference, mapping,
        or transformation without registering an external step.

        Returning ``None`` drops the frame; pipeline steps are not called.
        """
        return frame

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_pipeline(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Apply _apply() then each registered step in order."""
        result: Optional[np.ndarray] = self._safe_call(self._apply, frame, "_apply")
        if result is None:
            return None

        for step in self._steps:
            result = self._safe_call(
                step, result, label=getattr(step, "__name__", repr(step))
            )
            if result is None:
                return None

        return result

    def _should_skip(self) -> bool:
        """
        Return True when this frame should be skipped.

        Logic: process frames 1, 1+N, 1+2N, ... (first frame always processed).
        Equivalent to: skip when (frame_count - 1) % skip_interval != 0.

        Example with skip_interval=3:
            frame #1 → (0 % 3 == 0) → PROCESS
            frame #2 → (1 % 3 != 0) → skip
            frame #3 → (2 % 3 != 0) → skip
            frame #4 → (3 % 3 == 0) → PROCESS
        """
        return (
            self._skip_interval > 1
            and (self._frame_count - 1) % self._skip_interval != 0
        )

    @staticmethod
    def _safe_call(
        fn: ProcessingStep,
        frame: np.ndarray,
        label: str,
    ) -> Optional[np.ndarray]:
        """
        Call a pipeline step with per-step exception isolation.

        A bug in one step must not crash the loop for subsequent frames.
        On exception: logs ERROR, drops the frame, continues.
        """
        try:
            return fn(frame)
        except Exception as exc:
            logger.error(
                "Pipeline step '%s' raised %s: %s — frame dropped.",
                label,
                type(exc).__name__,
                exc,
                exc_info=True,
            )
            return None
