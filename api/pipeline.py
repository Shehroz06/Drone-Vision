"""
Pipeline orchestrator for the API server.

Owns and manages the three-stage background pipeline:

    [rtsp-capture thread]         [pipeline-worker thread]
    ─────────────────────         ────────────────────────
    RTSPStream._capture_loop()    PipelineManager._run()
         │                             │
     cap.read()                   queue.pop(timeout=1s)
         │                             │
     queue.push(frame)            _check_reconnect() → processor.reset()
                                       │
                                  processor.process(frame)      ← duplicate filter + steps
                                       │
                                  planner.plan(frame)           ← extract pixel waypoints
                                       │
                                  store.push(points)            ← thread-safe; broadcasts to WS

Lifecycle
---------
    pipeline = PipelineManager()
    pipeline.start()    # called from FastAPI lifespan startup
    ...
    pipeline.stop()     # called from FastAPI lifespan shutdown

The display (cv2.imshow) is intentionally absent — this is headless API mode.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

from api.state import store
from core.path_planning import PathPlanner
from core.processor import FrameProcessor
from core.queue_manager import FrameQueue
from core.stream import RTSPStream
from utils.logger import get_logger, log_performance
from utils.time_utils import FPSCounter

logger = get_logger(__name__)

_PERF_LOG_INTERVAL:  float = 30.0   # seconds between perf.log snapshots
_QUEUE_LOG_INTERVAL: float =  5.0   # seconds between live queue/fps console prints
_WORKER_JOIN_TIMEOUT: float = 5.0   # seconds to wait for worker thread on shutdown
_DROP_RATE_WARN:      float = 0.50  # queue_size=1 is designed to evict; warn only above 50%


class PipelineMetrics:
    """Point-in-time snapshot of pipeline health. Used by GET /status."""

    __slots__ = (
        "capture_fps", "processing_fps",
        "queue_size", "queue_max", "drop_rate", "dropped_total",
        "reconnect_count", "last_update",
    )

    def __init__(
        self,
        capture_fps:    float,
        processing_fps: float,
        queue_size:     int,
        queue_max:      int,
        drop_rate:      float,
        dropped_total:  int,
        reconnect_count: int,
        last_update:    Optional[str],
    ) -> None:
        self.capture_fps     = capture_fps
        self.processing_fps  = processing_fps
        self.queue_size      = queue_size
        self.queue_max       = queue_max
        self.drop_rate       = drop_rate
        self.dropped_total   = dropped_total
        self.reconnect_count = reconnect_count
        self.last_update     = last_update

    def as_dict(self) -> dict:
        return {
            "capture_fps":     self.capture_fps,
            "processing_fps":  self.processing_fps,
            "queue_size":      self.queue_size,
            "queue_max":       self.queue_max,
            "drop_rate":       round(self.drop_rate, 4),
            "dropped_total":   self.dropped_total,
            "reconnect_count": self.reconnect_count,
            "last_update":     self.last_update,
        }


class PipelineManager:
    """
    Headless frame-processing pipeline for the API server.

    Wires RTSPStream → FrameQueue → FrameProcessor → PathPlanner → PathStore.
    Owns the 'pipeline-worker' background thread. Thread-safe.

    Usage::

        manager = PipelineManager()
        manager.start()
        ...
        manager.stop()
    """

    def __init__(self) -> None:
        self._queue     = FrameQueue()
        self._stream    = RTSPStream(self._queue)
        self._processor = FrameProcessor(skip_interval=2)   # process 1 of every 2 frames
        self._planner   = PathPlanner()
        self._proc_fps  = FPSCounter()

        # Telemetry — updated externally via update_telemetry()
        self._lat: Optional[float] = None
        self._lon: Optional[float] = None

        self._stop_event = threading.Event()
        self._worker = threading.Thread(
            target=self._run,
            name="pipeline-worker",
            daemon=True,
        )

        # Mutable state — only read/written inside the worker thread
        # (except _last_perf_ts which is set in start() on the main thread
        #  before the worker begins — no race).
        self._last_reconnect:    int            = 0
        self._last_perf_ts:      float          = 0.0
        self._last_queue_log_ts: float          = 0.0
        self._last_tele_log_ts:  float          = 0.0
        # Last telemetry tuple sent to store.push(); None means not yet sent.
        # Prevents broadcasting identical telemetry on every processed frame
        # (pipeline runs at ~15 fps; OCR updates telemetry every 2–9 s).
        self._last_sent_tele:    Optional[tuple] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        Start the capture thread and the pipeline worker.

        Non-blocking — returns as soon as both threads are scheduled.
        Call from the FastAPI lifespan startup handler.
        """
        logger.info("Pipeline starting.")
        self._last_perf_ts = time.monotonic()
        self._stream.start()
        self._worker.start()
        logger.info(
            "Pipeline running — threads: [rtsp-capture] [pipeline-worker]"
        )

    def stop(self) -> None:
        """
        Signal all threads to stop and wait for them to exit.

        Blocking — waits up to 5 s for the worker, then up to 10 s for the
        capture thread. Call from the FastAPI lifespan shutdown handler.
        """
        logger.info("Pipeline stopping.")
        self._stop_event.set()

        self._worker.join(timeout=_WORKER_JOIN_TIMEOUT)
        if self._worker.is_alive():
            logger.warning(
                "Worker did not exit within %.0fs — continuing shutdown.",
                _WORKER_JOIN_TIMEOUT,
            )

        self._stream.stop()    # signals and joins rtsp-capture (up to 10 s)
        self._queue.clear()    # release frame memory
        logger.info("Pipeline stopped.")

    # ------------------------------------------------------------------
    # Telemetry (call from your sensor thread — thread-safe primitive write)
    # ------------------------------------------------------------------

    def update_telemetry(
        self,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
    ) -> None:
        """
        Update lat/lon from the OCR worker thread.

        Float/None assignment is atomic in CPython — no lock required.
        """
        self._lat = lat
        self._lon = lon

    # ------------------------------------------------------------------
    # Metrics (called from GET /status — any thread, any time)
    # ------------------------------------------------------------------

    def metrics(self) -> PipelineMetrics:
        """Return a consistent snapshot of pipeline health."""
        stats = self._queue.stats
        latest = store.get_latest()
        return PipelineMetrics(
            capture_fps     = round(self._stream.capture_fps, 1),
            processing_fps  = round(self._proc_fps.fps, 1),
            queue_size      = stats.current_size,
            queue_max       = stats.max_size,
            drop_rate       = stats.drop_rate,
            dropped_total   = stats.dropped,
            reconnect_count = self._stream.reconnect_count,
            last_update     = latest["timestamp"] if latest else None,
        )

    # ------------------------------------------------------------------
    # Worker thread — pipeline loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        """
        Core pipeline loop.  Runs on the 'pipeline-worker' thread.

        Each iteration:
            1. Pop a frame from the queue (blocks ≤ 1 s; loop re-checks stop flag)
            2. Detect stream reconnect → reset processor state
            3. FrameProcessor.process() — duplicate filter + registered steps
            4. PathPlanner.plan()       — extract waypoints from the result frame
            5. store.push()             — update REST snapshot + broadcast to WebSockets
            6. Emit perf metrics every _PERF_LOG_INTERVAL seconds
        """
        logger.info("Pipeline worker started.")

        while not self._stop_event.is_set():

            # ── 1. Pop ───────────────────────────────────────────────────────
            frame = self._queue.pop(timeout=1.0)
            if frame is None:
                continue   # timed out; loop back to check stop flag

            # ── Queue diagnostic — runs unconditionally so it fires even when
            #    frames are being skipped by FrameProcessor.
            now = time.monotonic()
            if now - self._last_queue_log_ts >= _QUEUE_LOG_INTERVAL:
                self._last_queue_log_ts = now
                self._log_queue()

            # ── 2. Reconnect detection ───────────────────────────────────────
            # reconnect_count starts at 0, increments on each successful open().
            # Comparing against our local last_reconnect detects every reconnect
            # without coupling RTSPStream to FrameProcessor.
            rc = self._stream.reconnect_count
            if rc != self._last_reconnect:
                logger.info(
                    "Stream reconnect #%d — resetting processor state.", rc
                )
                self._processor.reset()
                self._last_reconnect = rc

            # ── 3. Process ───────────────────────────────────────────────────
            # Returns None for duplicates, skipped frames, or step failures.
            result = self._processor.process(frame)
            if result is None:
                continue

            # ── 4. Plan path ─────────────────────────────────────────────────
            # PathPlanner.plan() is exception-safe; returns [] on any failure.
            # Output is raw pixel [x, y] pairs — geo conversion happens upstream.
            points = self._planner.plan(result)

            # ── 5. Publish ───────────────────────────────────────────────────
            # Only broadcast when telemetry actually changed.  Without this
            # guard, store.push() fires on every processed frame (~15/s) and
            # WebSocket clients receive 45–135 identical messages between OCR
            # updates.  Points are also included in the key so a path change
            # alone still triggers a push even with static telemetry.
            tele_key = (self._lat, self._lon)
            if tele_key != self._last_sent_tele:
                self._last_sent_tele = tele_key
                self._proc_fps.tick()
                store.push(points=points, lat=self._lat, lon=self._lon)

            # ── Telemetry console log (once per second, only when lat/lon known)
            now = time.monotonic()
            if self._lat is not None and now - self._last_tele_log_ts >= 1.0:
                self._last_tele_log_ts = now
                print(f"[pipeline] lat={self._lat:.6f}  lon={self._lon:.6f}")

            # ── 6. Performance metrics ───────────────────────────────────────
            now = time.monotonic()
            if now - self._last_perf_ts >= _PERF_LOG_INTERVAL:
                self._last_perf_ts = now
                self._emit_perf()

        logger.info("Pipeline worker stopped.")

    def _log_queue(self) -> None:
        """Print a one-line queue health summary to stdout every 5 s."""
        stats    = self._queue.stats
        cap_fps  = self._stream.capture_fps
        proc_fps = self._proc_fps.fps
        rate_pct = stats.drop_rate * 100

        flag = "  *** DROP RATE HIGH ***" if stats.drop_rate > _DROP_RATE_WARN else ""
        print(
            f"[queue] size={stats.current_size}/{stats.max_size}"
            f"  capture={cap_fps:.1f}fps  processing={proc_fps:.1f}fps"
            f"  dropped={stats.dropped}  rate={rate_pct:.1f}%{flag}"
        )

    def _emit_perf(self) -> None:
        """Write a performance snapshot to perf.log; warn on high drop rate."""
        stats = self._queue.stats

        log_performance(
            "fps",
            capture    = round(self._stream.capture_fps, 1),
            processing = round(self._proc_fps.fps, 1),
        )
        log_performance(
            "queue",
            size      = stats.current_size,
            dropped   = stats.dropped,
            drop_rate = round(stats.drop_rate, 4),
        )

        if stats.drop_rate > _DROP_RATE_WARN:
            logger.warning(
                "Queue drop rate %.1f%% exceeds %.0f%% — raise skip_interval "
                "or lower fps_limit in config.",
                stats.drop_rate * 100,
                _DROP_RATE_WARN * 100,
            )
