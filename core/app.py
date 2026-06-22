"""
Application orchestrator.

Owns the worker thread that drives the pop → process → display pipeline.
Wires RTSPStream, FrameQueue, FrameProcessor, and Display together.

Thread model:
    main thread    — installs signal handlers, blocks on _stop_event
    rtsp-capture   — RTSPStream; continuously pushes raw frames into FrameQueue
    frame-worker   — App; pops frames, processes them, and displays the result

Shutdown paths:
    1. Operator presses 'q' → worker calls _initiate_shutdown()
    2. SIGINT / SIGTERM     → _on_signal()  → _initiate_shutdown()

In both cases _initiate_shutdown() only sets _stop_event (non-blocking).
All resource cleanup (stream.stop, display.close, queue.clear) runs on the
main thread after the worker exits, avoiding nested thread joins.
"""

from __future__ import annotations

import signal
import threading
import time

from core.processor import FrameProcessor
from core.queue_manager import FrameQueue
from core.stream import RTSPStream
from ui.display import Display
from utils.logger import get_logger, log_performance
from utils.time_utils import FPSCounter

logger = get_logger(__name__)

_PERF_LOG_INTERVAL: float = 30.0   # seconds between perf metric snapshots
_WORKER_EXIT_TIMEOUT: float = 5.0  # seconds to wait for worker thread on shutdown
_DROP_RATE_WARN_THRESHOLD: float = 0.05  # warn when > 5 % of frames are evicted


class App:
    def __init__(self) -> None:
        self._queue     = FrameQueue()
        self._stream    = RTSPStream(self._queue)
        self._processor = FrameProcessor()
        self._display   = Display()

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="frame-worker",
            daemon=True,
        )

        self._proc_fps  = FPSCounter()
        self._last_perf_ts: float = 0.0
        self._last_reconnect_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """
        Start all threads, install signal handlers, and block until exit.

        Shutdown sequence (triggered by signal or 'q' keypress):
            1. _stop_event is set  → this method unblocks
            2. Worker thread exits → joined here (5 s timeout)
            3. Stream thread stops → stream.stop() joins it (10 s timeout)
            4. Display and queue are cleaned up
        """
        signal.signal(signal.SIGINT,  self._on_signal)
        signal.signal(signal.SIGTERM, self._on_signal)

        logger.info("Application starting.")
        self._last_perf_ts = time.monotonic()

        self._stream.start()
        self._worker_thread.start()

        # Block main thread until a shutdown is triggered from any path.
        self._stop_event.wait()

        logger.info("Shutdown triggered — waiting for worker thread.")
        self._worker_thread.join(timeout=_WORKER_EXIT_TIMEOUT)
        if self._worker_thread.is_alive():
            logger.warning(
                "Worker thread did not exit within %.0fs.", _WORKER_EXIT_TIMEOUT
            )

        self._stream.stop()     # signals + joins the capture thread (up to 10 s)
        self._display.close()
        self._queue.clear()
        logger.info("Application exited cleanly.")

    # ------------------------------------------------------------------
    # Worker thread — pop → process → display
    # ------------------------------------------------------------------

    def _worker_loop(self) -> None:
        """
        Core pipeline loop, running on the frame-worker thread.

        Each iteration:
            1. Pop a frame from the queue (blocks ≤ 1 s, then re-checks stop flag).
            2. Detect stream reconnects; reset processor state if one occurred.
            3. Pass frame through FrameProcessor.process() (duplicate filter + pipeline).
            4. Display the result; exit if the operator closes the window or presses 'q'.
            5. Emit periodic performance metrics every _PERF_LOG_INTERVAL seconds.
        """
        while not self._stop_event.is_set():
            frame = self._queue.pop(timeout=1.0)
            if frame is None:
                continue

            self._check_reconnect()

            result = self._processor.process(frame)
            if result is None:
                continue

            self._proc_fps.tick()

            if not self._display.show(result):
                logger.info("Operator closed the display window — shutting down.")
                self._initiate_shutdown()
                break   # cleanup is handled in run(), not here

            self._maybe_log_performance()

    def _check_reconnect(self) -> None:
        """
        Reset processor state whenever the stream has reconnected.

        RTSPStream.reconnect_count starts at 0 and is incremented on every
        successful VideoCapture open. Comparing against the last seen value
        lets us detect reconnects without any coupling between the two classes.

        Resetting clears _last_frame so the first frame after a reconnect is
        never wrongly dropped as a duplicate of a frame from the previous session.
        """
        current = self._stream.reconnect_count
        if current != self._last_reconnect_count:
            logger.info(
                "Stream reconnect detected (count=%d) — resetting processor state.",
                current,
            )
            self._processor.reset()
            self._last_reconnect_count = current

    def _maybe_log_performance(self) -> None:
        """
        Write performance metrics to perf.log every _PERF_LOG_INTERVAL seconds.

        Metrics emitted:
            fps   — capture FPS (from RTSPStream) and processing FPS (worker)
            queue — current size, total dropped frames, drop rate

        Also logs a WARNING when the queue drop rate exceeds the threshold,
        which signals that the processor cannot keep up with the capture rate.
        """
        now = time.monotonic()
        if now - self._last_perf_ts < _PERF_LOG_INTERVAL:
            return
        self._last_perf_ts = now

        stats    = self._queue.stats
        cap_fps  = round(self._stream.capture_fps, 1)
        proc_fps = round(self._proc_fps.fps, 1)

        log_performance("fps",   capture=cap_fps, processing=proc_fps)
        log_performance(
            "queue",
            size=stats.current_size,
            dropped=stats.dropped,
            drop_rate=round(stats.drop_rate, 4),
        )

        if stats.drop_rate > _DROP_RATE_WARN_THRESHOLD:
            logger.warning(
                "Queue drop rate %.1f%% exceeds %.0f%% — frames are being evicted "
                "faster than they are processed. Consider raising skip_interval "
                "or lowering fps_limit in config.",
                stats.drop_rate * 100,
                _DROP_RATE_WARN_THRESHOLD * 100,
            )

    # ------------------------------------------------------------------
    # Shutdown helpers
    # ------------------------------------------------------------------

    def _on_signal(self, signum: int, _frame: object) -> None:
        logger.info("Signal %d received.", signum)
        self._initiate_shutdown()

    def _initiate_shutdown(self) -> None:
        """
        Signal all threads to stop. Non-blocking — safe to call from any thread.

        Only sets _stop_event. Resource cleanup (stream.stop, display.close,
        queue.clear) is the responsibility of run() on the main thread, which
        unblocks as soon as _stop_event is set.
        """
        if self._stop_event.is_set():
            return
        logger.info("Initiating shutdown.")
        self._stop_event.set()
