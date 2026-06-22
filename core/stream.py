"""
RTSP stream capture module.

Single responsibility: open an RTSP connection, read frames continuously
on a daemon thread, and push raw BGR frames into a queue. All reconnect
logic lives here. Frame processing is explicitly out of scope.

Changes from initial version:
    - fps_limit throttle in _read_loop via stop_event.wait() (non-blocking sleep)
    - reconnect_count property so App can detect reconnects and reset the processor
    - capture_fps property exposing a rolling average from FPSCounter
    - join() public method — callers no longer need to touch _thread directly
    - read_frame() raises RuntimeError when the thread is live (thread-safety)
    - _redact_url() masks credentials in every log message
"""

from __future__ import annotations

import threading
import time
from typing import Optional, Protocol, Tuple
from urllib.parse import urlparse, urlunparse

import cv2
import numpy as np

from config.constants import MAX_RECONNECT_ATTEMPTS
from config.settings import settings
from utils.logger import get_logger
from utils.time_utils import FPSCounter

logger = get_logger(__name__)

# OPENCV_FFMPEG_CAPTURE_OPTIONS is set centrally in api/main.py before any
# VideoCapture is created.  Do NOT set it here — a module-level setdefault
# would race against (and win over) the full flag set in main.py.

__all__ = ["RTSPStream"]


def _redact_url(url: str) -> str:
    """Replace any password in an RTSP URL with *** for safe log output."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            host = f"{parsed.hostname}:{parsed.port}" if parsed.port else (parsed.hostname or "")
            return urlunparse(parsed._replace(netloc=f"{parsed.username}:***@{host}"))
    except Exception:
        pass
    return url


class FrameQueue(Protocol):
    """
    Structural interface the queue must satisfy.
    Depends on Protocol so RTSPStream and QueueManager are fully decoupled.
    """
    def push(self, frame: np.ndarray) -> None: ...


class RTSPStream:
    """
    Captures frames from an RTSP source on a dedicated daemon thread.

    Lifecycle::

        stream = RTSPStream(queue_manager)
        stream.start()   # returns immediately
        ...
        stream.stop()    # signals thread and blocks until it exits
        stream.join()    # idempotent; safe to call again after stop()
    """

    _STOP_JOIN_TIMEOUT: float = 10.0

    def __init__(
        self,
        queue_manager: FrameQueue,
        url: Optional[str] = None,
    ) -> None:
        self._queue = queue_manager
        self._url: str = url or settings.rtsp_url
        self._stop_event = threading.Event()
        self._cap: Optional[cv2.VideoCapture] = None
        self._cap_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="rtsp-capture",
            daemon=True,
        )
        # Incremented on each successful VideoCapture open.
        # App polls this to detect reconnects and call processor.reset().
        self._reconnect_count: int = 0
        self._capture_fps = FPSCounter()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the capture thread. Returns immediately (non-blocking)."""
        if self._thread.is_alive():
            logger.warning("start() called on an already-running stream — ignored.")
            return
        self._stop_event.clear()
        logger.info("Starting RTSP capture — url=%s", _redact_url(self._url))
        self._thread.start()

    def stop(self) -> None:
        """Signal the capture thread to exit and wait for it to finish."""
        logger.info("Stop requested — signalling capture thread.")
        self._stop_event.set()
        self._thread.join(timeout=self._STOP_JOIN_TIMEOUT)
        if self._thread.is_alive():
            logger.error(
                "Capture thread did not exit within %.0fs — possible deadlock.",
                self._STOP_JOIN_TIMEOUT,
            )
        self._release_cap()
        logger.info("RTSP capture stopped.")

    def join(self) -> None:
        """
        Block until the capture thread exits.

        Provided so callers never need to access ``self._thread`` directly.
        Safe to call multiple times and after stop() has already joined.
        """
        self._thread.join()

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read one frame directly from the active VideoCapture.

        For test and one-shot use only.

        Raises:
            RuntimeError: if called while the capture thread is running.
                          Both paths would share the same VideoCapture object,
                          which is not thread-safe.
        """
        if self.is_running:
            raise RuntimeError(
                "read_frame() must not be called while the capture thread is running."
            )
        with self._cap_lock:
            if self._cap is None or not self._cap.isOpened():
                return False, None
            ret, frame = self._cap.read()
            return (ret, frame) if ret else (False, None)

    @property
    def is_running(self) -> bool:
        """True while the thread is alive and the stop flag has not been set."""
        return self._thread.is_alive() and not self._stop_event.is_set()

    @property
    def reconnect_count(self) -> int:
        """
        Number of successful RTSP connections made so far.

        Starts at 0 (not yet connected). Becomes 1 on first connect, 2 on
        the first reconnect, and so on. The worker thread in App compares
        this against its last-seen value to detect reconnects.
        """
        return self._reconnect_count

    @property
    def capture_fps(self) -> float:
        """Rolling average capture FPS over the last FPS_SAMPLE_WINDOW frames."""
        return self._capture_fps.fps

    # ------------------------------------------------------------------
    # Capture thread — outer reconnect loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        """
        Outer reconnect loop that wraps the inner frame-read loop.

        On failure: waits with exponential backoff then retries. Never gives
        up permanently — logs CRITICAL after MAX_RECONNECT_ATTEMPTS consecutive
        failures but keeps retrying so the system recovers if the network returns.
        """
        consecutive_failures = 0

        while not self._stop_event.is_set():
            cap = self._connect()

            if cap is None:
                consecutive_failures += 1
                if consecutive_failures == MAX_RECONNECT_ATTEMPTS:
                    logger.critical(
                        "Stream unreachable after %d consecutive attempts — "
                        "still retrying. Verify RTSP URL and network connectivity.",
                        consecutive_failures,
                    )
                delay = self._backoff_delay(consecutive_failures)
                logger.info(
                    "Will retry in %.1fs (failure #%d).", delay, consecutive_failures
                )
                self._stop_event.wait(timeout=delay)
                continue

            consecutive_failures = 0
            self._reconnect_count += 1
            logger.info(
                "RTSP stream connected: %s (session #%d)",
                _redact_url(self._url),
                self._reconnect_count,
            )

            with self._cap_lock:
                self._cap = cap

            try:
                self._read_loop(cap)
            finally:
                with self._cap_lock:
                    self._cap = None
                cap.release()
                logger.debug("VideoCapture released after read loop exit.")

    # ------------------------------------------------------------------
    # Capture thread — inner frame-read loop
    # ------------------------------------------------------------------

    def _read_loop(self, cap: cv2.VideoCapture) -> None:
        """
        Inner tight loop: read one frame per iteration and push it to the queue.

        FPS throttle: when settings.fps_limit > 0, each iteration sleeps via
        stop_event.wait() for the remaining time in its frame slot. This avoids
        busy-waiting and wakes immediately when stop() is called.
        """
        frame_interval = (1.0 / settings.fps_limit) if settings.fps_limit > 0 else 0.0
        next_frame_time = time.monotonic()
        _telemetry_probed = False   # fire the diagnostic once per connection

        while not self._stop_event.is_set():

            # --- FPS throttle (skipped entirely when fps_limit == 0) ---
            if frame_interval > 0:
                now  = time.monotonic()
                wait = next_frame_time - now
                if wait > 0 and self._stop_event.wait(timeout=wait):
                    break   # stop was signalled during the wait; exit cleanly
                # Advance the target time by one interval.
                # max(now, next_frame_time) prevents drift if reading was slow.
                next_frame_time = max(time.monotonic(), next_frame_time) + frame_interval

            # --- Read frame ---
            try:
                ret, frame = cap.read()
            except Exception as exc:
                logger.error(
                    "Unexpected exception during cap.read(): %s", exc, exc_info=True
                )
                break

            if not ret:
                logger.warning(
                    "cap.read() returned False — stream dropped, triggering reconnect."
                )
                break

            # --- Telemetry probe (runs once per connection, first good frame) ---
            if not _telemetry_probed:
                _telemetry_probed = True
                self._probe_telemetry(cap)

            self._capture_fps.tick()
            self._queue.push(frame)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _connect(self) -> Optional[cv2.VideoCapture]:
        """
        Open the RTSP URL via the FFmpeg backend.

        CAP_PROP_BUFFERSIZE=1 limits OpenCV's internal frame queue to a single
        frame. Without this, frames pile up inside OpenCV and the processor
        always sees stale data. Correct tradeoff for real-time video.
        """
        logger.debug("Connecting to: %s", _redact_url(self._url))
        try:
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
        except Exception as exc:
            logger.error("VideoCapture() raised: %s", exc, exc_info=True)
            return None

        if not cap.isOpened():
            logger.warning("Failed to open: %s", _redact_url(self._url))
            cap.release()
            return None

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings.frame_height)

        logger.debug(
            "Connected — backend=%s  buffer=1  res=%dx%d",
            cap.getBackendName(),
            settings.frame_width,
            settings.frame_height,
        )

        return cap

    @staticmethod
    def _probe_telemetry(cap: cv2.VideoCapture) -> None:
        """
        Diagnostic probe — runs once per connection on the first good frame.

        Queries every OpenCV CAP_PROP that might carry stream metadata and
        prints a Telemetry summary line.  All values are read-only; nothing
        is modified.

        Logs stream properties (resolution, FPS, backend) on first connect.
        OpenCV's VideoCapture does not surface GPS/KLV metadata — telemetry
        comes exclusively from the OCR worker.
        """
        # OpenCV does not have CAP_PROP constants for GPS/telemetry.
        # The closest properties that *might* carry side-data on exotic backends:
        _PROPS = {
            "pos_msec":    cv2.CAP_PROP_POS_MSEC,        # stream timestamp (ms)
            "frame_width": cv2.CAP_PROP_FRAME_WIDTH,
            "frame_height":cv2.CAP_PROP_FRAME_HEIGHT,
            "fps":         cv2.CAP_PROP_FPS,
            "backend":     cv2.CAP_PROP_BACKEND,
        }

        prop_values = {}
        for name, prop_id in _PROPS.items():
            try:
                prop_values[name] = cap.get(prop_id)
            except Exception:
                prop_values[name] = None

        print(
            f"[stream] Stream props: "
            + "  ".join(f"{k}={v}" for k, v in prop_values.items())
        )
        logger.info("Stream props: %s", prop_values)

    def _release_cap(self) -> None:
        with self._cap_lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                logger.debug("VideoCapture released by stop().")

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """
        Exponential backoff: base × 2^(attempt-1), capped at base × 8.

        attempt 1 →  base × 1
        attempt 2 →  base × 2
        attempt 3 →  base × 4
        attempt 4+ → base × 8  (ceiling)
        """
        base: float = float(settings.reconnect_delay)
        return min(base * (2 ** min(attempt - 1, 3)), base * 8)
