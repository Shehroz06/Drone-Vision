"""
RTSP Vision — Production Entry Point
=====================================

Initialises every subsystem, wires them together, owns the processing
worker thread, and handles graceful shutdown.

Data flow
---------

  [rtsp-capture thread]              [frame-worker thread]         [main thread]
  ─────────────────────              ─────────────────────         ─────────────
  RTSPStream._capture_loop()         _worker_loop()                main()
       │                                  │                             │
   cap.read()                        queue.pop(timeout=1s)        _stop_event.wait()
       │                                  │
   capture_fps.tick()                _check_reconnect()           ← signal handler
       │                             (calls processor.reset()          │
   queue.push(frame)                  if reconnect detected)      _on_signal()
                                          │                            │
                                     processor.process(frame)     _stop_event.set()
                                          │
                                     proc_fps.tick()
                                          │
                                     display.show(result)  ──── 'q' → _stop_event.set()
                                          │
                                     _emit_performance()    ← every 30 s → perf.log


Shutdown paths
--------------
  Path A: Operator presses 'q'  → display.show() returns False → _stop_event.set()
  Path B: SIGINT / SIGTERM       → _on_signal()                → _stop_event.set()

In both cases the main thread unblocks, joins the worker (5 s), stops
the stream (up to 10 s), closes the display, and drains the queue.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time

# ── Logging must be initialised before any module that uses it is imported ──
# setup_logging() triggers config/settings.py to load. The settings module
# fires a WARNING during load; installing handlers first ensures that message
# is routed to app.log rather than silently dropped by the stdlib lastResort.
from utils.logger import get_logger, log_performance, setup_logging

setup_logging()
logger = get_logger(__name__)

# ── All other imports happen after logging is ready ──────────────────────────
from config.settings import settings       # singleton, already loaded above
from core.processor import FrameProcessor
from core.queue_manager import FrameQueue
from core.stream import RTSPStream
from ui.display import Display
from utils.time_utils import FPSCounter


# ─────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ─────────────────────────────────────────────────────────────────────────────

_PERF_LOG_INTERVAL:   float = 30.0   # seconds between perf.log snapshots
_WORKER_JOIN_TIMEOUT: float = 5.0    # seconds main thread waits for worker exit
_DROP_RATE_WARN:      float = 0.05   # warn when queue evicts > 5 % of frames

# Shared stop signal between all threads.
# Any thread sets this to trigger an orderly shutdown.
_stop_event = threading.Event()


# ─────────────────────────────────────────────────────────────────────────────
# Signal handler
# ─────────────────────────────────────────────────────────────────────────────

def _on_signal(signum: int, _frame: object) -> None:
    """
    Handle SIGINT (Ctrl-C) and SIGTERM (kill / systemd stop).

    Only sets the stop event — no blocking work in a signal handler.
    All resource teardown happens on the main thread after _stop_event.wait()
    returns.
    """
    logger.info("Signal %d received — initiating shutdown.", signum)
    _stop_event.set()


# ─────────────────────────────────────────────────────────────────────────────
# Performance helpers
# ─────────────────────────────────────────────────────────────────────────────

def _emit_performance(
    stream:   RTSPStream,
    proc_fps: FPSCounter,
    queue:    FrameQueue,
) -> None:
    """
    Write a structured performance snapshot to perf.log.

    Emits two log lines:
        fps   — rolling capture FPS and processing FPS
        queue — current depth, cumulative drops, drop rate

    Also issues a WARNING to app.log when the drop rate exceeds the
    threshold, which signals that the processor cannot keep up with
    the capture rate and the operator should tune skip_interval or fps_limit.
    """
    stats        = queue.stats
    cap_fps      = round(stream.capture_fps, 1)
    proc_fps_val = round(proc_fps.fps, 1)

    log_performance("fps",   capture=cap_fps, processing=proc_fps_val)
    log_performance(
        "queue",
        size=stats.current_size,
        dropped=stats.dropped,
        drop_rate=round(stats.drop_rate, 4),
    )

    if stats.drop_rate > _DROP_RATE_WARN:
        logger.warning(
            "Queue drop rate %.1f%% exceeds %.0f%% — processor is slower than capture. "
            "Raise skip_interval or lower fps_limit in config.",
            stats.drop_rate * 100,
            _DROP_RATE_WARN * 100,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Worker thread  — THE PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def _worker_loop(
    queue:     FrameQueue,
    processor: FrameProcessor,
    display:   Display,
    stream:    RTSPStream,
    proc_fps:  FPSCounter,
) -> None:
    """
    Frame-worker thread: drives the complete pop → process → display pipeline.

    This is the second thread in the system (the first is RTSPStream's
    rtsp-capture thread). It runs until _stop_event is set by a signal
    handler or the operator closing the display window.

    Loop steps
    ----------
    1. Pop     — blocking queue.pop(timeout=1s); re-checks stop flag on timeout.
    2. Reconnect check — resets processor state when the stream has reconnected,
                         preventing the first post-reconnect frame from being
                         wrongly flagged as a duplicate.
    3. Process — FrameProcessor.process() applies duplicate filter + pipeline.
                 Returns None for dropped frames (skip_interval, duplicate, etc.).
    4. FPS tick — record frame for rolling processing-FPS average.
    5. Display  — render to OpenCV window; returns False if 'q' pressed.
    6. Perf log — emit metrics to perf.log every _PERF_LOG_INTERVAL seconds.
    """
    logger.info("Worker thread started.")

    last_reconnect: int   = 0
    last_perf_ts:   float = time.monotonic()

    while not _stop_event.is_set():

        # ── Step 1: Pop a frame (non-blocking overall — 1 s wait max) ─────────
        frame = queue.pop(timeout=1.0)
        if frame is None:
            # Queue was empty for 1 s. Loop back and re-check the stop flag.
            continue

        # ── Step 2: Reconnect detection ────────────────────────────────────────
        # RTSPStream.reconnect_count is 0 before first connect, then incremented
        # on every successful VideoCapture.open(). Comparing against our local
        # last_reconnect detects both the initial connect and all subsequent
        # reconnects — without any coupling between RTSPStream and FrameProcessor.
        current_reconnect = stream.reconnect_count
        if current_reconnect != last_reconnect:
            logger.info(
                "Stream reconnect #%d detected — resetting processor state.",
                current_reconnect,
            )
            processor.reset()   # clears _last_frame so no false-duplicate on first frame back
            last_reconnect = current_reconnect

        # ── Step 3: Process frame ──────────────────────────────────────────────
        result = processor.process(frame)
        if result is None:
            # Frame was dropped (duplicate, skip_interval, or a pipeline step).
            # Continue without displaying or counting.
            continue

        # ── Step 4: Track processing FPS ──────────────────────────────────────
        proc_fps.tick()

        # ── Step 5: Display ────────────────────────────────────────────────────
        # show() calls cv2.waitKey(1) internally — required for the OpenCV event
        # loop to render the window. Returns False when 'q' is pressed.
        if not display.show(result):
            logger.info("Operator closed the display window — shutting down.")
            _stop_event.set()   # trigger main-thread cleanup
            break

        # ── Step 6: Performance metrics ────────────────────────────────────────
        now = time.monotonic()
        if now - last_perf_ts >= _PERF_LOG_INTERVAL:
            last_perf_ts = now
            _emit_performance(stream, proc_fps, queue)

    logger.info("Worker thread stopped.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    """
    Parse optional command-line overrides.

    All arguments are optional — the system runs fine with no flags, using
    values from config.yaml (or built-in defaults if no file is found).
    CLI values override individual config fields AFTER the config file is loaded.
    """
    parser = argparse.ArgumentParser(
        prog="rtsp-vision",
        description="Real-time RTSP stream processor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  rtsp-vision\n"
            "  rtsp-vision --rtsp-url rtsp://192.168.1.10:554/live\n"
            "  rtsp-vision --fps-limit 25 --skip-interval 2\n"
        ),
    )
    parser.add_argument(
        "--rtsp-url",
        metavar="URL",
        default=None,
        help="Override RTSP stream URL (e.g. rtsp://cam.local/live).",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        metavar="FPS",
        default=None,
        help="Cap capture rate in frames/second. 0 = unlimited.",
    )
    parser.add_argument(
        "--skip-interval",
        type=int,
        metavar="N",
        default=None,
        help="Process every Nth frame. 1 = every frame (default).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> int:
    """
    Initialise the system, start threads, and block until shutdown.

    Returns
    -------
    0 — clean exit
    1 — unhandled exception (detail in app.log / stderr)
    """
    # ── 1. Parse CLI arguments ────────────────────────────────────────────────
    args = _parse_args()

    logger.info("=" * 56)
    logger.info("RTSP Vision starting.")

    # ── 2. Apply CLI overrides to the settings singleton ─────────────────────
    # ``settings`` is the module-level singleton created when config/settings.py
    # was imported above. Mutating its fields here applies overrides before any
    # component reads them — no re-load required.
    if args.rtsp_url is not None:
        settings.rtsp_url = args.rtsp_url
        logger.info("CLI override  rtsp_url   = %s", args.rtsp_url)

    if args.fps_limit is not None:
        if args.fps_limit < 0:
            logger.error("--fps-limit must be >= 0. Got: %s", args.fps_limit)
            return 1
        settings.fps_limit = args.fps_limit
        logger.info(
            "CLI override  fps_limit  = %s",
            args.fps_limit if args.fps_limit > 0 else "unlimited",
        )

    # ── 3. Build components ───────────────────────────────────────────────────
    # Each component reads from the ``settings`` singleton; overrides above
    # are already in place when constructors run.
    queue     = FrameQueue()            # bounded thread-safe frame buffer
    stream    = RTSPStream(queue)       # capture thread + reconnect logic
    processor = FrameProcessor()        # pure processing (no thread, no queue)
    display   = Display()               # OpenCV window renderer
    proc_fps  = FPSCounter()            # rolling average for processing FPS

    # Apply skip_interval after processor is created so the setter validates it.
    if args.skip_interval is not None:
        if args.skip_interval < 1:
            logger.error("--skip-interval must be >= 1. Got: %s", args.skip_interval)
            return 1
        processor.skip_interval = args.skip_interval
        logger.info("CLI override  skip_interval = %d", args.skip_interval)

    logger.info(
        "Components ready — queue_max=%d  fps_limit=%s  skip_interval=%d",
        queue.maxsize,
        settings.fps_limit if settings.fps_limit > 0 else "unlimited",
        processor.skip_interval,
    )

    # ── 4. Install signal handlers ────────────────────────────────────────────
    # Must be called from the main thread (Python restriction).
    # _on_signal() only sets _stop_event; no blocking work inside it.
    signal.signal(signal.SIGINT,  _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    # ── 5. Build and start the worker thread ──────────────────────────────────
    # daemon=True means Python will not wait for this thread when the
    # interpreter exits — the finally block below handles clean join instead.
    worker = threading.Thread(
        target=_worker_loop,
        args=(queue, processor, display, stream, proc_fps),
        name="frame-worker",
        daemon=True,
    )

    try:
        stream.start()   # launches the 'rtsp-capture' daemon thread inside RTSPStream
        worker.start()   # launches the 'frame-worker' thread defined above

        logger.info(
            "System running — threads: [rtsp-capture] [frame-worker]"
        )
        logger.info("Press 'q' in the display window or Ctrl-C to stop.")

        # ── 6. Block main thread ──────────────────────────────────────────────
        # _stop_event is set by either:
        #   • _on_signal()          (SIGINT / SIGTERM)
        #   • _worker_loop()        (operator presses 'q')
        _stop_event.wait()

    except Exception as exc:
        logger.critical("Unhandled exception on main thread: %s", exc, exc_info=True)
        _stop_event.set()
        return 1

    finally:
        # ── 7. Graceful shutdown ──────────────────────────────────────────────
        # Order matters:
        #   a) Join worker — it sees _stop_event and exits its loop promptly.
        #   b) Stop stream — RTSPStream.stop() sets its own event + joins
        #                    the rtsp-capture thread (up to 10 s).
        #   c) Close display — destroys all OpenCV windows.
        #   d) Clear queue  — releases frame memory promptly.
        logger.info("Shutdown in progress.")

        _stop_event.set()   # idempotent; covers the exception path

        worker.join(timeout=_WORKER_JOIN_TIMEOUT)
        if worker.is_alive():
            logger.warning(
                "Worker did not exit within %.0fs — continuing shutdown.",
                _WORKER_JOIN_TIMEOUT,
            )

        stream.stop()       # public method; no access to private _thread
        display.close()
        queue.clear()

        logger.info("RTSP Vision stopped cleanly.")
        logger.info("=" * 56)

    return 0


if __name__ == "__main__":
    sys.exit(main())
