"""
Frame buffer module.

Sits between RTSPStream (producer) and FrameProcessor (consumer) as the
sole communication channel. Decouples capture rate from processing rate so
neither thread needs to know about the other's speed.

Design principles:
  - push() NEVER blocks  — the producer is never stalled by a slow consumer
  - pop()  blocks briefly — the consumer sleeps efficiently until a frame arrives
  - When full, the OLDEST frame is evicted so the processor always sees recent data
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import NamedTuple, Optional

import numpy as np

from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)

__all__ = ["FrameQueue"]


class QueueStats(NamedTuple):
    """Point-in-time snapshot of queue health. Used for monitoring."""

    pushed: int
    popped: int
    dropped: int
    current_size: int
    max_size: int
    drop_rate: float    # dropped / pushed; 0.0 when nothing has been pushed yet


class FrameQueue:
    """
    Thread-safe bounded frame buffer backed by ``collections.deque``.

    One producer thread (RTSPStream) calls push(); one consumer thread
    (FrameProcessor worker) calls pop(). Both sides are safe to call
    concurrently from different threads.

    Drop policy: when the buffer is full, the oldest (leftmost) frame is
    silently evicted to make room for the new one. This means the consumer
    always processes the most recent frame available — correct behaviour
    for real-time video where staleness matters more than completeness.

    Usage::

        q = FrameQueue(maxsize=10)

        # producer thread
        q.push(frame)

        # consumer thread
        frame = q.pop(timeout=1.0)
        if frame is not None:
            processor.process(frame)
    """

    def __init__(self, maxsize: Optional[int] = None) -> None:
        """
        Args:
            maxsize:  Maximum number of frames the buffer holds.
                      Defaults to ``settings.max_queue_size``.
                      Smaller values reduce memory and latency; larger values
                      absorb short processing spikes without dropping frames.
        """
        self._maxsize: int = maxsize if maxsize is not None else settings.max_queue_size
        if self._maxsize < 1:
            raise ValueError(f"maxsize must be >= 1, got {self._maxsize!r}")

        # deque with maxlen is the core data structure.
        # deque.append() on a full deque atomically evicts deque[0] (oldest).
        # This is the single-operation "drop oldest" behaviour we need.
        self._deque: deque = deque(maxlen=self._maxsize)

        # Condition combines a lock + a wait/notify channel.
        # push() notifies; pop() waits. This lets the consumer sleep without
        # busy-spinning when the queue is empty.
        self._condition = threading.Condition(threading.Lock())

        # Monotonic counters — never decremented, never reset.
        self._frames_pushed: int = 0
        self._frames_popped: int = 0
        self._frames_dropped: int = 0

        logger.debug("FrameQueue initialised — maxsize=%d", self._maxsize)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def push(self, frame: np.ndarray) -> None:
        """
        Enqueue a frame. Never blocks.

        If the buffer is full the oldest frame is evicted first.
        The eviction and the new append happen inside the same lock
        acquisition, so the consumer cannot observe a transient empty state
        between the two operations.

        Args:
            frame:  BGR ndarray from RTSPStream.  Must be non-None.
        """
        with self._condition:
            is_full = len(self._deque) == self._maxsize
            self._deque.append(frame)          # evicts oldest automatically when full
            self._frames_pushed += 1

            if is_full:
                self._frames_dropped += 1
                logger.debug(
                    "Queue full — oldest frame evicted (dropped=%d).",
                    self._frames_dropped,
                )

            # Wake the consumer if it is waiting in pop().
            self._condition.notify()

    def pop(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Dequeue the oldest frame. Blocks until a frame is available or
        the timeout elapses.

        The timeout exists so the caller's loop can periodically check a
        stop event without being stuck forever on an empty queue.

        Args:
            timeout:  Maximum seconds to wait. Must be > 0.

        Returns:
            np.ndarray — the next frame in FIFO order.
            None       — timeout elapsed with no frame available.
        """
        deadline = time.monotonic() + timeout

        with self._condition:
            while not self._deque:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                # wait() atomically releases _condition's lock and blocks.
                # It re-acquires the lock before returning, whether woken by
                # notify() or by a timeout. The while-loop guard handles
                # spurious wakeups (permitted by POSIX condition variables).
                self._condition.wait(timeout=remaining)

            frame = self._deque.popleft()
            self._frames_popped += 1
            return frame

    def size(self) -> int:
        """
        Current number of frames in the buffer.

        The value is a snapshot; it may change immediately after the call
        returns. Use it for monitoring and logging, not for flow control.
        """
        with self._condition:
            return len(self._deque)

    def clear(self) -> None:
        """
        Discard all frames and reset the buffer to empty.

        Call during shutdown after both producer and consumer threads have
        been stopped, to release frame memory promptly.
        """
        with self._condition:
            count = len(self._deque)
            self._deque.clear()
            self._condition.notify_all()    # unblock any thread still in pop()
        logger.debug("FrameQueue cleared — %d frame(s) discarded.", count)

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    @property
    def is_empty(self) -> bool:
        with self._condition:
            return len(self._deque) == 0

    @property
    def is_full(self) -> bool:
        with self._condition:
            return len(self._deque) == self._maxsize

    @property
    def maxsize(self) -> int:
        return self._maxsize

    @property
    def stats(self) -> QueueStats:
        """
        Return a consistent snapshot of queue metrics.

        All four counter reads happen inside a single lock acquisition so
        the ratios (e.g. drop_rate) are meaningful and internally consistent.
        """
        with self._condition:
            pushed = self._frames_pushed
            dropped = self._frames_dropped
            return QueueStats(
                pushed=pushed,
                popped=self._frames_popped,
                dropped=dropped,
                current_size=len(self._deque),
                max_size=self._maxsize,
                drop_rate=dropped / pushed if pushed > 0 else 0.0,
            )


# Backward-compatible alias — core/app.py currently imports QueueManager.
# Remove once app.py is updated to use FrameQueue directly.
QueueManager = FrameQueue
