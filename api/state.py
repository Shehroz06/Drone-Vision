"""
Shared state between the processing pipeline (sync threads) and the API (async).

The processing worker thread calls push() with new path data.
FastAPI handlers read the data via get_latest() and broadcast it to WebSocket clients.

Thread-safety contract
----------------------
_latest        — protected by threading.Lock (read from async, written from thread)
_connections   — only ever touched inside the event loop (add/remove/iterate all
                 come from async context), so no async lock is needed
_broadcast()   — a coroutine scheduled onto the event loop via
                 asyncio.run_coroutine_threadsafe(); never called directly from a thread
"""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Optional

from utils.time_utils import now_iso

if TYPE_CHECKING:
    from fastapi import WebSocket


class PathStore:
    """
    In-process message broker between the frame-worker thread and API clients.

    Usage in the processing pipeline (sync thread)::

        from api.state import store
        store.push(points=[[x1, y1], [x2, y2]], timestamp=now_iso())

    The API registers the event loop on startup; after that, every push()
    automatically broadcasts the new data to all connected WebSocket clients.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Optional[dict] = None
        self._connections: set[WebSocket] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    # ------------------------------------------------------------------
    # Called once from FastAPI lifespan (async context)
    # ------------------------------------------------------------------

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Register the running event loop so push() can schedule broadcasts."""
        self._loop = loop

    # ------------------------------------------------------------------
    # Called from the worker thread (sync context)
    # ------------------------------------------------------------------

    def push(
        self,
        points:    list,
        lat:       Optional[float] = None,
        lon:       Optional[float] = None,
        timestamp: Optional[str] = None,
    ) -> None:
        """
        Store a new path update and broadcast it to all WebSocket clients.

        Thread-safe. Safe to call from any thread at any time.
        """
        data = {
            "points":    points,
            "lat":       round(lat, 7) if lat is not None else None,
            "lon":       round(lon, 7) if lon is not None else None,
            "timestamp": timestamp or now_iso(),
        }
        with self._lock:
            self._latest = data

        # Schedule the async broadcast on the event loop from this sync thread.
        # run_coroutine_threadsafe is the only thread-safe way to call a coroutine
        # from outside the event loop. We don't await the future — fire and forget.
        if self._loop and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(self._broadcast(data), self._loop)

    # ------------------------------------------------------------------
    # Called from GET /path (async context)
    # ------------------------------------------------------------------

    def get_latest(self) -> Optional[dict]:
        """Return the most recent path update, or None if none has arrived yet."""
        with self._lock:
            return self._latest

    # ------------------------------------------------------------------
    # WebSocket management (async context only)
    # ------------------------------------------------------------------

    async def connect(self, ws: WebSocket) -> None:
        """Accept a WebSocket handshake and register the client for broadcasts."""
        await ws.accept()
        self._connections.add(ws)

    def disconnect(self, ws: WebSocket) -> None:
        """Unregister a WebSocket client (idempotent)."""
        self._connections.discard(ws)

    async def _broadcast(self, data: dict) -> None:
        """
        Send data to every connected WebSocket client.

        Dead connections (closed without a clean disconnect frame) are silently
        removed. Called exclusively from the event loop via run_coroutine_threadsafe.
        """
        if not self._connections:
            return

        dead: set[WebSocket] = set()
        for ws in list(self._connections):   # snapshot — safe to modify set during iteration
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)

        self._connections -= dead


# Module-level singleton shared by the API and the processing pipeline.
store = PathStore()
