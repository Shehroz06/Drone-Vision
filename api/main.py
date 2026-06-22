"""
RTSP Vision — FastAPI Backend
==============================

Single entry point for the API server.  Owns the full pipeline lifecycle:
starts the background processing thread on startup, tears it down on shutdown.

Data flow
---------

  [rtsp-capture thread]      [pipeline-worker thread]       [asyncio event loop]
  ─────────────────────      ────────────────────────       ────────────────────
  RTSPStream                 PipelineManager._run()         FastAPI handlers
       │                           │                              │
   cap.read()                 queue.pop()               GET  /health  → {"status":"ok"}
       │                           │                    GET  /path    → store.get_latest()
   queue.push(frame)          processor.process(frame)  GET  /status  → pipeline.metrics()
                                   │                    WS   /ws      ← store._broadcast()
                              planner.plan(frame)
                                   │
                              store.push(points)
                                   │
                     ┌─────────────┴──────────────┐
                     │                            │
              threading.Lock               asyncio.run_coroutine_threadsafe
              update _latest               schedule _broadcast()
                     │                            │
              GET /path reads            WS clients receive JSON
              this under lock

Endpoints
---------
GET  /health   — liveness probe
GET  /path     — latest path snapshot
GET  /status   — pipeline health (fps, queue, reconnects)
WS   /ws       — real-time path stream (push on every update)

How to run
----------
    # From the project root:
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

    # Or with explicit workers (production, no --reload):
    python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1

    Note: use exactly 1 worker. Multiple workers each spawn their own
    pipeline thread and their own in-process PathStore — they will NOT
    share state.  For horizontal scaling, use a shared broker (Redis).
"""

from __future__ import annotations

import os
# Set FFmpeg capture options before any cv2.VideoCapture is constructed.
# stream.py and ocr_worker.py both used os.environ.setdefault(), but setdefault
# is a no-op after the first writer wins.  Setting it once here, before any
# module import triggers VideoCapture, guarantees both connections use all flags.
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = (
    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|rtbufsize;102400"
)

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import requests as _requests
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from api.pipeline import PipelineManager
from api.state import store
from config.settings import settings
from core.ocr_worker import OcrWorker
from utils.logger import get_logger, setup_logging

_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
_TILES_DIR    = Path(__file__).parent.parent / "tiles"
_OSM_TILE_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"

# Logging must be ready before any other module fires startup messages.
setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — owns pipeline startup and shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    FastAPI lifespan handler.

    Startup:
        1. Register the running event loop with PathStore so the pipeline
           worker thread can schedule WebSocket broadcasts onto it.
        2. Create and start the PipelineManager (spawns both threads).
        3. Store the manager on app.state so endpoints can query metrics.

    Shutdown (reverse order):
        1. Stop the PipelineManager (signals threads, joins them).
        2. PathStore loop reference becomes stale — no more broadcasts,
           which is correct since the server is exiting.
    """
    # Wire PathStore to the running event loop BEFORE starting the pipeline,
    # so the very first store.push() already has a loop to broadcast onto.
    store.set_loop(asyncio.get_running_loop())

    pipeline = PipelineManager()
    app.state.pipeline = pipeline   # expose to endpoints via request.app.state
    pipeline.start()

    ocr = OcrWorker(rtsp_url=settings.rtsp_url, pipeline=pipeline)
    app.state.ocr = ocr
    ocr.start()

    logger.info("API server ready.")

    yield   # ── server is running ──

    ocr.stop()
    pipeline.stop()
    logger.info("API server shut down.")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="RTSP Vision API",
    description="Real-time path data from the RTSP processing pipeline.",
    version="1.0.0",
    lifespan=lifespan,
)

# Restrict origins in production; "*" is fine for local dev / drone GCS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["monitoring"])
async def health() -> dict:
    """
    Liveness probe.

    Returns {"status": "ok"} as long as the server process is alive.
    Use this for Docker HEALTHCHECK and load balancer probes.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# GET /path
# ---------------------------------------------------------------------------

@app.get("/path", tags=["data"])
async def get_path():
    """
    Latest path snapshot (REST / polling).

    Returns the most recent waypoints produced by the processing pipeline.
    Suitable for clients that poll on a timer rather than hold a WebSocket.

    Response 200:
        {
            "points": [[x1, y1], [x2, y2], ...],
            "timestamp": "2026-06-20T12:00:00.000Z"
        }
        Coordinates are normalised to [0.0, 1.0] relative to frame size.

    Response 503:
        Pipeline has started but no frame has been processed yet.
    """
    data = store.get_latest()
    if data is None:
        return JSONResponse(
            status_code=503,
            content={"detail": "No path data yet — pipeline may still be connecting."},
        )
    return data


# ---------------------------------------------------------------------------
# GET /status
# ---------------------------------------------------------------------------

@app.get("/config", tags=["data"])
async def get_config():
    """
    Static configuration consumed by the frontend on startup.

    Returns the geographic bounding box so the Leaflet map can centre itself
    on the correct area without hard-coded values in the HTML.

    Response 200:
        {
            "geo_bounds": {
                "lat_min": 33.684,
                "lat_max": 33.685,
                "lon_min": 73.047,
                "lon_max": 73.048
            }
        }
    """
    return {
        "geo_bounds": {
            "lat_min": settings.geo_lat_min,
            "lat_max": settings.geo_lat_max,
            "lon_min": settings.geo_lon_min,
            "lon_max": settings.geo_lon_max,
        }
    }


@app.get("/status", tags=["monitoring"])
async def get_status(request: Request):
    """
    Pipeline health snapshot.

    Returns real-time metrics about the processing pipeline.

    Response 200:
        {
            "capture_fps":     25.1,
            "processing_fps":  24.8,
            "queue_size":      2,
            "queue_max":       10,
            "drop_rate":       0.002,
            "dropped_total":   4,
            "reconnect_count": 1,
            "last_update":     "2026-06-20T12:00:00.000Z"
        }
    """
    pipeline: PipelineManager = request.app.state.pipeline
    return pipeline.metrics().as_dict()


# ---------------------------------------------------------------------------
# WebSocket /ws
# ---------------------------------------------------------------------------

@app.websocket("/ws")
async def websocket_stream(ws: WebSocket):
    """
    Real-time path stream.

    The server PUSHES a JSON message to the client on every path update.
    Clients do not need to send anything; the connection is receive-only.

    On connect: the client immediately receives the latest snapshot (if any),
    so it does not have to wait for the next pipeline update.

    On each pipeline update:
        {
            "points": [[x1, y1], ...],
            "timestamp": "2026-06-20T12:00:00.000Z"
        }
    """
    await store.connect(ws)
    client = ws.client
    logger.info(
        "WebSocket connected: %s  (active=%d)", client, len(store._connections)
    )

    # Send the current snapshot immediately so the client has data right away.
    latest = store.get_latest()
    if latest:
        try:
            await ws.send_json(latest)
        except Exception:
            pass   # client may have already disconnected; _broadcast will clean up

    try:
        # Drain any messages the client sends (ignored — server-push only).
        # Loop exits via WebSocketDisconnect when the client closes.
        while True:
            await ws.receive_text()

    except WebSocketDisconnect:
        pass  # normal close

    except Exception as exc:
        logger.warning("WebSocket %s error: %s", client, exc)

    finally:
        store.disconnect(ws)
        logger.info(
            "WebSocket disconnected: %s  (active=%d)", client, len(store._connections)
        )


# ---------------------------------------------------------------------------
# GET /tiles/{z}/{x}/{y}.png  — tile proxy with local disk cache
# ---------------------------------------------------------------------------

# One asyncio.Lock per (z, x, y) tile coordinate.
# asyncio is single-threaded, so dict access without a mutex is safe — no
# context switch can occur between the key check and the lock creation.
_tile_locks: dict[tuple[int, int, int], asyncio.Lock] = {}


@app.get("/tiles/{z}/{x}/{y}.png", tags=["tiles"])
async def get_tile(z: int, x: int, y: int):
    """
    Map tile proxy with transparent local disk cache.

    Flow:
        1. Cache hit  → serve tiles/{z}/{x}/{y}.png directly from disk.
        2. Cache miss → acquire a per-tile lock, re-check (another request
           may have fetched while we waited), then fetch from OpenStreetMap.
        3. Write atomically (tmp → rename) so a crashed write never leaves a
           corrupt file on disk.
        4. Release lock — any concurrent requests for the same tile now take
           the fast path on re-check and find the file already cached.
    """
    tile_path = _TILES_DIR / str(z) / str(x) / f"{y}.png"

    # ── Fast path: already cached ─────────────────────────────────────────────
    if tile_path.is_file():
        return FileResponse(
            tile_path,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    # ── Slow path: fetch from OSM under a per-tile lock ───────────────────────
    key = (z, x, y)
    if key not in _tile_locks:
        _tile_locks[key] = asyncio.Lock()

    async with _tile_locks[key]:
        # Double-checked: a concurrent request may have downloaded while we
        # were waiting for the lock — serve from cache if so.
        if tile_path.is_file():
            return FileResponse(
                tile_path,
                media_type="image/png",
                headers={"Cache-Control": "public, max-age=86400"},
            )

        # Fetch in a thread pool — requests is synchronous and must not block
        # the event loop.
        url = _OSM_TILE_URL.format(z=z, x=x, y=y)

        def _fetch() -> bytes:
            resp = _requests.get(
                url,
                headers={
                    "User-Agent": "rtsp-vision-tilecache/1.0",
                    "Accept":     "image/png",
                },
                timeout=10,
            )
            resp.raise_for_status()
            return resp.content

        try:
            content: bytes = await asyncio.get_running_loop().run_in_executor(
                None, _fetch
            )
        except _requests.HTTPError as exc:
            logger.warning("OSM tile HTTP %s z=%d x=%d y=%d", exc.response.status_code, z, x, y)
            return Response(status_code=exc.response.status_code)
        except Exception as exc:
            logger.warning("Tile fetch failed z=%d x=%d y=%d: %s", z, x, y, exc)
            return Response(status_code=502)

        # Atomic write: write to a per-tile .tmp then rename into place.
        # rename() is atomic on POSIX — readers never see a partial file.
        tile_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = tile_path.with_suffix(".tmp")
        try:
            tmp.write_bytes(content)
            tmp.rename(tile_path)
            logger.debug("Cached tile z=%d x=%d y=%d", z, x, y)
        except OSError as exc:
            logger.warning("Failed to cache tile z=%d x=%d y=%d: %s", z, x, y, exc)
            tmp.unlink(missing_ok=True)

    return Response(
        content=content,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


# ---------------------------------------------------------------------------
# Static frontend — must be mounted AFTER all API routes so routes take priority
# ---------------------------------------------------------------------------

if _FRONTEND_DIR.is_dir():
    app.mount("/", StaticFiles(directory=_FRONTEND_DIR, html=True), name="frontend")
    logger.info("Serving frontend from %s", _FRONTEND_DIR)
else:
    logger.warning(
        "Frontend directory not found at %s — UI will not be served.", _FRONTEND_DIR
    )
