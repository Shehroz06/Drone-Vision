#!/usr/bin/env python3
"""
RTSP Vision — single-command launcher.

Starts the FastAPI server (which owns the full processing pipeline and
serves the frontend).  The only required input is the RTSP stream URL.

Usage
-----
    # Environment variable (recommended for production / credentials):
    RTSP_URL=rtsp://user:pass@192.168.1.10:554/live python run.py

    # Command-line flag (convenient for dev):
    python run.py --rtsp-url rtsp://192.168.1.10:554/live

    # Development hot-reload (reloads on every .py file change):
    python run.py --rtsp-url rtsp://... --reload

    # Custom host/port:
    python run.py --rtsp-url rtsp://... --host 0.0.0.0 --port 9000

After starting, open:
    http://localhost:8000          ← live path map (frontend)
    http://localhost:8000/status   ← pipeline health (JSON)
    http://localhost:8000/docs     ← API docs (Swagger UI)
"""

from __future__ import annotations

import argparse
import os
import sys


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RTSP Vision — real-time path visualisation server.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--rtsp-url",
        metavar="URL",
        default=None,
        help=(
            "RTSP stream URL, e.g. rtsp://user:pass@192.168.1.10:554/live. "
            "Overrides the RTSP_URL env var and config file."
        ),
    )
    p.add_argument(
        "--host",
        default="0.0.0.0",
        help="IP address to bind the server to (default: 0.0.0.0 = all interfaces).",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="TCP port (default: 8000).",
    )
    p.add_argument(
        "--reload",
        action="store_true",
        help="Enable hot-reload on .py file changes (development only; uses extra CPU).",
    )
    p.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Uvicorn log level (default: info).",
    )
    return p.parse_args()


def main() -> None:
    # Required on Windows when the app is frozen with PyInstaller (--onedir or
    # --onefile).  Must be called before any other code in the main block.
    import multiprocessing
    multiprocessing.freeze_support()

    args = _parse_args()

    # Apply RTSP URL override before any module imports so that config/settings.py
    # (which reads RTSP_URL on import) picks it up.
    if args.rtsp_url:
        os.environ["RTSP_URL"] = args.rtsp_url

    try:
        import uvicorn
    except ImportError:
        print("ERROR: uvicorn is not installed.  Run:  pip install -r requirements.txt")
        sys.exit(1)

    print(f"\n  RTSP Vision starting on  http://{args.host}:{args.port}")
    print(f"  Frontend map:            http://localhost:{args.port}")
    print(f"  API docs:                http://localhost:{args.port}/docs")
    print(f"  Pipeline health:         http://localhost:{args.port}/status\n")

    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,              # must be 1 — pipeline state is in-process
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
