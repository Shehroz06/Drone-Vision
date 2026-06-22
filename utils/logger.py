"""
Centralized logging module.

Configures the root logger once and provides three output channels:
  - Console     : INFO and above, human-readable with level colours
  - app.log     : configurable level (from settings.log_level), rotating
  - error.log   : ERROR and CRITICAL only, rotating — for quick incident review
  - perf.log    : performance metrics only, structured key=value format

Every other module gets its logger via::

    from utils.logger import get_logger
    logger = get_logger(__name__)

Performance events are emitted via::

    from utils.logger import log_performance
    log_performance("fps", capture=29.8, processing=24.1, queue=3)
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

__all__ = ["get_logger", "log_performance", "setup_logging", "LOG_DIR"]

# ---------------------------------------------------------------------------
# Log directory — PyInstaller-aware
# ---------------------------------------------------------------------------

def _resolve_log_dir() -> Path:
    """
    Return the logs/ directory.

    When frozen as a .exe, write logs next to the executable so operators
    can find them. In development, write to the repo's logs/ folder.
    """
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "logs"
    return Path(__file__).parent.parent / "logs"

LOG_DIR: Path = _resolve_log_dir()

# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

_FILE_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(threadName)-16s | %(name)-28s | %(message)s"
)
_CONSOLE_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_PERF_FORMAT    = "%(asctime)s | %(message)s"
_DATE_FORMAT    = "%Y-%m-%d %H:%M:%S"

# ANSI colour codes per level — used by the console handler only.
_LEVEL_COLOURS = {
    logging.DEBUG:    "\033[36m",    # cyan
    logging.INFO:     "\033[32m",    # green
    logging.WARNING:  "\033[33m",    # yellow
    logging.ERROR:    "\033[31m",    # red
    logging.CRITICAL: "\033[35;1m",  # bright magenta
}
_RESET = "\033[0m"


class _ColourFormatter(logging.Formatter):
    """
    Wraps the level name in ANSI colour codes.

    Colour is applied per-level and reset afterward so the rest of the
    record — message, module name — is printed in the terminal's default
    colour. Avoids colouring entire lines, which makes them hard to read.
    """

    def format(self, record: logging.LogRecord) -> str:
        colour = _LEVEL_COLOURS.get(record.levelno, "")
        record = logging.makeLogRecord(record.__dict__)   # shallow copy — do not mutate original
        record.levelname = f"{colour}{record.levelname:<8}{_RESET}"
        return super().format(record)


def _supports_colour() -> bool:
    """
    True when stdout is a real terminal that likely supports ANSI codes.

    Respects the NO_COLOR env var (https://no-color.org/).
    Falls back to False in CI, redirected pipes, and Windows without ANSI.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if not hasattr(sys.stdout, "isatty") or not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        # Windows 10 build 1511+ supports ANSI via Virtual Terminal Processing.
        # Enable it programmatically so we don't need a third-party library.
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except Exception:
            return False
    return True


# ---------------------------------------------------------------------------
# Handler factories
# ---------------------------------------------------------------------------

def _make_rotating_handler(
    path: Path,
    level: int,
    fmt: str = _FILE_FORMAT,
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB
    backup_count: int = 5,
) -> logging.handlers.RotatingFileHandler:
    handler = logging.handlers.RotatingFileHandler(
        path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
        delay=True,     # don't create the file until the first write
    )
    handler.setLevel(level)
    handler.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
    return handler


def _make_console_handler(level: int) -> logging.StreamHandler:
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)
    formatter_class = _ColourFormatter if _supports_colour() else logging.Formatter
    handler.setFormatter(formatter_class(_CONSOLE_FORMAT, datefmt=_DATE_FORMAT))
    return handler


# ---------------------------------------------------------------------------
# One-time setup — thread-safe, idempotent
# ---------------------------------------------------------------------------

_setup_done = False
_setup_lock = threading.Lock()


def setup_logging(level: Optional[str] = None) -> None:
    """
    Configure the root logger. Idempotent — safe to call multiple times.

    Args:
        level:  Override the log level (e.g. "DEBUG").  When None, reads
                ``settings.log_level``.  Pass explicitly in tests so this
                function does not need to import ``config.settings``.
    """
    global _setup_done
    with _setup_lock:
        if _setup_done:
            return
        _setup_done = True

    if level is None:
        # Late import to avoid a circular dependency at module load time.
        # config.settings uses stdlib logging (not this module) during its
        # own loading, so the import here is always safe.
        from config.settings import settings
        level = settings.log_level

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    # Root captures everything; individual handlers enforce their own floor.
    root.setLevel(logging.DEBUG)

    # ---- output channels ---------------------------------------------------
    root.addHandler(_make_console_handler(level=max(numeric_level, logging.INFO)))
    root.addHandler(_make_rotating_handler(LOG_DIR / "app.log",   level=numeric_level))
    root.addHandler(_make_rotating_handler(LOG_DIR / "error.log", level=logging.ERROR))
    # ------------------------------------------------------------------------

    # Silence third-party libraries that flood the log at DEBUG level.
    for noisy in ("urllib3", "requests", "PIL", "charset_normalizer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).info(
        "Logging initialised — level=%s  dir=%s", level.upper(), LOG_DIR
    )


# ---------------------------------------------------------------------------
# Performance logger — isolated, does not propagate to root
# ---------------------------------------------------------------------------

_perf_logger: Optional[logging.Logger] = None
_perf_lock = threading.Lock()


def _get_perf_logger() -> logging.Logger:
    """
    Return the performance logger, creating it on first call.

    propagate=False keeps perf metrics out of app.log — they have their
    own file and their volume (one line per second) would drown app events.
    """
    global _perf_logger
    if _perf_logger is not None:
        return _perf_logger

    with _perf_lock:
        if _perf_logger is not None:   # double-checked under lock
            return _perf_logger

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        logger = logging.getLogger("performance")
        logger.propagate = False
        logger.setLevel(logging.DEBUG)
        logger.addHandler(
            _make_rotating_handler(
                LOG_DIR / "perf.log",
                level=logging.DEBUG,
                fmt=_PERF_FORMAT,
            )
        )
        _perf_logger = logger

    return _perf_logger


def log_performance(event: str, **metrics: Any) -> None:
    """
    Write a structured performance metric line to perf.log.

    Usage::

        log_performance("fps",   capture=29.8, processing=24.1)
        log_performance("queue", size=7, dropped=42, drop_rate=0.014)
        log_performance("frame", latency_ms=12.4)

    Each call produces one line::

        2024-01-15 10:23:45 | fps  capture=29.8  processing=24.1
    """
    metric_str = "  ".join(f"{k}={v}" for k, v in metrics.items())
    _get_perf_logger().info("%-12s  %s", event, metric_str)


# ---------------------------------------------------------------------------
# Public factory — the only import every other module needs
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger, configuring the root logger on the first call.

    Usage::

        logger = get_logger(__name__)
        logger.info("Stream connected: %s", url)
        logger.debug("Frame #%d received.", count)
        logger.error("cap.read() failed.", exc_info=True)
    """
    setup_logging()
    return logging.getLogger(name)
