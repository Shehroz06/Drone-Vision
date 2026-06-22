"""
Configuration loading module.

Loads application settings from a YAML or JSON file, applies environment
variable overrides, validates every field, and falls back to built-in
defaults on any invalid value.  Exposes a ready-to-use ``settings``
singleton that every other module imports.

Priority chain (highest → lowest):
    OS environment variables  >  .env file  >  config file  >  built-in defaults
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import dataclass, fields as dc_fields
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, get_type_hints
from urllib.parse import urlparse, urlunparse

import yaml
from dotenv import load_dotenv

__all__ = ["ConfigLoader", "Settings", "settings"]

# Use stdlib logger directly.
# utils.logger imports this module (indirectly via get_logger → setup_logging),
# so importing from utils.logger here would create a circular dependency.
_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _redact_url(url: str) -> str:
    """Replace any password in a URL with *** before logging."""
    try:
        parsed = urlparse(url)
        if parsed.password:
            host = f"{parsed.hostname}:{parsed.port}" if parsed.port else (parsed.hostname or "")
            return urlunparse(parsed._replace(netloc=f"{parsed.username}:***@{host}"))
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# Settings dataclass — the single authoritative shape of the config
# ---------------------------------------------------------------------------

@dataclass
class Settings:
    """
    All runtime configuration in one place.

    Fields carry built-in Python defaults as the last-resort fallback.
    Do not read environment variables here; that is ConfigLoader's job.
    """

    rtsp_url: str        = "rtsp://localhost:554/stream"
    frame_width: int     = 1280
    frame_height: int    = 720
    max_queue_size: int  = 10
    log_level: str       = "INFO"
    reconnect_delay: int = 5
    fps_limit: float     = 0.0   # 0.0 = unlimited; positive value caps capture FPS

    # Geographic bounding box for pixel → lat/lon conversion.
    # These must match the real-world area visible in the camera frame.
    geo_lat_min: float = 33.6840
    geo_lat_max: float = 33.6850
    geo_lon_min: float = 73.0470
    geo_lon_max: float = 73.0480


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------

_VALID_LOG_LEVELS: frozenset = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})

_FIELD_RULES: Dict[str, Tuple[Any, str]] = {
    "rtsp_url": (
        lambda v: isinstance(v, str) and v.startswith("rtsp://") and len(v) > len("rtsp://"),
        "must be a non-empty rtsp:// URL",
    ),
    "frame_width": (
        lambda v: isinstance(v, int) and 64 <= v <= 7680,
        "must be an integer between 64 and 7680",
    ),
    "frame_height": (
        lambda v: isinstance(v, int) and 64 <= v <= 4320,
        "must be an integer between 64 and 4320",
    ),
    "max_queue_size": (
        lambda v: isinstance(v, int) and 1 <= v <= 500,
        "must be an integer between 1 and 500",
    ),
    "log_level": (
        lambda v: isinstance(v, str) and v.upper() in _VALID_LOG_LEVELS,
        f"must be one of {sorted(_VALID_LOG_LEVELS)}",
    ),
    "reconnect_delay": (
        lambda v: isinstance(v, (int, float)) and 1 <= float(v) <= 300,
        "must be a number between 1 and 300 (seconds)",
    ),
    "fps_limit": (
        lambda v: isinstance(v, (int, float)) and float(v) >= 0,
        "must be >= 0  (0 = unlimited)",
    ),
    "geo_lat_min": (
        lambda v: isinstance(v, (int, float)) and -90  <= float(v) <= 90,
        "must be a latitude in [-90, 90]",
    ),
    "geo_lat_max": (
        lambda v: isinstance(v, (int, float)) and -90  <= float(v) <= 90,
        "must be a latitude in [-90, 90]",
    ),
    "geo_lon_min": (
        lambda v: isinstance(v, (int, float)) and -180 <= float(v) <= 180,
        "must be a longitude in [-180, 180]",
    ),
    "geo_lon_max": (
        lambda v: isinstance(v, (int, float)) and -180 <= float(v) <= 180,
        "must be a longitude in [-180, 180]",
    ),
}

_ENV_MAP: Dict[str, str] = {
    "RTSP_URL":        "rtsp_url",
    "FRAME_WIDTH":     "frame_width",
    "FRAME_HEIGHT":    "frame_height",
    "MAX_QUEUE_SIZE":  "max_queue_size",
    "LOG_LEVEL":       "log_level",
    "RECONNECT_DELAY": "reconnect_delay",
    "FPS_LIMIT":       "fps_limit",
    "GEO_LAT_MIN":    "geo_lat_min",
    "GEO_LAT_MAX":    "geo_lat_max",
    "GEO_LON_MIN":    "geo_lon_min",
    "GEO_LON_MAX":    "geo_lon_max",
}


# ---------------------------------------------------------------------------
# ConfigLoader
# ---------------------------------------------------------------------------

class ConfigLoader:
    """
    Loads, merges, and validates application configuration.

    Config file search order:
        1. Explicit ``path`` argument (if provided)
        2. config.yaml / config.json next to the running exe (operator override)
        3. assets/config_default.yaml inside the PyInstaller bundle
        4. assets/config_default.yaml relative to the project root (development)

    If no file is found, built-in ``Settings`` defaults are used.
    """

    _CANDIDATE_NAMES = ("config.yaml", "config.json", "config_default.yaml")

    def __init__(self, path: Optional[str] = None) -> None:
        self._explicit_path: Optional[Path] = Path(path) if path else None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_config(self) -> Settings:
        """
        Load and return a fully validated Settings instance.

        Never raises on bad config values — falls back to defaults and logs
        a warning so non-technical operators get a working system even if
        they make a typo in the config file.
        """
        raw: Dict[str, Any] = self._load_file()

        load_dotenv()
        raw = self._apply_env_overrides(raw)
        raw = self._coerce_types(raw)
        raw = self._validate_and_fill_defaults(raw)

        cfg = Settings(**raw)

        # WARNING level is used here (not INFO) so this line is visible via
        # the stdlib lastResort handler during early boot, before
        # utils.logger.setup_logging() has installed the file/console handlers.
        _log.warning(
            "Config loaded — url=%s  res=%dx%d  queue=%d  fps=%s  log=%s",
            _redact_url(cfg.rtsp_url),
            cfg.frame_width,
            cfg.frame_height,
            cfg.max_queue_size,
            cfg.fps_limit if cfg.fps_limit > 0 else "unlimited",
            cfg.log_level,
        )
        return cfg

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _load_file(self) -> Dict[str, Any]:
        path = self._resolve_config_path()
        if path is None:
            _log.warning("No config file found — using built-in defaults.")
            return {}
        _log.warning("Loading config from: %s", path)
        try:
            return self._parse_file(path)
        except Exception as exc:
            _log.error(
                "Failed to read config '%s': %s — using built-in defaults.", path, exc
            )
            return {}

    def _resolve_config_path(self) -> Optional[Path]:
        for candidate in self._search_candidates():
            if candidate.is_file():
                return candidate
        return None

    def _search_candidates(self):
        if self._explicit_path is not None:
            yield self._explicit_path
            return

        exe_dir = Path(sys.executable).parent
        for name in self._CANDIDATE_NAMES:
            yield exe_dir / name

        bundle_assets = self._bundle_assets_dir()
        yield bundle_assets / "config_default.yaml"
        yield bundle_assets / "config_default.json"

    @staticmethod
    def _bundle_assets_dir() -> Path:
        if getattr(sys, "frozen", False):
            return Path(sys._MEIPASS) / "assets"
        return Path(__file__).parent.parent / "assets"

    @staticmethod
    def _parse_file(path: Path) -> Dict[str, Any]:
        suffix = path.suffix.lower()
        with path.open("r", encoding="utf-8") as fh:
            if suffix in {".yaml", ".yml"}:
                data = yaml.safe_load(fh)
            elif suffix == ".json":
                data = json.load(fh)
            else:
                raise ValueError(f"Unsupported config extension: '{suffix}'")
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Environment variable overrides
    # ------------------------------------------------------------------

    def _apply_env_overrides(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        overridden = dict(raw)
        for env_key, field_name in _ENV_MAP.items():
            value = os.environ.get(env_key)
            if value is not None:
                overridden[field_name] = value
                _log.debug("Env override: %s → settings.%s", env_key, field_name)
        return overridden

    # ------------------------------------------------------------------
    # Type coercion
    # ------------------------------------------------------------------

    def _coerce_types(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Convert raw values to Python types declared on Settings."""
        type_hints = get_type_hints(Settings)
        coerced = {}
        for key, value in raw.items():
            if key not in type_hints:
                continue
            coerced[key] = self._coerce_value(key, value, type_hints[key])
        return coerced

    @staticmethod
    def _coerce_value(key: str, value: Any, target_type: type) -> Any:
        _CONVERTERS = {
            str:   str,
            int:   lambda v: int(float(v)),
            float: float,
            bool:  lambda v: str(v).strip().lower() in {"1", "true", "yes", "on"},
        }
        converter = _CONVERTERS.get(target_type)
        if converter is None or isinstance(value, target_type):
            return value
        try:
            return converter(value)
        except (ValueError, TypeError) as exc:
            _log.warning(
                "Cannot coerce '%s' value %r to %s: %s",
                key, value, target_type.__name__, exc,
            )
            return value

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_and_fill_defaults(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate each field and replace invalid values with their defaults.

        Strategy: warn loudly but never crash. Non-technical operators must
        get a running application even if they mistype a value.
        """
        defaults = {f.name: f.default for f in dc_fields(Settings)}
        validated: Dict[str, Any] = {}

        for f in dc_fields(Settings):
            name  = f.name
            value = raw.get(name, defaults[name])
            rule  = _FIELD_RULES.get(name)

            if rule is None:
                validated[name] = value
                continue

            predicate, description = rule
            try:
                is_valid = predicate(value)
            except Exception:
                is_valid = False

            if not is_valid:
                _log.warning(
                    "Config field '%s' has invalid value %r (%s). "
                    "Using default: %r.",
                    name, value, description, defaults[name],
                )
                validated[name] = defaults[name]
            else:
                validated[name] = value

        validated["log_level"] = str(validated["log_level"]).upper()
        return validated


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

settings: Settings = ConfigLoader().load_config()
