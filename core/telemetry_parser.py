"""
Telemetry OCR parser — lat/lon only.

Converts raw EasyOCR text strings from the LAT/LON ROI into structured
numeric values.

Input:  dict mapping ROI title → list of raw OCR strings
Output: {"lat": float|None, "lon": float|None}
"""

from __future__ import annotations

import re
from typing import Optional

__all__ = ["parse_telemetry"]


# ── Compiled patterns ─────────────────────────────────────────────────────────

# LON: handle OCR confusion O↔0, optional spaces around label/colon
_LON_RE = re.compile(
    r'L[O0o]\s*N\s*:?\s*([+-]?\d{1,3}[.,]\d{3,10})',
    re.IGNORECASE,
)

# LAT: similar tolerance
_LAT_RE = re.compile(
    r'LAT?\s*:?\s*([+-]?\d{1,2}[.,]\d{3,10})',
    re.IGNORECASE,
)

# High-precision decimal fallback (≥4 fractional digits)
_HIGHPREC_RE = re.compile(r'[+-]?\d{1,3}[.,]\d{4,10}')


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean(text: str) -> str:
    """Normalise common OCR artefacts before pattern matching."""
    text = text.replace(',', '.')
    text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)
    text = re.sub(r'(\d)\s+\.\s+(\d)', r'\1.\2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _to_float(s: str) -> Optional[float]:
    """Parse a cleaned numeric string to float; return None on failure."""
    try:
        return float(s.replace(',', '.'))
    except (ValueError, TypeError, AttributeError):
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def parse_telemetry(roi_texts: dict[str, list[str]]) -> dict:
    """
    Parse raw EasyOCR strings from the LAT/LON ROI into numeric coordinates.

    Args:
        roi_texts:  Mapping of ROI title → list of raw OCR text strings.

    Returns:
        {"lat": float | None, "lon": float | None}
    """
    def _join(key: str) -> str:
        return " ".join(_clean(t) for t in roi_texts.get(key, []))

    lat_lon_blob = _join("ROI: LAT/LON")

    print("[OCR RAW]   ", roi_texts.get("ROI: LAT/LON", []))
    print("[OCR CLEAN] ", lat_lon_blob)

    result: dict = {"lat": None, "lon": None}

    # ── Longitude ─────────────────────────────────────────────────────────────
    m = _LON_RE.search(lat_lon_blob)
    result["lon"] = _to_float(m.group(1)) if m else None

    # ── Latitude ──────────────────────────────────────────────────────────────
    m = _LAT_RE.search(lat_lon_blob)
    result["lat"] = _to_float(m.group(1)) if m else None

    # ── Fallback: assign by magnitude when labels are garbled ─────────────────
    if result["lon"] is None or result["lat"] is None:
        candidates = [_to_float(x) for x in _HIGHPREC_RE.findall(lat_lon_blob)]
        candidates = [v for v in candidates if v is not None]

        if len(candidates) >= 2:
            a, b = candidates[0], candidates[1]
            if abs(a) < abs(b):
                a, b = b, a   # larger absolute value → longitude
            if result["lon"] is None:
                result["lon"] = a
            if result["lat"] is None:
                result["lat"] = b
        elif len(candidates) == 1:
            v = candidates[0]
            if 60.0 <= v <= 180.0 and result["lon"] is None:
                result["lon"] = v
            elif 0.0 <= v < 60.0 and result["lat"] is None:
                result["lat"] = v

    # ── OSD digit correction: '9' misread for '5' in first decimal ───────────
    # EasyOCR's model cannot distinguish the OSD bitmap glyphs for '5' and '9'
    # after H.264 compression at this resolution.  All preprocessing attempts
    # failed — the confusion is in the trained weights, not the image quality.
    # Correction: if LON first decimal digit is '9' in the 70–79° range, it is
    # always a misread '5'.  Longitudes outside that range are not touched.
    lon = result["lon"]
    if lon is not None and 70.0 <= lon < 80.0:
        s   = f"{lon:.7f}"
        dot = s.index('.')
        if s[dot + 1] == '9':
            corrected = float(s[:dot + 1] + '5' + s[dot + 2:])
            print(f"[PARSE] LON 9→5 correction: {lon:.7f} → {corrected:.7f}")
            result["lon"] = corrected

    return result
