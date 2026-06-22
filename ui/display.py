from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np
import easyocr

from core.telemetry_parser import parse_telemetry

if TYPE_CHECKING:
    from api.pipeline import PipelineManager

# ── Telemetry ROI definitions (frame resolution: 1920×1080) ──────────────────
# Each entry: (window_title, x, y, w, h)
# Adjust coordinates here if the OSD layout shifts between streams.
_TELE_ROIS = [
    ("ROI: LAT/LON",  1350, 720, 570, 130),  # bottom-right — "LON …" + "LAT …"
    ("ROI: Speed",    1580, 355, 340,  80),  # right-center — "≈154 km/h"
    ("ROI: Altitude",    0,   0, 280, 160),  # top-left     — "AIR 226M"
]

_OCR_EVERY_N_FRAMES = 10


def _preprocess_roi(roi: np.ndarray) -> np.ndarray:
    """
    Prepare a raw BGR ROI crop for OCR.

    Uses adaptive Gaussian threshold instead of Otsu so bright-background
    ROIs (e.g. altitude / sky) produce clean binary output even when the
    histogram has no bimodal peak.

    1. Grayscale  2. Adaptive threshold (BINARY_INV)  3. 2× upscale
    Returns a single-channel binary image (0 or 255).
    """
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    proc = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        15, 4,
    )
    return cv2.resize(proc, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


class Display:
    def __init__(
        self,
        window_name: str = "RTSP Stream",
        debug_rois:  bool = True,
        pipeline: Optional[PipelineManager] = None,
    ):
        self._window      = window_name
        self._debug_rois  = debug_rois
        self._pipeline    = pipeline
        self._frame_count = 0
        self._smoothed: Optional[dict] = None  # EMA state; doubles as between-frame cache

        # EasyOCR reader — created once; first run downloads model weights (~50 MB).
        print("[display] Loading EasyOCR model (first run downloads weights)…")
        self._reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        print("[display] EasyOCR ready.")

    def show(self, frame) -> bool:
        cv2.imshow(self._window, frame)
        self._frame_count += 1

        if self._debug_rois:
            self._process_rois(frame)

        return cv2.waitKey(1) & 0xFF != ord("q")

    def _process_rois(self, frame: np.ndarray) -> None:
        h, w = frame.shape[:2]
        run_ocr = (self._frame_count % _OCR_EVERY_N_FRAMES == 0)
        roi_texts: dict[str, list[str]] = {}

        for title, x, y, rw, rh in _TELE_ROIS:
            x1 = max(0, min(x, w))
            y1 = max(0, min(y, h))
            x2 = max(0, min(x + rw, w))
            y2 = max(0, min(y + rh, h))

            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue

            processed = _preprocess_roi(roi)
            cv2.imshow(title, roi)
            cv2.imshow(title + " [proc]", processed)

            if not run_ocr:
                continue

            # reader.readtext() → [(bbox, text, confidence), ...]
            results = self._reader.readtext(processed, detail=1)

            raw_strings = [text for _, text, conf in results if conf >= 0.4]
            roi_texts[title] = raw_strings

            raw_log = " | ".join(
                f"{text!r} ({conf:.2f})" for _, text, conf in results
            )
            print(f"[OCR raw] {title}: {raw_log or '(nothing)'}")

        if run_ocr and roi_texts:
            raw      = parse_telemetry(roi_texts)
            smoothed = self._smooth(raw)
            print(f"[telemetry raw]    {raw}")
            print(f"[telemetry smooth] {smoothed}")
            self._push_telemetry(smoothed)
        elif self._smoothed is not None:
            self._push_telemetry(self._smoothed)

    def _smooth(self, raw: dict) -> dict:
        """
        Exponential moving average over telemetry fields.

        weight for new reading = 0.2  (α)
        weight for history     = 0.8  (1 - α)

        Rules:
          - First call: initialise smoothed state directly from raw values.
          - Subsequent calls: blend only fields that are not None in raw;
            if a field is None (OCR miss this cycle) keep the previous value.
        """
        if self._smoothed is None:
            self._smoothed = dict(raw)
            return self._smoothed

        updated: dict = {}
        for key in ("lat", "lon", "speed", "altitude"):
            new  = raw.get(key)
            prev = self._smoothed.get(key)
            if new is not None and prev is not None:
                updated[key] = 0.8 * prev + 0.2 * new
            elif new is not None:
                updated[key] = new    # first valid reading for this field
            else:
                updated[key] = prev   # OCR miss — hold last smoothed value
        self._smoothed = updated
        return updated

    def _push_telemetry(self, telemetry: dict) -> None:
        """Validate parsed values and forward them to the pipeline if one is wired up."""
        if self._pipeline is None:
            return

        lat = telemetry.get("lat")
        lon = telemetry.get("lon")

        # lat/lon are mandatory — skip if OCR failed to extract either
        if lat is None or lon is None:
            print("[telemetry] skip update: lat/lon not parsed")
            return

        # Sanity-check WGS-84 range — catches OCR garbage like "999.0" or "-0.003"
        if not (-90.0 <= lat <= 90.0):
            print(f"[telemetry] skip update: lat={lat} out of range")
            return
        if not (-180.0 <= lon <= 180.0):
            print(f"[telemetry] skip update: lon={lon} out of range")
            return

        # speed/altitude are optional — fall back to 0.0 so pipeline always
        # receives a valid numeric value even when those ROIs fail OCR
        speed    = telemetry.get("speed")    or 0.0
        altitude = telemetry.get("altitude") or 0.0

        try:
            self._pipeline.update_telemetry(
                lat      = lat,
                lon      = lon,
                speed    = speed,
                altitude = altitude,
            )
        except Exception as exc:
            print(f"[telemetry] update_telemetry error: {exc}")

    def close(self):
        cv2.destroyAllWindows()
