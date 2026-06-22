"""
Headless OCR worker — single-thread pipeline.

Reads RTSP frames independently, drains stale buffered frames via grab(),
runs EasyOCR on the latest live frame, applies sanity-filter + moving-average
smoothing, and forwards validated lat/lon telemetry to PipelineManager.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from core.telemetry_parser import parse_telemetry

if TYPE_CHECKING:
    from api.pipeline import PipelineManager

# ── ROI configuration ─────────────────────────────────────────────────────────
_TELE_ROIS = [
    ("ROI: LAT/LON", 1350, 720, 570, 130),
]

# ── Timing ────────────────────────────────────────────────────────────────────
_RECONNECT_DELAY = 3.0
_FRESH_THRESHOLD = 0.020  # grab() ≥ this seconds → live frame (not buffered)

# ── OCR ───────────────────────────────────────────────────────────────────────
_MIN_CONFIDENCE    = 0.35
_LON_LAT_ALLOWLIST = '0123456789. LONATlonat'

# ── Sanity filter ─────────────────────────────────────────────────────────────
_FIELD_RANGES: dict[str, tuple[float, float]] = {
    "lat": (-90.0,   90.0),
    "lon": (-180.0, 180.0),
}

_MAX_DELTA: dict[str, float] = {
    "lat": 0.005,
    "lon": 0.005,
}

_MA_WINDOW    = 5
_DELTA_WARMUP = 1


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(roi: np.ndarray, roi_title: str) -> tuple[np.ndarray, dict]:
    """Return (processed_image, readtext_kwargs) tuned per ROI type."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if "LAT" in roi_title:
        _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright = cv2.dilate(bright, np.ones((2, 2), np.uint8), iterations=1)
        proc = cv2.bitwise_not(bright)
        kwargs = {
            "allowlist":   _LON_LAT_ALLOWLIST,
            "width_ths":   2.0,
            "ycenter_ths": 0.8,
        }
    else:
        proc   = gray
        kwargs = {}
    return proc, kwargs


def _ocr_single(args: tuple) -> tuple[str, list]:
    """Run readtext on one pre-processed ROI."""
    reader, title, proc, rtkw = args
    results = reader.readtext(proc, detail=1, batch_size=4, **rtkw)
    return title, results


# ── Worker class ──────────────────────────────────────────────────────────────

class OcrWorker:
    """
    Single daemon thread: RTSP → drain → EasyOCR → parse → filter → pipeline.

    Usage::

        worker = OcrWorker(rtsp_url=settings.rtsp_url, pipeline=pipeline)
        worker.start()   # no-op if easyocr not installed
        ...
        worker.stop()
    """

    def __init__(self, rtsp_url: str, pipeline: "PipelineManager") -> None:
        self._url      = rtsp_url
        self._pipeline = pipeline
        self._stop     = threading.Event()
        self._thread   = threading.Thread(
            target=self._run, name="ocr-worker", daemon=True
        )
        # Filter state
        self._ma_buffers: dict[str, deque] = {
            k: deque(maxlen=_MA_WINDOW) for k in _FIELD_RANGES
        }
        self._last_valid: dict[str, Optional[float]] = {k: None for k in _FIELD_RANGES}
        self._accepted:   dict[str, int]             = {k: 0    for k in _FIELD_RANGES}
        self._last_pushed: Optional[dict]            = None

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        try:
            import easyocr as _  # noqa: F401
        except ImportError:
            print("[ocr-worker] easyocr not installed — OCR telemetry disabled.")
            return
        self._thread.start()
        print("[ocr-worker] Thread started.")

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=15.0)

    # ── Filter state reset ────────────────────────────────────────────────────

    def _reset_filter_state(self) -> None:
        self._ma_buffers  = {k: deque(maxlen=_MA_WINDOW) for k in _FIELD_RANGES}
        self._last_valid  = {k: None for k in _FIELD_RANGES}
        self._accepted    = {k: 0    for k in _FIELD_RANGES}
        self._last_pushed = None

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        import easyocr

        print("[ocr-worker] Loading EasyOCR model…")
        t0 = time.perf_counter()
        try:
            reader = easyocr.Reader(["en"], gpu=True, verbose=False)
            print(f"[ocr-worker] EasyOCR ready (GPU) in {time.perf_counter()-t0:.1f}s")
        except Exception:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            print(f"[ocr-worker] EasyOCR ready (CPU) in {time.perf_counter()-t0:.1f}s")

        while not self._stop.is_set():
            cap = cv2.VideoCapture(self._url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            if not cap.isOpened():
                cap.release()
                print(f"[ocr-worker] Cannot open {self._url!r} — retry in {_RECONNECT_DELAY}s")
                self._stop.wait(timeout=_RECONNECT_DELAY)
                continue

            print(f"[ocr-worker] Connected to {self._url!r}")
            self._reset_filter_state()

            try:
                self._stream_loop(cap, reader)
            finally:
                cap.release()

            if not self._stop.is_set():
                self._stop.wait(timeout=_RECONNECT_DELAY)

        print("[ocr-worker] Stopped.")

    # ── Inner OCR loop ────────────────────────────────────────────────────────

    def _stream_loop(self, cap: cv2.VideoCapture, reader) -> None:
        _res_warned = False

        while not self._stop.is_set():
            t_cycle = time.perf_counter()
            ts      = time.strftime('%H:%M:%S')

            # 1. Drain buffered frames to the live edge
            drained = 0
            while not self._stop.is_set():
                t0 = time.perf_counter()
                if not cap.grab():
                    print(f"[{ts}] [RTSP] grab() failed — reconnecting.")
                    return
                if time.perf_counter() - t0 >= _FRESH_THRESHOLD:
                    break
                drained += 1

            if self._stop.is_set():
                return

            t_drain = time.perf_counter()
            print(f"[{ts}] [DRAIN] {drained} stale  {int((t_drain-t_cycle)*1000)}ms")

            # 2. Decode the live frame
            ret, frame = cap.retrieve()
            if not ret:
                print(f"[{ts}] [RTSP] retrieve() failed — reconnecting.")
                return

            h, w = frame.shape[:2]
            if not _res_warned:
                _res_warned = True
                if w != 1920 or h != 1080:
                    print(f"[{ts}] WARNING: frame {w}×{h}, ROIs assume 1920×1080")

            # 3. Crop ROI(s)
            tasks = []
            for title, x, y, rw, rh in _TELE_ROIS:
                x1, y1 = max(0, min(x, w)),      max(0, min(y, h))
                x2, y2 = max(0, min(x + rw, w)), max(0, min(y + rh, h))
                roi = frame[y1:y2, x1:x2]
                if roi.size == 0:
                    continue
                proc, rtkw = _preprocess(roi, title)
                tasks.append((reader, title, proc, rtkw))

            # 4. OCR
            t_ocr_start = time.perf_counter()
            ocr_outputs = [_ocr_single(t) for t in tasks]
            t_ocr_end   = time.perf_counter()
            ts = time.strftime('%H:%M:%S')
            print(f"[{ts}] [OCR] {len(tasks)} ROI  {int((t_ocr_end-t_ocr_start)*1000)}ms")

            # 5. Collect tokens above confidence threshold
            roi_texts: dict[str, list[str]] = {}
            for title, results in ocr_outputs:
                tokens = " | ".join(f"{t!r}({c:.2f})" for _, t, c in results)
                print(f"[{ts}] [RAW] {title}: {tokens or '(nothing)'}")
                kept = [t for _, t, c in results if c >= _MIN_CONFIDENCE]
                if kept:
                    roi_texts[title] = kept

            if not roi_texts:
                continue

            # 6. Parse
            t_parse_start = time.perf_counter()
            raw = parse_telemetry(roi_texts)
            t_parse_end   = time.perf_counter()
            ts = time.strftime('%H:%M:%S')
            print(
                f"[{ts}] [PARSE] lat={raw.get('lat')}  lon={raw.get('lon')}  "
                f"{int((t_parse_end-t_parse_start)*1000)}ms"
            )

            # 7. Filter + smooth
            clean = self._filter_and_smooth(raw)
            ts = time.strftime('%H:%M:%S')
            print(
                f"[{ts}] [FILTER] lat={clean.get('lat')}  lon={clean.get('lon')}"
            )

            # 8. Push to pipeline
            t_push_start = time.perf_counter()
            self._push(clean)
            t_push_end = time.perf_counter()

            ts      = time.strftime('%H:%M:%S')
            elapsed = time.perf_counter() - t_cycle
            print(
                f"[{ts}] [CYCLE] total={int(elapsed*1000)}ms  "
                f"drain={drained}f  ocr={int((t_ocr_end-t_ocr_start)*1000)}ms  "
                f"send={int((t_push_end-t_push_start)*1000)}ms"
            )

    # ── Filter + moving average ───────────────────────────────────────────────

    def _filter_and_smooth(self, raw: dict) -> dict:
        out: dict = {}
        for field, (lo, hi) in _FIELD_RANGES.items():
            value: Optional[float] = raw.get(field)

            if value is not None and not (lo <= value <= hi):
                print(f"[FILTER] REJECT {field}={value} out of range [{lo},{hi}]")
                value = None

            if value is not None:
                prev  = self._last_valid[field]
                max_d = _MAX_DELTA[field]
                if (prev is not None
                        and self._accepted[field] >= _DELTA_WARMUP
                        and abs(value - prev) > max_d):
                    print(
                        f"[FILTER] REJECT {field}={value:.5g} "
                        f"delta={abs(value-prev):.5g} > {max_d} (prev={prev:.5g})"
                    )
                    value = None

            if value is not None:
                self._last_valid[field] = value
                self._accepted[field]  += 1
                self._ma_buffers[field].append(value)

            buf = self._ma_buffers[field]
            out[field] = sum(buf) / len(buf) if buf else None

        return out

    # ── Push to pipeline ──────────────────────────────────────────────────────

    def _push(self, telemetry: dict) -> None:
        lat = telemetry.get("lat")
        lon = telemetry.get("lon")

        if lat is None or lon is None:
            print("[PUSH] skip — lat/lon not parsed")
            return

        candidate = {"lat": lat, "lon": lon}
        if candidate == self._last_pushed:
            return
        self._last_pushed = candidate

        ts = time.strftime('%H:%M:%S')
        print(f"[{ts}] [WS-PUSH] lat={lat:.7f}  lon={lon:.7f}")

        try:
            self._pipeline.update_telemetry(lat=lat, lon=lon)
        except Exception as exc:
            print(f"[PUSH] update_telemetry error: {exc}")
