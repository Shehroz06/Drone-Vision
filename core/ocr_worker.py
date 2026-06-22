"""
Headless OCR worker — single-thread pipeline.

Reads RTSP frames independently, drains stale buffered frames via grab(),
runs EasyOCR on the latest live frame, applies sanity-filter + moving-average
smoothing, and forwards validated lat/lon telemetry to PipelineManager.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import cv2
import numpy as np

from core.telemetry_parser import parse_telemetry

if TYPE_CHECKING:
    from api.pipeline import PipelineManager


def _easyocr_model_dir() -> str:
    """
    Return the EasyOCR model directory.

    Dev mode  → project root / easyocr_models   (populated by prepare_models.py)
    Frozen    → sys._MEIPASS / easyocr_models    (bundled by PyInstaller spec)

    If the directory does not exist, returns None so EasyOCR falls back to its
    default ~/.EasyOCR/model/ and downloads on first use.
    """
    if getattr(sys, 'frozen', False):
        base = Path(sys._MEIPASS)
    else:
        base = Path(__file__).parent.parent
    d = base / 'easyocr_models'
    return str(d) if d.is_dir() else None


# ── ROI configuration ─────────────────────────────────────────────────────────
_TELE_ROIS = [
    ("ROI: LAT/LON", 1350, 720, 570, 130),
]

# ── Timing ────────────────────────────────────────────────────────────────────
_RECONNECT_DELAY  = 3.0
_FRESH_THRESHOLD  = 0.020  # grab() ≥ this seconds → live frame (not buffered)
_MAX_DRAIN        = 200    # cap buffered-frame drain; prevents OCR stall after large bursts

# ── OCR ───────────────────────────────────────────────────────────────────────
_MIN_CONFIDENCE    = 0.20
_LON_LAT_ALLOWLIST = '0123456789. LONATlonat'
# CRAFT (text detector) runs every Nth OCR cycle; CRNN (recognizer) runs every cycle.
# OSD bounding-box positions are fixed frame-to-frame — only digit values change —
# so caching the boxes saves ~1.2 s of CRAFT compute on cache-hit frames.
_CRAFT_INTERVAL   = 5

# ── Sanity filter ─────────────────────────────────────────────────────────────
_FIELD_RANGES: dict[str, tuple[float, float]] = {
    "lat": (-90.0,   90.0),
    "lon": (-180.0, 180.0),
}

_MAX_DELTA: dict[str, float] = {
    "lat": 0.010,
    "lon": 0.010,
}

_MA_WINDOW    = 5
_DELTA_WARMUP = 1
# Sliding-window reset: if the last _REJECT_RESET rejected values for a field all
# fall within _REJECT_CLUSTER_TOL of each other, the filter seeded on a wrong value
# and the new cluster IS the real location — reset and accept it.
# Random OCR garbage spans tens of degrees; a real stuck-seed spans < 0.02°.
_REJECT_RESET       = 5
_REJECT_CLUSTER_TOL = 0.02


# ── Image preprocessing ───────────────────────────────────────────────────────

def _preprocess(roi: np.ndarray, roi_title: str) -> tuple[np.ndarray, dict]:
    """Return (processed_image, readtext_kwargs) tuned per ROI type."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    if "LAT" in roi_title:
        _, bright = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
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
    results = reader.readtext(proc, detail=1, batch_size=1, decoder='greedy', **rtkw)
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
        self._last_valid:     dict[str, Optional[float]] = {k: None for k in _FIELD_RANGES}
        self._accepted:       dict[str, int]             = {k: 0    for k in _FIELD_RANGES}
        self._consec_rejects: dict[str, int]             = {k: 0    for k in _FIELD_RANGES}
        # Sliding window of rejected values — used to detect a stable new location
        # vs scattered OCR garbage before deciding to reset the anchor.
        self._rejected_vals: dict[str, deque] = {
            k: deque(maxlen=_REJECT_RESET) for k in _FIELD_RANGES
        }
        self._last_pushed: Optional[dict] = None
        # CRAFT bbox cache — populated on first frame and every _CRAFT_INTERVAL frames
        self._bbox_cache: Optional[tuple] = None   # (h_list, f_list)
        self._bbox_age:   int             = 0      # frames since last CRAFT run

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
        self._ma_buffers     = {k: deque(maxlen=_MA_WINDOW) for k in _FIELD_RANGES}
        self._last_valid     = {k: None for k in _FIELD_RANGES}
        self._accepted       = {k: 0    for k in _FIELD_RANGES}
        self._consec_rejects = {k: 0    for k in _FIELD_RANGES}
        self._rejected_vals  = {k: deque(maxlen=_REJECT_RESET) for k in _FIELD_RANGES}
        self._last_pushed    = None
        self._bbox_cache     = None
        self._bbox_age       = 0

    # ── Worker thread ─────────────────────────────────────────────────────────

    def _run(self) -> None:
        import torch
        import easyocr

        # Limit PyTorch to 4 threads.  Using all cores causes sustained thermal
        # throttling on laptops — the first OCR call is 2s then spikes to 18s as
        # the CPU overheats.  4 threads keeps clock speed stable.
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)

        model_dir = _easyocr_model_dir()
        model_kwargs = {"model_storage_directory": model_dir} if model_dir else {}

        print("[ocr-worker] Loading EasyOCR model…")
        t0 = time.perf_counter()
        try:
            reader = easyocr.Reader(["en"], gpu=True, verbose=False, **model_kwargs)
            print(f"[ocr-worker] EasyOCR ready (GPU) in {time.perf_counter()-t0:.1f}s")
        except Exception:
            reader = easyocr.Reader(["en"], gpu=False, verbose=False, **model_kwargs)
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

            # 1. Drain buffered frames to the live edge (capped at _MAX_DRAIN)
            drained = 0
            while not self._stop.is_set():
                t0 = time.perf_counter()
                if not cap.grab():
                    print(f"[{ts}] [RTSP] grab() failed — reconnecting.")
                    return
                if time.perf_counter() - t0 >= _FRESH_THRESHOLD or drained >= _MAX_DRAIN:
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

            # 3. Crop the single telemetry ROI and preprocess
            _title, _rx, _ry, _rw, _rh = _TELE_ROIS[0]
            x1 = max(0, min(_rx,        w));  x2 = max(0, min(_rx + _rw, w))
            y1 = max(0, min(_ry,        h));  y2 = max(0, min(_ry + _rh, h))
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                continue
            proc, rtkw = _preprocess(roi, _title)

            # CRAFT (detect) needs a BGR image; CRNN (recognize) uses grayscale.
            # Our preprocessed `proc` is already grayscale — convert once per cycle.
            img_grey = proc
            img_bgr  = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)

            # 4. OCR — CRAFT/CRNN split
            #
            # Strategy: CRAFT finds text bounding-box positions; for a fixed OSD
            # overlay those positions are stable across frames.  We cache them and
            # only re-run CRAFT every _CRAFT_INTERVAL frames.  CRNN recognition
            # runs every frame on the current image content using the cached boxes.
            #
            #   Full cycle  (CRAFT + CRNN): ~2 s   (every _CRAFT_INTERVAL frames)
            #   Cache cycle (CRNN only)   : ~0.4 s  (all other frames)
            #   Average latency @ interval=5:  (2 + 0.4×4) / 5 ≈ 0.72 s
            t_ocr_start = time.perf_counter()

            # 4a. CRAFT detection
            need_craft = (self._bbox_cache is None or
                          self._bbox_age  >= _CRAFT_INTERVAL)
            if need_craft:
                t_craft = time.perf_counter()
                try:
                    h_lists, f_lists = reader.detect(
                        img_bgr,
                        reformat=False,
                        width_ths=rtkw.get("width_ths",   0.5),
                        ycenter_ths=rtkw.get("ycenter_ths", 0.5),
                    )
                    h_list = h_lists[0] if h_lists else []
                    f_list = f_lists[0] if f_lists else []
                except Exception as exc:
                    print(f"[{ts}] [CRAFT] detect error: {exc}")
                    h_list, f_list = [], []

                craft_ms = int((time.perf_counter() - t_craft) * 1000)
                if h_list or f_list:
                    self._bbox_cache = (h_list, f_list)
                    self._bbox_age   = 0
                    print(f"[{ts}] [CRAFT] detect={craft_ms}ms  boxes={len(h_list)}")
                else:
                    # Detection found nothing — don't update cache; try again next frame
                    print(f"[{ts}] [CRAFT] detect={craft_ms}ms  no boxes — skip")
                    continue
            else:
                h_list, f_list = self._bbox_cache
                self._bbox_age += 1

            # 4b. CRNN recognition on cached bboxes + current image
            t_crnn = time.perf_counter()
            try:
                ocr_results = reader.recognize(
                    img_grey,
                    h_list,
                    f_list,
                    decoder='greedy',
                    batch_size=1,
                    allowlist=rtkw.get("allowlist"),
                    detail=1,
                )
            except Exception as exc:
                print(f"[{ts}] [CRNN] recognize error: {exc}")
                # Force CRAFT re-run next cycle in case the error is bbox-related
                self._bbox_cache = None
                continue

            t_ocr_end = time.perf_counter()
            crnn_ms   = int((t_ocr_end  - t_crnn)     * 1000)
            total_ms  = int((t_ocr_end  - t_ocr_start) * 1000)
            ts = time.strftime('%H:%M:%S')
            mode = "craft+crnn" if need_craft else f"crnn(cached×{self._bbox_age})"
            print(f"[{ts}] [OCR] {mode}  crnn={crnn_ms}ms  total={total_ms}ms")

            # 5. Collect tokens above confidence threshold
            roi_texts: dict[str, list[str]] = {}
            tokens = " | ".join(f"{t!r}({c:.2f})" for _, t, c in ocr_results)
            print(f"[{ts}] [RAW] {_title}: {tokens or '(nothing)'}")
            kept = [t for _, t, c in ocr_results if c >= _MIN_CONFIDENCE]
            if kept:
                roi_texts[_title] = kept

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
                f"drain={drained}f  ocr={total_ms}ms  crnn={crnn_ms}ms  "
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

                    self._consec_rejects[field] += 1
                    self._rejected_vals[field].append(value)

                    # Check whether the rejected values cluster together.
                    # If they do, the current anchor is the wrong seed and
                    # the cluster is the real location → reset and accept.
                    # If they scatter (random OCR garbage), keep rejecting.
                    rvals = list(self._rejected_vals[field])
                    clustered = (
                        len(rvals) >= _REJECT_RESET
                        and (max(rvals) - min(rvals)) <= _REJECT_CLUSTER_TOL
                    )

                    if clustered:
                        new_anchor = sum(rvals) / len(rvals)
                        print(
                            f"[FILTER] RESET {field}: last {_REJECT_RESET} rejects "
                            f"cluster near {new_anchor:.5g} "
                            f"(span={max(rvals)-min(rvals):.4g}) "
                            f"— was={prev:.5g} → accepting"
                        )
                        self._last_valid[field]     = None
                        self._accepted[field]       = 0
                        self._consec_rejects[field] = 0
                        self._rejected_vals[field].clear()
                        self._ma_buffers[field].clear()
                        # value stays set → accepted by the block below
                    else:
                        print(
                            f"[FILTER] REJECT {field}={value:.5g} "
                            f"delta={abs(value-prev):.5g} > {max_d} "
                            f"(prev={prev:.5g}) [{self._consec_rejects[field]}]"
                        )
                        value = None

            if value is not None:
                self._consec_rejects[field] = 0
                self._last_valid[field]     = value
                self._accepted[field]      += 1
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
