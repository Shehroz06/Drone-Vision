# Drone Vision

Real-time drone flight-path monitoring from a live RTSP video feed.

Drone Vision reads GPS coordinates directly from a drone's on-screen display (OSD) via its RTSP video stream, extracts them with OCR, and renders the live flight path on an interactive map — no telemetry link or SDK integration required.

---

## Overview

| | |
|---|---|
| **License** | Proprietary — All Rights Reserved |
| **Language** | Python 3.9+ |
| **Interface** | Web (FastAPI + Leaflet.js) |
| **Platform** | Windows, Linux, macOS |

---

## Features

- **No telemetry integration required** — reads position data straight from the video feed's OSD overlay
- **Live map view** — flight path streamed to the browser over WebSocket and rendered in real time
- **Configurable pipeline** — resolution, frame rate, buffer depth, and reconnect behavior are all tunable
- **Resilient streaming** — automatic RTSP reconnect with exponential backoff
- **Noise filtering** — sanity checks and moving-average smoothing remove spurious OCR reads
- **Standalone deployment** — packaged as a single Windows executable via PyInstaller

---

## Architecture

```
RTSP Stream
    │
    ▼
Frame Capture (OpenCV)
    │
    ▼
ROI Crop → EasyOCR → Regex Parser
    │
    ▼
Sanity Filter + Moving Average
    │
    ▼
WebSocket (FastAPI) → Frontend (Leaflet.js)
```

| Component | Responsibility |
|---|---|
| `core/stream.py` | RTSP connection and frame capture |
| `core/ocr_worker.py` | OCR extraction from the OSD region |
| `core/telemetry_parser.py` | Parses raw OCR text into structured coordinates |
| `core/processor.py` | Filtering and smoothing of the coordinate stream |
| `api/` | FastAPI server, WebSocket endpoint, pipeline state |
| `frontend/` | Browser-based live map (Leaflet.js) |

---

## Requirements

- Python 3.9 or newer
- A drone (or camera) exposing an RTSP stream with a visible GPS OSD overlay

Core dependencies: OpenCV, EasyOCR, PyTorch, FastAPI, Uvicorn. See [requirements.txt](requirements.txt) for exact versions.

---

## Installation

```bash
git clone https://github.com/Shehroz06/Drone-Vision.git
cd Drone-Vision
pip install -r requirements.txt
```

---

## Usage

### Quick start

```bash
python run.py --rtsp-url "rtsp://192.168.1.1:554/live"
```

Then open:

| URL | Purpose |
|---|---|
| `http://localhost:8000` | Live flight-path map |
| `http://localhost:8000/status` | Pipeline health (JSON) |
| `http://localhost:8000/docs` | API documentation (Swagger UI) |

### Configuration options

The RTSP URL and other settings can be supplied via CLI flag, environment variable, or config file (checked in that order of precedence).

**CLI flags**

```bash
python run.py --rtsp-url rtsp://192.168.1.10:554/live --host 0.0.0.0 --port 9000
```

**Environment variables**

Copy [.env.example](.env.example) to `.env` and fill in your values:

```
RTSP_URL=rtsp://username:password@192.168.1.1:554/stream
FRAME_WIDTH=1280
FRAME_HEIGHT=720
MAX_QUEUE_SIZE=10
LOG_LEVEL=INFO
RECONNECT_DELAY=5
```

**Config file**

Create `config.yaml` in the project root, including only the values you want to override. See [assets/config_default.yaml](assets/config_default.yaml) for the full list of options and their defaults.

```yaml
rtsp_url: "rtsp://192.168.1.1:554/live"
frame_width: 1920
frame_height: 1080
fps_limit: 30.0
reconnect_delay: 5
```

### Finding your drone's RTSP URL

Stream addresses vary by manufacturer. Common formats:

```
rtsp://192.168.1.1:554/live                    ← most drones (no login)
rtsp://admin:password@192.168.1.1:554/stream   ← if login is required
rtsp://192.168.42.1:554/                       ← some DJI models
```

1. Connect your PC to the drone's WiFi hotspot.
2. Find the drone's IP address in its companion app or manual.
3. Substitute that IP for `192.168.1.1` in the URL.
4. Confirm the stream path (`/live`, `/stream`, `/video`, etc.) in the manual or app.

---

## Testing

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Building a standalone executable

A PyInstaller spec is provided for producing a self-contained Windows build.

```bash
pip install -r requirements-dev.txt
pyinstaller drone_vision.spec
```

Output is written to `dist/drone-vision/drone-vision.exe`.

---

## Known Limitations

- Currently extracts **latitude and longitude** only; other OSD fields (altitude, heading, etc.) are not yet parsed.
- Accuracy depends on OSD legibility — low contrast, motion blur, or non-standard overlay fonts can reduce OCR reliability.

---

## License

This software is proprietary and all rights are reserved. See [LICENSE](LICENSE) for full terms. Unauthorized copying, modification, or distribution is prohibited.

For licensing inquiries, contact **shehrozndm22@gmail.com**.
