# RTSP Vision — Real-Time Drone Path Monitor

Reads live GPS coordinates from a drone's OSD (on-screen display) via RTSP, and plots the flight path on an interactive map in real time.
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

---

## Quick Start

**1. Clone and install**

```bash
git clone https://github.com/Shehroz06/RTSP-VISION.git
cd RTSP-VISION
pip install -r requirements.txt
```

**2. Run**

Replace the URL below with your drone's stream address (see next section):

```bash
python run.py --rtsp-url "rtsp://192.168.1.1:554/live"
```

**3. Open the map**

```
http://localhost:8000
```

---

## Finding Your RTSP URL

Every drone has a different stream address. Common formats:

```
rtsp://192.168.1.1:554/live                       ← most drones (no login)
rtsp://admin:password@192.168.1.1:554/stream      ← if login is required
rtsp://192.168.42.1:554/                           ← some DJI models
```

**Steps:**
1. Connect your PC to the drone's WiFi hotspot
2. Find the drone's IP address in its companion app or manual
3. Replace `192.168.1.1` in the URL with that IP
4. Check the manual or app for the correct stream path (e.g. `/live`, `/stream`, `/video`)

---

## Changing Settings

Create a file called `config.yaml` in the project folder. You only need to include the lines you want to change:

```yaml
rtsp_url: "rtsp://192.168.1.1:554/live"   # your stream address
frame_width: 1920                          # video width
frame_height: 1080                         # video height
fps_limit: 30.0                            # max frames per second
reconnect_delay: 5                         # seconds to wait before reconnecting
```

See `assets/config_default.yaml` for all available options and their defaults.

---

## Notes

- Currently reads **latitude and longitude** only.
- Requires Python 3.9 or newer.
