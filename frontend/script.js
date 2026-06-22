'use strict';

/* ── WebSocket config ────────────────────────────────────────────────────────
   URL is derived from the page host so it works without code changes on any
   machine (localhost in dev, the GCS IP in the field).                      */

const _proto      = location.protocol === 'https:' ? 'wss:' : 'ws:';
const WS_URL      = `${_proto}//${location.host}/ws`;
const WS_RETRY_MS = 2000;

/* ── Tile layer config ───────────────────────────────────────────────────────
   LOCAL_TILE_URL is served by FastAPI from the /tiles directory.
   If a tile is missing (404), Leaflet falls back to ONLINE_TILE_URL.
   Set OFFLINE_ONLY = true to suppress the online fallback (e.g. on a GCS
   with no internet connection — missing tiles will show as blank squares).  */

const LOCAL_TILE_URL  = '/tiles/{z}/{x}/{y}.png';
const ONLINE_TILE_URL = 'https://tile.openstreetmap.org/{z}/{x}/{y}.png';
const OFFLINE_ONLY    = false;

/* ── Map ─────────────────────────────────────────────────────────────────────
   Standard Leaflet with real geographic projection (EPSG:3857).
   View is set after /config resolves so we centre on the correct area.      */

const map = L.map('map', {
  zoomSnap:           0.25,
  attributionControl: false,
  zoomControl:        false,
});

L.control.zoom({ position: 'bottomright' }).addTo(map);

// Real-world scale bar in the bottom-left (replaces the old "0.25 units" bar)
L.control.scale({ position: 'bottomleft', imperial: false, maxWidth: 120 }).addTo(map);

// Tile layer — local proxy with OSM fallback on tile error
L.tileLayer(LOCAL_TILE_URL, {
  maxZoom:      19,
  errorTileUrl: ONLINE_TILE_URL,
  attribution:  '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
}).addTo(map);

/* ── Path layers ─────────────────────────────────────────────────────────────
   Created once, updated in-place on every accepted GPS fix.
   setLatLngs() edits the SVG path attribute directly — no flicker.         */

const _polyline = L.polyline([], {
  color:       '#1a73e8',
  weight:       4,
  opacity:      0.9,
  lineJoin:    'round',
  lineCap:     'round',
  interactive:  false,
}).addTo(map);

// Position marker: pulsing blue dot (Google-Maps puck style)
const _posMarker = L.marker([0, 0], {
  icon: L.divIcon({
    html:       '<div class="pos-dot"></div>',
    className:  '',
    iconSize:   [16, 16],
    iconAnchor: [8, 8],
  }),
  zIndexOffset: 1000,
  interactive:  false,
}).addTo(map);

_posMarker.setOpacity(0);   // hidden until first real data arrives

/* ── Draw ────────────────────────────────────────────────────────────────────
   lat/lon come directly from OCR-extracted telemetry.
   Every accepted point is appended to _pathCoords so the full flight trail
   is drawn as a polyline; the marker shows the current position.            */

// Pan only when the drone moves more than this distance (degrees ~1 km)
const _PAN_THRESHOLD_DEG  = 0.005;

let _lastDrawLat = null;
let _lastDrawLon = null;
const _pathCoords = [];   // accumulates every accepted [lat, lon]

function draw(lat, lon) {
  if (lat == null || lon == null) return;

  _pathCoords.push([lat, lon]);
  _polyline.setLatLngs(_pathCoords);

  _posMarker.setLatLng([lat, lon]);
  _posMarker.setOpacity(1);

  // Pan only when the drone has actually moved enough to warrant it
  if (_lastDrawLat == null ||
      Math.abs(lat - _lastDrawLat) > _PAN_THRESHOLD_DEG ||
      Math.abs(lon - _lastDrawLon) > _PAN_THRESHOLD_DEG) {
    map.panTo([lat, lon]);
  }

  _lastDrawLat = lat;
  _lastDrawLon = lon;
}

/* ── Telemetry panel ─────────────────────────────────────────────────────────
   Updates the top-right overlay from the speed / altitude / coord fields
   sent in every WebSocket message.                                          */

function updateTelemetry(data) {
  document.getElementById('tele-lat').textContent =
    data.lat != null ? Number(data.lat).toFixed(6) : '—';
  document.getElementById('tele-lon').textContent =
    data.lon != null ? Number(data.lon).toFixed(6) : '—';
}

/* ── Status helpers ──────────────────────────────────────────────────────── */

function setStatus(state) {
  const states = {
    connecting:   { cls: 'yellow', text: 'Connecting…'    },
    connected:    { cls: 'green',  text: 'Connected'       },
    reconnecting: { cls: 'yellow', text: 'Reconnecting…'   },
    error:        { cls: 'red',    text: 'Connection error' },
    'no-data':    { cls: 'yellow', text: 'Waiting for data' },
  };
  const s = states[state] ?? states.connecting;
  document.getElementById('dot').className           = `dot ${s.cls}`;
  document.getElementById('status-text').textContent = s.text;
}

/* ── Message handler ─────────────────────────────────────────────────────── */

let _updateCount = 0;

function onMessage(data) {
  const lat = data.lat;
  const lon = data.lon;

  draw(lat, lon);
  updateTelemetry(data);

  const d = data.timestamp ? new Date(data.timestamp) : new Date();
  document.getElementById('foot-time').textContent  = d.toLocaleTimeString();
  document.getElementById('foot-count').textContent = ++_updateCount;

  // Keep status as 'connected' once WS is open; only flag 'no-data' before first connection
  if (document.getElementById('dot').className.includes('green')) return;
  setStatus(lat != null && lon != null ? 'connected' : 'no-data');
}

/* ── WebSocket ───────────────────────────────────────────────────────────── */

let _ws         = null;
let _retryTimer = null;

function connect() {
  clearTimeout(_retryTimer);
  setStatus('connecting');

  _ws = new WebSocket(WS_URL);

  _ws.onopen = () => {
    console.log('[WS] connected to', WS_URL);
    setStatus('connected');
  };

  _ws.onmessage = ({ data }) => {
    try {
      const msg = JSON.parse(data);
      console.log('[WS] message lat=' + msg.lat + ' lon=' + msg.lon);
      onMessage(msg);
    }
    catch (e) { console.error('[WS] parse error', e, data); }
  };

  _ws.onerror = (e) => {
    console.error('[WS] error', e);
    setStatus('error');
  };

  _ws.onclose = (e) => {
    console.warn('[WS] closed code=' + e.code + ' reason=' + e.reason + ' clean=' + e.wasClean);
    _ws = null;
    setStatus('reconnecting');
    _retryTimer = setTimeout(connect, WS_RETRY_MS);
  };
}

/* ── Startup — fetch geo bounds, centre map, open WebSocket ─────────────────
   /config returns the same geo_bounds configured in config/settings.py so
   the map is always centred on the correct area without duplicating values. */

fetch('/config')
  .then(r => r.json())
  .then(({ geo_bounds: b }) => {
    const sw = [b.lat_min, b.lon_min];
    const ne = [b.lat_max, b.lon_max];
    map.fitBounds([sw, ne], { padding: [60, 60] });
    connect();
  })
  .catch(() => {
    // Server not yet ready — fall back to world view and connect anyway
    map.setView([20, 0], 2);
    connect();
  });
