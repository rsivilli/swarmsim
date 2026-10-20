"""ZSim Detection Monitor — serves a dashboard at http://localhost:9090.

Uses a native Zenoh subscriber to capture live detection data from ZSim,
plus the REST API for admin info (sessions, router stats).

Each message on tracks/{sensorName} is a JSON array of detection objects:
  [{"simTime":1.05,"sensorId":0,"entityIndex":42,"position":[120.5,20.0,340.2],
    "range":210.50,"bearingDeg":45.30,"elevationDeg":5.20,"snr":18.50}, ...]

Usage:
    python zenoh-monitor.py [--zenoh-router tcp/localhost:7447] [--zenoh-url http://localhost:8000] [--port 9090]

Requires: pip install eclipse-zenoh
"""

import argparse
import http.server
import json
import threading
import time
import urllib.request
import urllib.error
from collections import OrderedDict

ZENOH_URL = "http://localhost:8000"

# Shared state for detection samples received via native subscriber
_track_lock = threading.Lock()
_track_latest: OrderedDict[str, dict] = OrderedDict()  # key_expr -> per-key stats
_track_stats = {
    "total_received": 0,
    "msg_per_sec": 0.0,
    "start_time": 0.0,
    "total_detections": 0,
    "det_per_sec": 0.0,
    "unique_entities": 0,
}
_global_entities: set[int] = set()
_MAX_TRACK_KEYS = 200


def _start_zenoh_subscriber(router_endpoint: str):
    """Background thread: subscribe to tracks/** via native Zenoh."""
    import zenoh

    config = zenoh.Config()
    config.insert_json5("mode", json.dumps("client"))
    config.insert_json5("connect/endpoints", json.dumps([router_endpoint]))
    config.insert_json5("scouting/multicast/enabled", json.dumps(False))

    session = zenoh.open(config)
    print(f"Zenoh subscriber connected to {router_endpoint}")

    _track_stats["start_time"] = time.time()
    _last_msg_count = [0]
    _last_det_count = [0]
    _last_time = [time.time()]

    def _on_sample(sample):
        key = str(sample.key_expr)
        try:
            value = json.loads(bytes(sample.payload))
        except Exception:
            value = None

        with _track_lock:
            _track_stats["total_received"] += 1
            now = time.time()

            # Parse detection array
            batch_size = 0
            batch_entities: set[int] = set()
            snr_min = snr_max = range_min = range_max = None
            if isinstance(value, list):
                batch_size = len(value)
                for det in value:
                    if isinstance(det, dict):
                        eidx = det.get("entityIndex")
                        if eidx is not None:
                            batch_entities.add(eidx)
                            _global_entities.add(eidx)
                        snr = det.get("snr")
                        if snr is not None:
                            snr_min = snr if snr_min is None else min(snr_min, snr)
                            snr_max = snr if snr_max is None else max(snr_max, snr)
                        rng = det.get("range")
                        if rng is not None:
                            range_min = rng if range_min is None else min(range_min, rng)
                            range_max = rng if range_max is None else max(range_max, rng)

            _track_stats["total_detections"] += batch_size
            _track_stats["unique_entities"] = len(_global_entities)

            # Smoothed rate calculation (every 1s)
            dt = now - _last_time[0]
            if dt >= 1.0:
                _track_stats["msg_per_sec"] = (
                    _track_stats["total_received"] - _last_msg_count[0]
                ) / dt
                _track_stats["det_per_sec"] = (
                    _track_stats["total_detections"] - _last_det_count[0]
                ) / dt
                _last_msg_count[0] = _track_stats["total_received"]
                _last_det_count[0] = _track_stats["total_detections"]
                _last_time[0] = now

            if key in _track_latest:
                entry = _track_latest[key]
                entry["count"] += 1
                entry["ts"] = now
                entry["det_count"] += batch_size
                entry["last_batch_size"] = batch_size
                entry["entities"].update(batch_entities)
                entry["last_snr_min"] = snr_min
                entry["last_snr_max"] = snr_max
                entry["last_range_min"] = range_min
                entry["last_range_max"] = range_max
                _track_latest.move_to_end(key)
            else:
                if len(_track_latest) >= _MAX_TRACK_KEYS:
                    _track_latest.popitem(last=False)
                _track_latest[key] = {
                    "count": 1,
                    "ts": now,
                    "det_count": batch_size,
                    "last_batch_size": batch_size,
                    "entities": set(batch_entities),
                    "last_snr_min": snr_min,
                    "last_snr_max": snr_max,
                    "last_range_min": range_min,
                    "last_range_max": range_max,
                }

    session.declare_subscriber("tracks/**", _on_sample)
    print("Subscribed to tracks/**")

    # Keep thread alive
    while True:
        time.sleep(1)


HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>ZSim Detection Monitor</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #0a0e17; color: #c9d1d9; padding: 20px; }
  h1 { color: #58a6ff; margin-bottom: 16px; font-size: 22px; }
  h2 { color: #8b949e; font-size: 14px; text-transform: uppercase;
       letter-spacing: 1px; margin: 20px 0 8px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #161b22; border: 1px solid #30363d; border-radius: 8px;
          padding: 16px; }
  .card.full { grid-column: 1 / -1; }
  .stat { display: inline-block; background: #21262d; border-radius: 6px;
          padding: 8px 14px; margin: 4px; }
  .stat .label { font-size: 11px; color: #8b949e; }
  .stat .value { font-size: 20px; font-weight: 600; color: #58a6ff; }
  .stat .value.green { color: #3fb950; }
  .stat .value.orange { color: #d29922; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; color: #8b949e; font-weight: 500; padding: 6px 8px;
       border-bottom: 1px solid #30363d; }
  td { padding: 6px 8px; border-bottom: 1px solid #21262d; font-family: monospace; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 11px; font-weight: 600; }
  .badge-client { background: #1f3a2e; color: #3fb950; }
  .badge-router { background: #2a1f3a; color: #a371f7; }
  .badge-peer { background: #3a2f1f; color: #d29922; }
  .fresh { animation: flash 0.5s; }
  @keyframes flash { from { background: #1f3a2e; } to { background: transparent; } }
  #status { position: fixed; top: 12px; right: 20px; font-size: 12px; color: #8b949e; }
  .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%;
         margin-right: 6px; }
  .dot.ok { background: #3fb950; }
  .dot.err { background: #f85149; }
  .refresh-control { position: fixed; top: 34px; right: 20px; font-size: 12px; }
  .refresh-control select { background: #21262d; color: #c9d1d9; border: 1px solid #30363d;
                            border-radius: 4px; padding: 2px 6px; }
  .ago { color: #484f58; font-size: 11px; }
</style>
</head>
<body>
<h1>ZSim Detection Monitor</h1>
<div id="status"><span class="dot" id="dot"></span><span id="status-text">connecting...</span></div>
<div class="refresh-control">
  <label>Refresh:
    <select id="interval" onchange="setInterval_()">
      <option value="500">0.5s</option>
      <option value="1000" selected>1s</option>
      <option value="2000">2s</option>
      <option value="5000">5s</option>
    </select>
  </label>
</div>
<div class="grid">
  <div class="card" id="router-card">
    <h2>Router</h2>
    <div id="router-info">loading...</div>
  </div>
  <div class="card" id="stats-card">
    <h2>Throughput</h2>
    <div id="stats-info">loading...</div>
  </div>
  <div class="card full" id="sessions-card">
    <h2>Connected Sessions</h2>
    <div id="sessions-info">loading...</div>
  </div>
  <div class="card full" id="tracks-card">
    <h2>Live Detection Feed</h2>
    <div id="tracks-info">waiting for data...</div>
  </div>
</div>

<script>
let timer = null;
let prevCounts = {};

async function fetchJ(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(r.statusText);
  return r.json();
}

function setStatus(ok, msg) {
  document.getElementById('dot').className = 'dot ' + (ok ? 'ok' : 'err');
  document.getElementById('status-text').textContent = msg;
}

function stat(label, value, cls) {
  return `<div class="stat"><div class="label">${label}</div><div class="value ${cls||''}">${value}</div></div>`;
}

function badge(type) {
  return `<span class="badge badge-${type}">${type}</span>`;
}

function ago(ts) {
  const s = (Date.now()/1000 - ts);
  if (s < 1) return 'just now';
  if (s < 60) return Math.floor(s) + 's ago';
  return Math.floor(s/60) + 'm ago';
}

async function refresh() {
  try {
    // Admin info via REST proxy
    const all = await fetchJ('/z/@/**');
    setStatus(true, 'connected — ' + new Date().toLocaleTimeString());

    const sessions = [];
    const links = {};
    let pluginInfo = [];

    for (const item of all) {
      const k = item.key;
      if (k.includes('/session/transport/unicast/') && !k.includes('/link/')) {
        sessions.push({ zid: item.value.zid, whatami: item.value.whatami });
      } else if (k.includes('/link/')) {
        const m = k.match(/unicast\\/([a-f0-9]+)\\/link/);
        if (m) links[m[1]] = item.value;
      } else if (k.includes('/router/plugins/')) {
        pluginInfo.push(item.value);
      }
    }

    // Router card
    const routerZid = all.length > 0 ? all[0].key.split('/')[1] : 'unknown';
    const plugins = pluginInfo.map(p => `${p.name} v${p.version}`).join(', ') || 'none';
    document.getElementById('router-info').innerHTML =
      `<div style="font-family:monospace;font-size:12px;margin-bottom:8px">ZID: ${routerZid.substring(0,16)}...</div>` +
      `<div style="font-size:13px">Plugins: ${plugins}</div>` +
      `<div style="font-size:13px;margin-top:4px">Sessions: ${sessions.length}</div>`;

    // Sessions table
    if (sessions.length === 0) {
      document.getElementById('sessions-info').innerHTML = '<div style="color:#8b949e;padding:8px">No sessions</div>';
    } else {
      let html = '<table><tr><th>ZID</th><th>Type</th><th>Remote Address</th></tr>';
      for (const s of sessions) {
        const link = links[s.zid];
        const addr = link ? link.dst : '—';
        html += `<tr><td>${s.zid.substring(0,20)}...</td><td>${badge(s.whatami)}</td><td>${addr}</td></tr>`;
      }
      html += '</table>';
      document.getElementById('sessions-info').innerHTML = html;
    }

    // Detection stats from native subscriber
    const trackData = await fetchJ('/tracks');
    const st = trackData.stats;
    const now = Date.now() / 1000;
    const activeSensors = Object.values(trackData.tracks).filter(t => (now - t.ts) < 5).length;

    document.getElementById('stats-info').innerHTML =
      stat('msg/sec', st.msg_per_sec.toFixed(1), 'green') +
      stat('det/sec', st.det_per_sec.toFixed(1), 'green') +
      stat('Total Det', st.total_detections.toLocaleString()) +
      stat('Entities', st.unique_entities.toLocaleString(), 'orange') +
      stat('Active Sensors', activeSensors, activeSensors > 0 ? 'green' : '');

    // Detection feed table
    const entries = Object.entries(trackData.tracks);
    if (entries.length === 0) {
      document.getElementById('tracks-info').innerHTML = '<div style="color:#8b949e;padding:8px">No detection data yet — start ZSim with communication enabled</div>';
    } else {
      entries.sort((a, b) => b[1].ts - a[1].ts);
      let html = '<table><tr><th>Key</th><th>Msgs</th><th>Det</th><th>Batch</th><th>Entities</th><th>SNR</th><th>Range</th><th>Last Seen</th></tr>';
      for (const [key, info] of entries.slice(0, 50)) {
        const isNew = (prevCounts[key] || 0) < info.count;
        const snr = (info.last_snr_min != null && info.last_snr_max != null)
          ? `${info.last_snr_min.toFixed(1)}–${info.last_snr_max.toFixed(1)}`
          : '—';
        const rng = (info.last_range_min != null && info.last_range_max != null)
          ? `${info.last_range_min.toFixed(0)}–${info.last_range_max.toFixed(0)}`
          : '—';
        html += `<tr class="${isNew ? 'fresh' : ''}">` +
          `<td>${key}</td>` +
          `<td>${info.count.toLocaleString()}</td>` +
          `<td>${info.det_count.toLocaleString()}</td>` +
          `<td>${info.last_batch_size}</td>` +
          `<td>${info.entity_count}</td>` +
          `<td>${snr}</td>` +
          `<td>${rng}</td>` +
          `<td class="ago">${ago(info.ts)}</td></tr>`;
        prevCounts[key] = info.count;
      }
      if (entries.length > 50) html += `<tr><td colspan="8" style="color:#8b949e">...and ${entries.length - 50} more</td></tr>`;
      html += '</table>';
      document.getElementById('tracks-info').innerHTML = html;
    }

  } catch(e) {
    setStatus(false, 'error: ' + e.message);
  }
}

function setInterval_() {
  if (timer) clearInterval(timer);
  const ms = parseInt(document.getElementById('interval').value);
  timer = setInterval(refresh, ms);
}

refresh();
setInterval_();
</script>
</body>
</html>
"""


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())
        elif self.path == "/tracks":
            # Return native subscriber data (convert sets to counts for JSON)
            with _track_lock:
                tracks_out = {}
                for k, v in _track_latest.items():
                    tracks_out[k] = {
                        "count": v["count"],
                        "ts": v["ts"],
                        "det_count": v["det_count"],
                        "last_batch_size": v["last_batch_size"],
                        "entity_count": len(v["entities"]),
                        "last_snr_min": v["last_snr_min"],
                        "last_snr_max": v["last_snr_max"],
                        "last_range_min": v["last_range_min"],
                        "last_range_max": v["last_range_max"],
                    }
                data = {
                    "stats": dict(_track_stats),
                    "tracks": tracks_out,
                }
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(data).encode())
        elif self.path.startswith("/z/"):
            # Proxy to Zenoh REST API for admin info
            zenoh_path = self.path[3:]
            url = f"{ZENOH_URL}/{zenoh_path}"
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=5) as resp:
                    body = resp.read()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)
            except urllib.error.URLError as e:
                self.send_response(502)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    global ZENOH_URL
    parser = argparse.ArgumentParser(description="ZSim Detection Monitor dashboard")
    parser.add_argument(
        "--zenoh-router",
        default="tcp/localhost:7447",
        help="Zenoh router endpoint for native subscriber",
    )
    parser.add_argument(
        "--zenoh-url", default="http://localhost:8000", help="Zenoh REST API URL for admin info"
    )
    parser.add_argument("--port", type=int, default=9090, help="Dashboard port")
    args = parser.parse_args()
    ZENOH_URL = args.zenoh_url

    # Start native subscriber in background thread
    sub_thread = threading.Thread(
        target=_start_zenoh_subscriber, args=(args.zenoh_router,), daemon=True
    )
    sub_thread.start()

    server = http.server.HTTPServer(("0.0.0.0", args.port), Handler)
    print(f"ZSim Detection Monitor → http://localhost:{args.port}")
    print(f"Admin API: {ZENOH_URL}")
    print(f"Native sub: {args.zenoh_router}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
