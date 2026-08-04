"""Export a browser-native, SUMO-style comparison of three controllers.

The tool replays existing decision records in headless SUMO through libsumo,
captures vehicle positions, traffic-light states, and comparable metrics, then
embeds the resulting data and the static SUMO network geometry into one HTML
file.  The browser redraws the three synchronized scenes on Canvas; no
SUMO-GUI, X server, or Python web server is required to view the result.

This is an offline presentation/export tool.  It never changes source records,
network files, simulator configs, or experiment output directories.
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.sumo_gui_comparison import (
    METHODS, METHOD_LABELS, _configure_registry, _ensure_sumo_home,
    _load_replay, _make_world_config,
)


METHOD_COLORS = {
    "online_dqn": "#1976d2",
    "hadhoa": "#d32f2f",
    "fixedtime": "#388e3c",
}


def _resolve_source_path(source_config, value):
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    root = Path(__file__).resolve().parents[1]
    config_dir = Path(source_config).resolve().parent
    candidates = (config_dir / path, root / path, root / "data" / path)
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[0].resolve()


def _parse_shape(value):
    points = []
    for item in value.split():
        x, y = item.split(",")[:2]
        points.append([float(x), float(y)])
    return points


def load_network_geometry(source_config):
    """Extract drawable SUMO lane geometry and traffic-light positions."""
    source_config = Path(source_config).expanduser().resolve()
    with source_config.open(encoding="utf-8") as handle:
        config = json.load(handle)
    net_path = _resolve_source_path(source_config, config["roadnetFile"])
    if not net_path.is_file():
        raise FileNotFoundError(f"SUMO network file does not exist: {net_path}")
    root = ET.parse(net_path).getroot()
    lanes = []
    coordinates = []
    for edge in root.findall("edge"):
        internal = edge.get("function") == "internal"
        for lane in edge.findall("lane"):
            shape = _parse_shape(lane.get("shape", ""))
            if len(shape) < 2:
                continue
            lanes.append({"id": lane.get("id"), "internal": internal, "shape": shape})
            coordinates.extend(shape)
    junctions = []
    for junction in root.findall("junction"):
        x = junction.get("x")
        y = junction.get("y")
        if x is None or y is None:
            continue
        junctions.append({
            "id": junction.get("id"),
            "type": junction.get("type"),
            "x": float(x),
            "y": float(y),
        })
        coordinates.append([float(x), float(y)])
    if not coordinates:
        raise ValueError(f"No drawable geometry found in {net_path}")
    xs = [point[0] for point in coordinates]
    ys = [point[1] for point in coordinates]
    return {
        "network_file": str(net_path),
        "lanes": lanes,
        "junctions": junctions,
        "bounds": {
            "min_x": min(xs), "max_x": max(xs),
            "min_y": min(ys), "max_y": max(ys),
        },
    }


def _record_actions(inputs, scene, selectors):
    replay = {}
    intervals = set()
    for method in METHODS:
        replay[method], interval = _load_replay(
            inputs[method], method, scene, **selectors.get(method, {})
        )
        intervals.add(interval)
    if len(intervals) != 1:
        raise ValueError(f"Controllers use different action intervals: {intervals}")
    return replay, intervals.pop()


def _traffic_light_states(engine):
    return {
        str(tl_id): {
            "phase": int(engine.trafficlight.getPhase(tl_id)),
            "state": str(engine.trafficlight.getRedYellowGreenState(tl_id)),
        }
        for tl_id in engine.trafficlight.getIDList()
    }


def _vehicle_states(engine):
    states = []
    for vehicle_id in engine.vehicle.getIDList():
        x, y = engine.vehicle.getPosition(vehicle_id)
        states.append({
            "id": str(vehicle_id),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "angle": round(float(engine.vehicle.getAngle(vehicle_id)), 2),
            "speed": round(float(engine.vehicle.getSpeed(vehicle_id)), 3),
            "lane_id": str(engine.vehicle.getLaneID(vehicle_id)),
        })
    return states


def _metric_state(world, action, time_seconds):
    lane_queue = world.get_lane_waiting_vehicle_count()
    queue = float(sum(lane_queue.values()))
    return {
        "time": int(time_seconds),
        "vehicles": _vehicle_states(world.eng),
        "traffic_lights": _traffic_light_states(world.eng),
        "queue_network_sum": queue,
        "throughput_cumulative": int(world.get_cur_throughput()),
        "actions": [int(value) for value in action],
    }


def _capture_method(method, records, interval, source_config, seed, start, end,
                    sample_every, output_dir):
    method_dir = Path(output_dir) / method
    method_dir.mkdir(parents=True, exist_ok=True)
    runtime_dir = method_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    world_config = _make_world_config(source_config, runtime_dir, method, gui=False)
    world_class = _configure_registry(world_config, seed, method_dir)
    world = None
    states = []
    current_time = 0
    try:
        world = world_class(str(world_config), interface="libsumo")
        world.reset()
        for record in records:
            action = record["actions"]
            if len(action) != len(world.intersections):
                raise ValueError(
                    f"{method} action count {len(action)} does not match SUMO "
                    f"intersection count {len(world.intersections)}"
                )
            for _ in range(interval):
                world.step(action)
                current_time = int(round(float(world.get_current_time())))
                if (
                    start <= current_time <= end
                    and (current_time - start) % sample_every == 0
                ):
                    states.append(_metric_state(world, action, current_time))
            if current_time >= end:
                break
        if len(states) < 2:
            raise ValueError(f"{method} produced fewer than two HTML states")
        return states
    finally:
        if world is not None:
            world.close()
        try:
            world_config.unlink()
        except FileNotFoundError:
            pass


def _common_states(states_by_method):
    time_sets = [set(state["time"] for state in states) for states in states_by_method.values()]
    common = sorted(set.intersection(*time_sets))
    if len(common) < 2:
        raise ValueError("Fewer than two common HTML state timestamps")
    return common


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>SUMO HTML comparison - __SCENE__</title>
<style>
:root { color-scheme: light; font-family: Arial, sans-serif; }
body { margin: 0; background: #eef1f4; color: #17202a; }
header { padding: 14px 20px 8px; background: #17202a; color: white; }
h1 { margin: 0 0 4px; font-size: 20px; }
#meta { font-size: 13px; opacity: .84; }
.toolbar { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; padding: 12px 20px; background: white; border-bottom: 1px solid #ccd3da; }
button, select, input { font: inherit; }
button { padding: 5px 12px; cursor: pointer; }
#timeline { flex: 1 1 300px; }
#timeLabel { min-width: 90px; font-variant-numeric: tabular-nums; }
#boards { display: grid; grid-template-columns: repeat(3, minmax(240px, 1fr)); gap: 10px; padding: 10px; }
.panel { min-width: 0; background: white; border: 1px solid #c8d0d8; border-top: 4px solid var(--method-color); box-shadow: 0 1px 2px #0001; }
.panel h2 { margin: 0; padding: 7px 10px; font-size: 16px; color: var(--method-color); }
canvas { display: block; width: 100%; height: auto; background: #fafafa; }
.panel .stats { padding: 7px 10px; font-size: 13px; display: flex; justify-content: space-between; border-top: 1px solid #e0e4e8; }
#charts { margin: 0 10px 14px; background: white; border: 1px solid #c8d0d8; padding: 10px; }
#charts canvas { width: 100%; }
.note { margin: 0 20px 14px; font-size: 12px; color: #5c6770; }
@media (max-width: 900px) { #boards { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header><h1>SUMO-style traffic control comparison</h1><div id="meta"></div></header>
<div class="toolbar">
  <button id="play">Play</button><button id="stepBack">−</button><button id="stepForward">+</button>
  <input id="timeline" type="range" min="0" max="0" value="0">
  <span id="timeLabel"></span>
  <label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
</div>
<main id="boards"></main>
<section id="charts"><canvas id="metricChart" width="1200" height="300"></canvas></section>
<p class="note">Replay mode: recorded actions applied at the original action interval. Vehicles and signals are redrawn from headless SUMO state export; no smoothing or interpolation is applied.</p>
<script>
const PAYLOAD = __PAYLOAD__;
const METHODS = PAYLOAD.methods;
const COLORS = PAYLOAD.colors;
const labels = PAYLOAD.method_labels;
const boards = document.getElementById('boards');
const panels = {};
for (const method of METHODS) {
  const panel = document.createElement('section'); panel.className = 'panel'; panel.style.setProperty('--method-color', COLORS[method]);
  panel.innerHTML = `<h2>${labels[method]}</h2><canvas width="640" height="480"></canvas><div class="stats"><span class="queue"></span><span class="throughput"></span></div>`;
  boards.appendChild(panel); panels[method] = {panel, canvas: panel.querySelector('canvas'), queue: panel.querySelector('.queue'), throughput: panel.querySelector('.throughput')};
}
document.getElementById('meta').textContent = `Scene: ${PAYLOAD.scene} · evaluation seed: ${PAYLOAD.evaluation_seed} · interval: ${PAYLOAD.action_interval_seconds}s`;
const times = PAYLOAD.times;
const byMethod = Object.fromEntries(METHODS.map(method => [method, Object.fromEntries(PAYLOAD.states[method].map(state => [state.time, state]))]));
const timeline = document.getElementById('timeline'); timeline.max = String(times.length - 1);
const timeLabel = document.getElementById('timeLabel');
const bounds = PAYLOAD.network.bounds;
const margin = 20;
function transform(canvas, x, y) {
  const w = canvas.width, h = canvas.height;
  const sx = (w - 2 * margin) / Math.max(1, bounds.max_x - bounds.min_x);
  const sy = (h - 2 * margin) / Math.max(1, bounds.max_y - bounds.min_y);
  const scale = Math.min(sx, sy);
  return [margin + (x - bounds.min_x) * scale, h - margin - (y - bounds.min_y) * scale];
}
function drawNetwork(ctx, canvas) {
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#fafafa'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  for (const lane of PAYLOAD.network.lanes) {
    ctx.beginPath();
    lane.shape.forEach((point, i) => { const [x, y] = transform(canvas, point[0], point[1]); if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); });
    ctx.strokeStyle = lane.internal ? '#b0b7bd' : '#69747d'; ctx.lineWidth = lane.internal ? 2 : 5; ctx.stroke();
  }
  for (const junction of PAYLOAD.network.junctions) {
    const [x, y] = transform(canvas, junction.x, junction.y); ctx.fillStyle = junction.type === 'traffic_light' ? '#d9dee2' : '#c7cdd2'; ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI * 2); ctx.fill();
  }
}
function drawLights(ctx, canvas, states) {
  for (const junction of PAYLOAD.network.junctions) {
    if (!states[junction.id]) continue;
    const [x, y] = transform(canvas, junction.x, junction.y); const state = states[junction.id].state || '';
    const active = state.split('').find(c => c.toLowerCase() !== 'r') || 'r';
    ctx.fillStyle = active.toLowerCase() === 'g' ? '#20a050' : active.toLowerCase() === 'y' ? '#e0a100' : '#d53939';
    ctx.beginPath(); ctx.arc(x, y, 9, 0, Math.PI * 2); ctx.fill(); ctx.strokeStyle = '#17202a'; ctx.lineWidth = 1; ctx.stroke();
  }
}
function drawVehicles(ctx, canvas, vehicles) {
  for (const vehicle of vehicles) {
    const [x, y] = transform(canvas, vehicle.x, vehicle.y); const angle = (90 - vehicle.angle) * Math.PI / 180;
    ctx.save(); ctx.translate(x, y); ctx.rotate(angle); ctx.fillStyle = vehicle.speed < 0.1 ? '#e65151' : '#2367a8'; ctx.fillRect(-5, -2.5, 10, 5); ctx.restore();
  }
}
function drawMetricChart(index) {
  const canvas = document.getElementById('metricChart'), ctx = canvas.getContext('2d'); const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  const left = 54, right = 18, top = 20, bottom = 38, plotW = w - left - right, plotH = h - top - bottom;
  const maxQueue = Math.max(1, ...METHODS.flatMap(method => PAYLOAD.states[method].map(state => state.queue_network_sum)));
  const maxThroughput = Math.max(1, ...METHODS.flatMap(method => PAYLOAD.states[method].map(state => state.throughput_cumulative)));
  const xAt = i => left + i * plotW / Math.max(1, times.length - 1);
  const yAt = value => top + plotH * (1 - value / maxQueue);
  ctx.strokeStyle = '#d7dce0'; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(left, top); ctx.lineTo(left, top + plotH); ctx.lineTo(w - right, top + plotH); ctx.stroke();
  ctx.fillStyle = '#45515a'; ctx.font = '14px Arial'; ctx.fillText('Queue (vehicles)', left, 15); ctx.fillText(`Throughput max ${maxThroughput}`, w - 170, 15);
  for (const method of METHODS) { const states = PAYLOAD.states[method]; ctx.beginPath(); states.forEach((state, i) => { const x = xAt(i), y = yAt(state.queue_network_sum); if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y); }); ctx.strokeStyle = COLORS[method]; ctx.lineWidth = 2; ctx.stroke(); }
  ctx.fillStyle = '#45515a'; ctx.fillText(`t = ${times[index]} s`, left, h - 12);
  const currentX = xAt(index); ctx.strokeStyle = '#17202a'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(currentX, top); ctx.lineTo(currentX, top + plotH); ctx.stroke(); ctx.setLineDash([]);
}
function render(index) {
  index = Math.max(0, Math.min(times.length - 1, index)); timeline.value = String(index); const time = times[index]; timeLabel.textContent = `t = ${time} s`;
  for (const method of METHODS) { const state = byMethod[method][time]; const view = panels[method]; const ctx = view.canvas.getContext('2d'); drawNetwork(ctx, view.canvas); drawLights(ctx, view.canvas, state.traffic_lights); drawVehicles(ctx, view.canvas, state.vehicles); view.queue.textContent = `Queue: ${state.queue_network_sum}`; view.throughput.textContent = `Throughput: ${state.throughput_cumulative}`; }
  drawMetricChart(index);
}
let timer = null;
function stop() { if (timer) { clearInterval(timer); timer = null; } document.getElementById('play').textContent = 'Play'; }
document.getElementById('play').onclick = () => { if (timer) { stop(); return; } document.getElementById('play').textContent = 'Pause'; timer = setInterval(() => { let next = Number(timeline.value) + 1; if (next >= times.length) next = 0; render(next); }, 1000 / Number(document.getElementById('speed').value)); };
document.getElementById('stepBack').onclick = () => { stop(); render(Number(timeline.value) - 1); };
document.getElementById('stepForward').onclick = () => { stop(); render(Number(timeline.value) + 1); };
timeline.oninput = () => { stop(); render(Number(timeline.value)); };
document.getElementById('speed').onchange = () => { if (timer) { stop(); document.getElementById('play').click(); } };
render(0);
</script>
</body>
</html>
"""


def build_html(payload):
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\u003c")
    return HTML_TEMPLATE.replace("__SCENE__", str(payload["scene"])).replace("__PAYLOAD__", encoded)


def export_html_comparison(inputs, scene, source_config, evaluation_seed, output,
                           start, end, selectors=None, sample_every=1):
    if start < 0 or end <= start or sample_every <= 0:
        raise ValueError("Require 0 <= start < end and sample_every > 0")
    if int(start) != start or int(end) != end:
        raise ValueError("start/end must be integer simulation seconds")
    selectors = selectors or {}
    replay, interval = _record_actions(inputs, scene, selectors)
    if start % interval != 0:
        raise ValueError("start must align with the common action interval")
    geometry = load_network_geometry(source_config)
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    states = {}
    _ensure_sumo_home()
    for method in METHODS:
        states[method] = _capture_method(
            method, replay[method], interval, source_config, evaluation_seed,
            int(start), int(end), int(sample_every), output_dir,
        )
    common_times = _common_states(states)
    by_time = {
        method: {state["time"]: state for state in states[method]}
        for method in METHODS
    }
    payload = {
        "schema_version": 1,
        "scene": scene,
        "evaluation_seed": int(evaluation_seed),
        "source_config": str(Path(source_config).resolve()),
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "colors": METHOD_COLORS,
        "action_interval_seconds": interval,
        "sample_every_seconds": int(sample_every),
        "times": common_times,
        "network": geometry,
        "states": {
            method: [by_time[method][time_seconds] for time_seconds in common_times]
            for method in METHODS
        },
        "replay_mode": "recorded_actions_exact_interval_headless_sumo",
    }
    html_path = output_dir / "sumo_html_comparison.html"
    html_path.write_text(build_html(payload), encoding="utf-8")
    states_path = output_dir / "comparison_states.json"
    states_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    # Keep a compact machine-readable manifest next to the page without
    # duplicating the potentially large embedded state payload.
    manifest = {
        "schema_version": 1, "scene": scene,
        "evaluation_seed": int(evaluation_seed),
        "methods": list(METHODS), "action_interval_seconds": interval,
        "sample_every_seconds": int(sample_every),
        "timestamps": {"count": len(common_times),
                       "start": common_times[0], "end": common_times[-1]},
        "network_file": geometry["network_file"],
        "html": str(html_path),
        "states": str(states_path),
        "state_counts": {method: len(states[method]) for method in METHODS},
        "replay_mode": payload["replay_mode"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--sim-config", default=None)
    parser.add_argument("--online-dqn", required=True)
    parser.add_argument("--hadhoa", required=True)
    parser.add_argument("--fixedtime", required=True)
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-controller-id")
        parser.add_argument(f"--{method.replace('_', '-')}-training-seed", type=int)
        parser.add_argument(f"--{method.replace('_', '-')}-evaluation-seed", type=int)
    parser.add_argument("--evaluation-seed", type=int, required=True)
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--output", required=True)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    source_config = args.sim_config or str(
        Path(__file__).resolve().parents[1] / "configs" / "sim" / f"{args.scene}.cfg"
    )
    selectors = {
        method: {
            "controller_id": getattr(args, f"{method}_controller_id"),
            "training_seed": getattr(args, f"{method}_training_seed"),
            "evaluation_seed": getattr(args, f"{method}_evaluation_seed"),
        }
        for method in METHODS
    }
    manifest = export_html_comparison(
        {method: getattr(args, method) for method in METHODS}, args.scene,
        source_config, args.evaluation_seed, args.output, args.start, args.end,
        selectors, args.sample_every,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
