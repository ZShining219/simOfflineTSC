"""Export a browser-native, SUMO-style replay with optional intersection focus.

The tool replays existing decision records in headless SUMO through libsumo,
captures vehicle positions, traffic-light states, and comparable metrics, then
embeds the resulting data and the static SUMO network geometry into one HTML
file.  The browser redraws the synchronized scenes on Canvas; no
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
    "cont_o2_final": "#7b1fa2",
}
METHOD_LABELS_DEFAULT = {
    **METHOD_LABELS,
    "online_dqn": "online DQN",
    "fixedtime": "fixedtime",
    "hadhoa": "DHOA",
    "cont_o2_final": "CONT DQN",
}
PRESENTATION_METHOD_ORDER = (
    "online_dqn", "fixedtime", "hadhoa", "cont_o2_final",
)

SCENE_ALIASES = {
    "S1": "sumohz1x1_config2",
    "S2": "sumohz1x1",
    "S3": "sumohz1x1_config4",
    "S4": "sumohz1x1_config3",
}
DEFAULT_S2_FIXEDTIME_INPUT = (
    Path(__file__).resolve().parents[1]
    / "output_data/evaluations/plan1/"
      "plan1_best_checkpoint_reevaluation_v1_20260723/records.jsonl"
)


def canonical_scene(scene):
    """Accept S1--S4 labels as a convenience while preserving network names."""
    return SCENE_ALIASES.get(str(scene).upper(), str(scene))


def _presentation_methods(methods):
    """Return methods in the stable four-panel presentation order."""
    available = list(dict.fromkeys(methods))
    ordered = [
        method for method in PRESENTATION_METHOD_ORDER
        if method in available
    ]
    ordered.extend(method for method in available if method not in ordered)
    return ordered


def _presentation_payload(payload):
    """Copy a payload with stable panel order and concise panel titles."""
    prepared = dict(payload)
    methods = _presentation_methods(payload["methods"])
    prepared["methods"] = methods

    existing_labels = dict(payload.get("method_labels", {}))
    prepared["method_labels"] = {
        method: METHOD_LABELS_DEFAULT.get(
            method, existing_labels.get(method, method)
        )
        for method in methods
    }
    for field in (
        "colors", "states", "method_evaluation_seeds", "method_sources",
    ):
        values = payload.get(field)
        if not isinstance(values, dict):
            continue
        reordered = {
            method: values[method] for method in methods if method in values
        }
        reordered.update({
            method: value for method, value in values.items()
            if method not in reordered
        })
        prepared[field] = reordered
    return prepared


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


def _record_actions(inputs, scene, selectors, methods=None):
    methods = tuple(methods or inputs)
    replay = {}
    intervals = set()
    for method in methods:
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


def _intersection_catalog(engine):
    """Return stable lane/link metadata for detailed browser-side tracking."""
    catalog = {}
    for tl_id in engine.trafficlight.getIDList():
        try:
            x, y = engine.junction.getPosition(tl_id)
        except Exception:
            x, y = 0.0, 0.0
        controlled = []
        for lane_id in engine.trafficlight.getControlledLanes(tl_id):
            if lane_id not in controlled:
                controlled.append(str(lane_id))
        incoming = []
        try:
            links = engine.trafficlight.getControlledLinks(tl_id)
            def add_link(candidate):
                if candidate is None:
                    return
                if (
                    isinstance(candidate, (tuple, list))
                    and candidate
                    and isinstance(candidate[0], str)
                ):
                    if candidate[0] not in incoming:
                        incoming.append(str(candidate[0]))
                    return
                if isinstance(candidate, (tuple, list)):
                    for nested in candidate:
                        add_link(nested)

            add_link(links)
        except Exception:
            incoming = list(controlled)
        catalog[str(tl_id)] = {
            "id": str(tl_id),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "controlled_lanes": controlled,
            "incoming_lanes": incoming or controlled,
        }
    return catalog


def _vehicle_states(engine):
    states = []
    for vehicle_id in engine.vehicle.getIDList():
        x, y = engine.vehicle.getPosition(vehicle_id)
        lane_id = str(engine.vehicle.getLaneID(vehicle_id))
        try:
            lane_position = float(engine.vehicle.getLanePosition(vehicle_id))
            lane_length = float(engine.lane.getLength(lane_id))
        except Exception:
            lane_position = None
            lane_length = None
        states.append({
            "id": str(vehicle_id),
            "x": round(float(x), 3),
            "y": round(float(y), 3),
            "angle": round(float(engine.vehicle.getAngle(vehicle_id)), 2),
            "speed": round(float(engine.vehicle.getSpeed(vehicle_id)), 3),
            "lane_id": lane_id,
            "lane_position": round(lane_position, 3) if lane_position is not None else None,
            "distance_to_stop": (
                round(max(0.0, lane_length - lane_position), 3)
                if lane_position is not None and lane_length is not None else None
            ),
        })
    return states


def _intersection_details(world, catalog, vehicles, lights):
    details = {}
    for intersection_id, item in catalog.items():
        lane_ids = set(item["incoming_lanes"])
        focused_by_lane = {}
        focused = [vehicle for vehicle in vehicles if vehicle["lane_id"] in lane_ids]
        for vehicle in focused:
            focused_by_lane.setdefault(vehicle["lane_id"], []).append(vehicle["id"])
        lane_rows = []
        for lane_id in item["incoming_lanes"]:
            try:
                lane_rows.append({
                    "id": lane_id,
                    "vehicles": int(world.eng.lane.getLastStepVehicleNumber(lane_id)),
                    "halting": int(world.eng.lane.getLastStepHaltingNumber(lane_id)),
                    "mean_speed": round(float(world.eng.lane.getLastStepMeanSpeed(lane_id)), 3),
                    "occupancy": round(float(world.eng.lane.getLastStepOccupancy(lane_id)), 3),
                    "vehicle_ids": focused_by_lane.get(lane_id, []),
                })
            except Exception:
                lane_rows.append({
                    "id": lane_id, "vehicles": 0, "halting": 0,
                    "mean_speed": 0.0, "occupancy": 0.0,
                    "vehicle_ids": focused_by_lane.get(lane_id, []),
                })
        speeds = [vehicle["speed"] for vehicle in focused]
        light = lights.get(intersection_id, {"phase": None, "state": ""})
        details[intersection_id] = {
            "id": intersection_id,
            "phase": light.get("phase"),
            "signal_state": light.get("state", ""),
            "lane_count": len(lane_rows),
            "vehicles": len(focused),
            "halting": sum(row["halting"] for row in lane_rows),
            "mean_speed": round(sum(speeds) / len(speeds), 3) if speeds else 0.0,
            "lanes": lane_rows,
        }
    return details


def _metric_state(world, action, time_seconds, catalog):
    lane_queue = world.get_lane_waiting_vehicle_count()
    queue = float(sum(lane_queue.values()))
    vehicles = _vehicle_states(world.eng)
    lights = _traffic_light_states(world.eng)
    return {
        "time": int(time_seconds),
        "vehicles": vehicles,
        "traffic_lights": lights,
        "intersections": _intersection_details(world, catalog, vehicles, lights),
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
    intersection_catalog = {}
    try:
        world = world_class(str(world_config), interface="libsumo")
        world.reset()
        intersection_catalog = _intersection_catalog(world.eng)
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
                    states.append(_metric_state(world, action, current_time, intersection_catalog))
            if current_time >= end:
                break
        if len(states) < 2:
            raise ValueError(f"{method} produced fewer than two HTML states")
        return {"states": states, "intersections": intersection_catalog}
    finally:
        if world is not None:
            world.close()
        try:
            world_config.unlink()
        except FileNotFoundError:
            pass


def _common_states(states_by_method):
    time_sets = [
        set(state["time"] for state in bundle["states"])
        for bundle in states_by_method.values()
    ]
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
#boards { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 10px; padding: 10px; }
.panel { min-width: 0; background: white; border: 1px solid #c8d0d8; border-top: 4px solid var(--method-color); box-shadow: 0 1px 2px #0001; }
.panel h2 { margin: 0; padding: 7px 10px; font-size: 16px; color: var(--method-color); overflow-wrap: anywhere; }
.panel .viewMode { padding: 0 10px 6px; color: #53616a; font-size: 11px; }
.panelViewTools { display: flex; align-items: center; gap: 5px; padding: 0 10px 7px; color: #53616a; font-size: 11px; }
.panelViewTools button { padding: 2px 8px; font-size: 13px; line-height: 1.2; }
.panelViewTools .viewHint { margin-left: 3px; }
.topCanvas { touch-action: none; cursor: grab; background: #f4f7f8; }
.topCanvas.dragging { cursor: grabbing; }
canvas { display: block; width: 100%; height: auto; background: #fafafa; }
.panel .stats { padding: 7px 10px; font-size: 13px; display: flex; gap: 8px; flex-wrap: wrap; justify-content: space-between; border-top: 1px solid #e0e4e8; }
.panel .jam { color: #b71c1c; font-weight: 700; }
#legend { margin: 0 10px 10px; padding: 8px 10px; display: flex; gap: 14px; flex-wrap: wrap; align-items: center; background: white; border: 1px solid #c8d0d8; font-size: 12px; color: #45515a; }
.legendItem { display: inline-flex; align-items: center; gap: 6px; white-space: nowrap; }
.legendMark { display: inline-block; width: 17px; height: 10px; border-radius: 3px; box-sizing: border-box; }
.legendNormal { background: #1976d2; border: 2px solid #0d3c68; }
.legendJam { background: #e53935; border: 2px solid #8f1d1d; box-shadow: 0 0 0 3px #ffb3b333; }
.legendRoad { background: #697780; border: 2px solid #34424a; }
.legendIntersection { width: 13px; height: 13px; border-radius: 50%; background: #f4b400; border: 3px solid #17202a; }
#charts { margin: 0 10px 14px; background: white; border: 1px solid #c8d0d8; padding: 10px; }
#charts canvas { width: 100%; }
.note { margin: 0 20px 14px; font-size: 12px; color: #5c6770; }
#focus { margin: 0 10px 14px; padding: 10px; background: white; border: 1px solid #c8d0d8; }
#focusControls { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; }
#focusGrid { display: grid; grid-template-columns: minmax(480px, 2fr) minmax(300px, 1fr); gap: 12px; }
#focusCanvas, #focusChart { width: 100%; height: auto; border: 1px solid #d8dee3; background: #fafafa; }
#focusInfo { min-width: 0; }
#focusMetrics { display: grid; grid-template-columns: repeat(2, minmax(120px, 1fr)); gap: 7px; margin-bottom: 10px; }
.metric { padding: 7px; background: #f1f4f6; border-left: 3px solid #536875; font-size: 13px; }
.metric strong { display: block; font-size: 16px; color: #17202a; }
#laneTable { width: 100%; border-collapse: collapse; font-size: 12px; }
#laneTable th, #laneTable td { padding: 5px 4px; text-align: right; border-bottom: 1px solid #e3e7ea; }
#laneTable th:first-child, #laneTable td:first-child { text-align: left; }
#focusCaption { margin: 0 0 8px; color: #53616a; font-size: 13px; }
@media (max-width: 1280px) { #boards { grid-template-columns: repeat(2, minmax(360px, 1fr)); } }
@media (max-width: 900px) { #boards { grid-template-columns: 1fr; } }
@media (max-width: 900px) { #focusGrid { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<header><h1>SUMO-style traffic control comparison</h1><div id="meta"></div></header>
<div class="toolbar">
  <button id="play">Play</button><button id="stepBack">−</button><button id="stepForward">+</button>
  <input id="timeline" type="range" min="0" max="0" value="0">
  <span id="timeLabel"></span>
  <label>Speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option><option value="4">4×</option></select></label>
  <label>Focus intersection <select id="intersectionSelect"></select></label>
  <label>Detail method <select id="detailMethod"></select></label>
  <label><input id="linkTopViews" type="checkbox" checked> 三窗视角联动</label>
  <label><input id="overlayTopViews" type="checkbox" checked> 路口覆盖层</label>
</div>
<main id="boards"></main>
<section id="legend" aria-label="Visual legend">
  <span class="legendItem"><span class="legendMark legendNormal"></span>正常行驶</span>
  <span class="legendItem"><span class="legendMark legendJam"></span>堵塞/停止（speed &lt; 0.1 m/s）</span>
  <span class="legendItem"><span class="legendMark legendRoad"></span>道路与车道层</span>
  <span class="legendItem"><span class="legendMark legendIntersection"></span>重点信号路口</span>
</section>
<section id="charts"><canvas id="metricChart" width="1200" height="300"></canvas></section>
<section id="focus">
  <p id="focusCaption"></p>
  <div id="focusGrid">
    <div><canvas id="focusCanvas" width="1000" height="620"></canvas><canvas id="focusChart" width="1000" height="240"></canvas></div>
    <div id="focusInfo"><div id="focusMetrics"></div><table id="laneTable"><thead><tr><th>Incoming lane</th><th>Vehicles</th><th>Halting</th><th>Mean speed</th><th>Occupancy</th></tr></thead><tbody></tbody></table></div>
  </div>
</section>
<p class="note">Replay mode: recorded actions applied at the original action interval. Vehicles and signals are redrawn from headless SUMO state export; no smoothing or interpolation is applied.</p>
<script>
const PAYLOAD = __PAYLOAD__;
const METHODS = PAYLOAD.methods;
const COLORS = PAYLOAD.colors;
const labels = PAYLOAD.method_labels;
const CATALOG = PAYLOAD.intersections || {};
const boards = document.getElementById('boards');
const panels = {};
const topViews = Object.fromEntries(METHODS.map(method => [method, {zoom: 1, panX: 0, panY: 0, dragging: false, pointerId: null, lastX: 0, lastY: 0}]));
for (const method of METHODS) {
  const panel = document.createElement('section'); panel.className = 'panel'; panel.style.setProperty('--method-color', COLORS[method]);
  panel.innerHTML = `<h2>${labels[method]}</h2><div class="viewMode">选定路口局部视角 · 远端四向道路已裁剪</div><div class="panelViewTools"><button class="zoomOut" type="button" title="缩小">−</button><button class="resetView" type="button">重置</button><button class="zoomIn" type="button" title="放大">+</button><span class="viewHint">滚轮缩放 · 拖动移动</span></div><canvas class="topCanvas" width="800" height="600"></canvas><div class="stats"><span class="queue"></span><span class="jam"></span><span class="throughput"></span></div>`;
  boards.appendChild(panel); panels[method] = {panel, canvas: panel.querySelector('canvas'), viewMode: panel.querySelector('.viewMode'), viewHint: panel.querySelector('.viewHint'), queue: panel.querySelector('.queue'), jam: panel.querySelector('.jam'), throughput: panel.querySelector('.throughput')};
}
document.getElementById('meta').textContent = [
  `Scene: ${PAYLOAD.scene}`,
  `evaluation seed: ${PAYLOAD.evaluation_seed}`,
  `interval: ${PAYLOAD.action_interval_seconds}s`,
  PAYLOAD.method_evaluation_seeds
    ? `SUMO seeds: ${METHODS.map(method => `${method}=${PAYLOAD.method_evaluation_seeds[method] == null ? 'fixed_default' : PAYLOAD.method_evaluation_seeds[method]}`).join(', ')}`
    : null,
  PAYLOAD.comparison_label,
].filter(Boolean).join(' · ');
const intersectionSelect = document.getElementById('intersectionSelect');
const detailMethod = document.getElementById('detailMethod');
const linkTopViews = document.getElementById('linkTopViews');
const overlayTopViews = document.getElementById('overlayTopViews');
for (const id of Object.keys(CATALOG)) { const option = document.createElement('option'); option.value = id; option.textContent = id; intersectionSelect.appendChild(option); }
for (const method of METHODS) { const option = document.createElement('option'); option.value = method; option.textContent = labels[method]; detailMethod.appendChild(option); }
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
function focusBounds(intersectionId, radius = 72) {
  const item = CATALOG[intersectionId], points = [];
  if (!item) return bounds;
  for (const lane of PAYLOAD.network.lanes) {
    for (const point of lane.shape) {
      if (Math.hypot(point[0] - item.x, point[1] - item.y) <= radius) points.push(point);
    }
  }
  points.push([item.x, item.y]);
  if (!points.length) return bounds;
  const xs = points.map(point => point[0]), ys = points.map(point => point[1]);
  const span = Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys));
  const pad = Math.max(14, Math.min(24, span * 0.16));
  return {min_x: Math.min(...xs) - pad, max_x: Math.max(...xs) + pad, min_y: Math.min(...ys) - pad, max_y: Math.max(...ys) + pad};
}
function transformWith(canvas, x, y, localBounds) {
  const margin = 34, scale = Math.min((canvas.width - 2 * margin) / Math.max(1, localBounds.max_x - localBounds.min_x), (canvas.height - 2 * margin) / Math.max(1, localBounds.max_y - localBounds.min_y));
  return [margin + (x - localBounds.min_x) * scale, canvas.height - margin - (y - localBounds.min_y) * scale];
}
const laneById = Object.fromEntries(PAYLOAD.network.lanes.map(lane => [lane.id, lane]));
function focusedLanePredicate(intersectionId, localBounds) {
  const laneIds = new Set((CATALOG[intersectionId] && CATALOG[intersectionId].incoming_lanes) || []);
  return lane => laneIds.has(lane.id) || lane.shape.some(point => point[0] >= localBounds.min_x && point[0] <= localBounds.max_x && point[1] >= localBounds.min_y && point[1] <= localBounds.max_y);
}
function mapPoint(canvas, point, localBounds) {
  return localBounds ? transformWith(canvas, point[0], point[1], localBounds) : transform(canvas, point[0], point[1]);
}
function mapScale(canvas, localBounds) {
  const inset = localBounds ? 34 : margin;
  const area = localBounds || bounds;
  return Math.min(
    (canvas.width - 2 * inset) / Math.max(1, area.max_x - area.min_x),
    (canvas.height - 2 * inset) / Math.max(1, area.max_y - area.min_y),
  );
}
function traceShape(ctx, canvas, shape, localBounds) {
  ctx.beginPath();
  shape.forEach((point, i) => {
    const [x, y] = mapPoint(canvas, point, localBounds);
    if (!i) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
}
function laneWorldWidth(lane) {
  const specified = Number(lane.width);
  if (Number.isFinite(specified) && specified > 0) return specified;
  return lane.internal ? 1.6 : 3.2;
}
function lanePixelWidth(canvas, lane, localBounds) {
  return Math.max(localBounds ? 3 : 2, laneWorldWidth(lane) * mapScale(canvas, localBounds));
}
function drawLaneSurface(ctx, canvas, lane, localBounds, emphasized) {
  const width = lanePixelWidth(canvas, lane, localBounds);
  traceShape(ctx, canvas, lane.shape, localBounds);
  if (lane.internal) {
    ctx.strokeStyle = '#7f8b92'; ctx.lineWidth = width + (localBounds ? 2 : 1.5); ctx.stroke();
    traceShape(ctx, canvas, lane.shape, localBounds);
    ctx.strokeStyle = emphasized ? '#aebdc5' : '#b7c1c6'; ctx.lineWidth = width; ctx.stroke();
  } else {
    ctx.strokeStyle = '#34424a'; ctx.lineWidth = width + (localBounds ? 7 : 5); ctx.stroke();
    traceShape(ctx, canvas, lane.shape, localBounds);
    ctx.strokeStyle = emphasized ? '#5e7887' : '#697780'; ctx.lineWidth = width; ctx.stroke();
  }
}
function laneGroupKey(id) {
  return String(id).replace(/_\d+$/, '');
}
function laneIndex(id) {
  const match = String(id).match(/_(\d+)$/);
  return match ? Number(match[1]) : 0;
}
function drawLaneSeparators(ctx, canvas, lanes, localBounds, emphasized) {
  const groups = {};
  for (const lane of lanes) {
    if (lane.internal) continue;
    (groups[laneGroupKey(lane.id)] ||= []).push(lane);
  }
  ctx.lineCap = 'butt';
  for (const group of Object.values(groups)) {
    group.sort((a, b) => laneIndex(a.id) - laneIndex(b.id));
    for (let index = 1; index < group.length; index += 1) {
      const left = group[index - 1], right = group[index];
      const count = Math.min(left.shape.length, right.shape.length);
      if (count < 2) continue;
      const separator = [];
      for (let pointIndex = 0; pointIndex < count; pointIndex += 1) {
        separator.push([
          (left.shape[pointIndex][0] + right.shape[pointIndex][0]) / 2,
          (left.shape[pointIndex][1] + right.shape[pointIndex][1]) / 2,
        ]);
      }
      traceShape(ctx, canvas, separator, localBounds);
      ctx.strokeStyle = emphasized ? '#f4f8fa' : '#d9e0e4';
      ctx.lineWidth = localBounds ? 1.6 : 1.1;
      ctx.setLineDash(localBounds ? [10, 8] : [7, 6]);
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }
  ctx.lineCap = 'round';
}
function drawDirectionArrow(ctx, canvas, lane, localBounds, emphasized) {
  if (lane.internal || lane.shape.length < 2) return;
  const endIndex = lane.shape.length - 1;
  const first = lane.shape[Math.max(0, Math.floor(endIndex * 0.38))];
  const second = lane.shape[Math.min(endIndex, Math.max(1, Math.floor(endIndex * 0.58)))];
  const start = mapPoint(canvas, first, localBounds), end = mapPoint(canvas, second, localBounds);
  const dx = end[0] - start[0], dy = end[1] - start[1], distance = Math.hypot(dx, dy);
  if (distance < 2) return;
  const size = localBounds ? 10 : 7;
  const angle = Math.atan2(dy, dx);
  ctx.save(); ctx.translate(end[0], end[1]); ctx.rotate(angle);
  ctx.beginPath(); ctx.moveTo(size, 0); ctx.lineTo(-size * 0.75, size * 0.62); ctx.lineTo(-size * 0.75, -size * 0.62); ctx.closePath();
  ctx.fillStyle = emphasized ? '#eaf3f7' : '#dce5e9'; ctx.strokeStyle = '#34424a'; ctx.lineWidth = 1; ctx.fill(); ctx.stroke();
  ctx.restore();
}
function drawStopBars(ctx, canvas, localBounds, intersectionId) {
  const selected = intersectionId && CATALOG[intersectionId] ? CATALOG[intersectionId].incoming_lanes : Object.values(CATALOG).flatMap(item => item.incoming_lanes || []);
  const laneIds = [...new Set(selected || [])];
  for (const laneId of laneIds) {
    const lane = laneById[laneId];
    if (!lane || lane.internal || lane.shape.length < 2) continue;
    const last = lane.shape[lane.shape.length - 1], before = lane.shape[lane.shape.length - 2];
    const dx = last[0] - before[0], dy = last[1] - before[1], length = Math.hypot(dx, dy);
    if (length < 0.1) continue;
    const ux = dx / length, uy = dy / length;
    const stopDistance = Math.min(4, length * 0.25);
    const center = [last[0] - ux * stopDistance, last[1] - uy * stopDistance];
    const halfWidth = laneWorldWidth(lane) * 0.52;
    const endpoints = [
      [center[0] - uy * halfWidth, center[1] + ux * halfWidth],
      [center[0] + uy * halfWidth, center[1] - ux * halfWidth],
    ].map(point => mapPoint(canvas, point, localBounds));
    ctx.beginPath(); ctx.moveTo(...endpoints[0]); ctx.lineTo(...endpoints[1]);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = localBounds ? 3 : 2; ctx.lineCap = 'butt'; ctx.stroke(); ctx.lineCap = 'round';
  }
}
function drawRoadLayer(ctx, canvas, localBounds, predicate, emphasized) {
  const lanes = PAYLOAD.network.lanes.filter(predicate || (() => true));
  lanes.sort((a, b) => Number(a.internal) - Number(b.internal));
  for (const lane of lanes) drawLaneSurface(ctx, canvas, lane, localBounds, emphasized && !lane.internal);
  drawLaneSeparators(ctx, canvas, lanes, localBounds, emphasized);
  for (const lane of lanes) drawDirectionArrow(ctx, canvas, lane, localBounds, emphasized && !lane.internal);
  drawStopBars(ctx, canvas, localBounds, emphasized ? intersectionSelect.value : null);
}
function drawTag(ctx, text, x, y, color = '#17202a') {
  ctx.font = 'bold 12px Arial';
  const metrics = ctx.measureText(text), width = metrics.width + 10, height = 19;
  ctx.fillStyle = '#ffffffeb'; ctx.fillRect(x, y - height + 3, width, height);
  ctx.strokeStyle = '#536875'; ctx.lineWidth = 1; ctx.strokeRect(x, y - height + 3, width, height);
  ctx.fillStyle = color; ctx.fillText(text, x + 5, y - 3);
}
function drawJunctions(ctx, canvas, localBounds, focusIntersectionId) {
  for (const junction of PAYLOAD.network.junctions) {
    if (localBounds && (junction.x < localBounds.min_x || junction.x > localBounds.max_x || junction.y < localBounds.min_y || junction.y > localBounds.max_y)) continue;
    const [x, y] = mapPoint(canvas, [junction.x, junction.y], localBounds);
    const tracked = Boolean(CATALOG[junction.id]) || junction.id === focusIntersectionId;
    const radius = tracked ? (localBounds ? 18 : 12) : (localBounds ? 8 : 7);
    if (tracked) {
      ctx.fillStyle = '#f4b40033'; ctx.beginPath(); ctx.arc(x, y, radius + 8, 0, Math.PI * 2); ctx.fill();
    }
    ctx.fillStyle = tracked ? '#17202a' : '#8e9aa1'; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = tracked ? '#fff4c2' : '#52616a'; ctx.lineWidth = tracked ? 3 : 1.5; ctx.stroke();
    if (tracked) drawTag(ctx, junction.id, x + radius + 6, y - radius - 2, '#17202a');
  }
}
function drawLights(ctx, canvas, states, localBounds) {
  for (const junction of PAYLOAD.network.junctions) {
    if (!states[junction.id]) continue;
    if (localBounds && (junction.x < localBounds.min_x || junction.x > localBounds.max_x || junction.y < localBounds.min_y || junction.y > localBounds.max_y)) continue;
    const [x, y] = mapPoint(canvas, [junction.x, junction.y], localBounds);
    const light = states[junction.id], state = light.state || '';
    const active = state.split('').find(c => c.toLowerCase() !== 'r') || 'r';
    const color = active.toLowerCase() === 'g' ? '#20a050' : active.toLowerCase() === 'y' ? '#e0a100' : '#d53939';
    const radius = CATALOG[junction.id] ? (localBounds ? 14 : 11) : (localBounds ? 10 : 8);
    ctx.fillStyle = `${color}33`; ctx.beginPath(); ctx.arc(x, y, radius + 6, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = '#17202a'; ctx.lineWidth = localBounds ? 2.5 : 2; ctx.stroke();
    if (CATALOG[junction.id]) drawTag(ctx, `P${light.phase ?? '-'} · ${active.toUpperCase()}`, x + radius + 6, y + radius + 20, color);
  }
}
function isJammed(vehicle) {
  return Number(vehicle.speed) < 0.1;
}
function drawVehicle(ctx, canvas, vehicle, localBounds) {
  const [x, y] = mapPoint(canvas, [vehicle.x, vehicle.y], localBounds);
  const angle = (90 - vehicle.angle) * Math.PI / 180, jammed = isJammed(vehicle);
  const length = localBounds ? 38 : 28, width = localBounds ? 22 : 16;
  ctx.save(); ctx.translate(x, y); ctx.rotate(angle);
  if (jammed) {
    ctx.fillStyle = '#e5393538'; ctx.beginPath(); ctx.arc(0, 0, length * 0.72, 0, Math.PI * 2); ctx.fill();
  }
  ctx.fillStyle = jammed ? '#e53935' : '#1976d2'; ctx.strokeStyle = jammed ? '#8f1d1d' : '#0d3c68'; ctx.lineWidth = jammed ? 2 : 1.4;
  ctx.fillRect(-length / 2, -width / 2, length, width); ctx.strokeRect(-length / 2, -width / 2, length, width);
  ctx.fillStyle = jammed ? '#ffd1d1' : '#cce8ff'; ctx.fillRect(-length * 0.06, -width * 0.31, length * 0.34, width * 0.62);
  ctx.fillStyle = jammed ? '#fff3f3' : '#eaf5ff'; ctx.beginPath(); ctx.moveTo(length / 2 + 2, 0); ctx.lineTo(length / 2 - 2, -width * 0.34); ctx.lineTo(length / 2 - 2, width * 0.34); ctx.closePath(); ctx.fill();
  ctx.restore();
  if (localBounds && jammed) drawTag(ctx, '堵', x + length / 2 + 5, y - width / 2, '#b71c1c');
}
function drawVehicles(ctx, canvas, vehicles) {
  for (const vehicle of vehicles) drawVehicle(ctx, canvas, vehicle, null);
}
function applyViewport(ctx, canvas, view) {
  ctx.translate(view.panX, view.panY);
  ctx.translate(canvas.width / 2, canvas.height / 2);
  ctx.scale(view.zoom, view.zoom);
  ctx.translate(-canvas.width / 2, -canvas.height / 2);
}
function drawIntersectionOverlay(ctx, canvas, localBounds, intersectionId) {
  const item = CATALOG[intersectionId];
  if (!item) return;
  const [x, y] = mapPoint(canvas, [item.x, item.y], localBounds), radius = 23 * mapScale(canvas, localBounds);
  ctx.save();
  ctx.fillStyle = '#3f5159e8'; ctx.beginPath(); ctx.arc(x, y, radius, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = '#dce6eacc'; ctx.lineWidth = Math.max(2, radius * 0.025); ctx.stroke();
  const grid = radius * 0.68;
  ctx.strokeStyle = '#e9f0f255'; ctx.lineWidth = Math.max(1.5, radius * 0.018); ctx.setLineDash([radius * 0.08, radius * 0.08]);
  ctx.beginPath(); ctx.moveTo(x - grid, y); ctx.lineTo(x + grid, y); ctx.moveTo(x, y - grid); ctx.lineTo(x, y + grid); ctx.stroke(); ctx.setLineDash([]);
  ctx.strokeStyle = '#f7fbfcaa'; ctx.lineWidth = Math.max(2, radius * 0.035);
  for (const offset of [-0.36, -0.18, 0.18, 0.36]) {
    ctx.beginPath(); ctx.moveTo(x - radius * 0.92, y + radius * offset); ctx.lineTo(x - radius * 0.62, y + radius * offset); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + radius * 0.62, y + radius * offset); ctx.lineTo(x + radius * 0.92, y + radius * offset); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + radius * offset, y - radius * 0.92); ctx.lineTo(x + radius * offset, y - radius * 0.62); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(x + radius * offset, y + radius * 0.62); ctx.lineTo(x + radius * offset, y + radius * 0.92); ctx.stroke();
  }
  ctx.restore();
}
function drawPanelScene(ctx, canvas, state, intersectionId, view) {
  const localBounds = focusBounds(intersectionId, 56);
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#f4f7f8'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.save(); applyViewport(ctx, canvas, view);
  const focusedPredicate = focusedLanePredicate(intersectionId, localBounds);
  const lanePredicate = overlayTopViews.checked ? lane => !lane.internal && focusedPredicate(lane) : focusedPredicate;
  drawRoadLayer(ctx, canvas, localBounds, lanePredicate, true);
  if (overlayTopViews.checked) {
    drawIntersectionOverlay(ctx, canvas, localBounds, intersectionId);
    drawStopBars(ctx, canvas, localBounds, intersectionId);
  }
  drawJunctions(ctx, canvas, localBounds, intersectionId);
  drawLights(ctx, canvas, state.traffic_lights, localBounds);
  for (const vehicle of state.vehicles) drawVehicle(ctx, canvas, vehicle, localBounds);
  ctx.restore();
}
const trails = Object.fromEntries(METHODS.map(method => [method, {}]));
function drawFocusChart(intersectionId, index) {
  const canvas = document.getElementById('focusChart'), ctx = canvas.getContext('2d'), w = canvas.width, h = canvas.height;
  const left = 54, right = 18, top = 24, bottom = 28, plotW = w - left - right, plotH = h - top - bottom;
  const values = METHODS.flatMap(method => PAYLOAD.states[method].map(state => ((state.intersections[intersectionId] || {}).halting || 0)));
  const maxValue = Math.max(1, ...values);
  const xAt = i => left + i * plotW / Math.max(1, times.length - 1);
  const yAt = value => top + plotH * (1 - value / maxValue);
  ctx.clearRect(0, 0, w, h); ctx.fillStyle = '#fff'; ctx.fillRect(0, 0, w, h);
  ctx.fillStyle = '#45515a'; ctx.font = '14px Arial'; ctx.fillText('Focused intersection halting vehicles', left, 16);
  for (const method of METHODS) {
    ctx.beginPath(); PAYLOAD.states[method].forEach((state, i) => { const p = [xAt(i), yAt((state.intersections[intersectionId] || {}).halting || 0)]; if (!i) ctx.moveTo(...p); else ctx.lineTo(...p); });
    ctx.strokeStyle = COLORS[method]; ctx.lineWidth = 2; ctx.stroke();
  }
  const currentX = xAt(index); ctx.strokeStyle = '#17202a'; ctx.setLineDash([4, 3]); ctx.beginPath(); ctx.moveTo(currentX, top); ctx.lineTo(currentX, top + plotH); ctx.stroke(); ctx.setLineDash([]);
}
function drawFocus(index) {
  const intersectionId = intersectionSelect.value || Object.keys(CATALOG)[0];
  const method = detailMethod.value || METHODS[0];
  const state = byMethod[method][times[index]];
  const detail = state && state.intersections && state.intersections[intersectionId];
  const canvas = document.getElementById('focusCanvas'), ctx = canvas.getContext('2d'), localBounds = focusBounds(intersectionId);
  const laneIds = new Set((CATALOG[intersectionId] && CATALOG[intersectionId].incoming_lanes) || []);
  ctx.clearRect(0, 0, canvas.width, canvas.height); ctx.fillStyle = '#f4f7f8'; ctx.fillRect(0, 0, canvas.width, canvas.height);
  drawRoadLayer(ctx, canvas, localBounds, focusedLanePredicate(intersectionId, localBounds), true);
  drawJunctions(ctx, canvas, localBounds, intersectionId);
  drawLights(ctx, canvas, state ? state.traffic_lights : {}, localBounds);
  const visible = (state && state.vehicles || []).filter(vehicle => laneIds.has(vehicle.lane_id));
  const trail = trails[method];
  for (const vehicle of visible) trail[vehicle.id] = (trail[vehicle.id] || []).concat([[vehicle.x, vehicle.y]]).slice(-30);
  for (const points of Object.values(trail)) {
    if (points.length < 2) continue;
    ctx.beginPath(); points.forEach((point, i) => { const p = transformWith(canvas, point[0], point[1], localBounds); if (!i) ctx.moveTo(...p); else ctx.lineTo(...p); });
    ctx.strokeStyle = '#f0a34b'; ctx.globalAlpha = 0.72; ctx.lineWidth = 3; ctx.lineCap = 'round'; ctx.stroke(); ctx.globalAlpha = 1;
  }
  for (const vehicle of visible) drawVehicle(ctx, canvas, vehicle, localBounds);
  const rows = detail || {vehicles: 0, halting: 0, mean_speed: 0, phase: '-', signal_state: ''};
  document.getElementById('focusCaption').textContent = `${labels[method]} · intersection ${intersectionId} · local view · t = ${times[index]} s`;
  document.getElementById('focusMetrics').innerHTML = [['Vehicles', rows.vehicles], ['Halting / red', rows.halting], ['Mean speed', `${Number(rows.mean_speed).toFixed(2)} m/s`], ['Phase', rows.phase], ['Signal', rows.signal_state || '-']].map(item => `<div class="metric">${item[0]}<strong>${item[1]}</strong></div>`).join('');
  document.querySelector('#laneTable tbody').innerHTML = (detail ? detail.lanes : []).map(lane => `<tr><td>${lane.id}</td><td>${lane.vehicles}</td><td>${lane.halting}</td><td>${Number(lane.mean_speed).toFixed(2)}</td><td>${Number(lane.occupancy).toFixed(1)}%</td></tr>`).join('');
  drawFocusChart(intersectionId, index);
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
function clampZoom(value) {
  return Math.max(0.5, Math.min(5, value));
}
function clampPan(view, canvas) {
  const maxX = canvas.width * Math.max(0.15, (view.zoom - 1) * 0.5) + 80;
  const maxY = canvas.height * Math.max(0.15, (view.zoom - 1) * 0.5) + 60;
  view.panX = Math.max(-maxX, Math.min(maxX, view.panX));
  view.panY = Math.max(-maxY, Math.min(maxY, view.panY));
}
function canvasPoint(event, canvas) {
  const rect = canvas.getBoundingClientRect();
  return [
    (event.clientX - rect.left) * canvas.width / Math.max(1, rect.width),
    (event.clientY - rect.top) * canvas.height / Math.max(1, rect.height),
  ];
}
function targetMethods(method) {
  return linkTopViews.checked ? METHODS : [method];
}
function renderTopPanels(index) {
  const time = times[index], intersectionId = intersectionSelect.value || Object.keys(CATALOG)[0];
  for (const method of METHODS) {
    const state = byMethod[method][time], panel = panels[method], view = topViews[method];
    const ctx = panel.canvas.getContext('2d');
    drawPanelScene(ctx, panel.canvas, state, intersectionId, view);
    panel.viewMode.textContent = `路口局部视角 · ${intersectionId} · ${overlayTopViews.checked ? '覆盖层开启' : '原始路网'}`;
    panel.viewHint.textContent = `${Math.round(view.zoom * 100)}% · 滚轮缩放 · 拖动移动`;
    panel.queue.textContent = `Queue: ${state.queue_network_sum}`;
    panel.jam.textContent = `红色堵塞: ${state.vehicles.filter(isJammed).length}`;
    panel.throughput.textContent = `Throughput: ${state.throughput_cumulative}`;
  }
}
function resetTopViews(method) {
  for (const target of targetMethods(method)) {
    topViews[target].zoom = 1; topViews[target].panX = 0; topViews[target].panY = 0;
    clampPan(topViews[target], panels[target].canvas);
  }
  renderTopPanels(Number(timeline.value));
}
function zoomTopView(method, factor, anchor) {
  for (const target of targetMethods(method)) {
    const view = topViews[target], canvas = panels[target].canvas, oldZoom = view.zoom, newZoom = clampZoom(oldZoom * factor);
    const ax = anchor ? anchor[0] : canvas.width / 2, ay = anchor ? anchor[1] : canvas.height / 2;
    const cx = canvas.width / 2, cy = canvas.height / 2;
    view.panX = ax - cx - (ax - cx - view.panX) * newZoom / oldZoom;
    view.panY = ay - cy - (ay - cy - view.panY) * newZoom / oldZoom;
    view.zoom = newZoom; clampPan(view, canvas);
  }
  renderTopPanels(Number(timeline.value));
}
function bindTopView(method) {
  const panel = panels[method], canvas = panel.canvas, view = topViews[method];
  panel.panel.querySelector('.zoomIn').onclick = () => zoomTopView(method, 1.25);
  panel.panel.querySelector('.zoomOut').onclick = () => zoomTopView(method, 0.8);
  panel.panel.querySelector('.resetView').onclick = () => resetTopViews(method);
  canvas.addEventListener('wheel', event => {
    event.preventDefault();
    zoomTopView(method, event.deltaY < 0 ? 1.15 : 1 / 1.15, canvasPoint(event, canvas));
  }, {passive: false});
  canvas.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    const point = canvasPoint(event, canvas);
    view.dragging = true; view.pointerId = event.pointerId; view.lastX = point[0]; view.lastY = point[1];
    canvas.classList.add('dragging'); canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener('pointermove', event => {
    if (!view.dragging || view.pointerId !== event.pointerId) return;
    const point = canvasPoint(event, canvas), dx = point[0] - view.lastX, dy = point[1] - view.lastY;
    for (const target of targetMethods(method)) {
      const targetView = topViews[target]; targetView.panX += dx; targetView.panY += dy; clampPan(targetView, panels[target].canvas);
    }
    view.lastX = point[0]; view.lastY = point[1]; renderTopPanels(Number(timeline.value));
  });
  const endDrag = event => {
    if (view.pointerId !== event.pointerId) return;
    view.dragging = false; view.pointerId = null; canvas.classList.remove('dragging');
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  };
  canvas.addEventListener('pointerup', endDrag); canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('dblclick', event => { event.preventDefault(); resetTopViews(method); });
}
function render(index) {
  index = Math.max(0, Math.min(times.length - 1, index)); timeline.value = String(index); const time = times[index]; timeLabel.textContent = `t = ${time} s`;
  renderTopPanels(index); drawMetricChart(index); drawFocus(index);
}
let timer = null;
function stop() { if (timer) { clearInterval(timer); timer = null; } document.getElementById('play').textContent = 'Play'; }
document.getElementById('play').onclick = () => { if (timer) { stop(); return; } document.getElementById('play').textContent = 'Pause'; timer = setInterval(() => { let next = Number(timeline.value) + 1; if (next >= times.length) next = 0; render(next); }, 1000 / Number(document.getElementById('speed').value)); };
document.getElementById('stepBack').onclick = () => { stop(); render(Number(timeline.value) - 1); };
document.getElementById('stepForward').onclick = () => { stop(); render(Number(timeline.value) + 1); };
timeline.oninput = () => { stop(); render(Number(timeline.value)); };
intersectionSelect.onchange = () => { for (const method of METHODS) { topViews[method].zoom = 1; topViews[method].panX = 0; topViews[method].panY = 0; } render(Number(timeline.value)); };
detailMethod.onchange = () => render(Number(timeline.value));
linkTopViews.onchange = () => renderTopPanels(Number(timeline.value));
overlayTopViews.onchange = () => renderTopPanels(Number(timeline.value));
document.getElementById('speed').onchange = () => { if (timer) { stop(); document.getElementById('play').click(); } };
for (const method of METHODS) bindTopView(method);
render(0);
</script>
</body>
</html>
"""


def build_html(payload):
    payload = _presentation_payload(payload)
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).replace("<", "\u003c")
    return HTML_TEMPLATE.replace("__SCENE__", str(payload["scene"])).replace("__PAYLOAD__", encoded)


def _write_payload_bundle(payload, output_dir):
    """Write one self-contained HTML/state/manifest bundle."""
    payload = _presentation_payload(payload)
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(payload["methods"])
    times = list(payload["times"])
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
        "schema_version": 1,
        "scene": payload["scene"],
        "evaluation_seed": int(payload["evaluation_seed"]),
        "methods": methods,
        "action_interval_seconds": int(payload["action_interval_seconds"]),
        "method_labels": payload["method_labels"],
        "method_evaluation_seeds": payload.get("method_evaluation_seeds"),
        "comparison_label": payload.get("comparison_label"),
        "sample_every_seconds": int(payload["sample_every_seconds"]),
        "timestamps": {
            "count": len(times),
            "start": times[0],
            "end": times[-1],
        },
        "network_file": payload["network"]["network_file"],
        "html": str(html_path),
        "states": str(states_path),
        "state_counts": {
            method: len(payload["states"][method]) for method in methods
        },
        "replay_mode": payload["replay_mode"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def _load_html_payload(value):
    """Load a comparison state payload from a directory or JSON file."""
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "comparison_states.json"
    if not path.is_file():
        raise FileNotFoundError(f"Comparison state file does not exist: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not payload.get("methods"):
        raise ValueError(f"Invalid comparison state payload: {path}")
    return path, payload


def append_html_method(base, method, input_value, output, selector=None,
                       method_label=None, source_config=None,
                       evaluation_seed=None):
    """Replay one new method and append it to an existing HTML payload.

    Existing method state arrays are copied from ``base`` and are never
    replayed.  This is intended for adding a validated revisit record without
    changing already-confirmed comparison data.
    """
    _, base_payload = _load_html_payload(base)
    method = str(method)
    if method in base_payload["methods"]:
        raise ValueError(f"Method is already present in base payload: {method}")
    if method not in METHOD_COLORS:
        raise ValueError(
            f"Unknown method color: {method}; add it to METHOD_COLORS before export"
        )
    scene = str(base_payload["scene"])
    source_config = source_config or base_payload["source_config"]
    # ``None`` is intentional here: CONT formal evaluations use SUMO's
    # fixed-default realization and leave evaluation_seed null in the record.
    if evaluation_seed is not None:
        evaluation_seed = int(evaluation_seed)
    sample_every = int(base_payload["sample_every_seconds"])
    start, end = int(base_payload["times"][0]), int(base_payload["times"][-1])
    selector = dict(selector or {})
    replay, interval = _record_actions(
        {method: input_value}, scene, {method: selector}, methods=(method,)
    )
    expected_interval = int(base_payload["action_interval_seconds"])
    if interval != expected_interval:
        raise ValueError(
            f"New method interval {interval} does not match base interval "
            f"{expected_interval}"
        )
    output_dir = Path(output).expanduser().resolve()
    _ensure_sumo_home()
    captured = _capture_method(
        method, replay[method], interval, source_config, evaluation_seed,
        start, end, sample_every, output_dir,
    )
    new_by_time = {state["time"]: state for state in captured["states"]}
    base_times = list(base_payload["times"])
    if set(new_by_time) != set(base_times):
        raise ValueError(
            "New method timestamps do not exactly match the existing HTML "
            f"timeline: expected {base_times[0]}..{base_times[-1]} "
            f"({len(base_times)}), got {len(new_by_time)} states"
        )

    methods = list(base_payload["methods"]) + [method]
    labels = dict(base_payload.get("method_labels", {}))
    labels[method] = method_label or METHOD_LABELS_DEFAULT.get(method, method)
    colors = dict(base_payload.get("colors", {}))
    colors[method] = METHOD_COLORS[method]
    states = {
        existing: base_payload["states"][existing]
        for existing in base_payload["methods"]
    }
    states[method] = [new_by_time[time_seconds] for time_seconds in base_times]
    payload = dict(base_payload)
    payload["methods"] = methods
    payload["method_labels"] = labels
    payload["colors"] = colors
    payload["states"] = states
    method_evaluation_seeds = dict(
        base_payload.get("method_evaluation_seeds", {})
    )
    for existing in base_payload["methods"]:
        method_evaluation_seeds.setdefault(
            existing, int(base_payload["evaluation_seed"])
        )
    method_evaluation_seeds[method] = evaluation_seed
    payload["method_evaluation_seeds"] = method_evaluation_seeds
    payload["method_sources"] = dict(base_payload.get("method_sources", {}))
    payload["method_sources"][method] = {
        "decision_records": str(Path(input_value).expanduser().resolve()),
        "selector": selector,
        "append_mode": "new_method_only",
        "sumo_seed": evaluation_seed,
        "sumo_seed_mode": (
            "fixed_default" if evaluation_seed is None else "explicit"
        ),
    }
    return _write_payload_bundle(payload, output_dir)


def export_html_comparison(inputs, scene, source_config, evaluation_seed, output,
                           start, end, selectors=None, sample_every=1,
                           methods=None, method_labels=None,
                           comparison_label=None):
    if start < 0 or end <= start or sample_every <= 0:
        raise ValueError("Require 0 <= start < end and sample_every > 0")
    if int(start) != start or int(end) != end:
        raise ValueError("start/end must be integer simulation seconds")
    selectors = selectors or {}
    methods = tuple(methods or inputs.keys())
    if not methods:
        raise ValueError("At least one replay method is required")
    if set(methods) - set(inputs):
        raise ValueError(f"Missing replay input(s): {sorted(set(methods) - set(inputs))}")
    display_labels = {
        method: (method_labels or {}).get(method, METHOD_LABELS_DEFAULT.get(method, method))
        for method in methods
    }
    replay, interval = _record_actions(inputs, scene, selectors, methods)
    if start % interval != 0:
        raise ValueError("start must align with the common action interval")
    geometry = load_network_geometry(source_config)
    output_dir = Path(output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    states = {}
    _ensure_sumo_home()
    for method in methods:
        states[method] = _capture_method(
            method, replay[method], interval, source_config, evaluation_seed,
            int(start), int(end), int(sample_every), output_dir,
        )
    common_times = _common_states(states)
    by_time = {
        method: {state["time"]: state for state in states[method]["states"]}
        for method in methods
    }
    scene_intersections = next(iter(states.values()))["intersections"]
    payload = {
        "schema_version": 2,
        "scene": scene,
        "evaluation_seed": int(evaluation_seed),
        "source_config": str(Path(source_config).resolve()),
        "methods": list(methods),
        "method_labels": display_labels,
        "method_evaluation_seeds": {
            method: int(evaluation_seed) for method in methods
        },
        "comparison_label": comparison_label,
        "colors": {method: METHOD_COLORS[method] for method in methods},
        "action_interval_seconds": interval,
        "sample_every_seconds": int(sample_every),
        "times": common_times,
        "network": geometry,
        "states": {
            method: [by_time[method][time_seconds] for time_seconds in common_times]
            for method in methods
        },
        "intersections": scene_intersections,
        "tracking": {
            "focus_supported": True,
            "trail_length_states": 30,
            "detail_metrics": ["vehicles", "halting", "mean_speed", "occupancy", "phase", "signal_state"],
        },
        "replay_mode": "recorded_actions_exact_interval_headless_sumo",
    }
    return _write_payload_bundle(payload, output_dir)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", default="S2", help="Scene label or SUMO network name (default: S2)")
    parser.add_argument("--sim-config", default=None)
    parser.add_argument(
        "--append-to", default=None,
        help="Append one method to an existing comparison_states.json or output directory",
    )
    parser.add_argument("--append-method", default=None)
    parser.add_argument("--append-input", default=None)
    parser.add_argument("--append-method-label", default=None)
    parser.add_argument("--append-controller-id", default=None)
    parser.add_argument("--append-training-seed", type=int, default=None)
    parser.add_argument("--append-evaluation-seed", type=int, default=None)
    parser.add_argument("--online-dqn")
    parser.add_argument("--hadhoa")
    parser.add_argument("--fixedtime", default=str(DEFAULT_S2_FIXEDTIME_INPUT))
    parser.add_argument("--fixedtime-only", action="store_true", help="Replay only FixedTime; defaults to S2 when no scene is supplied")
    parser.add_argument("--hadhoa-label", default=None,
                        help="Display label for the HADHOA panel")
    parser.add_argument("--hadhoa-ignore-evaluation-seed", action="store_true",
                        help="Do not apply the global evaluation seed filter to HADHOA records")
    parser.add_argument("--comparison-label", default=None,
                        help="Optional label shown in the page metadata")
    for method in METHODS:
        parser.add_argument(f"--{method.replace('_', '-')}-controller-id")
        parser.add_argument(f"--{method.replace('_', '-')}-training-seed", type=int)
        parser.add_argument(f"--{method.replace('_', '-')}-evaluation-seed", type=int)
    parser.add_argument(
        "--evaluation-seed", type=int,
        help="Explicit SUMO/evaluation seed; omit in append mode for fixed_default",
    )
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--output")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.append_to is not None:
        missing = [
            name for name, value in (
                ("--append-method", args.append_method),
                ("--append-input", args.append_input),
                ("--output", args.output),
            ) if value is None
        ]
        if missing:
            raise ValueError("Append mode requires: " + ", ".join(missing))
        append_selector = {
            "controller_id": args.append_controller_id,
            "training_seed": args.append_training_seed,
            "evaluation_seed": args.append_evaluation_seed,
        }
        append_selector = {
            key: value for key, value in append_selector.items()
            if value is not None
        }
        manifest = append_html_method(
            args.append_to,
            args.append_method,
            args.append_input,
            args.output,
            selector=append_selector,
            method_label=args.append_method_label,
            source_config=args.sim_config,
            evaluation_seed=args.evaluation_seed,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    missing = [
        name for name, value in (
            ("--evaluation-seed", args.evaluation_seed),
            ("--start", args.start),
            ("--end", args.end),
            ("--output", args.output),
        ) if value is None
    ]
    if missing:
        raise ValueError("Normal export requires: " + ", ".join(missing))
    scene = canonical_scene(args.scene)
    if args.fixedtime_only:
        methods = ("fixedtime",)
    else:
        methods = tuple(method for method, value in (("online_dqn", args.online_dqn), ("hadhoa", args.hadhoa), ("fixedtime", args.fixedtime)) if value)
    if not methods:
        raise ValueError("Provide at least one replay input, or use --fixedtime-only")
    inputs = {method: getattr(args, method) for method in methods}
    source_config = args.sim_config or str(
        Path(__file__).resolve().parents[1] / "configs" / "sim" / f"{scene}.cfg"
    )
    selectors = {
        method: {
            "controller_id": (
                getattr(args, f"{method}_controller_id")
                if getattr(args, f"{method}_controller_id") is not None
                else (f"fixedtime_{scene}" if method == "fixedtime" else None)
            ),
            "training_seed": getattr(args, f"{method}_training_seed"),
            "evaluation_seed": (
                getattr(args, f"{method}_evaluation_seed")
                if getattr(args, f"{method}_evaluation_seed") is not None
                else (
                    None
                    if method == "hadhoa" and args.hadhoa_ignore_evaluation_seed
                    else args.evaluation_seed
                )
            ),
        }
        for method in methods
    }
    manifest = export_html_comparison(
        inputs, scene,
        source_config, args.evaluation_seed, args.output, args.start, args.end,
        selectors, args.sample_every, methods,
        method_labels=(
            {"hadhoa": args.hadhoa_label}
            if args.hadhoa_label is not None else None
        ),
        comparison_label=args.comparison_label,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
