"""Render a reproducible SUMO-vs-aerial Xiasha comparison panel.

The left panel is a schematic SUMO view reconstructed from ``.net.xml`` and
one FCD snapshot.  It draws vehicle lanes, internal turning lanes, signal
link states, and vehicles.  The right panel is the supplied aerial reference.
This is intentionally a static evidence renderer: it does not claim that the
photo has a surveyed coordinate transform.
"""

from __future__ import annotations

import argparse
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


def _shape(value: str | None) -> list[tuple[float, float]]:
    result = []
    for token in (value or "").split():
        try:
            x, y = token.split(",")[:2]
            result.append((float(x), float(y)))
        except (ValueError, IndexError):
            continue
    return result


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _parse_network(path: Path):
    root = ET.parse(path).getroot()
    lanes = []
    lane_by_id = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        internal = edge.get("function") == "internal" or edge_id.startswith(":")
        for lane in edge.findall("lane"):
            points = _shape(lane.get("shape"))
            if len(points) < 2:
                continue
            item = {
                "id": lane.get("id", ""),
                "edge": edge_id,
                "internal": internal,
                "pedestrian": lane.get("allow") == "pedestrian" and not lane.get("disallow"),
                "width": float(lane.get("width", "3.2")),
                "shape": points,
            }
            lanes.append(item)
            lane_by_id[item["id"]] = item
    junctions = []
    for junction in root.findall("junction"):
        if junction.get("type") == "internal":
            continue
        try:
            junctions.append((junction.get("id", ""), float(junction.get("x")), float(junction.get("y"))))
        except (TypeError, ValueError):
            pass
    tl = root.find("tlLogic")
    phases = []
    offset = 0.0
    if tl is not None:
        offset = float(tl.get("offset", "0"))
        for phase in tl.findall("phase"):
            phases.append((float(phase.get("duration", "0")), phase.get("state", ""), phase.get("name", "")))
    controlled = []
    for connection in root.findall("connection"):
        if connection.get("tl") != "J" or connection.get("linkIndex") is None:
            continue
        lane_id = f"{connection.get('from')}_{connection.get('fromLane')}"
        if lane_id in lane_by_id:
            controlled.append({"link": int(connection.get("linkIndex")), "lane": lane_id})
    return lanes, lane_by_id, junctions, phases, offset, controlled


def _parse_fcd(path: Path, target: float) -> tuple[float, list[dict]]:
    best_time = None
    best_rows: list[dict] = []
    for event, element in ET.iterparse(path, events=("end",)):
        if element.tag != "timestep":
            continue
        try:
            current = float(element.get("time", "0"))
        except ValueError:
            element.clear()
            continue
        if best_time is None or abs(current - target) < abs(best_time - target):
            rows = []
            for vehicle in element.findall("vehicle"):
                try:
                    rows.append({
                        "id": vehicle.get("id", ""),
                        "x": float(vehicle.get("x")),
                        "y": float(vehicle.get("y")),
                        "angle": float(vehicle.get("angle", "0")),
                        "speed": float(vehicle.get("speed", "0")),
                    })
                except (TypeError, ValueError):
                    continue
            best_time, best_rows = current, rows
        element.clear()
    if best_time is None:
        raise ValueError(f"FCD file has no timesteps: {path}")
    return best_time, best_rows


def _phase_state(phases, offset: float, time: float) -> str:
    if not phases:
        return ""
    cycle = sum(duration for duration, _, _ in phases)
    position = (time - offset) % cycle
    cursor = 0.0
    for duration, state, _ in phases:
        cursor += duration
        if position < cursor:
            return state
    return phases[-1][1]


def _world_to_panel(bounds, size: tuple[int, int], margin: int = 46):
    width, height = size
    min_x, max_x, min_y, max_y = bounds
    sx = (width - 2 * margin) / max(max_x - min_x, 1e-6)
    sy = (height - 2 * margin) / max(max_y - min_y, 1e-6)
    scale = min(sx, sy)
    ox = (width - scale * (max_x - min_x)) / 2.0 - scale * min_x
    oy = (height + scale * (max_y + min_y)) / 2.0
    return lambda x, y: (ox + scale * x, oy - scale * y), scale


def _arrow(draw: ImageDraw.ImageDraw, a, b, color, width=2):
    draw.line((a, b), fill=color, width=width)
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    tip = b
    left = (b[0] - ux * 10 + px * 4, b[1] - uy * 10 + py * 4)
    right = (b[0] - ux * 10 - px * 4, b[1] - uy * 10 - py * 4)
    draw.polygon((tip, left, right), fill=color)


def render(network: Path, aerial: Path, output: Path, fcd: Path | None, time: float) -> dict:
    lanes, lane_by_id, junctions, phases, offset, controlled = _parse_network(network)
    vehicles: list[dict] = []
    snapshot_time = None
    if fcd is not None:
        snapshot_time, vehicles = _parse_fcd(fcd, time)
    bounds = (0.0, 240.0, 0.0, 240.0)
    left_w, panel_h = 820, 760
    right = Image.open(aerial).convert("RGB")
    right.thumbnail((820, 700), Image.Resampling.LANCZOS)
    right_panel = Image.new("RGB", (left_w, panel_h), "#f4f4f4")
    right_panel.paste(right, ((left_w - right.width) // 2, 50 + (690 - right.height) // 2))
    left = Image.new("RGB", (left_w, panel_h), "#20252a")
    draw = ImageDraw.Draw(left)
    project, scale = _world_to_panel(bounds, (left_w, panel_h - 50))
    # Draw broad road surfaces behind lane centerlines.
    for lane in lanes:
        points = [project(*p) for p in lane["shape"]]
        if lane["pedestrian"]:
            color, width = "#6f7780", max(2, round(lane["width"] * scale))
        elif lane["internal"]:
            color, width = "#4b3a2e", max(3, round(lane["width"] * scale))
        else:
            color, width = "#343b42", max(3, round(lane["width"] * scale))
        draw.line(points, fill=color, width=width, joint="curve")
    # Vehicle lane centerlines and internal turning paths.
    for lane in lanes:
        points = [project(*p) for p in lane["shape"]]
        if lane["pedestrian"]:
            draw.line(points, fill="#a8b0b8", width=1)
        elif lane["internal"]:
            draw.line(points, fill="#f09a4b", width=2)
        else:
            draw.line(points, fill="#49c4e8", width=1)
            if len(points) >= 2:
                _arrow(draw, points[-2], points[-1], "#d9f7ff", 2)
    # Signal dots are located at the controlled incoming lane ends.
    state = _phase_state(phases, offset, snapshot_time if snapshot_time is not None else time)
    signal_colors = {"G": "#27d17f", "g": "#70e4a6", "y": "#f3c64d", "r": "#e05252", "o": "#e05252", "O": "#e05252"}
    seen = set()
    for item in controlled:
        lane = lane_by_id.get(item["lane"])
        if lane is None or item["link"] in seen or not lane["shape"]:
            continue
        seen.add(item["link"])
        x, y = project(*lane["shape"][-1])
        signal = state[item["link"]] if item["link"] < len(state) else "r"
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=signal_colors.get(signal, "#e05252"), outline="white", width=1)
    # Vehicles are coloured by speed; red marks stopped/queued vehicles.
    for vehicle in vehicles:
        x, y = project(vehicle["x"], vehicle["y"])
        angle = math.radians(90.0 - vehicle["angle"])
        ux, uy = math.cos(angle), math.sin(angle)
        vx, vy = -uy, ux
        length, width = 10.0, 5.0
        points = [(x + ux * length / 2 + vx * width / 2, y + uy * length / 2 + vy * width / 2),
                  (x + ux * length / 2 - vx * width / 2, y + uy * length / 2 - vy * width / 2),
                  (x - ux * length / 2 - vx * width / 2, y - uy * length / 2 - vy * width / 2),
                  (x - ux * length / 2 + vx * width / 2, y - uy * length / 2 + vy * width / 2)]
        draw.polygon(points, fill="#ef5350" if vehicle["speed"] < 0.2 else "#ffd166", outline="#111820")
    draw.text((16, 12), "SUMO network / vehicles / signal links", fill="white", font=_font(24))
    label_time = "" if snapshot_time is None else f"  t={snapshot_time:.1f}s  vehicles={len(vehicles)}"
    draw.text((16, 42), f"J: 12-phase cycle, green/yellow/red dots show link state{label_time}", fill="#d4dbe1", font=_font(15))
    # Right panel labels and a compact legend.
    right_draw = ImageDraw.Draw(right_panel)
    right_draw.text((16, 12), "Aerial reference (approximate geometry)", fill="#111820", font=_font(24))
    right_draw.text((16, panel_h - 34), "Reference: a1俯视底图.png", fill="#3b4650", font=_font(15))
    canvas = Image.new("RGB", (left_w * 2, panel_h), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right_panel, (left_w, 0))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return {
        "network": str(network.resolve()),
        "aerial": str(aerial.resolve()),
        "fcd": None if fcd is None else str(fcd.resolve()),
        "requested_time_seconds": time,
        "snapshot_time_seconds": snapshot_time,
        "vehicle_count": len(vehicles),
        "phase_count": len(phases),
        "controlled_links": len(controlled),
        "output": str(output.resolve()),
    }


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", type=Path, required=True)
    parser.add_argument("--aerial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fcd", type=Path, default=None)
    parser.add_argument("--time", type=float, default=600.0)
    args = parser.parse_args(argv)
    import json

    manifest = render(args.network, args.aerial, args.output, args.fcd, args.time)
    args.output.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
