from __future__ import annotations

import argparse
import csv
import json
import math
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIR = ROOT / "data/raw_data/xiasha1*1"
DEFAULT_INPUT = DEFAULT_DIR / "原始数据/语义事件表(13min).csv"
DEFAULT_NET = DEFAULT_DIR / "原始数据/real_scene.net.xml"
DEFAULT_CYCLE = DEFAULT_DIR / "原始数据/红绿灯周期"

PHASES = [
    ("NS直行绿", 27, "G"),
    ("NS直行黄", 3, "y"),
    ("NS过渡红", 2, "r"),
    ("NS左转绿", 22, "G"),
    ("NS左转黄", 3, "y"),
    ("南北东西全红", 5, "r"),
    ("EW直行绿", 34, "G"),
    ("EW直行黄", 3, "y"),
    ("EW过渡红", 2, "r"),
    ("EW左转绿", 19, "G"),
    ("EW左转黄", 3, "y"),
    ("东西南北全红", 5, "r"),
]

REQUIRED_FIELDS = ("entry_edge", "exit_edge", "entry_time")
CONTROLLED_ENTRY_EDGES = {"N2J", "S2J", "E2J", "W2J"}
VEHICLE_TYPE_ATTRIBUTES = {
    "id": "xiasha_passenger",
    "vClass": "passenger",
    "accel": "2.6",
    "decel": "4.5",
    "sigma": "0.5",
    "length": "5",
    "minGap": "2.5",
    "maxSpeed": "13.89",
}


def _format_number(value: float) -> str:
    return f"{value:.2f}"


def _report_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_xml(tree: ET.ElementTree, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def _skip_detail_reason(row, errors):
    missing = [key for key in REQUIRED_FIELDS if not row.get(key, "").strip()]
    if missing:
        return "missing_" + "_and_".join(missing)
    if "invalid_time" in errors:
        return "invalid_entry_time"
    return "unknown_skip_reason"


def _skip_category(errors):
    if len(errors) > 1:
        return "multiple_errors"
    return errors[0]


def parse_rows(path: Path):
    """Read event rows and classify every skipped record explicitly."""

    with path.open(encoding="utf-8-sig", newline="") as file_obj:
        rows = list(csv.DictReader(file_obj))

    detail_reasons = Counter()
    category_counts = Counter()
    component_counts = Counter()
    valid = []
    skipped = []

    for index, row in enumerate(rows, start=2):
        errors = []
        if not row.get("entry_edge", "").strip():
            errors.append("missing_entry")
        if not row.get("exit_edge", "").strip():
            errors.append("missing_exit")

        entry_time_text = row.get("entry_time", "").strip()
        if not entry_time_text:
            errors.append("invalid_time")
            entry_time = None
        else:
            try:
                entry_time = float(entry_time_text)
            except ValueError:
                errors.append("invalid_time")
                entry_time = None

        if errors:
            category = _skip_category(errors)
            component_counts.update(errors)
            detail_reasons[_skip_detail_reason(row, errors)] += 1
            category_counts[category] += 1
            skipped.append(
                {
                    "row_number": index,
                    "vehicle_id": row.get("vehicle_id", ""),
                    "semantic_id": row.get("semantic_id", ""),
                    "category": category,
                    "errors": ";".join(errors),
                    "detail_reason": _skip_detail_reason(row, errors),
                    "entry_edge": row.get("entry_edge", ""),
                    "exit_edge": row.get("exit_edge", ""),
                    "observed_event_time": row.get("observed_event_time", ""),
                    "entry_time": row.get("entry_time", ""),
                    "exit_time": row.get("exit_time", ""),
                }
            )
            continue

        row["_entry_time"] = entry_time
        valid.append(row)

    shift = max(0.0, -min((row["_entry_time"] for row in valid), default=0.0))
    for row in valid:
        row["_depart"] = row["_entry_time"] + shift

    return rows, valid, shift, detail_reasons, category_counts, component_counts, skipped


def _network_geometry(root: ET.Element):
    junctions = {}
    for junction in root.findall("junction"):
        junction_id = junction.get("id")
        if junction_id and junction.get("x") is not None and junction.get("y") is not None:
            junctions[junction_id] = (
                float(junction.get("x")),
                float(junction.get("y")),
            )

    edges = {}
    for edge in root.findall("edge"):
        edge_id = edge.get("id")
        from_id = edge.get("from")
        to_id = edge.get("to")
        if edge_id and from_id and to_id and edge.get("function") != "internal":
            edges[edge_id] = (from_id, to_id)
    return junctions, edges


def classify_movement(entry_edge, exit_edge, junctions, edges, junction_id="J"):
    """Classify a movement from network geometry, including U-turns.

    The incoming vector points from the source junction to the signalized
    junction. The outgoing vector points from the signalized junction to the
    destination junction. This avoids assumptions about edge-name spelling.
    """

    if entry_edge not in edges or exit_edge not in edges:
        raise ValueError(f"Unknown edge pair: {entry_edge} -> {exit_edge}")
    source_id, incoming_to = edges[entry_edge]
    outgoing_from, target_id = edges[exit_edge]
    if incoming_to != junction_id or outgoing_from != junction_id:
        raise ValueError(
            f"Edges do not form a movement through {junction_id}: "
            f"{entry_edge} -> {exit_edge}"
        )
    if source_id not in junctions or target_id not in junctions or junction_id not in junctions:
        raise ValueError(f"Missing junction geometry for {entry_edge} -> {exit_edge}")

    center_x, center_y = junctions[junction_id]
    source_x, source_y = junctions[source_id]
    target_x, target_y = junctions[target_id]
    incoming_x, incoming_y = center_x - source_x, center_y - source_y
    outgoing_x, outgoing_y = target_x - center_x, target_y - center_y
    incoming_norm = math.hypot(incoming_x, incoming_y)
    outgoing_norm = math.hypot(outgoing_x, outgoing_y)
    if incoming_norm == 0 or outgoing_norm == 0:
        raise ValueError(f"Zero-length movement vector: {entry_edge} -> {exit_edge}")

    dot = (incoming_x * outgoing_x + incoming_y * outgoing_y) / (
        incoming_norm * outgoing_norm
    )
    cross = (incoming_x * outgoing_y - incoming_y * outgoing_x) / (
        incoming_norm * outgoing_norm
    )
    if dot > 0.5:
        return "straight"
    if dot < -0.5:
        return "u_turn"
    if cross > 0:
        return "left"
    if cross < 0:
        return "right"
    raise ValueError(f"Cannot classify movement: {entry_edge} -> {exit_edge}")


def _approach_group(entry_edge, junctions, edges, junction_id="J"):
    source_id, incoming_to = edges[entry_edge]
    if incoming_to != junction_id:
        raise ValueError(f"Edge is not incoming to {junction_id}: {entry_edge}")
    center_x, center_y = junctions[junction_id]
    source_x, source_y = junctions[source_id]
    dx, dy = source_x - center_x, source_y - center_y
    return "NS" if abs(dy) > abs(dx) else "EW"


def write_routes(rows, path: Path, flows=False, window=60.0):
    if window <= 0:
        raise ValueError("flow window must be positive")

    root = ET.Element("routes")
    ET.SubElement(root, "vType", attrib=VEHICLE_TYPE_ATTRIBUTES)
    if flows:
        groups = defaultdict(list)
        for row in rows:
            begin = math.floor(row["_depart"] / window) * window
            groups[(row["entry_edge"], row["exit_edge"], begin)].append(row)
        ordered_groups = sorted(
            groups.items(), key=lambda item: (item[0][2], item[0][0], item[0][1])
        )
        flow_elements = []
        for index, ((entry, exit_edge, begin), group) in enumerate(ordered_groups):
            route_id = f"route_{index:04d}"
            flow_elements.append(
                ET.Element(
                    "flow",
                    {
                        "id": f"flow_{index:04d}",
                        "type": VEHICLE_TYPE_ATTRIBUTES["id"],
                        "route": route_id,
                        "begin": _format_number(begin),
                        "end": _format_number(begin + window),
                        "number": str(len(group)),
                    },
                )
            )
            ET.SubElement(root, "route", id=route_id, edges=f"{entry} {exit_edge}")
        for flow in flow_elements:
            root.append(flow)
    else:
        vehicle_elements = []
        for index, row in enumerate(sorted(rows, key=lambda item: item["_depart"])):
            route_id = f"route_{index:04d}"
            vehicle_id = row.get("vehicle_id") or f"vehicle_{index:04d}"
            vehicle_elements.append(
                ET.Element(
                    "vehicle",
                    {
                        "id": vehicle_id,
                        "type": VEHICLE_TYPE_ATTRIBUTES["id"],
                        "route": route_id,
                        "depart": _format_number(row["_depart"]),
                    },
                )
            )
            ET.SubElement(
                root,
                "route",
                id=route_id,
                edges=f"{row['entry_edge']} {row['exit_edge']}",
            )
        for vehicle in vehicle_elements:
            root.append(vehicle)
    _write_xml(ET.ElementTree(root), path)


def _phase_is_active(name, movement, approach_group):
    if movement == "right":
        return True
    if movement == "u_turn":
        return False
    if approach_group == "NS" and not name.startswith("NS"):
        return False
    if approach_group == "EW" and not name.startswith("EW"):
        return False
    if "直行" in name:
        return movement == "straight"
    if "左转" in name:
        return movement == "left"
    return False


def _build_tl_logic(states, offset="0"):
    tl_logic = ET.Element(
        "tlLogic",
        id="J",
        type="static",
        programID="xiasha_cycle",
        offset=str(offset),
    )
    for (name, duration, _), state in zip(PHASES, states):
        ET.SubElement(
            tl_logic,
            "phase",
            duration=str(duration),
            state=state,
            name=name,
        )
    return tl_logic


def write_signal_network(net_path: Path, out_path: Path, add_path: Path):
    tree = ET.parse(net_path)
    root = tree.getroot()
    junction = next((item for item in root.findall("junction") if item.get("id") == "J"), None)
    if junction is None:
        raise ValueError("network does not contain central junction J")
    junction.set("type", "traffic_light")
    junctions, edges = _network_geometry(root)

    controlled = []
    candidate_connections = []
    for connection in root.findall("connection"):
        from_edge = connection.get("from", "")
        to_edge = connection.get("to", "")
        if from_edge in CONTROLLED_ENTRY_EDGES and to_edge.startswith("J2"):
            candidate_connections.append(connection)
            movement = classify_movement(from_edge, to_edge, junctions, edges)
            connection.set("tl", "J")
            connection.set("linkIndex", str(len(controlled)))
            controlled.append({"element": connection, "movement": movement})

    states = []
    for name, _, mark in PHASES:
        chars = []
        for item in controlled:
            connection = item["element"]
            movement = item["movement"]
            group = _approach_group(connection.get("from", ""), junctions, edges)
            active = _phase_is_active(name, movement, group)
            chars.append(mark if active else "r")
        states.append("".join(chars))

    # SUMO versions used by this repository require the effective tlLogic to
    # be present in the network. The additional file is retained as an audit
    # and portable signal definition, while generated cfg files use the net.
    for existing in list(root.findall("tlLogic")):
        if existing.get("id") == "J":
            root.remove(existing)
    tl_logic = _build_tl_logic(states)
    children = list(root)
    first_connection = next(
        (index for index, child in enumerate(children) if child.tag == "connection"),
        len(children),
    )
    root.insert(first_connection, tl_logic)
    _write_xml(tree, out_path)

    additional = ET.Element("additional")
    additional.append(_build_tl_logic(states))
    _write_xml(ET.ElementTree(additional), add_path)
    return {
        "controlled": controlled,
        "candidate_connections": candidate_connections,
        "states": states,
    }


def write_cfg(path, net, routes, begin=0.0, end=900.0):
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=net.name)
    ET.SubElement(inputs, "route-files", value=routes.name)
    time = ET.SubElement(root, "time")
    ET.SubElement(time, "begin", value=_format_number(begin))
    ET.SubElement(time, "end", value=_format_number(end))
    _write_xml(ET.ElementTree(root), path)


def _phase_state_text(states, link_index):
    return ";".join(
        f"{name}={states[phase_index][link_index]}"
        for phase_index, (name, _, _) in enumerate(PHASES)
    )


def write_connection_audit(path: Path, controlled, states):
    fieldnames = [
        "linkIndex",
        "from_edge",
        "to_edge",
        "from_lane",
        "to_lane",
        "movement",
        "phase_states",
        "green_or_yellow_phases",
    ]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        for item in controlled:
            connection = item["element"]
            link_index = int(connection.get("linkIndex", "-1"))
            phase_states = _phase_state_text(states, link_index)
            active_phases = ";".join(
                name
                for phase_index, (name, _, _) in enumerate(PHASES)
                if states[phase_index][link_index] in {"G", "y"}
            )
            writer.writerow(
                {
                    "linkIndex": link_index,
                    "from_edge": connection.get("from", ""),
                    "to_edge": connection.get("to", ""),
                    "from_lane": connection.get("fromLane", ""),
                    "to_lane": connection.get("toLane", ""),
                    "movement": item["movement"],
                    "phase_states": phase_states,
                    "green_or_yellow_phases": active_phases,
                }
            )


def write_phase_matrix(path_csv: Path, path_txt: Path, controlled, states):
    link_indices = list(range(len(controlled)))
    columns = [f"link_{index:02d}" for index in link_indices]
    with path_csv.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["phase_index", "phase_name", "duration", *columns])
        for phase_index, (name, duration, _) in enumerate(PHASES):
            writer.writerow([phase_index, name, duration, *states[phase_index]])

    with path_txt.open("w", encoding="utf-8") as file_obj:
        file_obj.write("linkIndex\t" + "\t".join(f"{index:02d}" for index in link_indices) + "\n")
        file_obj.write(
            "movement\t"
            + "\t".join(item["movement"] for item in controlled)
            + "\n"
        )
        file_obj.write(
            "connection\t"
            + "\t".join(
                f"{item['element'].get('from', '')}->{item['element'].get('to', '')}"
                for item in controlled
            )
            + "\n"
        )
        for phase_index, (name, duration, _) in enumerate(PHASES):
            file_obj.write(
                f"{phase_index:02d} {name} ({duration}s)\t"
                + "\t".join(states[phase_index])
                + "\n"
            )


def _route_definitions(root):
    return {
        route.get("id"): tuple(route.get("edges", "").split())
        for route in root.findall("route")
        if route.get("id") and route.get("edges")
    }


def _artifact_vehicle_counts(path: Path, window: float):
    root = ET.parse(path).getroot()
    definitions = _route_definitions(root)
    od_counts = Counter()
    time_counts = Counter()
    od_time_counts = Counter()
    vehicle_ids = []
    for vehicle in root.findall("vehicle"):
        route = definitions.get(vehicle.get("route"))
        if route is None or len(route) != 2:
            raise ValueError(f"Vehicle references an invalid route: {vehicle.get('id')}")
        depart = float(vehicle.get("depart", ""))
        begin = math.floor(depart / window) * window
        key = (route[0], route[1])
        od_counts[key] += 1
        time_counts[begin] += 1
        od_time_counts[(route[0], route[1], begin)] += 1
        vehicle_ids.append(vehicle.get("id", ""))
    return {
        "count": len(root.findall("vehicle")),
        "od_counts": od_counts,
        "time_counts": time_counts,
        "od_time_counts": od_time_counts,
        "duplicate_vehicle_ids": [
            vehicle_id
            for vehicle_id, count in Counter(vehicle_ids).items()
            if vehicle_id and count > 1
        ],
    }


def _artifact_flow_counts(path: Path, window: float):
    root = ET.parse(path).getroot()
    definitions = _route_definitions(root)
    od_counts = Counter()
    time_counts = Counter()
    od_time_counts = Counter()
    group_count = 0
    for flow in root.findall("flow"):
        route = definitions.get(flow.get("route"))
        if route is None or len(route) != 2:
            raise ValueError(f"Flow references an invalid route: {flow.get('id')}")
        begin = float(flow.get("begin", ""))
        number = int(flow.get("number", ""))
        if not math.isclose(begin / window, round(begin / window)):
            raise ValueError(f"Flow begin is not aligned to the window: {begin}")
        key = (route[0], route[1])
        od_counts[key] += number
        time_counts[begin] += number
        od_time_counts[(route[0], route[1], begin)] += number
        group_count += 1
    return {
        "group_count": group_count,
        "count": sum(od_counts.values()),
        "od_counts": od_counts,
        "time_counts": time_counts,
        "od_time_counts": od_time_counts,
    }


def _counter_records(counter, keys):
    records = []
    for key in sorted(counter):
        values = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        values["count"] = counter[key]
        records.append(values)
    return records


def write_demand_audit(path: Path, vehicle_path: Path | None, flow_path: Path | None, rows, window):
    expected_od = Counter((row["entry_edge"], row["exit_edge"]) for row in rows)
    expected_time = Counter(
        math.floor(row["_depart"] / window) * window for row in rows
    )
    expected_od_time = Counter(
        (
            row["entry_edge"],
            row["exit_edge"],
            math.floor(row["_depart"] / window) * window,
        )
        for row in rows
    )

    vehicle = _artifact_vehicle_counts(vehicle_path, window) if vehicle_path and vehicle_path.is_file() else None
    flow = _artifact_flow_counts(flow_path, window) if flow_path and flow_path.is_file() else None

    fieldnames = [
        "scope",
        "entry_edge",
        "exit_edge",
        "window_begin",
        "window_end",
        "valid_count",
        "vehicle_count",
        "flow_count",
        "vehicle_delta",
        "flow_delta",
        "status",
    ]
    audit_rows = []

    def add_row(scope, key, expected, vehicle_count, flow_count):
        entry_edge = exit_edge = ""
        window_begin = ""
        window_end = ""
        if scope == "od":
            entry_edge, exit_edge = key
        elif scope == "time_window":
            window_begin = _format_number(key)
            window_end = _format_number(key + window)
        else:
            entry_edge, exit_edge, window_value = key
            window_begin = _format_number(window_value)
            window_end = _format_number(window_value + window)
        status = "OK"
        if vehicle_count is None or flow_count is None:
            status = "NOT_AVAILABLE"
        elif expected != vehicle_count or expected != flow_count:
            status = "MISMATCH"
        audit_rows.append(
            {
                "scope": scope,
                "entry_edge": entry_edge,
                "exit_edge": exit_edge,
                "window_begin": window_begin,
                "window_end": window_end,
                "valid_count": expected,
                "vehicle_count": "" if vehicle_count is None else vehicle_count,
                "flow_count": "" if flow_count is None else flow_count,
                "vehicle_delta": "" if vehicle_count is None else vehicle_count - expected,
                "flow_delta": "" if flow_count is None else flow_count - expected,
                "status": status,
            }
        )

    all_od_keys = set(expected_od)
    if vehicle:
        all_od_keys.update(vehicle["od_counts"])
    if flow:
        all_od_keys.update(flow["od_counts"])
    for key in sorted(all_od_keys):
        add_row(
            "od",
            key,
            expected_od[key],
            vehicle["od_counts"].get(key) if vehicle else None,
            flow["od_counts"].get(key) if flow else None,
        )

    all_time_keys = set(expected_time)
    if vehicle:
        all_time_keys.update(vehicle["time_counts"])
    if flow:
        all_time_keys.update(flow["time_counts"])
    for key in sorted(all_time_keys):
        add_row(
            "time_window",
            key,
            expected_time[key],
            vehicle["time_counts"].get(key) if vehicle else None,
            flow["time_counts"].get(key) if flow else None,
        )

    all_od_time_keys = set(expected_od_time)
    if vehicle:
        all_od_time_keys.update(vehicle["od_time_counts"])
    if flow:
        all_od_time_keys.update(flow["od_time_counts"])
    for key in sorted(all_od_time_keys):
        add_row(
            "od_time_window",
            key,
            expected_od_time[key],
            vehicle["od_time_counts"].get(key) if vehicle else None,
            flow["od_time_counts"].get(key) if flow else None,
        )

    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit_rows)

    vehicle_count = vehicle["count"] if vehicle else None
    flow_count = flow["count"] if flow else None
    conservation_passed = (
        vehicle_count is not None
        and flow_count is not None
        and len(rows) == vehicle_count == flow_count
        and all(item["status"] == "OK" for item in audit_rows)
    )
    return {
        "valid_count": len(rows),
        "vehicle_count": vehicle_count,
        "flow_count": flow_count,
        "flow_group_count": flow["group_count"] if flow else None,
        "conservation_equation": (
            f"{len(rows)}={vehicle_count if vehicle_count is not None else '?'}="
            f"{flow_count if flow_count is not None else '?'}"
        ),
        "passed": conservation_passed,
        "duplicate_vehicle_ids": vehicle["duplicate_vehicle_ids"] if vehicle else [],
        "od_counts": [
            {
                "entry_edge": key[0],
                "exit_edge": key[1],
                "valid_count": expected_od[key],
                "vehicle_count": vehicle["od_counts"].get(key) if vehicle else None,
                "flow_count": flow["od_counts"].get(key) if flow else None,
            }
            for key in sorted(all_od_keys)
        ],
        "time_window_counts": [
            {
                "window_begin": key,
                "window_end": key + window,
                "valid_count": expected_time[key],
                "vehicle_count": vehicle["time_counts"].get(key) if vehicle else None,
                "flow_count": flow["time_counts"].get(key) if flow else None,
            }
            for key in sorted(all_time_keys)
        ],
    }


def _observed_time_bin(value, window):
    try:
        return _format_number(math.floor(float(value) / window) * window)
    except (TypeError, ValueError):
        return "unknown"


def write_skip_audits(skip_path: Path, summary_path: Path, skipped, window):
    detail_fields = [
        "row_number",
        "vehicle_id",
        "semantic_id",
        "category",
        "errors",
        "detail_reason",
        "entry_edge",
        "exit_edge",
        "observed_event_time",
        "entry_time",
        "exit_time",
    ]
    with skip_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=detail_fields)
        writer.writeheader()
        writer.writerows(skipped)

    summary = Counter()
    for row in skipped:
        summary[
            (
                row["category"],
                row["entry_edge"] or "<missing>",
                row["exit_edge"] or "<missing>",
                _observed_time_bin(row["observed_event_time"], window),
            )
        ] += 1
    summary_fields = [
        "category",
        "entry_edge",
        "exit_edge",
        "observed_time_bin_start",
        "count",
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=summary_fields)
        writer.writeheader()
        for key in sorted(summary):
            writer.writerow(
                {
                    "category": key[0],
                    "entry_edge": key[1],
                    "exit_edge": key[2],
                    "observed_time_bin_start": key[3],
                    "count": summary[key],
                }
            )

    by_entry = Counter(row["entry_edge"] or "<missing>" for row in skipped)
    by_time = Counter(
        _observed_time_bin(row["observed_event_time"], window) for row in skipped
    )
    return {
        "by_entry_edge": dict(sorted(by_entry.items())),
        "by_observed_time_bin": dict(sorted(by_time.items())),
        "summary_rows": len(summary),
    }


def write_route_topology_audit(path: Path, rows, net_path: Path):
    root = ET.parse(net_path).getroot()
    junctions, edges = _network_geometry(root)
    direct_connections = {
        (connection.get("from"), connection.get("to"))
        for connection in root.findall("connection")
        if connection.get("from") in CONTROLLED_ENTRY_EDGES
        and connection.get("to", "").startswith("J2")
    }
    pairs = Counter((row["entry_edge"], row["exit_edge"]) for row in rows)
    fields = [
        "entry_edge",
        "exit_edge",
        "observed_count",
        "movement",
        "direct_connection",
        "status",
    ]
    missing_count = 0
    missing_pairs = []
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=fields)
        writer.writeheader()
        for entry_edge, exit_edge in sorted(pairs):
            movement = classify_movement(entry_edge, exit_edge, junctions, edges)
            direct = (entry_edge, exit_edge) in direct_connections
            if not direct:
                missing_count += pairs[(entry_edge, exit_edge)]
                missing_pairs.append(
                    {
                        "entry_edge": entry_edge,
                        "exit_edge": exit_edge,
                        "count": pairs[(entry_edge, exit_edge)],
                    }
                )
            writer.writerow(
                {
                    "entry_edge": entry_edge,
                    "exit_edge": exit_edge,
                    "observed_count": pairs[(entry_edge, exit_edge)],
                    "movement": movement,
                    "direct_connection": str(direct).lower(),
                    "status": "OK" if direct else "MISSING_CONNECTION",
                }
            )
    return {
        "observed_od_pairs": len(pairs),
        "missing_connection_records": missing_count,
        "missing_connection_pairs": missing_pairs,
    }


def _phase_matrix_audit(controlled, states, candidate_connections):
    link_indices = [int(item["element"].get("linkIndex", "-1")) for item in controlled]
    connection_keys = [
        (
            item["element"].get("from", ""),
            item["element"].get("to", ""),
            item["element"].get("fromLane", ""),
            item["element"].get("toLane", ""),
        )
        for item in controlled
    ]
    duplicate_keys = [
        list(key) for key, count in Counter(connection_keys).items() if count > 1
    ]
    duplicate_indices = [
        index for index, count in Counter(link_indices).items() if count > 1
    ]
    expected_indices = list(range(len(controlled)))
    return {
        "candidate_connections": len(candidate_connections),
        "controlled_connections": len(controlled),
        "omitted_connections": len(candidate_connections) - len(controlled),
        "duplicate_connections": duplicate_keys,
        "duplicate_link_indices": duplicate_indices,
        "link_indices_contiguous": link_indices == expected_indices,
        "phase_count": len(PHASES),
        "phase_state_lengths": sorted({len(state) for state in states}),
        "phase_duration_total": sum(duration for _, duration, _ in PHASES),
        "movement_counts": dict(
            Counter(item["movement"] for item in controlled)
        ),
        "u_turn_controlled_connections": sum(
            item["movement"] == "u_turn" for item in controlled
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Convert xiasha semantic events to auditable SUMO demand and signal files"
    )
    parser.add_argument("--mode", choices=["all", "vehicles", "flows"], default="all")
    parser.add_argument("--all", action="store_true", help="generate all artifacts")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--network", type=Path, default=DEFAULT_NET)
    parser.add_argument("--window", type=float, default=60.0)
    parser.add_argument("--shift", type=float, default=None)
    parser.add_argument(
        "--strict-topology",
        action="store_true",
        help="fail after writing artifacts if an observed OD lacks a network connection",
    )
    args = parser.parse_args(argv)
    mode = "all" if args.all else args.mode
    if args.window <= 0:
        raise ValueError("flow window must be positive")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (
        raw,
        valid,
        shift,
        detail_reasons,
        category_counts,
        component_counts,
        skipped,
    ) = parse_rows(args.input)
    shift = args.shift if args.shift is not None else shift
    for row in valid:
        row["_depart"] = row["_entry_time"] + shift

    vehicles_path = output_dir / "xiasha1_sumo_vehicles.rou.xml"
    flows_path = output_dir / "xiasha1_sumo_flows.rou.xml"
    network_path = output_dir / "xiasha1_sumo_signal.net.xml"
    additional_path = output_dir / "xiasha1_sumo_signal.add.xml"
    report_path = output_dir / "xiasha1_sumo_conversion_report.json"
    connection_audit_path = output_dir / "xiasha1_sumo_connection_movement_audit.csv"
    phase_matrix_csv_path = output_dir / "xiasha1_sumo_phase_matrix.csv"
    phase_matrix_txt_path = output_dir / "xiasha1_sumo_phase_matrix.txt"
    demand_audit_path = output_dir / "xiasha1_sumo_demand_audit.csv"
    skip_audit_path = output_dir / "xiasha1_sumo_skip_audit.csv"
    skip_summary_path = output_dir / "xiasha1_sumo_skip_summary.csv"
    topology_audit_path = output_dir / "xiasha1_sumo_route_topology_audit.csv"

    if mode in {"all", "vehicles"}:
        write_routes(valid, vehicles_path, window=args.window)
    if mode in {"all", "flows"}:
        write_routes(valid, flows_path, flows=True, window=args.window)

    skip_bias = write_skip_audits(skip_audit_path, skip_summary_path, skipped, args.window)
    topology_audit = write_route_topology_audit(topology_audit_path, valid, args.network)
    signal_audit = None
    demand_audit = None
    simulation_end = None

    if mode == "all":
        signal_audit = write_signal_network(args.network, network_path, additional_path)
        connection_audit = _phase_matrix_audit(
            signal_audit["controlled"],
            signal_audit["states"],
            signal_audit["candidate_connections"],
        )
        write_connection_audit(
            connection_audit_path,
            signal_audit["controlled"],
            signal_audit["states"],
        )
        write_phase_matrix(
            phase_matrix_csv_path,
            phase_matrix_txt_path,
            signal_audit["controlled"],
            signal_audit["states"],
        )
        simulation_end = max(
            900.0,
            math.ceil(max(row["_depart"] for row in valid) + 120.0)
            if valid
            else 900.0,
        )
        write_cfg(
            output_dir / "xiasha1_sumo_vehicles.sumocfg",
            network_path,
            vehicles_path,
            end=simulation_end,
        )
        write_cfg(
            output_dir / "xiasha1_sumo_flows.sumocfg",
            network_path,
            flows_path,
            end=simulation_end,
        )
    else:
        connection_audit = None

    demand_audit = write_demand_audit(
        demand_audit_path,
        vehicles_path if vehicles_path.is_file() else None,
        flows_path if flows_path.is_file() else None,
        valid,
        args.window,
    )

    report = {
        "schema_version": 2,
        "input": _report_path(args.input),
        "network": _report_path(args.network),
        "cycle_file": _report_path(DEFAULT_CYCLE),
        "total_records": len(raw),
        "valid_records": len(valid),
        "skipped_records": len(raw) - len(valid),
        "skip_category_counts": dict(sorted(category_counts.items())),
        "skip_error_component_counts": dict(sorted(component_counts.items())),
        "skip_category_conservation": {
            "equation": (
                f"{len(raw) - len(valid)}="
                + "+".join(
                    str(category_counts.get(category, 0))
                    for category in (
                        "missing_entry",
                        "missing_exit",
                        "invalid_time",
                        "multiple_errors",
                    )
                )
            ),
            "passed": sum(category_counts.values()) == len(raw) - len(valid),
        },
        "skip_reasons_detail": dict(sorted(detail_reasons.items())),
        "skip_audit_file": skip_audit_path.name,
        "skip_summary_file": skip_summary_path.name,
        "skip_bias_summary": skip_bias,
        "shift_seconds": shift,
        "depart_time_range": {
            "min": min((row["_depart"] for row in valid), default=None),
            "max": max((row["_depart"] for row in valid), default=None),
        },
        "flow_window_seconds": args.window,
        "demand_audit_file": demand_audit_path.name,
        "demand_conservation": demand_audit,
        "route_topology_audit_file": topology_audit_path.name,
        "route_topology_audit": topology_audit,
        "controlled_connections": (
            len(signal_audit["controlled"]) if signal_audit else 0
        ),
        "connection_movement_audit_file": connection_audit_path.name
        if signal_audit
        else None,
        "phase_matrix_csv_file": phase_matrix_csv_path.name if signal_audit else None,
        "phase_matrix_text_file": phase_matrix_txt_path.name if signal_audit else None,
        "connection_audit": connection_audit,
        "phase_durations_total": sum(duration for _, duration, _ in PHASES),
        "phase_count": len(PHASES),
        "simulation_begin_seconds": 0.0 if simulation_end is not None else None,
        "simulation_end_seconds": simulation_end,
        "exit_time_usage": "not used for departure or arrival calibration",
        "movement_classification": "network geometry vectors; straight/left/right/u_turn",
        "strict_topology_requested": args.strict_topology,
    }
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if args.strict_topology and topology_audit["missing_connection_records"]:
        raise ValueError(
            "Observed OD pairs without network connections: "
            f"{topology_audit['missing_connection_pairs']}"
        )


if __name__ == "__main__":
    main()
