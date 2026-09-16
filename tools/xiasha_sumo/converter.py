from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
import tempfile
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
SIGNAL_CYCLE_SECONDS = sum(duration for _, duration, _ in PHASES)
SIGNAL_ANCHOR_PHASE = "EW直行绿"
DEFAULT_SIGNAL_ANCHOR_VEHICLE_ID = "lv_0016"
# This run samples SUMO at one-tenth-second steps. Edge-exit timestamps are kept at
# the step boundary; diagnostics may use the interval midpoint explicitly.
DEFAULT_SIGNAL_STEP_LENGTH_SECONDS = 0.1
ALIGNMENT_WARMUP_CYCLES = 2
# Approximate free-flow time from the near-line injection point to the stop line.
# The evidence workflow replaces this coarse anchor with an all-event scan.
DEFAULT_SIGNAL_ANCHOR_TRAVEL_SECONDS = 2.20

# Approximate positions of the four coloured video counting lines, measured
# along each road arm from the junction-side end of the external SUMO edge.
# The source image has no ground-plane calibration, so these are deliberately
# coarse priors (roughly derived from lane width and several vehicle lengths),
# not surveyed geometry.  Event matching refines the timing interpretation
# while keeping every line within its disclosed image-derived interval.
LOGIC_LINE_MODEL = {
    "N": {"distance_m": 15.0, "min_m": 9.0, "max_m": 24.0},
    "S": {"distance_m": 14.0, "min_m": 8.0, "max_m": 23.0},
    "E": {"distance_m": 10.0, "min_m": 6.0, "max_m": 18.0},
    "W": {"distance_m": 10.0, "min_m": 6.0, "max_m": 18.0},
}
LOGIC_LINE_DEPART_SPEED_MPS = 10.0
# Start shortly upstream of the observed entry line. Traffic before that line
# is outside the video evidence and must not create unobserved approach queues.
LOGIC_LINE_PRE_ROLL_DISTANCE_METERS = 12.0
LOGIC_LINE_OUTPUT_MARGIN_METERS = 1.0
LOGIC_LINE_DETECTOR_FILE = "xiasha1_sumo_logic_line_events.xml"
LOGIC_LINE_ADDITIONAL_FILE = "xiasha1_sumo_logic_lines.add.xml"

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


def _kinematic_travel_time(distance, initial_speed, max_speed, acceleration):
    """Free-flow time for a short approach segment with capped acceleration."""

    if distance <= 0:
        return 0.0
    accelerate_time = max(0.0, (max_speed - initial_speed) / acceleration)
    accelerate_distance = (
        initial_speed * accelerate_time
        + 0.5 * acceleration * accelerate_time * accelerate_time
    )
    if distance <= accelerate_distance:
        return (
            -initial_speed
            + math.sqrt(initial_speed * initial_speed + 2 * acceleration * distance)
        ) / acceleration
    return accelerate_time + (distance - accelerate_distance) / max_speed


def logic_line_pre_roll_seconds(edge_id):
    if edge_id not in CONTROLLED_ENTRY_EDGES:
        raise ValueError(f"Pre-roll requires an incoming edge: {edge_id}")
    return _kinematic_travel_time(
        LOGIC_LINE_PRE_ROLL_DISTANCE_METERS,
        LOGIC_LINE_DEPART_SPEED_MPS,
        float(VEHICLE_TYPE_ATTRIBUTES["maxSpeed"]),
        float(VEHICLE_TYPE_ATTRIBUTES["accel"]),
    )


MAX_LOGIC_LINE_PRE_ROLL_SECONDS = max(
    logic_line_pre_roll_seconds(edge_id)
    for edge_id in ("N2J", "S2J", "E2J", "W2J")
)


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


def read_departure_adjustments(path: Path | None) -> dict[str, float]:
    """Read auditable per-event departure adjustments produced by evidence.py."""

    if path is None:
        return {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"event_id", "departure_adjustment_seconds"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(
                "departure adjustment file requires event_id and "
                "departure_adjustment_seconds"
            )
        adjustments = {}
        for row in reader:
            event_id = row["event_id"].strip()
            if event_id in adjustments:
                raise ValueError(f"Duplicate departure adjustment: {event_id}")
            adjustments[event_id] = float(row["departure_adjustment_seconds"])
    return adjustments


def _sumo_tool_command(name):
    candidates = [Path(sys.executable).with_name(name)]
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            scripted = handle.read(2) == b"#!"
        return [sys.executable, str(candidate)] if scripted else [str(candidate)]
    raise FileNotFoundError(f"SUMO tool '{name}' was not found")


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
        # A source ID is an annotation, not an identity constraint.  Each row
        # is an independent observed vehicle event even when either source ID
        # is repeated.
        row["_source_row_number"] = index
        row["_event_id"] = f"event_{index:06d}"
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

    base_shift = max(0.0, -min((row["_entry_time"] for row in valid), default=0.0))
    # Keep one second of observable motion before the vehicle crosses the
    # video entry line.  This is a clock-origin pad, not a fitted traffic lag.
    # Reserve two complete cycles before the first observed event. This keeps
    # one-to-one departure corrections non-negative without changing phase,
    # because the added warm-up is an exact multiple of the 128 s cycle.
    shift = (
        base_shift
        + MAX_LOGIC_LINE_PRE_ROLL_SECONDS
        + ALIGNMENT_WARMUP_CYCLES * SIGNAL_CYCLE_SECONDS
    )
    for row in valid:
        row["_observed_entry"] = row["_entry_time"] + shift
        row["_depart"] = row["_observed_entry"] - logic_line_pre_roll_seconds(
            row["entry_edge"]
        )

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


def _road_arm(edge_id):
    """Return the cardinal arm encoded by an external edge ID."""

    if edge_id in {"N2J", "J2N"}:
        return "N"
    if edge_id in {"S2J", "J2S"}:
        return "S"
    if edge_id in {"E2J", "J2E"}:
        return "E"
    if edge_id in {"W2J", "J2W"}:
        return "W"
    raise ValueError(f"Unsupported logic-line edge: {edge_id}")


def logic_line_distance(edge_id):
    return float(LOGIC_LINE_MODEL[_road_arm(edge_id)]["distance_m"])


def _edge_lane_lengths(net_path: Path):
    root = ET.parse(net_path).getroot()
    return {
        edge.get("id"): [float(lane.get("length")) for lane in edge.findall("lane")]
        for edge in root.findall("edge")
        if edge.get("function") != "internal"
    }


def _departure_lane_choices(net_path: Path):
    root = ET.parse(net_path).getroot()
    choices = defaultdict(set)
    for connection in root.findall("connection"):
        key = (connection.get("from", ""), connection.get("to", ""))
        if key[0] in CONTROLLED_ENTRY_EDGES and key[1].startswith("J2"):
            choices[key].add(int(connection.get("fromLane", "0")))
    return {key: sorted(values) for key, values in choices.items()}


def _route_spatial_attributes(row, net_path: Path, ordinal: int):
    lengths = _edge_lane_lengths(net_path)
    lane_choices = _departure_lane_choices(net_path)
    entry_edge = row["entry_edge"]
    exit_edge = row["exit_edge"]
    allowed_lanes = lane_choices.get((entry_edge, exit_edge))
    if not allowed_lanes:
        raise ValueError(f"No departure lane for {entry_edge} -> {exit_edge}")
    depart_lane = allowed_lanes[ordinal % len(allowed_lanes)]
    entry_lengths = lengths.get(entry_edge, [])
    exit_lengths = lengths.get(exit_edge, [])
    if depart_lane >= len(entry_lengths) or not exit_lengths:
        raise ValueError(f"Missing lane geometry for {entry_edge} -> {exit_edge}")
    entry_line_pos = min(entry_lengths) - logic_line_distance(entry_edge)
    depart_pos = max(0.1, entry_line_pos - LOGIC_LINE_PRE_ROLL_DISTANCE_METERS)
    arrival_pos = min(
        min(exit_lengths) - 0.1,
        logic_line_distance(exit_edge) + LOGIC_LINE_OUTPUT_MARGIN_METERS,
    )
    return {
        "departLane": str(depart_lane),
        "departPos": _format_number(depart_pos),
        "departSpeed": _format_number(LOGIC_LINE_DEPART_SPEED_MPS),
        "arrivalPos": _format_number(arrival_pos),
    }


def write_routes(rows, path: Path, net_path: Path, flows=False, window=60.0):
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
            vehicle = ET.Element(
                "vehicle",
                {
                    "id": row["_event_id"],
                    "type": VEHICLE_TYPE_ATTRIBUTES["id"],
                    "route": route_id,
                    "depart": _format_number(row["_depart"]),
                    **_route_spatial_attributes(
                        row, net_path, int(row["_source_row_number"])
                    ),
                },
            )
            ET.SubElement(
                vehicle,
                "param",
                key="source.vehicle_id",
                value=row.get("vehicle_id", ""),
            )
            ET.SubElement(
                vehicle,
                "param",
                key="source.semantic_id",
                value=row.get("semantic_id", ""),
            )
            ET.SubElement(
                vehicle,
                "param",
                key="alignment.departure_adjustment_seconds",
                value=_format_number(row.get("_departure_adjustment", 0.0)),
            )
            vehicle_elements.append(vehicle)
            ET.SubElement(
                root,
                "route",
                id=route_id,
                edges=f"{row['entry_edge']} {row['exit_edge']}",
            )
        for vehicle in vehicle_elements:
            root.append(vehicle)
    _write_xml(ET.ElementTree(root), path)


def write_logic_line_detectors(net_path: Path, path: Path):
    """Write per-lane detectors at the four video counting cross-sections."""

    root = ET.parse(net_path).getroot()
    additional = ET.Element("additional")
    detector_file = path.parent / LOGIC_LINE_DETECTOR_FILE
    detector_count = 0
    for edge in root.findall("edge"):
        edge_id = edge.get("id", "")
        if edge_id not in {
            "N2J", "S2J", "E2J", "W2J", "J2N", "J2S", "J2E", "J2W"
        }:
            continue
        incoming = edge_id in CONTROLLED_ENTRY_EDGES
        role = "entry" if incoming else "exit"
        for lane in edge.findall("lane"):
            # ``--sidewalks.guess`` adds a pedestrian-only lane at index 0.
            # Logic-line detectors represent vehicle observations, so leave
            # that sidewalk lane out of the detector file.
            if lane.get("allow") == "pedestrian" and not lane.get("disallow"):
                continue
            length = float(lane.get("length"))
            pos = (
                length - logic_line_distance(edge_id)
                if incoming
                else logic_line_distance(edge_id)
            )
            ET.SubElement(
                additional,
                "instantInductionLoop",
                id=f"logic_{role}_{lane.get('id')}",
                lane=lane.get("id"),
                pos=_format_number(pos),
                # SUMO resolves detector output paths from the launch working
                # directory.  A basename keeps the generated config runnable
                # when launched directly from its own output directory.
                file=detector_file.name,
                friendlyPos="false",
            )
            detector_count += 1
    _write_xml(ET.ElementTree(additional), path)
    return {
        "detector_count": detector_count,
        "output_file": str(detector_file),
    }


def _phase_is_active(name, movement, approach_group):
    all_red = {"NS过渡红", "南北东西全红", "EW过渡红", "东西南北全红"}
    if name in all_red:
        return False
    if movement == "u_turn":
        return False
    # The video event table does not identify a protected right-turn phase.
    # Model right turns as permissive/yielding outside explicit all-red time;
    # the lowercase signal state is assigned by write_signal_network.
    if movement == "right":
        return True
    if approach_group == "NS" and not name.startswith("NS"):
        return False
    if approach_group == "EW" and not name.startswith("EW"):
        return False
    if "直行" in name:
        return movement in {"straight", "right"}
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


def _phase_start_seconds(phase_name):
    elapsed = 0.0
    for name, duration, _ in PHASES:
        if name == phase_name:
            return elapsed
        elapsed += duration
    raise ValueError(f"Unknown signal phase: {phase_name}")


def _signal_offset_for_anchor(anchor_arrival_seconds, phase_name=SIGNAL_ANCHOR_PHASE):
    phase_start = _phase_start_seconds(phase_name)
    # SUMO evaluates a static program at (simulation_time - offset) mod cycle.
    return (anchor_arrival_seconds - phase_start) % SIGNAL_CYCLE_SECONDS


def signal_phase_position(
    simulation_time,
    offset,
    cycle_seconds=SIGNAL_CYCLE_SECONDS,
    sample_offset=0.0,
):
    """Return SUMO's phase position for a timestamp.

    ``offset`` follows SUMO's static ``tlLogic`` convention: the signal
    program is evaluated at ``(simulation_time - offset) mod cycle``.  The
    optional ``sample_offset`` is only for interval-valued observations such
    as SUMO edge-exit times; it must not be folded into the configured offset.
    """

    if cycle_seconds <= 0:
        raise ValueError("signal cycle must be positive")
    return (float(simulation_time) - float(offset) + float(sample_offset)) % cycle_seconds


def signal_phase_at(
    simulation_time,
    offset,
    cycle_seconds=SIGNAL_CYCLE_SECONDS,
    sample_offset=0.0,
):
    """Return the named signal phase at a timestamp using SUMO semantics."""

    return _phase_name_at(
        signal_phase_position(
            simulation_time,
            offset,
            cycle_seconds=cycle_seconds,
            sample_offset=sample_offset,
        )
    )


def _phase_name_at(position):
    elapsed = 0.0
    for name, duration, _ in PHASES:
        if elapsed <= position < elapsed + duration:
            return name
        elapsed += duration
    return PHASES[-1][0]


def _movement_matches_phase(movement, approach_group, phase_name):
    if movement in {"right", "u_turn"}:
        return None
    if approach_group == "NS" and not phase_name.startswith("NS"):
        return False
    if approach_group == "EW" and not phase_name.startswith("EW"):
        return False
    if movement == "straight":
        return "直行" in phase_name
    if movement == "left":
        return "左转" in phase_name
    return False


def calibrate_signal_offset(
    rows, net_path: Path, shift, resolution=0.1, reference_offset=None
):
    """Fit phase offset using observed exit time as a disclosed proxy.

    The source does not contain stop-line crossing timestamps or phase labels.
    Right turns are uninformative and U-turns are unsupported, so only
    straight/left complete trajectories participate in the scan.
    """

    if resolution <= 0:
        raise ValueError("signal offset resolution must be positive")
    root = ET.parse(net_path).getroot()
    junctions, edges = _network_geometry(root)
    observations = []
    for row in rows:
        exit_time = row.get("exit_time", "").strip()
        if not exit_time:
            continue
        try:
            timestamp = float(exit_time) + shift
        except ValueError:
            continue
        movement = classify_movement(
            row["entry_edge"], row["exit_edge"], junctions, edges
        )
        approach_group = _approach_group(
            row["entry_edge"], junctions, edges
        )
        if movement in {"right", "u_turn"}:
            continue
        observations.append((timestamp, movement, approach_group))

    steps = int(round(SIGNAL_CYCLE_SECONDS / resolution))
    scan = []
    for index in range(steps):
        offset = round(index * resolution, 10)
        matched = 0
        for timestamp, movement, approach_group in observations:
            phase_name = signal_phase_at(timestamp, offset)
            matched += bool(
                _movement_matches_phase(movement, approach_group, phase_name)
            )
        scan.append(
            {
                "offset_seconds": offset,
                "matched_records": matched,
                "calibration_records": len(observations),
                "match_rate": matched / len(observations) if observations else 0.0,
            }
        )
    def circular_distance(offset):
        if reference_offset is None:
            return 0.0
        delta = abs(offset - reference_offset) % SIGNAL_CYCLE_SECONDS
        return min(delta, SIGNAL_CYCLE_SECONDS - delta)

    best = max(
        scan,
        key=lambda item: (
            item["matched_records"],
            -circular_distance(item["offset_seconds"]),
            -item["offset_seconds"],
        ),
    )
    return best, scan


def write_signal_offset_scan(path: Path, scan):
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(
            file_obj,
            fieldnames=(
                "offset_seconds",
                "matched_records",
                "calibration_records",
                "match_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(scan)


def rebuild_parallel_straight_outbound_lanes(net_path: Path, output_path: Path):
    """Rebuild a SUMO net so parallel straight lanes do not merge needlessly."""

    source_root = ET.parse(net_path).getroot()
    junctions, edges = _network_geometry(source_root)
    netconvert = _sumo_tool_command("netconvert")
    changes = []
    with tempfile.TemporaryDirectory(prefix="xiasha_netconvert_") as directory:
        directory = Path(directory)
        prefix = directory / "plain"
        export = subprocess.run(
            netconvert
            + [
                "--sumo-net-file",
                str(net_path),
                "--plain-output-prefix",
                str(prefix),
            ],
            text=True,
            capture_output=True,
            check=True,
        )
        connection_path = prefix.with_suffix(".con.xml")
        edge_path = prefix.with_suffix(".edg.xml")
        node_path = prefix.with_suffix(".nod.xml")
        connection_tree = ET.parse(connection_path)
        connection_root = connection_tree.getroot()
        edge_root = ET.parse(edge_path).getroot()
        lane_counts = {
            edge.get("id"): int(edge.get("numLanes", "1"))
            for edge in edge_root.findall("edge")
        }
        groups = defaultdict(list)
        for connection in connection_root.findall("connection"):
            from_edge = connection.get("from", "")
            to_edge = connection.get("to", "")
            if from_edge not in CONTROLLED_ENTRY_EDGES or to_edge not in edges:
                continue
            if classify_movement(from_edge, to_edge, junctions, edges) == "straight":
                groups[(from_edge, to_edge)].append(connection)

        for (from_edge, to_edge), connections in sorted(groups.items()):
            available = lane_counts.get(to_edge, 1)
            if len(connections) < 2 or available < len(connections):
                continue
            for target_lane, connection in enumerate(
                sorted(connections, key=lambda item: int(item.get("fromLane", "0")))
            ):
                old_lane = int(connection.get("toLane", "0"))
                if old_lane == target_lane:
                    continue
                connection.set("toLane", str(target_lane))
                changes.append(
                    {
                        "from": from_edge,
                        "to": to_edge,
                        "from_lane": int(connection.get("fromLane", "0")),
                        "old_to_lane": old_lane,
                        "new_to_lane": target_lane,
                    }
                )

        connection_tree.write(
            connection_path, encoding="utf-8", xml_declaration=True
        )
        rebuild = subprocess.run(
            netconvert
            + [
                "--node-files",
                str(node_path),
                "--edge-files",
                str(edge_path),
                "--connection-files",
                str(connection_path),
                "--output-file",
                str(output_path),
                "--no-turnarounds",
                "true",
            ],
            text=True,
            capture_output=True,
            check=True,
        )
    return {
        "enabled": True,
        "changes": changes,
        "change_count": len(changes),
        "netconvert_command": netconvert,
        "export_stderr": export.stderr.strip(),
        "rebuild_stderr": rebuild.stderr.strip(),
    }


def write_signal_network(net_path: Path, out_path: Path, add_path: Path, offset=0.0):
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

    # Parallel incoming lanes can merge into the same outgoing lane. SUMO
    # permits exactly one protected (upper-case G) link for such a merge; the
    # remaining links and all right turns must yield (lower-case g).
    protected_merge_keys = set()
    for item in controlled:
        connection = item["element"]
        key = (
            connection.get("from", ""),
            connection.get("to", ""),
            connection.get("toLane", ""),
        )
        if item["movement"] != "right" and key not in protected_merge_keys:
            item["protected"] = True
            protected_merge_keys.add(key)
        else:
            item["protected"] = False

    states = []
    for name, _, mark in PHASES:
        chars = []
        for item in controlled:
            connection = item["element"]
            movement = item["movement"]
            group = _approach_group(connection.get("from", ""), junctions, edges)
            active = _phase_is_active(name, movement, group)
            signal_mark = mark
            if mark == "G" and (movement == "right" or not item["protected"]):
                signal_mark = "g"
            chars.append(signal_mark if active else "r")
        states.append("".join(chars))

    # SUMO versions used by this repository require the effective tlLogic to
    # be present in the network. The additional file is retained as an audit
    # and portable signal definition, while generated cfg files use the net.
    for existing in list(root.findall("tlLogic")):
        if existing.get("id") == "J":
            root.remove(existing)
    offset_text = _format_number(offset)
    tl_logic = _build_tl_logic(states, offset=offset_text)
    children = list(root)
    first_connection = next(
        (index for index, child in enumerate(children) if child.tag == "connection"),
        len(children),
    )
    root.insert(first_connection, tl_logic)
    _write_xml(tree, out_path)

    additional = ET.Element("additional")
    additional.append(_build_tl_logic(states, offset=offset_text))
    _write_xml(ET.ElementTree(additional), add_path)
    return {
        "controlled": controlled,
        "candidate_connections": candidate_connections,
        "states": states,
    }


def write_cfg(path, net, routes, begin=0.0, end=900.0, additional=None):
    root = ET.Element("configuration")
    inputs = ET.SubElement(root, "input")
    ET.SubElement(inputs, "net-file", value=net.name)
    ET.SubElement(inputs, "route-files", value=routes.name)
    if additional is not None:
        additional_files = (
            [additional]
            if isinstance(additional, (str, Path))
            else list(additional)
        )
        ET.SubElement(
            inputs,
            "additional-files",
            value=",".join(Path(item).name for item in additional_files),
        )
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
    direct_connections = _direct_connections(root)
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


def _direct_connections(root: ET.Element):
    return {
        (connection.get("from"), connection.get("to"))
        for connection in root.findall("connection")
        if connection.get("from") in CONTROLLED_ENTRY_EDGES
        and connection.get("to", "").startswith("J2")
    }


def filter_topology_rows(rows, net_path: Path):
    """Keep only field-valid rows that can be represented by the network."""

    root = ET.parse(net_path).getroot()
    direct_connections = _direct_connections(root)
    routeable = []
    excluded = []
    for row in rows:
        pair = (row["entry_edge"], row["exit_edge"])
        if pair in direct_connections:
            routeable.append(row)
        else:
            excluded.append(row)
    return routeable, excluded


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
        "--signal-anchor-vehicle",
        default=DEFAULT_SIGNAL_ANCHOR_VEHICLE_ID,
        help="vehicle used to anchor the signal cycle",
    )
    parser.add_argument(
        "--signal-anchor-travel-time",
        type=float,
        default=DEFAULT_SIGNAL_ANCHOR_TRAVEL_SECONDS,
        help="seconds from the anchor vehicle's SUMO depart to the stop line",
    )
    parser.add_argument(
        "--signal-offset",
        type=float,
        default=None,
        help="override the computed static signal offset in seconds",
    )
    parser.add_argument(
        "--signal-offset-method",
        choices=["observed_exit_proxy", "anchor"],
        default="observed_exit_proxy",
        help="derive the phase offset from all observed exits or use one anchor vehicle",
    )
    parser.add_argument(
        "--departure-adjustments",
        type=Path,
        default=None,
        help=(
            "CSV with event_id and departure_adjustment_seconds from matched "
            "entry-line calibration"
        ),
    )
    parser.add_argument(
        "--expand-parallel-straight-output-lanes",
        action="store_true",
        help=(
            "rebuild the network so parallel straight connections use distinct "
            "available outbound lanes"
        ),
    )
    parser.add_argument(
        "--strict-topology",
        action="store_true",
        help="fail after writing artifacts if an observed OD lacks a network connection",
    )
    parser.add_argument(
        "--visual-additional",
        type=Path,
        default=None,
        help="optional SUMO-GUI polygons (crossings, medians, channelization)",
    )
    args = parser.parse_args(argv)
    mode = "all" if args.all else args.mode
    if args.window <= 0:
        raise ValueError("flow window must be positive")
    if args.visual_additional is not None and not args.visual_additional.is_file():
        raise FileNotFoundError(args.visual_additional)

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
    departure_adjustments = read_departure_adjustments(args.departure_adjustments)
    known_event_ids = {row["_event_id"] for row in valid}
    unknown_adjustments = sorted(set(departure_adjustments) - known_event_ids)
    if unknown_adjustments:
        raise ValueError(
            "Departure adjustments contain unknown event IDs: "
            + ", ".join(unknown_adjustments[:5])
        )
    for row in valid:
        row["_observed_entry"] = row["_entry_time"] + shift
        row["_departure_adjustment"] = departure_adjustments.get(
            row["_event_id"], 0.0
        )
        row["_depart"] = (
            row["_observed_entry"]
            - logic_line_pre_roll_seconds(row["entry_edge"])
            + row["_departure_adjustment"]
        )
        if row["_depart"] < 0:
            raise ValueError(
                f"Departure adjustment places {row['_event_id']} before time zero"
            )

    vehicles_path = output_dir / "xiasha1_sumo_vehicles.rou.xml"
    flows_path = output_dir / "xiasha1_sumo_flows.rou.xml"
    network_path = output_dir / "xiasha1_sumo_signal.net.xml"
    additional_path = output_dir / "xiasha1_sumo_signal.add.xml"
    logic_line_path = output_dir / LOGIC_LINE_ADDITIONAL_FILE
    report_path = output_dir / "xiasha1_sumo_conversion_report.json"
    connection_audit_path = output_dir / "xiasha1_sumo_connection_movement_audit.csv"
    phase_matrix_csv_path = output_dir / "xiasha1_sumo_phase_matrix.csv"
    phase_matrix_txt_path = output_dir / "xiasha1_sumo_phase_matrix.txt"
    signal_offset_scan_path = output_dir / "xiasha1_sumo_signal_offset_scan.csv"
    demand_audit_path = output_dir / "xiasha1_sumo_demand_audit.csv"
    skip_audit_path = output_dir / "xiasha1_sumo_skip_audit.csv"
    skip_summary_path = output_dir / "xiasha1_sumo_skip_summary.csv"
    topology_audit_path = output_dir / "xiasha1_sumo_route_topology_audit.csv"

    topology_audit = write_route_topology_audit(topology_audit_path, valid, args.network)
    routeable, topology_excluded = filter_topology_rows(valid, args.network)
    topology_audit.update(
        {
            "routeable_records": len(routeable),
            "excluded_from_sumo_routes": len(topology_excluded),
            "exclusion_policy": "drop_missing_connection_records",
        }
    )

    anchor_row = next(
        (row for row in routeable if row.get("vehicle_id") == args.signal_anchor_vehicle),
        None,
    )
    if anchor_row is None:
        raise ValueError(
            "Signal anchor vehicle is not present in routeable records: "
            f"{args.signal_anchor_vehicle}"
        )
    if args.signal_anchor_travel_time < 0:
        raise ValueError("signal anchor travel time must be non-negative")
    anchor_depart = anchor_row["_depart"]
    anchor_arrival = anchor_depart + args.signal_anchor_travel_time
    target_phase_start = _phase_start_seconds(SIGNAL_ANCHOR_PHASE)
    anchor_offset = _signal_offset_for_anchor(anchor_arrival)
    calibration_best, calibration_scan = calibrate_signal_offset(
        routeable, args.network, shift, reference_offset=anchor_offset
    )
    tied_best_offsets = [
        item["offset_seconds"]
        for item in calibration_scan
        if item["matched_records"] == calibration_best["matched_records"]
    ]
    write_signal_offset_scan(signal_offset_scan_path, calibration_scan)
    if args.signal_offset is not None:
        signal_offset = args.signal_offset % SIGNAL_CYCLE_SECONDS
        offset_source = "explicit_cli_override"
    elif args.signal_offset_method == "anchor":
        signal_offset = anchor_offset
        offset_source = "single_vehicle_anchor_convention"
    else:
        signal_offset = calibration_best["offset_seconds"]
        offset_source = "all_complete_exit_time_proxy_scan"

    if mode in {"all", "vehicles"}:
        write_routes(routeable, vehicles_path, args.network, window=args.window)
    if mode in {"all", "flows"}:
        write_routes(routeable, flows_path, args.network, flows=True, window=args.window)

    skip_bias = write_skip_audits(skip_audit_path, skip_summary_path, skipped, args.window)
    signal_audit = None
    lane_connection_rebuild = {"enabled": False, "changes": [], "change_count": 0}
    demand_audit = None
    simulation_end = None

    if mode == "all":
        signal_network_source = args.network
        if args.expand_parallel_straight_output_lanes:
            rebuilt_path = output_dir / "xiasha1_sumo_lane_aligned_base.net.xml"
            lane_connection_rebuild = rebuild_parallel_straight_outbound_lanes(
                args.network, rebuilt_path
            )
            signal_network_source = rebuilt_path
        signal_audit = write_signal_network(
            signal_network_source, network_path, additional_path, offset=signal_offset
        )
        logic_line_audit = write_logic_line_detectors(network_path, logic_line_path)
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
            math.ceil(
                max(row["_depart"] for row in routeable)
                + 3 * SIGNAL_CYCLE_SECONDS
            )
            if routeable
            else 900.0,
        )
        config_additional = [logic_line_path]
        if args.visual_additional is not None:
            config_additional.append(args.visual_additional)
        write_cfg(
            output_dir / "xiasha1_sumo_vehicles.sumocfg",
            network_path,
            vehicles_path,
            end=simulation_end,
            additional=config_additional,
        )
        write_cfg(
            output_dir / "xiasha1_sumo_flows.sumocfg",
            network_path,
            flows_path,
            end=simulation_end,
            additional=config_additional,
        )
    else:
        connection_audit = None
        logic_line_audit = None

    demand_audit = write_demand_audit(
        demand_audit_path,
        vehicles_path if vehicles_path.is_file() else None,
        flows_path if flows_path.is_file() else None,
        routeable,
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
        "routeable_records": len(routeable),
        "topology_excluded_records": len(topology_excluded),
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
        "logic_line_alignment": {
            "source": "image-derived coarse prior; refined and audited by event matching",
            "distance_reference": "metres from the junction-side end of each external edge",
            "positions": LOGIC_LINE_MODEL,
            "pre_roll_seconds_by_entry": {
                edge_id: logic_line_pre_roll_seconds(edge_id)
                for edge_id in ("N2J", "S2J", "E2J", "W2J")
            },
            "depart_position_rule": (
                "entry logic-line position minus upstream pre-roll distance"
            ),
            "pre_roll_distance_meters": LOGIC_LINE_PRE_ROLL_DISTANCE_METERS,
            "depart_speed_mps": LOGIC_LINE_DEPART_SPEED_MPS,
            "clock_warmup_cycles": ALIGNMENT_WARMUP_CYCLES,
            "departure_adjustment_file": (
                _report_path(args.departure_adjustments)
                if args.departure_adjustments is not None
                else None
            ),
            "departure_adjustment_records": len(departure_adjustments),
            "departure_adjustment_range_seconds": (
                [
                    min(departure_adjustments.values()),
                    max(departure_adjustments.values()),
                ]
                if departure_adjustments
                else [0.0, 0.0]
            ),
            "detectors": logic_line_audit,
            "source_id_policy": (
                "each CSV row is an independent SUMO vehicle; source IDs are trace metadata"
            ),
        },
        "signal_alignment": {
            "anchor_vehicle_id": args.signal_anchor_vehicle,
            "anchor_route": (
                f"{anchor_row['entry_edge']} -> {anchor_row['exit_edge']}"
            ),
            "anchor_depart_seconds": anchor_depart,
            "anchor_travel_time_to_stop_line_seconds": args.signal_anchor_travel_time,
            "anchor_arrival_seconds": anchor_arrival,
            "target_phase": SIGNAL_ANCHOR_PHASE,
            "target_phase_start_in_cycle_seconds": target_phase_start,
            "anchor_convention_offset_seconds": anchor_offset,
            "cycle_seconds": SIGNAL_CYCLE_SECONDS,
            "signal_offset_seconds": signal_offset,
            "effective_phase_position_at_anchor_seconds": (
                anchor_arrival - signal_offset
            ) % SIGNAL_CYCLE_SECONDS,
            "sumo_offset_semantics": (
                "phase_position=(simulation_time-offset) mod cycle"
            ),
            "offset_source": offset_source,
            "offset_method": args.signal_offset_method,
            "calibration_proxy": "complete trajectory exit_time + normalized shift",
            "calibration_records": calibration_best["calibration_records"],
            "calibration_matched_records": calibration_best["matched_records"],
            "calibration_match_rate": calibration_best["match_rate"],
            "calibration_tied_best_count": len(tied_best_offsets),
            "calibration_tied_best_min_seconds": min(tied_best_offsets),
            "calibration_tied_best_max_seconds": max(tied_best_offsets),
            "calibration_scan_file": signal_offset_scan_path.name,
            "calibration_limitation": (
                "The source has no stop-line timestamp or phase label; exit_time is "
                "a proxy and cannot establish video-phase ground truth."
            ),
        },
        "depart_time_range": {
            "min": min((row["_depart"] for row in routeable), default=None),
            "max": max((row["_depart"] for row in routeable), default=None),
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
        "signal_offset_scan_file": signal_offset_scan_path.name,
        "lane_connection_rebuild": lane_connection_rebuild,
        "connection_audit": connection_audit,
        "phase_durations_total": sum(duration for _, duration, _ in PHASES),
        "phase_count": len(PHASES),
        "simulation_begin_seconds": 0.0 if simulation_end is not None else None,
        "simulation_end_seconds": simulation_end,
        "exit_time_usage": (
            "target time for crossing the same direction-specific video exit line; "
            "never treated directly as a stop-line timestamp"
        ),
        "movement_classification": "network geometry vectors; straight/left/right/u_turn",
        "strict_topology_requested": args.strict_topology,
        "visual_additional_file": (
            _report_path(args.visual_additional)
            if args.visual_additional is not None
            else None
        ),
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
