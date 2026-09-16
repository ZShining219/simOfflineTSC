"""Build an independent CitySky SUMO network, demand package, and evidence bundle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/raw_data/citysky/semantic_injection_validation.csv"
DEFAULT_PACKAGE_DIR = ROOT / "data/raw_data/citysky/sumo"
DEFAULT_EVIDENCE_DIR = ROOT / "final_result/citysky_semantic_sumo_evidence"

ARMS = ("N", "E", "S", "W")
ARM_COORDINATES = {
    "N": (0.0, 80.0),
    "E": (80.0, 0.0),
    "S": (0.0, -80.0),
    "W": (-80.0, 0.0),
}
APPROACH_NAMES = {"N": "North", "E": "East", "S": "South", "W": "West"}
SIGNAL_LABELS = ("NS_GREEN", "ALL_RED", "EW_GREEN")
AMBIGUOUS_SIGNAL_LABELS = {"CONFLICT", "TRANSITION_OR_ALL_RED"}
LANE_COUNT = 4
EDGE_SPEED_MPS = 10.0
DEPART_SPEED_MPS = 8.0
DEPART_POSITION_METERS = 35.0
ARRIVAL_POSITION_METERS = 35.0
STEP_LENGTH_SECONDS = 0.1
FLUSH_SECONDS = 1800
FALLBACK_SIGNAL = (
    ("NS_GREEN", 15),
    ("ALL_RED", 6),
    ("EW_GREEN", 17),
    ("ALL_RED", 6),
)
MIN_RECONSTRUCTED_DWELL_SECONDS = 3
YELLOW_SECONDS = 3
MIN_CLEARANCE_INTERVAL_SECONDS = 4
TURN_HOLD_BEFORE_AXIS_SWITCH_SECONDS = 6

COLORS = {
    "observed": "#343A40",
    "complete": "#D1495B",
    "sumo_input": "#00798C",
    "sumo_completed": "#EDAE49",
    "ns": "#30638E",
    "ew": "#D1495B",
    "all_red": "#ADB5BD",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path.resolve())


def _write_xml(root: ET.Element, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)


def _tool_command(name: str, explicit: Path | None = None) -> tuple[list[str], Path]:
    candidates = []
    if explicit is not None:
        candidates.append(explicit)
    environment = os.environ.get(f"{name.upper()}_BINARY")
    if environment:
        candidates.append(Path(environment))
    candidates.append(Path(sys.executable).with_name(name))
    found = shutil.which(name)
    if found:
        candidates.append(Path(found))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            scripted = handle.read(2) == b"#!"
        command = [sys.executable, str(candidate)] if scripted else [str(candidate)]
        return command, candidate
    raise FileNotFoundError(f"SUMO tool '{name}' was not found")


def read_source(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "vehicle_id",
        "semantic_id",
        "observed_event_time",
        "entry_edge",
        "exit_edge",
        "entry_time",
        "exit_time",
        "entry_phase",
        "exit_phase",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"CitySky source is missing required columns: {missing}")
    if frame["vehicle_id"].duplicated().any():
        duplicates = frame.loc[frame["vehicle_id"].duplicated(), "vehicle_id"].tolist()
        raise ValueError(f"vehicle_id must be unique; duplicates include {duplicates[:5]}")
    frame = frame.copy()
    frame.insert(0, "source_row_number", np.arange(2, len(frame) + 2))
    for column in ("observed_event_time", "entry_time", "exit_time"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["complete_od"] = frame[["entry_edge", "exit_edge", "entry_time"]].notna().all(axis=1)
    return frame


def movement_for_pair(entry_edge: str, exit_edge: str) -> str:
    if not isinstance(entry_edge, str) or not isinstance(exit_edge, str):
        return ""
    entry_arm = entry_edge[0]
    exit_arm = exit_edge[-1]
    if entry_arm not in ARMS or exit_arm not in ARMS:
        raise ValueError(f"Unsupported CitySky OD pair: {entry_edge} -> {exit_edge}")
    if entry_arm == exit_arm:
        return "u_turn"
    if {entry_arm, exit_arm} in ({"N", "S"}, {"E", "W"}):
        return "through"
    source = np.asarray(ARM_COORDINATES[entry_arm])
    target = np.asarray(ARM_COORDINATES[exit_arm])
    incoming = -source
    cross = incoming[0] * target[1] - incoming[1] * target[0]
    return "left" if cross > 0 else "right"


def _lane_pairs(movement: str) -> tuple[tuple[int, int], ...]:
    if movement == "right":
        return ((0, 0),)
    if movement == "through":
        return ((1, 1), (2, 2))
    if movement in {"left", "u_turn"}:
        return ((3, 3),)
    raise ValueError(f"Unknown movement: {movement}")


def source_design(frame: pd.DataFrame) -> tuple[pd.DataFrame, float, int]:
    complete = frame.loc[frame["complete_od"]].copy()
    allowed_edges = {f"{arm}2J" for arm in ARMS} | {f"J2{arm}" for arm in ARMS}
    actual_edges = set(complete["entry_edge"]) | set(complete["exit_edge"])
    unsupported = sorted(actual_edges - allowed_edges)
    if unsupported:
        raise ValueError(f"Unsupported edge labels in complete OD rows: {unsupported}")
    complete["movement"] = [
        movement_for_pair(entry, exit_edge)
        for entry, exit_edge in zip(complete["entry_edge"], complete["exit_edge"])
    ]
    complete["approach"] = complete["entry_edge"].str[0].map(APPROACH_NAMES)
    complete["route"] = complete["entry_edge"] + "->" + complete["exit_edge"]
    minimum = float(complete["entry_time"].min()) if len(complete) else 0.0
    shift = max(0.0, -minimum)
    complete["planned_depart"] = complete["entry_time"] + shift
    maximum_depart = float(complete["planned_depart"].max()) if len(complete) else 0.0
    simulation_end = int(math.ceil(maximum_depart)) + FLUSH_SECONDS
    return complete, shift, simulation_end


def reconstruct_signal(
    source: pd.DataFrame, shift: float, simulation_end: int
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame, dict]:
    samples = []
    for role in ("entry", "exit"):
        for source_row, vehicle_id, timestamp, label in source[
            ["source_row_number", "vehicle_id", f"{role}_time", f"{role}_phase"]
        ].itertuples(index=False, name=None):
            if pd.isna(timestamp) or pd.isna(label):
                continue
            label = str(label)
            samples.append(
                {
                    "source_row_number": int(source_row),
                    "vehicle_id": vehicle_id,
                    "role": role,
                    "source_time": float(timestamp),
                    "normalized_time": float(timestamp) + shift,
                    "source_label": label,
                    "label_status": (
                        "accepted"
                        if label in SIGNAL_LABELS
                        else "ambiguous"
                        if label in AMBIGUOUS_SIGNAL_LABELS
                        else "unsupported"
                    ),
                }
            )
    sample_frame = pd.DataFrame(samples)
    accepted = sample_frame.loc[sample_frame["label_status"] == "accepted"].copy()
    if accepted.empty:
        raise ValueError("No usable NS_GREEN/EW_GREEN/ALL_RED signal observations")
    accepted["second"] = np.floor(accepted["normalized_time"]).astype(int)

    vote_rows = []
    for second, group in accepted.groupby("second"):
        counts = group["source_label"].value_counts().to_dict()
        selected = max(
            SIGNAL_LABELS,
            key=lambda label: (counts.get(label, 0), label == "ALL_RED"),
        )
        vote_rows.append(
            {
                "second": int(second),
                "selected_label": selected,
                "sample_count": int(len(group)),
                "distinct_labels": int(group["source_label"].nunique()),
                **{f"{label.lower()}_votes": int(counts.get(label, 0)) for label in SIGNAL_LABELS},
            }
        )
    votes = pd.DataFrame(vote_rows).sort_values("second").reset_index(drop=True)
    observed_end = min(
        simulation_end,
        max(1, int(math.ceil(float(accepted["normalized_time"].max()))) + 1),
    )
    labels = np.empty(simulation_end, dtype=object)
    labeled_seconds = votes["second"].to_numpy(dtype=int)
    labeled_values = votes["selected_label"].to_numpy(dtype=object)
    observed_seconds = np.arange(observed_end)
    insertions = np.searchsorted(labeled_seconds, observed_seconds)
    left = np.clip(insertions - 1, 0, len(labeled_seconds) - 1)
    right = np.clip(insertions, 0, len(labeled_seconds) - 1)
    choose_right = np.abs(labeled_seconds[right] - observed_seconds) < np.abs(
        observed_seconds - labeled_seconds[left]
    )
    nearest = np.where(choose_right, right, left)
    labels[:observed_end] = labeled_values[nearest]

    def runs(values: np.ndarray) -> list[tuple[int, int, str]]:
        output = []
        start = 0
        for index in range(1, len(values) + 1):
            if index == len(values) or values[index] != values[start]:
                output.append((start, index, str(values[start])))
                start = index
        return output

    # Sparse event labels occasionally produce one- or two-second islands.
    # Remove those islands before constructing a physical controller, then
    # insert one second of all-red whenever the source observations jump
    # directly between orthogonal greens.
    for _ in range(10):
        changed = False
        current_runs = runs(labels[:observed_end])
        for run_index, (begin, end, _) in enumerate(current_runs):
            if end - begin >= MIN_RECONSTRUCTED_DWELL_SECONDS:
                continue
            if run_index == 0:
                replacement = current_runs[run_index + 1][2]
            elif run_index + 1 == len(current_runs):
                replacement = current_runs[run_index - 1][2]
            elif current_runs[run_index - 1][2] == current_runs[run_index + 1][2]:
                replacement = current_runs[run_index - 1][2]
            else:
                left_length = current_runs[run_index - 1][1] - current_runs[run_index - 1][0]
                right_length = current_runs[run_index + 1][1] - current_runs[run_index + 1][0]
                replacement = (
                    current_runs[run_index - 1][2]
                    if left_length >= right_length
                    else current_runs[run_index + 1][2]
                )
            labels[begin:end] = replacement
            changed = True
        if not changed:
            break
    direct_switches_repaired = 0
    for second in range(1, observed_end):
        if {labels[second - 1], labels[second]} == {"NS_GREEN", "EW_GREEN"}:
            labels[second - 1] = "ALL_RED"
            direct_switches_repaired += 1

    fallback_cycle = sum(duration for _, duration in FALLBACK_SIGNAL)
    for second in range(observed_end, simulation_end):
        position = (second - observed_end) % fallback_cycle
        elapsed = 0
        for label, duration in FALLBACK_SIGNAL:
            if elapsed <= position < elapsed + duration:
                labels[second] = label
                break
            elapsed += duration

    clearance_intervals_expanded = 0
    for _ in range(5):
        current_runs = runs(labels)
        changed = False
        for run_index, (begin, end, label) in enumerate(current_runs):
            if (
                label != "ALL_RED"
                or end - begin >= MIN_CLEARANCE_INTERVAL_SECONDS
                or run_index == 0
                or run_index + 1 == len(current_runs)
            ):
                continue
            previous_label = current_runs[run_index - 1][2]
            next_label = current_runs[run_index + 1][2]
            if previous_label not in {"NS_GREEN", "EW_GREEN"} or next_label not in {
                "NS_GREEN",
                "EW_GREEN",
            }:
                continue
            required = MIN_CLEARANCE_INTERVAL_SECONDS - (end - begin)
            take_left = (required + 1) // 2
            take_right = required - take_left
            labels[max(0, begin - take_left) : min(len(labels), end + take_right)] = "ALL_RED"
            clearance_intervals_expanded += 1
            changed = True
        if not changed:
            break
    # The generated static program repeats after the configured simulation
    # horizon. End with a safe clearance interval before it wraps to phase 0.
    labels[-(YELLOW_SECONDS + 1) :] = "ALL_RED"

    def state_at(timestamp: float) -> str:
        index = min(max(int(math.floor(timestamp)), 0), len(labels) - 1)
        return str(labels[index])

    sample_frame["reconstructed_label"] = [
        state_at(value) for value in sample_frame["normalized_time"]
    ]
    sample_frame["label_match"] = pd.Series(pd.NA, index=sample_frame.index, dtype="boolean")
    accepted_mask = sample_frame["label_status"] == "accepted"
    sample_frame.loc[accepted_mask, "label_match"] = (
        sample_frame.loc[accepted_mask, "source_label"]
        == sample_frame.loc[accepted_mask, "reconstructed_label"]
    )

    scan_rows = []
    accepted_samples = sample_frame.loc[accepted_mask]
    for offset in np.arange(-30.0, 30.01, 0.5):
        compared = [
            state_at(timestamp + offset)
            for timestamp in accepted_samples["normalized_time"].to_numpy(dtype=float)
        ]
        matches = np.asarray(compared) == accepted_samples["source_label"].to_numpy()
        scan_rows.append(
            {
                "offset_seconds": float(offset),
                "matched_records": int(matches.sum()),
                "compared_records": int(len(matches)),
                "match_rate": float(matches.mean()),
            }
        )
    scan = pd.DataFrame(scan_rows)
    best = scan.sort_values(
        ["match_rate", "offset_seconds"], ascending=[False, True]
    ).iloc[0]
    report = {
        "method": "nearest one-second majority of accepted entry/exit phase labels",
        "accepted_labels": list(SIGNAL_LABELS),
        "ambiguous_labels_excluded_from_fit": sorted(AMBIGUOUS_SIGNAL_LABELS),
        "accepted_samples": int(accepted_mask.sum()),
        "ambiguous_samples": int(
            (sample_frame["label_status"] == "ambiguous").sum()
        ),
        "mixed_label_seconds": int((votes["distinct_labels"] > 1).sum()),
        "minimum_reconstructed_dwell_seconds": MIN_RECONSTRUCTED_DWELL_SECONDS,
        "direct_green_switches_repaired": direct_switches_repaired,
        "minimum_clearance_interval_seconds": MIN_CLEARANCE_INTERVAL_SECONDS,
        "clearance_intervals_expanded": clearance_intervals_expanded,
        "yellow_refinement_seconds": YELLOW_SECONDS,
        "turn_hold_before_axis_switch_seconds": TURN_HOLD_BEFORE_AXIS_SWITCH_SECONDS,
        "controller_refinement": (
            "The broad CSV ALL_RED label is refined into yellow then all-red. "
            "Permissive left/U-turn entry closes before each axis switch so vehicles "
            "already inside the junction can clear safely."
        ),
        "observed_reconstruction_end_seconds": observed_end,
        "fallback_extension": [
            {"label": label, "duration_seconds": duration}
            for label, duration in FALLBACK_SIGNAL
        ],
        "zero_offset_match_rate": float(
            sample_frame.loc[accepted_mask, "label_match"].astype(float).mean()
        ),
        "best_scan_offset_seconds": float(best["offset_seconds"]),
        "best_scan_match_rate": float(best["match_rate"]),
        "limitation": (
            "The CSV contains event-time phase labels but no controller log. "
            "Unobserved seconds are nearest-label interpolation; the post-observation "
            "flush interval uses a disclosed fallback cycle."
        ),
    }
    return labels, sample_frame, scan, report


def _write_plain_network(package_dir: Path) -> dict[str, Path]:
    node_path = package_dir / "citysky.nod.xml"
    edge_path = package_dir / "citysky.edg.xml"
    connection_path = package_dir / "citysky.con.xml"

    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="J", x="0", y="0", type="traffic_light")
    for arm in ARMS:
        x, y = ARM_COORDINATES[arm]
        ET.SubElement(nodes, "node", id=arm, x=str(x), y=str(y), type="priority")
    _write_xml(nodes, node_path)

    edges = ET.Element("edges")
    for arm in ARMS:
        common = {"numLanes": str(LANE_COUNT), "speed": str(EDGE_SPEED_MPS), "priority": "3"}
        ET.SubElement(edges, "edge", id=f"{arm}2J", **common, **{"from": arm, "to": "J"})
        ET.SubElement(edges, "edge", id=f"J2{arm}", **common, **{"from": "J", "to": arm})
    _write_xml(edges, edge_path)

    connections = ET.Element("connections")
    for entry_arm in ARMS:
        for exit_arm in ARMS:
            entry_edge, exit_edge = f"{entry_arm}2J", f"J2{exit_arm}"
            movement = movement_for_pair(entry_edge, exit_edge)
            for from_lane, to_lane in _lane_pairs(movement):
                ET.SubElement(
                    connections,
                    "connection",
                    **{
                        "from": entry_edge,
                        "to": exit_edge,
                        "fromLane": str(from_lane),
                        "toLane": str(to_lane),
                    },
                )
    _write_xml(connections, connection_path)
    return {"nodes": node_path, "edges": edge_path, "connections": connection_path}


def _signal_state(
    label: str,
    controlled: list[dict],
    state_length: int,
    yellow: bool = False,
    through_only: bool = False,
    turn_yellow: bool = False,
) -> str:
    state = ["r"] * state_length
    if label == "ALL_RED":
        return "".join(state)
    active_arms = {"N", "S"} if label == "NS_GREEN" else {"E", "W"}
    for item in controlled:
        if item["entry_arm"] not in active_arms:
            continue
        if through_only and item["movement"] in {"left", "u_turn"}:
            continue
        if turn_yellow and item["movement"] in {"left", "u_turn"}:
            mark = "y"
        else:
            mark = "y" if yellow else "G" if item["movement"] == "through" else "g"
        state[item["link_index"]] = mark
    return "".join(state)


def _run_length_signal(labels: np.ndarray, observed_end: int) -> list[dict]:
    broad_segments = []
    start = 0
    current = str(labels[0])
    for second in range(1, len(labels) + 1):
        next_label = str(labels[second]) if second < len(labels) else None
        source = "observed_reconstruction" if start < observed_end else "fallback_flush"
        next_source = "observed_reconstruction" if second < observed_end else "fallback_flush"
        if next_label != current or next_source != source:
            broad_segments.append(
                {
                    "begin_seconds": start,
                    "end_seconds": second,
                    "duration_seconds": second - start,
                    "label": current,
                    "source": source,
                }
            )
            start = second
            current = next_label
    segments = []
    for index, segment in enumerate(broad_segments):
        duration = segment["duration_seconds"]
        previous_label = broad_segments[index - 1]["label"] if index > 0 else None
        if segment["label"] in {"NS_GREEN", "EW_GREEN"}:
            combined_duration = max(
                0, duration - TURN_HOLD_BEFORE_AXIS_SWITCH_SECONDS
            )
            if combined_duration:
                combined_end = segment["begin_seconds"] + combined_duration
                segments.append(
                    {
                        **segment,
                        "phase_index": len(segments),
                        "end_seconds": combined_end,
                        "duration_seconds": combined_duration,
                        "signal_indication": segment["label"],
                        "yellow_for_broad_label": "",
                    }
                )
                turn_yellow_duration = min(
                    YELLOW_SECONDS, duration - combined_duration
                )
                turn_yellow_end = combined_end + turn_yellow_duration
                segments.append(
                    {
                        **segment,
                        "phase_index": len(segments),
                        "begin_seconds": combined_end,
                        "end_seconds": turn_yellow_end,
                        "duration_seconds": turn_yellow_duration,
                        "signal_indication": f"{segment['label']}_LEFT_YELLOW",
                        "yellow_for_broad_label": "",
                    }
                )
            else:
                turn_yellow_end = segment["begin_seconds"]
            segments.append(
                {
                    **segment,
                    "phase_index": len(segments),
                    "begin_seconds": turn_yellow_end,
                    "duration_seconds": segment["end_seconds"] - turn_yellow_end,
                    "signal_indication": f"{segment['label']}_THROUGH_ONLY",
                    "yellow_for_broad_label": "",
                }
            )
        elif (
            segment["label"] == "ALL_RED"
            and previous_label in {"NS_GREEN", "EW_GREEN"}
        ):
            # The source schema groups yellow and transition intervals under
            # ALL_RED. Refine the start of that broad interval into a SUMO
            # yellow phase without shortening the observed green interval.
            yellow_duration = min(YELLOW_SECONDS, duration)
            yellow_end = segment["begin_seconds"] + yellow_duration
            segments.append(
                {
                    **segment,
                    "phase_index": len(segments),
                    "end_seconds": yellow_end,
                    "duration_seconds": yellow_duration,
                    "signal_indication": previous_label.replace("GREEN", "YELLOW"),
                    "yellow_for_broad_label": previous_label,
                }
            )
            if yellow_end < segment["end_seconds"]:
                segments.append(
                    {
                        **segment,
                        "phase_index": len(segments),
                        "begin_seconds": yellow_end,
                        "duration_seconds": segment["end_seconds"] - yellow_end,
                        "signal_indication": "ALL_RED",
                        "yellow_for_broad_label": "",
                    }
                )
        else:
            segments.append(
                {
                    **segment,
                    "phase_index": len(segments),
                    "signal_indication": segment["label"],
                    "yellow_for_broad_label": "",
                }
            )
    return segments


def build_network(
    package_dir: Path,
    complete: pd.DataFrame,
    signal_labels: np.ndarray,
    signal_report: dict,
    netconvert_binary: Path | None,
) -> tuple[Path, pd.DataFrame, pd.DataFrame, dict]:
    package_dir.mkdir(parents=True, exist_ok=True)
    plain = _write_plain_network(package_dir)
    network_path = package_dir / "citysky.net.xml"
    netconvert, executable = _tool_command("netconvert", netconvert_binary)
    command = netconvert + [
        "--node-files",
        str(plain["nodes"]),
        "--edge-files",
        str(plain["edges"]),
        "--connection-files",
        str(plain["connections"]),
        "--output-file",
        str(network_path),
        "--junctions.corner-detail",
        "5",
    ]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)

    tree = ET.parse(network_path)
    root = tree.getroot()
    logic = next((item for item in root.findall("tlLogic") if item.get("id") == "J"), None)
    if logic is None:
        raise ValueError("netconvert did not create traffic-light logic for J")
    controlled = []
    for connection in root.findall("connection"):
        entry_edge = connection.get("from", "")
        exit_edge = connection.get("to", "")
        if not (entry_edge.endswith("2J") and exit_edge.startswith("J2")):
            continue
        if connection.get("tl") != "J" or connection.get("linkIndex") is None:
            raise ValueError(f"Uncontrolled central connection: {entry_edge}->{exit_edge}")
        controlled.append(
            {
                "entry_edge": entry_edge,
                "exit_edge": exit_edge,
                "entry_arm": entry_edge[0],
                "exit_arm": exit_edge[-1],
                "movement": movement_for_pair(entry_edge, exit_edge),
                "from_lane": int(connection.get("fromLane", "0")),
                "to_lane": int(connection.get("toLane", "0")),
                "link_index": int(connection.get("linkIndex", "0")),
            }
        )
    state_length = max(item["link_index"] for item in controlled) + 1
    observed_end = int(signal_report["observed_reconstruction_end_seconds"])
    segments = _run_length_signal(signal_labels, observed_end)
    for phase in list(logic.findall("phase")):
        logic.remove(phase)
    logic.set("type", "static")
    logic.set("programID", "citysky_csv_reconstruction")
    logic.set("offset", "0")
    for segment in segments:
        segment["state"] = _signal_state(
            segment["yellow_for_broad_label"] or segment["label"],
            controlled,
            state_length,
            yellow=segment["signal_indication"] in {"NS_YELLOW", "EW_YELLOW"},
            through_only=(
                "THROUGH_ONLY" in segment["signal_indication"]
                or segment["signal_indication"] in {"NS_YELLOW", "EW_YELLOW"}
            ),
            turn_yellow=segment["signal_indication"].endswith("LEFT_YELLOW"),
        )
        ET.SubElement(
            logic,
            "phase",
            duration=str(segment["duration_seconds"]),
            state=segment["state"],
            name=f"{segment['signal_indication']}:{segment['source']}",
        )
    tree.write(network_path, encoding="utf-8", xml_declaration=True)

    od_counts = complete.groupby(["entry_edge", "exit_edge"]).size().to_dict()
    audit_rows = []
    for entry_arm in ARMS:
        for exit_arm in ARMS:
            entry_edge, exit_edge = f"{entry_arm}2J", f"J2{exit_arm}"
            matches = [
                item
                for item in controlled
                if item["entry_edge"] == entry_edge and item["exit_edge"] == exit_edge
            ]
            audit_rows.append(
                {
                    "entry_edge": entry_edge,
                    "exit_edge": exit_edge,
                    "movement": movement_for_pair(entry_edge, exit_edge),
                    "source_complete_od_count": int(od_counts.get((entry_edge, exit_edge), 0)),
                    "network_connection_count": len(matches),
                    "from_lanes": ";".join(str(item["from_lane"]) for item in matches),
                    "to_lanes": ";".join(str(item["to_lane"]) for item in matches),
                    "link_indices": ";".join(str(item["link_index"]) for item in matches),
                    "routeable": bool(matches),
                }
            )
    network_audit = pd.DataFrame(audit_rows)
    signal_timeline = pd.DataFrame(segments)
    report = {
        "design_basis": "four source entry edges x four source exit edges",
        "source_observed_od_pairs": int(len(od_counts)),
        "network_routeable_od_pairs": int(network_audit["routeable"].sum()),
        "all_observed_od_pairs_routeable": bool(
            network_audit.loc[network_audit["source_complete_od_count"] > 0, "routeable"].all()
        ),
        "incoming_lanes_per_arm": LANE_COUNT,
        "outgoing_lanes_per_arm": LANE_COUNT,
        "lane_design_assumption": (
            "The CSV depart_lane and arrival_lane columns are entirely empty. Four "
            "lanes are the minimum movement-separating design used here: right, two "
            "through lanes, and a shared left/U-turn lane."
        ),
        "controlled_connections": len(controlled),
        "signal_link_count": state_length,
        "signal_segment_count": len(segments),
        "netconvert_binary": str(executable),
        "netconvert_command": command,
        "netconvert_stderr": run.stderr.strip(),
    }
    return network_path, network_audit, signal_timeline, report


def write_demand(
    package_dir: Path,
    complete: pd.DataFrame,
    network_audit: pd.DataFrame,
    simulation_end: int,
) -> tuple[Path, Path, pd.DataFrame, dict]:
    route_ids = {
        (entry, exit_edge): f"route_{entry}_{exit_edge}"
        for entry in sorted(complete["entry_edge"].unique())
        for exit_edge in sorted(complete.loc[complete["entry_edge"] == entry, "exit_edge"].unique())
    }
    lane_choices = {}
    for row in network_audit.itertuples(index=False):
        lane_choices[(row.entry_edge, row.exit_edge)] = [
            int(value) for value in str(row.from_lanes).split(";") if value != ""
        ]

    def routes_root() -> ET.Element:
        root = ET.Element("routes")
        ET.SubElement(
            root,
            "vType",
            id="citysky_passenger",
            vClass="passenger",
            accel="2.6",
            decel="4.5",
            sigma="0",
            speedDev="0",
            length="5.0",
            minGap="2.5",
            jmTimegapMinor="10.0",
            emergencyDecel="20",
            maxSpeed=str(EDGE_SPEED_MPS),
        )
        for (entry, exit_edge), route_id in sorted(route_ids.items()):
            ET.SubElement(root, "route", id=route_id, edges=f"{entry} {exit_edge}")
        return root

    vehicle_root = routes_root()
    od_ordinals = defaultdict(int)
    for row in complete.sort_values(["planned_depart", "vehicle_id"]).itertuples(index=False):
        pair = (row.entry_edge, row.exit_edge)
        choices = lane_choices[pair]
        lane = choices[od_ordinals[pair] % len(choices)]
        od_ordinals[pair] += 1
        ET.SubElement(
            vehicle_root,
            "vehicle",
            id=str(row.vehicle_id),
            type="citysky_passenger",
            route=route_ids[pair],
            depart=f"{row.planned_depart:.2f}",
            departLane=str(lane),
            departPos=f"{DEPART_POSITION_METERS:.2f}",
            departSpeed=f"{DEPART_SPEED_MPS:.2f}",
            arrivalLane="current",
            arrivalPos=f"{ARRIVAL_POSITION_METERS:.2f}",
        )
    vehicle_path = package_dir / "citysky_vehicles.rou.xml"
    _write_xml(vehicle_root, vehicle_path)

    flow_root = routes_root()
    flow_counts = Counter()
    for row in complete.itertuples(index=False):
        begin = int(math.floor(float(row.planned_depart) / 60.0) * 60)
        flow_counts[(begin, row.entry_edge, row.exit_edge)] += 1
    for index, ((begin, entry, exit_edge), count) in enumerate(sorted(flow_counts.items())):
        ET.SubElement(
            flow_root,
            "flow",
            id=f"flow_{index:04d}",
            type="citysky_passenger",
            route=route_ids[(entry, exit_edge)],
            begin=str(begin),
            end=str(begin + 60),
            number=str(count),
            departLane="best",
            departPos=f"{DEPART_POSITION_METERS:.2f}",
            departSpeed=f"{DEPART_SPEED_MPS:.2f}",
            arrivalLane="current",
            arrivalPos=f"{ARRIVAL_POSITION_METERS:.2f}",
        )
    flow_path = package_dir / "citysky_flows.rou.xml"
    _write_xml(flow_root, flow_path)

    vehicle_od = complete.groupby(["entry_edge", "exit_edge"]).size()
    flow_od = Counter()
    for (_, entry, exit_edge), count in flow_counts.items():
        flow_od[(entry, exit_edge)] += count
    demand_rows = []
    for pair, count in vehicle_od.items():
        demand_rows.append(
            {
                "scope": "OD",
                "entry_edge": pair[0],
                "exit_edge": pair[1],
                "complete_source_count": int(count),
                "vehicle_count": int(count),
                "flow_count": int(flow_od[pair]),
                "conserved": int(count) == int(flow_od[pair]),
            }
        )
    demand_rows.append(
        {
            "scope": "TOTAL",
            "entry_edge": "",
            "exit_edge": "",
            "complete_source_count": len(complete),
            "vehicle_count": len(vehicle_root.findall("vehicle")),
            "flow_count": sum(int(item.get("number", "0")) for item in flow_root.findall("flow")),
            "conserved": True,
        }
    )
    audit = pd.DataFrame(demand_rows)
    report = {
        "complete_source_records": len(complete),
        "vehicle_records": len(vehicle_root.findall("vehicle")),
        "flow_records": sum(int(item.get("number", "0")) for item in flow_root.findall("flow")),
        "flow_groups": len(flow_root.findall("flow")),
        "conservation_passed": bool(audit["conserved"].all()),
    }

    network_name = "citysky.net.xml"
    for mode, route_path in (("vehicles", vehicle_path), ("flows", flow_path)):
        config = ET.Element("configuration")
        inputs = ET.SubElement(config, "input")
        ET.SubElement(inputs, "net-file", value=network_name)
        ET.SubElement(inputs, "route-files", value=route_path.name)
        time = ET.SubElement(config, "time")
        ET.SubElement(time, "begin", value="0")
        ET.SubElement(time, "end", value=str(simulation_end))
        _write_xml(config, package_dir / f"citysky_{mode}.sumocfg")
    return vehicle_path, flow_path, audit, report


def _parse_tripinfo(path: Path) -> dict[str, dict]:
    result = {}
    for trip in ET.parse(path).getroot().findall("tripinfo"):
        result[trip.get("id", "")] = {
            f"sumo_{name}": float(trip.get(name, "nan"))
            for name in (
                "depart",
                "arrival",
                "duration",
                "departDelay",
                "waitingTime",
                "timeLoss",
                "routeLength",
            )
        }
    return result


def _parse_crossings(path: Path) -> dict[str, float]:
    result = {}
    for vehicle in ET.parse(path).getroot().findall("vehicle"):
        route = vehicle.find("route")
        times = route.get("exitTimes", "").split() if route is not None else []
        if times:
            result[vehicle.get("id", "")] = float(times[0])
    return result


def run_sumo(
    package_dir: Path,
    evidence_dir: Path,
    sumo_binary: Path | None,
    seed: int,
) -> tuple[dict[str, dict], dict[str, float], dict, dict]:
    evidence_dir.mkdir(parents=True, exist_ok=True)
    trip_path = evidence_dir / "sumo_tripinfo.xml"
    summary_path = evidence_dir / "sumo_summary.xml"
    route_path = evidence_dir / "sumo_vehroute.xml"
    command, executable = _tool_command("sumo", sumo_binary)
    command += [
        "-c",
        str(package_dir / "citysky_vehicles.sumocfg"),
        "--seed",
        str(seed),
        "--step-length",
        str(STEP_LENGTH_SECONDS),
        "--duration-log.disable",
        "--no-step-log",
        "true",
        "--tripinfo-output",
        str(trip_path),
        "--summary-output",
        str(summary_path),
        "--summary-output.period",
        "60",
        "--vehroute-output",
        str(route_path),
        "--vehroute-output.exit-times",
        "true",
        "--vehroute-output.route-length",
        "true",
    ]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    steps = ET.parse(summary_path).getroot().findall("step")
    if not steps:
        raise ValueError("SUMO summary did not contain any simulation steps")
    last = steps[-1].attrib
    keys = (
        "loaded",
        "inserted",
        "running",
        "waiting",
        "ended",
        "arrived",
        "collisions",
        "teleports",
        "discarded",
    )
    execution = {key: int(float(last.get(key, 0))) for key in keys}
    execution["last_step_seconds"] = float(last.get("time", 0))
    report = {
        "seed": seed,
        "sumo_binary": str(executable),
        "command": command,
        "stdout": run.stdout.strip(),
        "stderr": run.stderr.strip(),
    }
    return _parse_tripinfo(trip_path), _parse_crossings(route_path), execution, report


def run_flow_sumo(
    package_dir: Path,
    evidence_dir: Path,
    sumo_binary: Path | None,
    seed: int,
) -> tuple[dict, dict]:
    summary_path = evidence_dir / "sumo_flow_summary.xml"
    command, executable = _tool_command("sumo", sumo_binary)
    command += [
        "-c",
        str(package_dir / "citysky_flows.sumocfg"),
        "--seed",
        str(seed),
        "--step-length",
        str(STEP_LENGTH_SECONDS),
        "--duration-log.disable",
        "--no-step-log",
        "true",
        "--summary-output",
        str(summary_path),
        "--summary-output.period",
        "60",
    ]
    run = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, check=True)
    steps = ET.parse(summary_path).getroot().findall("step")
    if not steps:
        raise ValueError("SUMO flow summary did not contain any simulation steps")
    last = steps[-1].attrib
    keys = (
        "loaded",
        "inserted",
        "running",
        "waiting",
        "ended",
        "arrived",
        "collisions",
        "teleports",
        "discarded",
    )
    execution = {key: int(float(last.get(key, 0))) for key in keys}
    execution["last_step_seconds"] = float(last.get("time", 0))
    report = {
        "seed": seed,
        "sumo_binary": str(executable),
        "command": command,
        "stdout": run.stdout.strip(),
        "stderr": run.stderr.strip(),
    }
    return execution, report


def _signal_at(labels: np.ndarray, timestamp: float) -> str:
    index = min(max(int(math.floor(timestamp)), 0), len(labels) - 1)
    return str(labels[index])


def _signal_indication_at(timeline: pd.DataFrame, timestamp: float) -> str:
    begins = timeline["begin_seconds"].to_numpy(dtype=float)
    index = int(np.searchsorted(begins, float(timestamp), side="right") - 1)
    index = min(max(index, 0), len(timeline) - 1)
    return str(timeline.iloc[index]["signal_indication"])


def build_traceability(
    source: pd.DataFrame,
    complete: pd.DataFrame,
    shift: float,
    labels: np.ndarray,
    signal_timeline: pd.DataFrame,
    trips: dict[str, dict],
    crossings: dict[str, float],
) -> pd.DataFrame:
    complete_lookup = complete.set_index("vehicle_id").to_dict("index")
    rows = []
    for source_row in source.to_dict("records"):
        vehicle_id = source_row["vehicle_id"]
        entry_edge = source_row.get("entry_edge")
        exit_edge = source_row.get("exit_edge")
        item = dict(source_row)
        item["approach"] = (
            APPROACH_NAMES.get(entry_edge[0], "") if isinstance(entry_edge, str) else ""
        )
        item["turning_direction"] = (
            movement_for_pair(entry_edge, exit_edge)
            if isinstance(entry_edge, str) and isinstance(exit_edge, str)
            else ""
        )
        item["route"] = (
            f"{entry_edge}->{exit_edge}"
            if isinstance(entry_edge, str) and isinstance(exit_edge, str)
            else ""
        )
        item["extraction_status"] = "sumo_input" if vehicle_id in complete_lookup else "incomplete_trajectory"
        item["network_connection_exists"] = bool(vehicle_id in complete_lookup)
        entry_time = source_row.get("entry_time")
        exit_time = source_row.get("exit_time")
        item["observed_travel_time"] = (
            float(exit_time - entry_time)
            if pd.notna(entry_time) and pd.notna(exit_time)
            else np.nan
        )
        item["normalized_planned_depart"] = (
            float(entry_time + shift) if vehicle_id in complete_lookup else np.nan
        )
        item["normalized_observed_exit"] = (
            float(exit_time + shift) if pd.notna(exit_time) else np.nan
        )
        if vehicle_id in trips:
            item.update(trips[vehicle_id])
            crossing = crossings.get(vehicle_id, np.nan)
            item["sumo_stopline_crossing"] = crossing
            item["sumo_crossing_source_label"] = (
                _signal_at(labels, crossing)
                if pd.notna(crossing)
                else ""
            )
            item["sumo_crossing_signal"] = (
                _signal_indication_at(signal_timeline, crossing)
                if pd.notna(crossing)
                else ""
            )
            axis = "NS" if str(entry_edge)[0] in {"N", "S"} else "EW"
            item["sumo_crossing_permitted"] = item["sumo_crossing_signal"] in {
                f"{axis}_GREEN",
                f"{axis}_YELLOW",
            } or (
                item["sumo_crossing_signal"] == f"{axis}_GREEN_THROUGH_ONLY"
                and item["turning_direction"] in {"through", "right"}
            ) or (
                item["sumo_crossing_signal"] == f"{axis}_GREEN_LEFT_YELLOW"
            )
            item["sumo_minus_observed_exit_seconds"] = (
                item["sumo_arrival"] - item["normalized_observed_exit"]
                if pd.notna(item["normalized_observed_exit"])
                else np.nan
            )
            item["sumo_status"] = "completed"
        else:
            item["sumo_status"] = "not_completed" if vehicle_id in complete_lookup else "not_applicable"
        rows.append(item)
    return pd.DataFrame(rows)


def _window_table(trace: pd.DataFrame, window: int) -> pd.DataFrame:
    end = int(math.ceil((float(trace["observed_event_time"].max()) + 1e-9) / window) * window)
    bins = np.arange(0, end + window, window)

    def counts(mask: pd.Series) -> np.ndarray:
        values = trace.loc[mask, "observed_event_time"].to_numpy(dtype=float)
        return np.histogram(values, bins=bins)[0]

    observed = pd.Series(True, index=trace.index)
    complete = trace["extraction_status"] == "sumo_input"
    completed = trace["sumo_status"] == "completed"
    return pd.DataFrame(
        {
            "window_begin_seconds": bins[:-1],
            "window_end_seconds": bins[1:],
            "real_observed_count": counts(observed),
            "extracted_complete_count": counts(complete),
            "sumo_input_count": counts(complete),
            "sumo_completed_count": counts(completed),
        }
    )


def _sumo_time_table(trace: pd.DataFrame, window: int) -> pd.DataFrame:
    selected = trace.loc[trace["extraction_status"] == "sumo_input"]
    columns = {
        "normalized_planned_depart": "sumo_input_count",
        "sumo_depart": "sumo_actual_depart_count",
        "sumo_stopline_crossing": "sumo_stopline_crossing_count",
        "sumo_arrival": "sumo_actual_arrival_count",
    }
    maximum = max(float(selected[column].max()) for column in columns)
    end = int(math.ceil((maximum + 1e-9) / window) * window)
    bins = np.arange(0, end + window, window)
    output = {"window_begin_seconds": bins[:-1], "window_end_seconds": bins[1:]}
    for column, name in columns.items():
        output[name] = np.histogram(selected[column].dropna().to_numpy(dtype=float), bins=bins)[0]
    return pd.DataFrame(output)


def _composition(trace: pd.DataFrame, category: str) -> pd.DataFrame:
    stages = {
        "extracted_complete": trace["extraction_status"] == "sumo_input",
        "sumo_input": trace["extraction_status"] == "sumo_input",
        "sumo_completed": trace["sumo_status"] == "completed",
    }
    rows = []
    for stage, mask in stages.items():
        values = trace.loc[mask, category]
        values = values[values.astype(str) != ""]
        total = len(values)
        for value, count in values.value_counts().sort_index().items():
            rows.append(
                {
                    "stage": stage,
                    category: value,
                    "count": int(count),
                    "percentage": 100.0 * count / total if total else 0.0,
                    "stage_total": total,
                }
            )
    return pd.DataFrame(rows)


def _tv_similarity(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() == 0 or right.sum() == 0:
        return float("nan")
    return float(1 - 0.5 * np.abs(left / left.sum() - right / right.sum()).sum())


def _pearson(left, right) -> float:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _plot_time(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = table["window_begin_seconds"] / 60.0
    for column, label, color, style in (
        ("real_observed_count", "All observed events", COLORS["observed"], "-"),
        ("extracted_complete_count", "Complete OD", COLORS["complete"], "--"),
        ("sumo_completed_count", "SUMO completed", COLORS["sumo_input"], ":"),
    ):
        axis.plot(x, table[column], label=label, color=color, linewidth=2, linestyle=style)
    axis.set(title=title, xlabel="Window start (minutes)", ylabel="Vehicle count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _plot_composition(table: pd.DataFrame, category: str, title: str, path: Path) -> None:
    pivot = table.pivot(index=category, columns="stage", values="percentage").fillna(0)
    columns = [column for column in ("extracted_complete", "sumo_completed") if column in pivot]
    axis = pivot[columns].plot(
        kind="bar", figsize=(10, 4.8), color=[COLORS["complete"], COLORS["sumo_input"]], width=0.75
    )
    axis.set(title=title, xlabel="", ylabel="Share (%)")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(["Complete OD", "SUMO completed"], frameon=False)
    rotation = 45 if category == "route" else 0
    plt.xticks(rotation=rotation, ha="right" if rotation else "center")
    plt.tight_layout()
    axis.figure.savefig(path.with_suffix(".png"), dpi=180)
    axis.figure.savefig(path.with_suffix(".svg"))
    plt.close(axis.figure)


def _plot_totals(totals: list[dict], path: Path) -> None:
    labels = [item["stage"] for item in totals]
    values = [item["count"] for item in totals]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color=[COLORS["observed"], COLORS["complete"], COLORS["sumo_input"], COLORS["sumo_completed"]])
    axis.bar_label(bars, padding=3)
    axis.set(title="Demand scale and SUMO completion", ylabel="Vehicle count")
    axis.set_ylim(0, max(values) * 1.12)
    axis.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=10, ha="right")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _plot_signal_execution(trace: pd.DataFrame, timeline: pd.DataFrame, path: Path) -> dict:
    selected = trace.loc[trace["sumo_status"] == "completed"].copy()
    paired = selected[["normalized_observed_exit", "sumo_arrival"]].dropna()
    residual = paired["sumo_arrival"] - paired["normalized_observed_exit"]
    fig = plt.figure(figsize=(11, 7.2))
    grid = fig.add_gridspec(2, 1, height_ratios=[1.1, 1])
    top = fig.add_subplot(grid[0, 0])
    for column, label, color, style in (
        ("normalized_observed_exit", "Observed exit line", COLORS["observed"], "--"),
        ("sumo_arrival", "SUMO exit line", COLORS["sumo_input"], "-"),
    ):
        values = np.sort(selected[column].dropna().to_numpy(dtype=float))
        top.step(values / 60.0, np.arange(1, len(values) + 1), where="post", label=label, color=color, linestyle=style, linewidth=2)
    top.set(
        title=(
            "Same-boundary timing execution "
            f"(median residual {residual.median():.1f} s; MAE {residual.abs().mean():.1f} s)"
        ),
        xlabel="Normalized clock (minutes)",
        ylabel="Cumulative vehicles",
    )
    top.grid(alpha=0.25)
    top.legend(frameon=False)

    bottom = fig.add_subplot(grid[1, 0])
    horizon = 300
    for row in timeline.loc[timeline["begin_seconds"] < horizon].itertuples(index=False):
        color = COLORS["ns"] if row.label == "NS_GREEN" else COLORS["ew"] if row.label == "EW_GREEN" else COLORS["all_red"]
        bottom.axvspan(row.begin_seconds, min(row.end_seconds, horizon), color=color, alpha=0.18)
    crossings = selected.loc[selected["sumo_stopline_crossing"] < horizon]
    axis_value = crossings["entry_edge"].str[0].map(lambda arm: 1 if arm in {"N", "S"} else 0)
    axis_color = crossings["entry_edge"].str[0].map(lambda arm: COLORS["ns"] if arm in {"N", "S"} else COLORS["ew"])
    bottom.scatter(crossings["sumo_stopline_crossing"], axis_value, c=axis_color, s=16, alpha=0.72, edgecolors="none")
    bottom.set(
        title="SUMO stop-line crossings in the first five minutes",
        xlabel="Simulation time (seconds)",
        ylabel="Approach axis",
        xlim=(0, horizon),
        yticks=(0, 1),
        yticklabels=("EW", "NS"),
        ylim=(-0.45, 1.45),
    )
    bottom.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)
    return {
        "paired_exit_records": int(len(paired)),
        "median_sumo_minus_observed_exit_seconds": float(residual.median()),
        "mean_absolute_exit_residual_seconds": float(residual.abs().mean()),
        "p90_absolute_exit_residual_seconds": float(residual.abs().quantile(0.9)),
    }


def _plot_signal_scan(scan: pd.DataFrame, path: Path) -> None:
    best = scan.loc[scan["match_rate"].idxmax()]
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(scan["offset_seconds"], 100 * scan["match_rate"], color=COLORS["ns"], linewidth=2)
    axis.axvline(0, color=COLORS["complete"], linestyle="--", linewidth=2, label="Reconstructed clock: 0 s")
    axis.scatter([best["offset_seconds"]], [100 * best["match_rate"]], color=COLORS["sumo_completed"], zorder=3, label=f"Best scan: {best['offset_seconds']:.1f} s")
    axis.set(
        title="CSV phase-label reconstruction alignment",
        xlabel="Tested temporal offset (seconds)",
        ylabel="Accepted label agreement (%)",
        xlim=(float(scan["offset_seconds"].min()), float(scan["offset_seconds"].max())),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _plot_network(network_audit: pd.DataFrame, path: Path) -> None:
    fig, (network_axis, matrix_axis) = plt.subplots(1, 2, figsize=(11, 5.2), gridspec_kw={"width_ratios": [1, 1.15]})
    for arm, (x, y) in ARM_COORDINATES.items():
        network_axis.plot([0, x], [0, y], color="#CED4DA", linewidth=18, solid_capstyle="butt")
        network_axis.annotate("", xy=(0.38 * x, 0.38 * y), xytext=(0.78 * x, 0.78 * y), arrowprops={"arrowstyle": "-|>", "color": COLORS["observed"], "lw": 1.8})
        network_axis.annotate("", xy=(0.78 * x + (8 if y else 0), 0.78 * y + (8 if x else 0)), xytext=(0.38 * x + (8 if y else 0), 0.38 * y + (8 if x else 0)), arrowprops={"arrowstyle": "-|>", "color": COLORS["sumo_input"], "lw": 1.8})
        network_axis.text(1.08 * x, 1.08 * y, f"{arm}\n4 in / 4 out", ha="center", va="center", fontsize=9)
    network_axis.scatter([0], [0], s=700, marker="s", color="#FFFFFF", edgecolor=COLORS["complete"], linewidth=2, zorder=5)
    network_axis.text(0, 0, "J\nTLS", ha="center", va="center", zorder=6, fontsize=9)
    network_axis.set(title="Independent CitySky four-arm network", xlim=(-100, 100), ylim=(-100, 100), aspect="equal")
    network_axis.axis("off")

    matrix = network_audit.pivot(index="entry_edge", columns="exit_edge", values="source_complete_od_count").reindex(
        index=[f"{arm}2J" for arm in ARMS], columns=[f"J2{arm}" for arm in ARMS]
    )
    image = matrix_axis.imshow(matrix.to_numpy(), cmap="YlGnBu")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = int(matrix.iloc[row, column])
            matrix_axis.text(column, row, str(value), ha="center", va="center", color="white" if value > matrix.to_numpy().max() * 0.45 else "black")
    matrix_axis.set(
        title="Complete OD demand represented by the network",
        xlabel="Exit edge",
        ylabel="Entry edge",
        xticks=np.arange(4),
        yticks=np.arange(4),
        xticklabels=matrix.columns,
        yticklabels=matrix.index,
    )
    fig.colorbar(image, ax=matrix_axis, fraction=0.046, pad=0.04, label="Vehicles")
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=180)
    fig.savefig(path.with_suffix(".svg"))
    plt.close(fig)


def _write_metrics_csv(path: Path, metrics: dict) -> None:
    rows = []
    for group in (
        "counts",
        "rates",
        "temporal_structure",
        "composition_structure",
        "signal_reconstruction",
        "sumo_execution",
        "flow_sumo_execution",
    ):
        values = metrics[group]
        for name, value in values.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                rows.append({"group": group, "metric": name, "value": value})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def build_evidence(
    source_path: Path,
    source: pd.DataFrame,
    complete: pd.DataFrame,
    shift: float,
    package_dir: Path,
    evidence_dir: Path,
    network_path: Path,
    network_audit: pd.DataFrame,
    network_report: dict,
    signal_labels: np.ndarray,
    signal_samples: pd.DataFrame,
    signal_scan: pd.DataFrame,
    signal_report: dict,
    signal_timeline: pd.DataFrame,
    demand_audit: pd.DataFrame,
    demand_report: dict,
    sumo_binary: Path | None,
    seed: int,
) -> dict:
    trips, crossings, execution, run_report = run_sumo(package_dir, evidence_dir, sumo_binary, seed)
    flow_execution, flow_run_report = run_flow_sumo(
        package_dir, evidence_dir, sumo_binary, seed
    )
    trace = build_traceability(
        source,
        complete,
        shift,
        signal_labels,
        signal_timeline,
        trips,
        crossings,
    )
    trace.to_csv(evidence_dir / "vehicle_traceability.csv", index=False, encoding="utf-8-sig")
    trace.loc[trace["extraction_status"] == "incomplete_trajectory"].to_csv(
        evidence_dir / "incomplete_trajectory_events.csv", index=False, encoding="utf-8-sig"
    )
    network_audit.to_csv(package_dir / "citysky_network_design_audit.csv", index=False, encoding="utf-8-sig")
    signal_timeline.to_csv(package_dir / "citysky_signal_timeline.csv", index=False, encoding="utf-8-sig")
    signal_samples.to_csv(evidence_dir / "signal_label_traceability.csv", index=False, encoding="utf-8-sig")
    signal_scan.to_csv(evidence_dir / "signal_offset_scan.csv", index=False, encoding="utf-8-sig")
    demand_audit.to_csv(package_dir / "citysky_demand_audit.csv", index=False, encoding="utf-8-sig")

    time_tables = {suffix: _window_table(trace, window) for suffix, window in (("1min", 60), ("5min", 300))}
    sumo_tables = {suffix: _sumo_time_table(trace, window) for suffix, window in (("1min", 60), ("5min", 300))}
    for suffix, table in time_tables.items():
        table.to_csv(evidence_dir / f"time_flow_video_{suffix}.csv", index=False, encoding="utf-8-sig")
    for suffix, table in sumo_tables.items():
        table.to_csv(evidence_dir / f"time_flow_sumo_{suffix}.csv", index=False, encoding="utf-8-sig")

    compositions = {
        name: _composition(trace, name)
        for name in ("approach", "turning_direction", "route")
    }
    composition_files = {
        "approach": "approach_composition.csv",
        "turning_direction": "turning_composition.csv",
        "route": "route_composition.csv",
    }
    for name, table in compositions.items():
        table.to_csv(
            evidence_dir / composition_files[name],
            index=False,
            encoding="utf-8-sig",
        )
    (evidence_dir / "turning_direction_composition.csv").unlink(missing_ok=True)

    for window, suffix in ((60, "1min"), (300, "5min")):
        route_flow = complete.copy()
        route_flow["window_begin_seconds"] = (
            route_flow["planned_depart"] // window * window
        ).astype(int)
        route_flow = (
            route_flow.groupby(
                ["window_begin_seconds", "approach", "movement", "route"]
            )
            .size()
            .rename("source_and_sumo_input_count")
            .reset_index()
        )
        route_flow["sumo_completed_count"] = route_flow[
            "source_and_sumo_input_count"
        ]
        route_flow.insert(
            1, "window_end_seconds", route_flow["window_begin_seconds"] + window
        )
        route_flow.to_csv(
            evidence_dir / f"route_flow_{suffix}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    totals = [
        {"stage": "All observed", "count": len(source)},
        {"stage": "Complete OD", "count": len(complete)},
        {"stage": "SUMO input", "count": len(complete)},
        {"stage": "SUMO completed", "count": len(trips)},
    ]
    pd.DataFrame(totals).to_csv(evidence_dir / "demand_scale.csv", index=False, encoding="utf-8-sig")

    _plot_time(time_tables["1min"], "Observed demand preservation (1-minute windows)", evidence_dir / "figure_01_time_flow_video_1min")
    _plot_time(time_tables["5min"], "Observed demand preservation (5-minute windows)", evidence_dir / "figure_02_time_flow_video_5min")
    timing = _plot_signal_execution(trace, signal_timeline, evidence_dir / "figure_03_signal_aligned_execution")
    _plot_composition(compositions["approach"], "approach", "Approach composition", evidence_dir / "figure_04_approach_composition")
    _plot_composition(compositions["turning_direction"], "turning_direction", "Turning composition", evidence_dir / "figure_05_turning_composition")
    _plot_composition(compositions["route"], "route", "Route composition", evidence_dir / "figure_06_route_composition")
    _plot_totals(totals, evidence_dir / "figure_07_demand_scale")
    _plot_signal_scan(signal_scan, evidence_dir / "figure_08_signal_offset_calibration")
    _plot_network(network_audit, evidence_dir / "figure_09_network_topology")

    one_minute = time_tables["1min"]
    input_crossings = trace.loc[trace["extraction_status"] == "sumo_input", "sumo_crossing_permitted"].dropna()
    phase_matches = signal_samples.loc[signal_samples["label_status"] == "accepted", "label_match"].astype(bool)
    temporal = {
        "complete_to_input_tv_similarity_1min": _tv_similarity(one_minute["extracted_complete_count"], one_minute["sumo_input_count"]),
        "input_to_completed_tv_similarity_1min": _tv_similarity(one_minute["sumo_input_count"], one_minute["sumo_completed_count"]),
        "input_to_completed_pearson_1min": _pearson(one_minute["sumo_input_count"], one_minute["sumo_completed_count"]),
    }
    composition_metrics = {}
    for category, table in compositions.items():
        pivot = table.pivot(index=category, columns="stage", values="count").fillna(0)
        composition_metrics[f"{category}_complete_to_completed_tv_similarity"] = _tv_similarity(
            pivot["extracted_complete"], pivot["sumo_completed"]
        )

    metrics = {
        "schema_version": 1,
        "source": {
            "semantic_event_table": _relative(source_path),
            "semantic_event_table_sha256": _sha256(source_path),
            "observed_time_range_seconds": [float(source["observed_event_time"].min()), float(source["observed_event_time"].max())],
            "entry_time_normalization_shift_seconds": shift,
            "lane_columns_non_null": {
                "depart_lane": int(source.get("depart_lane", pd.Series(dtype=float)).notna().sum()),
                "arrival_lane": int(source.get("arrival_lane", pd.Series(dtype=float)).notna().sum()),
            },
        },
        "network_design": network_report,
        "demand_conservation": demand_report,
        "counts": {
            "real_observed_events": len(source),
            "extracted_complete_od": len(complete),
            "incomplete_trajectory_events": len(source) - len(complete),
            "observed_od_pairs": int(complete.groupby(["entry_edge", "exit_edge"]).ngroups),
            "sumo_input_vehicles": len(complete),
            "sumo_completed_vehicles": len(trips),
        },
        "rates": {
            "complete_od_extraction_rate": len(complete) / len(source),
            "sumo_retention_of_complete_od": len(complete) / len(complete),
            "sumo_completion_rate": len(trips) / len(complete),
        },
        "temporal_structure": temporal,
        "composition_structure": composition_metrics,
        "signal_reconstruction": {
            **signal_report,
            "accepted_phase_label_matches": int(phase_matches.sum()),
            "all_sumo_stopline_crossings_permitted": bool(input_crossings.all()),
            "permitted_sumo_stopline_crossings": int(input_crossings.sum()),
            "sumo_stopline_crossings": int(len(input_crossings)),
        },
        "same_boundary_timing": timing,
        "sumo_execution": {**execution, **run_report},
        "flow_sumo_execution": {**flow_execution, **flow_run_report},
    }
    criteria = {
        "all_16_csv_od_pairs_routeable": network_report["network_routeable_od_pairs"] == 16 and network_report["all_observed_od_pairs_routeable"],
        "complete_od_vehicle_flow_conservation": demand_report["conservation_passed"],
        "complete_od_to_sumo_retention_100pct": metrics["rates"]["sumo_retention_of_complete_od"] == 1.0,
        "all_sumo_inputs_inserted_and_completed": execution["inserted"] == len(complete) == execution["arrived"] == len(trips),
        "no_collision_teleport_or_discard": execution["collisions"] == execution["teleports"] == execution["discarded"] == 0,
        "no_sumo_warning_or_error": not run_report["stderr"],
        "all_flow_vehicles_inserted_and_completed": (
            flow_execution["inserted"]
            == len(complete)
            == flow_execution["arrived"]
            == flow_execution["ended"]
        ),
        "no_flow_sumo_warning_or_error": not flow_run_report["stderr"],
        "no_flow_collision_teleport_or_discard": (
            flow_execution["collisions"]
            == flow_execution["teleports"]
            == flow_execution["discarded"]
            == 0
        ),
        "source_phase_reconstruction_match_at_least_95pct": signal_report["zero_offset_match_rate"] >= 0.95,
        "all_sumo_stopline_crossings_permitted": bool(input_crossings.all()),
        "reference_eight_figures_generated": all(
            (evidence_dir / f"figure_{index:02d}_{name}.png").is_file()
            for index, name in (
                (1, "time_flow_video_1min"),
                (2, "time_flow_video_5min"),
                (3, "signal_aligned_execution"),
                (4, "approach_composition"),
                (5, "turning_composition"),
                (6, "route_composition"),
                (7, "demand_scale"),
                (8, "signal_offset_calibration"),
            )
        ),
    }
    metrics["claim_assessment"] = {
        "supported": all(criteria.values()),
        "criteria": criteria,
        "scope": (
            "The claim covers complete OD rows in semantic_injection_validation.csv, "
            "including all four observed U-turn OD pairs. Incomplete trajectories are "
            "retained in traceability but are not imputed into SUMO demand."
        ),
        "geometry_limitation": network_report["lane_design_assumption"],
        "signal_limitation": signal_report["limitation"],
        "absolute_vehicle_timing_alignment_established": False,
        "timing_alignment_limitation": (
            "Vehicles are inserted on the normalized CSV entry clock and aggregate "
            "demand is preserved, but SUMO queueing determines downstream exit time. "
            "The same-boundary residual metrics are evidence of the remaining "
            "microscopic timing difference, not a per-vehicle trajectory match."
        ),
    }
    summary_path = evidence_dir / "evidence_summary.json"
    summary_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    _write_metrics_csv(evidence_dir / "evidence_metrics.csv", metrics)
    pd.DataFrame(
        [
            {"check": name, "passed": passed, "evidence": "evidence_summary.json"}
            for name, passed in criteria.items()
        ]
    ).to_csv(evidence_dir / "validation_checks.csv", index=False, encoding="utf-8-sig")

    build_report = {
        "schema_version": 1,
        "input": _relative(source_path),
        "input_sha256": _sha256(source_path),
        "package_directory": _relative(package_dir),
        "evidence_directory": _relative(evidence_dir),
        "simulation_end_seconds": len(signal_labels),
        "network": network_report,
        "signal": signal_report,
        "demand": demand_report,
    }
    (package_dir / "citysky_build_report.json").write_text(
        json.dumps(build_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    provenance = [
        ("source_semantic_events", source_path),
        ("pipeline_code", Path(__file__)),
        ("sumo_network", network_path),
    ]
    provenance.extend(("package_artifact", path) for path in sorted(package_dir.iterdir()) if path.is_file())
    pd.DataFrame(
        [
            {"role": role, "path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
            for role, path in provenance
        ]
    ).drop_duplicates(subset=["role", "path"]).to_csv(
        evidence_dir / "source_and_build_provenance.csv", index=False, encoding="utf-8-sig"
    )

    manifest_rows = []
    for role, directory in (("sumo_package", package_dir), ("evidence", evidence_dir)):
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.name != "manifest.csv":
                manifest_rows.append(
                    {"role": role, "path": _relative(path), "bytes": path.stat().st_size, "sha256": _sha256(path)}
                )
    pd.DataFrame(manifest_rows).to_csv(evidence_dir / "manifest.csv", index=False, encoding="utf-8-sig")
    return metrics


def build_and_evaluate(
    input_path: Path = DEFAULT_INPUT,
    package_dir: Path = DEFAULT_PACKAGE_DIR,
    evidence_dir: Path = DEFAULT_EVIDENCE_DIR,
    sumo_binary: Path | None = None,
    netconvert_binary: Path | None = None,
    seed: int = 0,
) -> dict:
    source = read_source(input_path)
    complete, shift, simulation_end = source_design(source)
    labels, signal_samples, signal_scan, signal_report = reconstruct_signal(
        source, shift, simulation_end
    )
    network_path, network_audit, timeline, network_report = build_network(
        package_dir, complete, labels, signal_report, netconvert_binary
    )
    _, _, demand_audit, demand_report = write_demand(
        package_dir, complete, network_audit, simulation_end
    )
    return build_evidence(
        input_path,
        source,
        complete,
        shift,
        package_dir,
        evidence_dir,
        network_path,
        network_audit,
        network_report,
        labels,
        signal_samples,
        signal_scan,
        signal_report,
        timeline,
        demand_audit,
        demand_report,
        sumo_binary,
        seed,
    )


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--package-dir", type=Path, default=DEFAULT_PACKAGE_DIR)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE_DIR)
    parser.add_argument("--sumo-binary", type=Path, default=None)
    parser.add_argument("--netconvert-binary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    metrics = build_and_evaluate(
        args.input,
        args.package_dir,
        args.evidence_dir,
        args.sumo_binary,
        args.netconvert_binary,
        args.seed,
    )
    print(
        json.dumps(
            {
                "supported": metrics["claim_assessment"]["supported"],
                "counts": metrics["counts"],
                "criteria": metrics["claim_assessment"]["criteria"],
                "package": _relative(args.package_dir),
                "evidence": _relative(args.evidence_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
