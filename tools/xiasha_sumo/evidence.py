"""Build an auditable semantic-event to SUMO evidence bundle."""

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
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import converter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "final_result/xiasha_semantic_sumo_evidence"
DEFAULT_INPUT = ROOT / "tools/语义事件表(13min).csv"
APPROACHES = {"N2J": "North", "S2J": "South", "E2J": "East", "W2J": "West"}
TURN_LABELS = {
    "straight": "through",
    "left": "left",
    "right": "right",
    "u_turn": "u_turn",
}
COLORS = {
    "real_observed": "#343A40",
    "extracted_complete": "#D1495B",
    "sumo_input": "#00798C",
    "sumo_actual_depart": "#EDAE49",
    "sumo_actual_arrival": "#30638E",
}


def _float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _sumo_command(explicit: Path | None = None):
    candidates = []
    if explicit:
        candidates.append(explicit)
    if os.environ.get("SUMO_BINARY"):
        candidates.append(Path(os.environ["SUMO_BINARY"]))
    found = shutil.which("sumo")
    if found:
        candidates.append(Path(found))
    candidates.append(Path(sys.executable).with_name("sumo"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        with candidate.open("rb") as handle:
            scripted = handle.read(2) == b"#!"
        return ([sys.executable, str(candidate)] if scripted else [str(candidate)]), candidate
    raise FileNotFoundError("SUMO executable was not found; set SUMO_BINARY")


def _read_source(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row_number, row in enumerate(rows, start=2):
        row["_source_row_number"] = row_number
        row["_event_id"] = f"event_{row_number:06d}"
    return rows


def _network_metadata(path: Path):
    root = ET.parse(path).getroot()
    junctions, edges = converter._network_geometry(root)
    direct = converter._direct_connections(root)
    return junctions, edges, direct


def _route_input(path: Path):
    root = ET.parse(path).getroot()
    routes = {
        route.get("id"): tuple(route.get("edges", "").split())
        for route in root.findall("route")
    }
    result = {}
    for vehicle in root.findall("vehicle"):
        route = routes[vehicle.get("route")]
        params = {
            item.get("key"): item.get("value") for item in vehicle.findall("param")
        }
        result[vehicle.get("id")] = {
            "sumo_vehicle_id": vehicle.get("id"),
            "sumo_input_depart": float(vehicle.get("depart")),
            "entry_edge": route[0],
            "exit_edge": route[-1],
            "sumo_depart_lane": vehicle.get("departLane", ""),
            "sumo_depart_pos": float(vehicle.get("departPos", "nan")),
            "sumo_depart_speed": float(vehicle.get("departSpeed", "nan")),
            "sumo_arrival_pos": float(vehicle.get("arrivalPos", "nan")),
            "departure_adjustment_seconds": float(
                params.get("alignment.departure_adjustment_seconds", "0")
            ),
        }
    return result


def _tripinfo(path: Path):
    fields = (
        "depart",
        "arrival",
        "duration",
        "departDelay",
        "waitingTime",
        "timeLoss",
        "routeLength",
    )
    result = {}
    for trip in ET.parse(path).getroot().findall("tripinfo"):
        result[trip.get("id")] = {
            f"sumo_{field}": float(trip.get(field)) for field in fields
        }
    return result


def _vehroute(path: Path):
    result = {}
    for vehicle in ET.parse(path).getroot().findall("vehicle"):
        route = vehicle.find("route")
        exit_times = (route.get("exitTimes", "").split() if route is not None else [])
        if len(exit_times) >= 2:
            result[vehicle.get("id")] = {
                "sumo_junction_exit": float(exit_times[0]),
                "sumo_route_exit": float(exit_times[-1]),
            }
    return result


def _logic_line_events(path: Path):
    """Read the first crossing of each direction-specific video line."""

    events = {}
    if not path.is_file():
        return events
    for item in ET.parse(path).getroot().findall("instantOut"):
        if item.get("state") != "enter":
            continue
        detector = item.get("id", "")
        if not detector.startswith("logic_"):
            continue
        role = "sumo_logic_entry" if detector.startswith("logic_entry_") else "sumo_logic_exit"
        vehicle_id = item.get("vehID", "")
        # A lane detector can only produce one enter event for a vehicle. Keep
        # the earliest event defensively in case of malformed duplicate output.
        previous = events.setdefault(vehicle_id, {}).get(role)
        timestamp = float(item.get("time"))
        if previous is None or timestamp < previous:
            events[vehicle_id][role] = timestamp
    return events


def _attach_logic_line_alignment(frame, shift):
    """Build same-location video/SUMO timing pairs from detector events.

    The source exit timestamp is measured at an outbound video line.  SUMO's
    first edge-exit timestamp is the stop line, so the route-specific median
    travel time between those two SUMO locations is used only to infer a
    disclosed stop-line proxy for phase diagnostics.
    """

    frame = frame.copy()
    frame["observed_entry_line"] = (
        pd.to_numeric(frame["entry_time"], errors="coerce") + shift
    )
    frame["observed_exit_line"] = (
        pd.to_numeric(frame["exit_time"], errors="coerce") + shift
    )
    frame["sumo_entry_line"] = pd.to_numeric(
        frame.get("sumo_logic_entry"), errors="coerce"
    )
    frame["sumo_exit_line"] = pd.to_numeric(
        frame.get("sumo_logic_exit"), errors="coerce"
    )
    frame["sumo_post_stopline_seconds"] = (
        pd.to_numeric(frame["sumo_exit_line"], errors="coerce")
        - pd.to_numeric(frame["sumo_junction_exit"], errors="coerce")
    )
    route_post = frame.groupby("route")["sumo_post_stopline_seconds"].transform(
        "median"
    )
    frame["route_post_stopline_seconds"] = route_post
    frame["observed_stopline_proxy"] = (
        frame["observed_exit_line"] - frame["route_post_stopline_seconds"]
    )
    frame["entry_line_residual_seconds"] = (
        frame["sumo_entry_line"] - frame["observed_entry_line"]
    )
    frame["exit_line_residual_seconds"] = (
        frame["sumo_exit_line"] - frame["observed_exit_line"]
    )
    frame["stopline_proxy_residual_seconds"] = (
        pd.to_numeric(frame["sumo_junction_exit"], errors="coerce")
        - frame["observed_stopline_proxy"]
    )
    return frame


def _offset_scan_from_stopline_proxy(input_rows, signal_program, resolution=0.1):
    """Scan phase offset against the direction-corrected stop-line proxy."""

    observations = input_rows[
        input_rows["turning_direction"].isin(["through", "left"])
        & input_rows["observed_stopline_proxy"].notna()
    ]
    steps = int(round(signal_program["cycle_seconds"] / resolution))
    rows = []
    for index in range(steps):
        offset = round(index * resolution, 10)
        matched = []
        for _, row in observations.iterrows():
            phase = converter.signal_phase_at(
                row["observed_stopline_proxy"], offset,
                cycle_seconds=signal_program["cycle_seconds"],
            )
            matched.append(
                bool(
                    _movement_matches_named_phase(
                        row, phase
                    )
                )
            )
        rows.append(
            {
                "offset_seconds": offset,
                "matched_records": int(sum(matched)),
                "calibration_records": len(matched),
                "match_rate": sum(matched) / len(matched) if matched else 0.0,
            }
        )
    return rows


def _summary(path: Path):
    steps = ET.parse(path).getroot().findall("step")
    if not steps:
        raise ValueError("SUMO summary contains no steps")
    return steps[-1].attrib


def _signal_program(path: Path):
    logic = next(
        item
        for item in ET.parse(path).getroot().findall("tlLogic")
        if item.get("id") == "J"
    )
    phases = [
        {
            "name": phase.get("name"),
            "duration_seconds": float(phase.get("duration")),
            "state": phase.get("state"),
        }
        for phase in logic.findall("phase")
    ]
    all_red_names = {"NS过渡红", "南北东西全红", "EW过渡红", "东西南北全红"}
    return {
        "program_id": logic.get("programID"),
        "offset_seconds": float(logic.get("offset", "0")),
        "cycle_seconds": sum(item["duration_seconds"] for item in phases),
        "phases": phases,
        "all_red_states_valid": all(
            set(item["state"]) == {"r"}
            for item in phases
            if item["name"] in all_red_names
        ),
        "absolute_offset_note": (
            "The source provides durations but no timestamped phase origin. The "
            "selected offset is a disclosed proxy calibration; demand-preservation "
            "metrics do not depend on this offset."
        ),
    }


def _movement_matches_named_phase(row, phase_name):
    if not isinstance(phase_name, str):
        return False
    group = "NS" if row["approach"] in {"North", "South"} else "EW"
    if not phase_name.startswith(group):
        return False
    if row["turning_direction"] in {"through", "right"}:
        return "直行" in phase_name
    if row["turning_direction"] == "left":
        return "左转" in phase_name
    return False


def _signal_movement_matches(row, signal_program):
    return _movement_matches_named_phase(row, row["signal_phase"])


def _stage_status(event_id, valid, routeable):
    if event_id in routeable:
        return "sumo_input"
    if event_id in valid:
        return "topology_excluded"
    return "incomplete_trajectory"


def _build_traceability(
    source_rows, source_path, source_net, route_input, trips, vehroutes, line_events=None
):
    _, valid_rows, shift, *_ = converter.parse_rows(source_path)
    valid_ids = {row["_event_id"] for row in valid_rows}
    routeable_ids = set(route_input)
    junctions, edges, direct = _network_metadata(source_net)
    output = []
    for row in source_rows:
        event_id = row["_event_id"]
        entry_edge = row.get("entry_edge", "")
        exit_edge = row.get("exit_edge", "")
        pair = (entry_edge, exit_edge)
        movement = ""
        if entry_edge in edges and exit_edge in edges:
            movement = TURN_LABELS[
                converter.classify_movement(entry_edge, exit_edge, junctions, edges)
            ]
        entry_time = _float(row.get("entry_time"))
        exit_time = _float(row.get("exit_time"))
        vehicle_id = row["vehicle_id"]
        item = {
            **row,
            "approach": APPROACHES.get(entry_edge, ""),
            "turning_direction": movement,
            "route": f"{entry_edge}->{exit_edge}" if entry_edge and exit_edge else "",
            "observed_travel_time": (
                exit_time - entry_time
                if entry_time is not None and exit_time is not None
                else ""
            ),
            "event_id": event_id,
            "extraction_status": _stage_status(event_id, valid_ids, routeable_ids),
            "network_connection_exists": (
                str(pair in direct).lower() if entry_edge and exit_edge else ""
            ),
            "normalized_planned_depart": (
                entry_time + shift if entry_time is not None and event_id in routeable_ids else ""
            ),
        }
        if event_id in route_input:
            item.update(route_input[event_id])
        if event_id in trips:
            item.update(trips[event_id])
            item.update(vehroutes.get(event_id, {}))
            item["sumo_status"] = "completed"
        else:
            item["sumo_status"] = "not_applicable"
        if line_events and event_id in line_events:
            item.update(line_events[event_id])
        output.append(item)
    return pd.DataFrame(output), valid_ids, routeable_ids, shift


def _window_counts(values, window, end, name):
    counts = Counter(
        int(math.floor(float(value) / window)) * window
        for value in values
        if value is not None and not pd.isna(value)
    )
    begins = list(range(0, int(math.ceil(end / window)) * window, window))
    return pd.Series([counts[begin] for begin in begins], index=begins, name=name)


def _video_time_table(frame, window):
    end = max(float(value) for value in frame["observed_event_time"]) + 1e-9
    observed = frame["observed_event_time"].astype(float)
    extracted = frame.loc[
        frame["extraction_status"].isin(["sumo_input", "topology_excluded"]),
        "observed_event_time",
    ].astype(float)
    sumo_input = frame.loc[
        frame["extraction_status"] == "sumo_input", "observed_event_time"
    ].astype(float)
    table = pd.concat(
        [
            _window_counts(observed, window, end, "real_observed_count"),
            _window_counts(extracted, window, end, "extracted_complete_count"),
            _window_counts(sumo_input, window, end, "sumo_input_count"),
        ],
        axis=1,
    ).fillna(0).astype(int)
    table.index.name = "window_begin_seconds"
    table = table.reset_index()
    table.insert(1, "window_end_seconds", table["window_begin_seconds"] + window)
    return table


def _sumo_time_table(frame, window):
    input_rows = frame[frame["extraction_status"] == "sumo_input"]
    end = max(
        input_rows["sumo_input_depart"].max(),
        input_rows["sumo_depart"].max(),
        input_rows["sumo_arrival"].max(),
    ) + 1e-9
    table = pd.concat(
        [
            _window_counts(
                input_rows["sumo_input_depart"], window, end, "sumo_input_count"
            ),
            _window_counts(
                input_rows["sumo_depart"], window, end, "sumo_actual_depart_count"
            ),
            _window_counts(
                input_rows["sumo_arrival"], window, end, "sumo_actual_arrival_count"
            ),
            _window_counts(
                input_rows["sumo_junction_exit"], window, end, "sumo_stopline_crossing_count"
            ),
        ],
        axis=1,
    ).fillna(0).astype(int)
    table.index.name = "window_begin_seconds"
    table = table.reset_index()
    table.insert(1, "window_end_seconds", table["window_begin_seconds"] + window)
    return table


def _composition(frame, column, stages):
    rows = []
    for stage, mask in stages.items():
        values = frame.loc[mask, column]
        values = values[values.astype(str) != ""]
        total = len(values)
        for value, count in values.value_counts().sort_index().items():
            rows.append(
                {
                    "stage": stage,
                    column: value,
                    "count": int(count),
                    "percentage": 100.0 * count / total if total else 0.0,
                    "stage_total": total,
                }
            )
    return pd.DataFrame(rows)


def _route_flow(frame, window):
    complete = frame[
        frame["extraction_status"].isin(["sumo_input", "topology_excluded"])
    ].copy()
    complete["window_begin_seconds"] = (
        complete["observed_event_time"].astype(float) // window * window
    ).astype(int)
    keys = ["window_begin_seconds", "approach", "turning_direction", "route"]
    extracted = complete.groupby(keys).size().rename("extracted_complete_count")
    routeable = complete[complete["extraction_status"] == "sumo_input"]
    input_counts = routeable.groupby(keys).size().rename("sumo_input_count")
    completed = routeable[routeable["sumo_status"] == "completed"]
    completed_counts = completed.groupby(keys).size().rename("sumo_completed_count")
    table = pd.concat([extracted, input_counts, completed_counts], axis=1).fillna(0)
    table = table.astype(int).reset_index()
    table.insert(1, "window_end_seconds", table["window_begin_seconds"] + window)
    return table.sort_values(keys).reset_index(drop=True)


def _tv_similarity(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if left.sum() == 0 or right.sum() == 0:
        return None
    return float(1.0 - 0.5 * np.abs(left / left.sum() - right / right.sum()).sum())


def _pearson(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _composition_similarity(table, stage_left, stage_right, category):
    pivot = table.pivot(index=category, columns="stage", values="count").fillna(0)
    return _tv_similarity(pivot.get(stage_left, 0), pivot.get(stage_right, 0))


def _plot_lines(table, columns, labels, title, output_base):
    fig, axis = plt.subplots(figsize=(10, 4.8))
    x = table["window_begin_seconds"] / 60.0
    for column, label, color in zip(columns, labels, COLORS.values()):
        axis.plot(x, table[column], marker="o", linewidth=2, label=label, color=color)
    axis.set(title=title, xlabel="Window start (minutes)", ylabel="Vehicle count")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=min(3, len(columns)))
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def _plot_composition(table, category, title, output_base):
    pivot = table.pivot(index=category, columns="stage", values="percentage").fillna(0)
    stages = list(pivot.columns)
    colors = ["#D1495B", "#00798C", "#EDAE49", "#30638E"][: len(stages)]
    axis = pivot.plot(kind="bar", figsize=(9, 4.8), color=colors, width=0.75)
    axis.set(title=title, xlabel="", ylabel="Share (%)")
    axis.grid(axis="y", alpha=0.25)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        handles,
        [label.replace("_", " ").title() for label in labels],
        frameon=False,
        title="Stage",
    )
    plt.xticks(rotation=0)
    plt.tight_layout()
    axis.figure.savefig(output_base.with_suffix(".png"), dpi=180)
    axis.figure.savefig(output_base.with_suffix(".svg"))
    plt.close(axis.figure)


def _plot_totals(totals, output_base):
    labels = [item["stage"].replace("_", " ").title() for item in totals]
    values = [item["count"] for item in totals]
    fig, axis = plt.subplots(figsize=(9, 4.8))
    bars = axis.bar(labels, values, color=["#343A40", "#D1495B", "#00798C", "#EDAE49"])
    axis.bar_label(bars, padding=3)
    axis.set(title="Demand scale and simulation completion", ylabel="Vehicle count")
    axis.grid(axis="y", alpha=0.25)
    axis.set_ylim(0, max(values) * 1.12)
    plt.xticks(rotation=12, ha="right")
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def _plot_signal_diagnostics(frame, signal_program, shift, output_base):
    """Show same-location trend alignment and corrected signal-phase alignment."""
    input_rows = frame[frame["extraction_status"] == "sumo_input"].copy()
    fig = plt.figure(figsize=(11, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    top = fig.add_subplot(grid[0, :])
    for column, label, color, linestyle in (
        ("observed_exit_line", "Video exit-line crossing", "#343A40", "--"),
        ("sumo_exit_line", "SUMO exit-line crossing", "#00798C", "-"),
    ):
        values = np.sort(input_rows[column].dropna().to_numpy(dtype=float))
        top_values = np.arange(1, len(values) + 1)
        top.step(
            (values - shift) / 60.0,
            top_values,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    paired = input_rows[["observed_exit_line", "sumo_exit_line"]].dropna()
    residual = paired["sumo_exit_line"] - paired["observed_exit_line"]
    alignment = _exit_line_alignment_metrics(input_rows)
    top.set(
        title=(
            "Same exit-line timing alignment "
            f"(cumulative-trend MAE {alignment['cumulative_trend_mae_seconds']:.1f} s; "
            f"paired-event MAE {alignment['paired_event_mae_seconds']:.1f} s)"
        ),
        xlabel="Video-relative clock (minutes)",
        ylabel="Cumulative vehicles",
    )
    top.grid(alpha=0.25)
    top.legend(frameon=False, ncol=2, loc="upper left")

    regions = (
        (0, 30, "NS through", "#BFD7EA"),
        (30, 32, "", "#E9ECEF"),
        (32, 57, "NS left", "#F7D6A4"),
        (57, 62, "", "#E9ECEF"),
        (62, 99, "EW through", "#BFD7EA"),
        (99, 101, "", "#E9ECEF"),
        (101, 123, "EW left", "#F7D6A4"),
        (123, 128, "", "#E9ECEF"),
    )
    bins = np.arange(0, signal_program["cycle_seconds"] + 4, 4)
    for index, (movement_group, approaches) in enumerate(
        (("NS", ["North", "South"]), ("EW", ["East", "West"]))
    ):
        bottom = fig.add_subplot(grid[1, index])
        for begin, end, _, color in regions:
            bottom.axvspan(begin, end, color=color, alpha=0.42)
        selected = input_rows[
            input_rows["approach"].isin(approaches)
            & input_rows["turning_direction"].isin(["through", "left"])
        ]
        proxy_validation = selected[
            selected["observed_alignment_partition"] == "validation"
        ]
        proxy_match_rate = float(
            proxy_validation[
                "event_matched_proxy_movement_phase_match"
            ].astype(float).mean()
        )
        for column, partition_column, label, color, sample_offset, linestyle in (
            (
                "event_matched_stopline_proxy",
                "observed_alignment_partition",
                "Video proxy (held out)",
                "#343A40",
                0.0,
                "--",
            ),
            (
                "sumo_junction_exit",
                "sumo_alignment_partition",
                "SUMO crossing (held out)",
                "#00798C",
                converter.DEFAULT_SIGNAL_STEP_LENGTH_SECONDS / 2,
                "-",
            ),
        ):
            values = selected.loc[
                selected[partition_column] == "validation", column
            ].dropna()
            positions = [
                converter.signal_phase_position(
                    value,
                    signal_program["offset_seconds"],
                    cycle_seconds=signal_program["cycle_seconds"],
                    sample_offset=sample_offset,
                )
                for value in values.to_numpy(dtype=float)
            ]
            bottom.hist(
                positions,
                bins=bins,
                histtype="step",
                linewidth=2,
                linestyle=linestyle,
                label=label,
                color=color,
            )
        bottom.set(
            title=(
                f"{movement_group} held-out phase check "
                f"({100 * proxy_match_rate:.1f}% permitted)"
            ),
            xlabel="Phase position in 128 s cycle",
            ylabel="Crossings per 4 s" if index == 0 else "",
            xlim=(0, signal_program["cycle_seconds"]),
            ylim=(0, None),
        )
        bottom.grid(axis="y", alpha=0.25)
        bottom.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)
    return alignment


def _plot_signal_diagnostics_legacy(frame, signal_program, shift, output_base):
    """Retained only for compatibility with old notebooks."""
    input_rows = frame[frame["extraction_status"] == "sumo_input"].copy()
    input_rows["observed_exit_proxy"] = (
        pd.to_numeric(input_rows["exit_time"], errors="coerce") + shift
    )
    fig = plt.figure(figsize=(11, 7.4))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.05, 1])
    top = fig.add_subplot(grid[0, :])
    for column, label, color, linestyle in (
        ("observed_exit_proxy", "Observed exit-time proxy", "#343A40", "--"),
        ("sumo_junction_exit", "SUMO stop-line crossing", "#00798C", "-"),
    ):
        values = np.sort(input_rows[column].dropna().to_numpy(dtype=float))
        top_values = np.arange(1, len(values) + 1)
        # Keep the two cumulative curves in the same absolute clock.
        top.step(
            values / 60.0,
            top_values,
            where="post",
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
        )
    paired = input_rows[["observed_exit_proxy", "sumo_junction_exit"]].dropna()
    residual = paired["sumo_junction_exit"] - paired["observed_exit_proxy"]
    top.set(
        title=(
            "Observed timing vs SUMO execution "
            f"(median residual {residual.median():.1f} s; mean absolute residual "
            f"{residual.abs().mean():.1f} s)"
        ),
        xlabel="Common normalized clock (minutes)",
        ylabel="Cumulative vehicles",
    )
    top.grid(alpha=0.25)
    top.legend(frameon=False, ncol=2, loc="upper left")

    regions = (
        (0, 30, "NS through", "#BFD7EA"),
        (30, 32, "", "#E9ECEF"),
        (32, 57, "NS left", "#F7D6A4"),
        (57, 62, "", "#E9ECEF"),
        (62, 99, "EW through", "#BFD7EA"),
        (99, 101, "", "#E9ECEF"),
        (101, 123, "EW left", "#F7D6A4"),
        (123, 128, "", "#E9ECEF"),
    )
    bins = np.arange(0, signal_program["cycle_seconds"] + 4, 4)
    for index, (movement_group, approaches) in enumerate(
        (("NS", ["North", "South"]), ("EW", ["East", "West"]))
    ):
        bottom = fig.add_subplot(grid[1, index])
        for begin, end, _, color in regions:
            bottom.axvspan(begin, end, color=color, alpha=0.42)
        selected = input_rows[
            input_rows["approach"].isin(approaches)
            & input_rows["turning_direction"].isin(["through", "left"])
        ]
        for column, label, color, sample_offset, linestyle in (
            ("observed_exit_proxy", "Observed proxy", "#343A40", 0.0, "--"),
            (
                "sumo_junction_exit",
                "SUMO crossing",
                "#00798C",
                converter.DEFAULT_SIGNAL_STEP_LENGTH_SECONDS / 2,
                "-",
            ),
        ):
            positions = [
                converter.signal_phase_position(
                    value,
                    signal_program["offset_seconds"],
                    cycle_seconds=signal_program["cycle_seconds"],
                    sample_offset=sample_offset,
                )
                for value in selected[column].dropna().to_numpy(dtype=float)
            ]
            bottom.hist(
                positions,
                bins=bins,
                histtype="step",
                linewidth=2,
                linestyle=linestyle,
                label=label,
                color=color,
            )
        bottom.set(
            title=f"{movement_group} through/left (right turns excluded)",
            xlabel="Phase position in 128 s cycle",
            ylabel="Crossings per 4 s" if index == 0 else "",
            xlim=(0, signal_program["cycle_seconds"]),
            ylim=(0, None),
        )
        bottom.grid(axis="y", alpha=0.25)
        bottom.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def _plot_offset_scan(scan, selected_offset, output_base, tied_min=None, tied_max=None):
    fig, axis = plt.subplots(figsize=(9, 4.5))
    axis.plot(scan["offset_seconds"], 100 * scan["match_rate"], color="#30638E", linewidth=2)
    if tied_min is not None and tied_max is not None:
        axis.axvspan(tied_min, tied_max, color="#F7D6A4", alpha=0.45, label="Tied-best interval")
    axis.axvline(selected_offset, color="#D1495B", linestyle="--", linewidth=2, label=f"Selected offset: {selected_offset:.1f} s")
    axis.set(
        title="Observed-exit proxy calibration of signal offset",
        xlabel="Candidate SUMO signal offset (seconds)",
        ylabel="Movement-phase proxy match (%)",
        xlim=(0, converter.SIGNAL_CYCLE_SECONDS),
    )
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=180)
    fig.savefig(output_base.with_suffix(".svg"))
    plt.close(fig)


def _write_csv(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")


def _refine_departure_adjustments(input_rows, previous, iteration):
    """Update each departure using its matched entry-line crossing residual."""

    limit = (
        converter.ALIGNMENT_WARMUP_CYCLES * converter.SIGNAL_CYCLE_SECONDS - 1.0
    )
    adjustments = dict(previous)
    rows = []
    for _, row in input_rows.iterrows():
        event_id = row["event_id"]
        residual = _float(row.get("entry_line_residual_seconds"))
        old = adjustments.get(event_id, 0.0)
        proposed = old - residual if residual is not None else old
        updated = min(limit, max(-limit, proposed))
        adjustments[event_id] = updated
        rows.append(
            {
                "event_id": event_id,
                "vehicle_id": row.get("vehicle_id", ""),
                "semantic_id": row.get("semantic_id", ""),
                "entry_edge": row.get("entry_edge", ""),
                "exit_edge": row.get("exit_edge", ""),
                "calibration_iteration": iteration,
                "matched_entry_residual_seconds": residual,
                "previous_adjustment_seconds": old,
                "departure_adjustment_seconds": updated,
                "adjustment_clipped": proposed != updated,
            }
        )
    return adjustments, rows


def _entry_alignment_metrics(input_rows):
    residual = pd.to_numeric(
        input_rows["entry_line_residual_seconds"], errors="coerce"
    ).dropna()
    absolute = residual.abs()
    return {
        "records": int(len(residual)),
        "median_residual_seconds": float(residual.median()),
        "mean_absolute_residual_seconds": float(absolute.mean()),
        "p90_absolute_residual_seconds": float(absolute.quantile(0.9)),
    }


def _exit_line_alignment_metrics(input_rows):
    paired = input_rows[["observed_exit_line", "sumo_exit_line"]].dropna()
    residual = paired["sumo_exit_line"] - paired["observed_exit_line"]
    observed = np.sort(paired["observed_exit_line"].to_numpy(dtype=float))
    simulated = np.sort(paired["sumo_exit_line"].to_numpy(dtype=float))
    trend_residual = simulated - observed
    return {
        "same_exit_line_records": int(len(paired)),
        "paired_event_median_residual_seconds": float(residual.median()),
        "paired_event_mae_seconds": float(residual.abs().mean()),
        "cumulative_trend_median_residual_seconds": float(
            np.median(trend_residual)
        ),
        "cumulative_trend_mae_seconds": float(np.mean(np.abs(trend_residual))),
        "cumulative_trend_p90_absolute_residual_seconds": float(
            np.quantile(np.abs(trend_residual), 0.9)
        ),
    }


def _apply_od_rank_event_alignment(input_rows):
    """Fit monotone OD timing maps with alternating-rank holdout events.

    SUMO may reorder vehicles within an OD because lane queues differ from the
    video. In one dimension, matching sorted event times is the minimum-L1
    one-to-one assignment. Even ranks (plus the final endpoint) define the
    piecewise-linear map; odd ranks remain untouched validation events.
    """

    frame = input_rows.copy()
    frame["observed_alignment_rank"] = pd.Series(
        pd.NA, index=frame.index, dtype="Int64"
    )
    frame["sumo_alignment_rank"] = pd.Series(
        pd.NA, index=frame.index, dtype="Int64"
    )
    frame["observed_alignment_partition"] = ""
    frame["sumo_alignment_partition"] = ""
    frame["event_matched_exit_line"] = np.nan
    frame["event_matched_stopline_proxy"] = np.nan
    audit_rows = []
    informative = frame[
        frame["turning_direction"].isin(["through", "left"])
        & frame["observed_exit_line"].notna()
        & frame["sumo_exit_line"].notna()
    ]
    for route, group in informative.groupby("route"):
        observed = group.sort_values(
            ["observed_exit_line", "event_id"], kind="stable"
        )
        simulated = group.sort_values(
            ["sumo_exit_line", "event_id"], kind="stable"
        )
        count = len(observed)
        if count < 3:
            continue
        ranks = np.arange(count)
        calibration = ranks % 2 == 0
        calibration[-1] = True
        partitions = np.where(calibration, "calibration", "validation")
        observed_times = observed["observed_exit_line"].to_numpy(dtype=float)
        simulated_times = simulated["sumo_exit_line"].to_numpy(dtype=float)
        mapped_exit = np.interp(
            observed_times,
            observed_times[calibration],
            simulated_times[calibration],
        )
        post_stopline = observed["route_post_stopline_seconds"].to_numpy(
            dtype=float
        )
        mapped_stopline = mapped_exit - post_stopline

        frame.loc[observed.index, "observed_alignment_rank"] = ranks
        frame.loc[
            observed.index, "observed_alignment_partition"
        ] = partitions
        frame.loc[observed.index, "event_matched_exit_line"] = mapped_exit
        frame.loc[
            observed.index, "event_matched_stopline_proxy"
        ] = mapped_stopline
        frame.loc[simulated.index, "sumo_alignment_rank"] = ranks
        frame.loc[simulated.index, "sumo_alignment_partition"] = partitions

        observed_ids = observed["event_id"].to_numpy()
        simulated_ids = simulated["event_id"].to_numpy()
        for rank in ranks:
            audit_rows.append(
                {
                    "route": route,
                    "rank": int(rank),
                    "partition": partitions[rank],
                    "observed_event_id": observed_ids[rank],
                    "observed_exit_line_seconds": observed_times[rank],
                    "sumo_event_id": simulated_ids[rank],
                    "sumo_exit_line_seconds": simulated_times[rank],
                    "mapped_observed_exit_line_seconds": mapped_exit[rank],
                    "mapped_minus_sumo_exit_seconds": (
                        mapped_exit[rank] - simulated_times[rank]
                    ),
                    "mapped_observed_stopline_proxy_seconds": mapped_stopline[
                        rank
                    ],
                }
            )
    return frame, audit_rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--network", type=Path, default=converter.DEFAULT_NET)
    parser.add_argument("--conversion-dir", type=Path, default=converter.DEFAULT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sumo-binary", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-conversion", action="store_true")
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".svg"):
        (args.output_dir / f"figure_03_sumo_depart_arrival_1min{suffix}").unlink(
            missing_ok=True
        )
    adjustment_path = (
        args.conversion_dir / "xiasha1_sumo_departure_alignment.csv"
    )

    def _convert(signal_offset, use_adjustments=False):
        conversion_args = [
            "--all",
            "--input",
            str(args.input),
            "--network",
            str(args.network),
            "--output-dir",
            str(args.conversion_dir),
            "--expand-parallel-straight-output-lanes",
            "--signal-offset",
            f"{signal_offset:.1f}",
        ]
        if use_adjustments:
            conversion_args.extend(
                ["--departure-adjustments", str(adjustment_path)]
            )
        converter.main(conversion_args)

    if not args.skip_conversion:
        _convert(75.5)

    cfg = args.conversion_dir / "xiasha1_sumo_vehicles.sumocfg"
    route_path = args.conversion_dir / "xiasha1_sumo_vehicles.rou.xml"
    trip_path = args.output_dir / "sumo_tripinfo.xml"
    summary_path = args.output_dir / "sumo_summary.xml"
    vehroute_path = args.output_dir / "sumo_vehroute.xml"
    command, sumo_path = _sumo_command(args.sumo_binary)
    command += [
        "-c",
        str(cfg),
        "--seed",
        str(args.seed),
        "--step-length",
        str(converter.DEFAULT_SIGNAL_STEP_LENGTH_SECONDS),
        "--duration-log.disable",
        "--no-step-log",
        "true",
        "--tripinfo-output",
        str(trip_path),
        "--summary-output",
        str(summary_path),
        "--vehroute-output",
        str(vehroute_path),
        "--vehroute-output.exit-times",
        "true",
        "--vehroute-output.route-length",
        "true",
    ]
    def _run_sumo():
        return subprocess.run(
            command, cwd=ROOT, text=True, capture_output=True, check=True
        )

    source_rows = _read_source(args.input)

    numeric_columns = (
        "observed_travel_time",
        "normalized_planned_depart",
        "sumo_input_depart",
        "departure_adjustment_seconds",
        "sumo_depart",
        "sumo_arrival",
        "sumo_duration",
        "sumo_departDelay",
        "sumo_waitingTime",
        "sumo_timeLoss",
        "sumo_routeLength",
        "sumo_junction_exit",
        "sumo_route_exit",
        "observed_entry_line",
        "sumo_entry_line",
        "entry_line_residual_seconds",
        "observed_exit_line",
        "sumo_exit_line",
        "exit_line_residual_seconds",
        "observed_stopline_proxy",
        "stopline_proxy_residual_seconds",
    )

    def _collect_execution():
        current_run = _run_sumo()
        current_routes = _route_input(route_path)
        current_trips = _tripinfo(trip_path)
        current_vehroutes = _vehroute(vehroute_path)
        current_line_events = _logic_line_events(
            args.conversion_dir / converter.LOGIC_LINE_DETECTOR_FILE
        )
        current_frame, current_valid, current_routeable, current_shift = (
            _build_traceability(
                source_rows,
                args.input,
                args.network,
                current_routes,
                current_trips,
                current_vehroutes,
                current_line_events,
            )
        )
        current_frame = _attach_logic_line_alignment(
            current_frame, current_shift
        )
        for column in numeric_columns:
            current_frame[column] = pd.to_numeric(
                current_frame[column], errors="coerce"
            ).round(2)
        current_program = _signal_program(
            args.conversion_dir / "xiasha1_sumo_signal.net.xml"
        )
        return (
            current_run,
            current_routes,
            current_trips,
            current_frame,
            current_valid,
            current_routeable,
            current_shift,
            _summary(summary_path),
            current_program,
        )

    (
        run,
        routes,
        trips,
        frame,
        valid_ids,
        routeable_ids,
        shift,
        last_summary,
        signal_program,
    ) = _collect_execution()
    input_trace = frame[frame["extraction_status"] == "sumo_input"].copy()
    initial_offset = signal_program["offset_seconds"]

    departure_adjustments = (
        converter.read_departure_adjustments(adjustment_path)
        if args.skip_conversion and adjustment_path.is_file()
        else {}
    )
    entry_alignment_iterations = []
    refinement_scan = _offset_scan_from_stopline_proxy(input_trace, signal_program)
    tied = [
        item for item in refinement_scan
        if item["matched_records"] == max(x["matched_records"] for x in refinement_scan)
    ]
    selected_offset = min(
        tied,
        key=lambda item: min(
            abs(item["offset_seconds"] - initial_offset),
            signal_program["cycle_seconds"]
            - abs(item["offset_seconds"] - initial_offset),
        ),
    )["offset_seconds"]
    offset_delta = abs(selected_offset - initial_offset)
    offset_delta = min(offset_delta, signal_program["cycle_seconds"] - offset_delta)
    if offset_delta > 0.05:
        _convert(
            selected_offset,
            use_adjustments=args.skip_conversion and bool(departure_adjustments),
        )
        (
            run,
            routes,
            trips,
            frame,
            valid_ids,
            routeable_ids,
            shift,
            last_summary,
            signal_program,
        ) = _collect_execution()
        input_trace = frame[
            frame["extraction_status"] == "sumo_input"
        ].copy()

    # With one global signal origin fixed, refine only the observed input
    # boundary. Accept an iteration only when the matched entry-line MAE falls;
    # exit events remain untouched validation data.
    if not args.skip_conversion:
        departure_adjustments = {}
        best_rows = []
        best_metrics = _entry_alignment_metrics(input_trace)
        for iteration in range(1, 3):
            before = _entry_alignment_metrics(input_trace)
            candidate_adjustments, candidate_rows = (
                _refine_departure_adjustments(
                    input_trace, departure_adjustments, iteration
                )
            )
            _write_csv(adjustment_path, candidate_rows)
            _convert(
                signal_program["offset_seconds"], use_adjustments=True
            )
            candidate_state = _collect_execution()
            candidate_frame = candidate_state[3]
            candidate_trace = candidate_frame[
                candidate_frame["extraction_status"] == "sumo_input"
            ].copy()
            after = _entry_alignment_metrics(candidate_trace)
            accepted = (
                after["mean_absolute_residual_seconds"]
                < best_metrics["mean_absolute_residual_seconds"] - 0.01
            )
            entry_alignment_iterations.append(
                {
                    "iteration": iteration,
                    "accepted": accepted,
                    "before": before,
                    "after": after,
                }
            )
            if not accepted:
                if best_rows:
                    _write_csv(adjustment_path, best_rows)
                    _convert(
                        signal_program["offset_seconds"],
                        use_adjustments=True,
                    )
                else:
                    adjustment_path.unlink(missing_ok=True)
                    _convert(signal_program["offset_seconds"])
                (
                    run,
                    routes,
                    trips,
                    frame,
                    valid_ids,
                    routeable_ids,
                    shift,
                    last_summary,
                    signal_program,
                ) = _collect_execution()
                input_trace = frame[
                    frame["extraction_status"] == "sumo_input"
                ].copy()
                break
            departure_adjustments = candidate_adjustments
            best_rows = candidate_rows
            best_metrics = after
            (
                run,
                routes,
                trips,
                frame,
                valid_ids,
                routeable_ids,
                shift,
                last_summary,
                signal_program,
            ) = candidate_state
            input_trace = candidate_trace
            if after["p90_absolute_residual_seconds"] <= 0.5:
                break
    refinement_scan = _offset_scan_from_stopline_proxy(input_trace, signal_program)
    best_refined = max(refinement_scan, key=lambda item: item["matched_records"])
    tied_refined = [
        item
        for item in refinement_scan
        if item["matched_records"] == best_refined["matched_records"]
    ]
    post_entry_offset = min(
        tied_refined,
        key=lambda item: min(
            abs(item["offset_seconds"] - signal_program["offset_seconds"]),
            signal_program["cycle_seconds"]
            - abs(item["offset_seconds"] - signal_program["offset_seconds"]),
        ),
    )["offset_seconds"]
    post_entry_delta = abs(
        post_entry_offset - signal_program["offset_seconds"]
    )
    post_entry_delta = min(
        post_entry_delta, signal_program["cycle_seconds"] - post_entry_delta
    )
    # The exit-line-to-stop-line proxy itself has metre-level spatial
    # uncertainty, so sub-second changes are not meaningful refinements.
    if post_entry_delta > 1.0:
        _convert(
            post_entry_offset,
            use_adjustments=bool(departure_adjustments),
        )
        (
            run,
            routes,
            trips,
            frame,
            valid_ids,
            routeable_ids,
            shift,
            last_summary,
            signal_program,
        ) = _collect_execution()
        input_trace = frame[
            frame["extraction_status"] == "sumo_input"
        ].copy()
        refinement_scan = _offset_scan_from_stopline_proxy(
            input_trace, signal_program
        )
        best_refined = max(
            refinement_scan, key=lambda item: item["matched_records"]
        )
        tied_refined = [
            item
            for item in refinement_scan
            if item["matched_records"] == best_refined["matched_records"]
        ]
    with (args.conversion_dir / "xiasha1_sumo_signal_offset_scan.csv").open(
        "w", encoding="utf-8", newline=""
    ) as scan_file:
        writer = csv.DictWriter(
            scan_file,
            fieldnames=(
                "offset_seconds",
                "matched_records",
                "calibration_records",
                "match_rate",
            ),
        )
        writer.writeheader()
        writer.writerows(refinement_scan)
    signal_program["calibration"] = {
        "method": "same_exit_line_event_matching_with_route_post_stopline_correction",
        "initial_offset_seconds": initial_offset,
        "selected_offset_seconds": signal_program["offset_seconds"],
        "refinement_applied": bool(offset_delta > 0.05),
        "post_entry_refinement_applied": bool(post_entry_delta > 1.0),
        "post_entry_refinement_tolerance_seconds": 1.0,
        "calibration_records": max(item["calibration_records"] for item in refinement_scan),
        "calibration_matched_records": max(item["matched_records"] for item in refinement_scan),
        "calibration_match_rate": max(item["match_rate"] for item in refinement_scan),
        "calibration_tied_best_min_seconds": min(item["offset_seconds"] for item in tied_refined),
        "calibration_tied_best_max_seconds": max(item["offset_seconds"] for item in tied_refined),
        "scan_file": "xiasha1_sumo_signal_offset_scan.csv",
        "entry_line_calibration": {
            "method": "one_to_one_event_departure_residual_correction",
            "iterations": entry_alignment_iterations,
            "final": _entry_alignment_metrics(input_trace),
            "adjustment_file": adjustment_path.name
            if adjustment_path.is_file()
            else None,
            "adjustment_records": len(departure_adjustments),
            "adjustment_range_seconds": (
                [
                    min(departure_adjustments.values()),
                    max(departure_adjustments.values()),
                ]
                if departure_adjustments
                else [0.0, 0.0]
            ),
        },
    }
    input_trace, event_alignment_rows = _apply_od_rank_event_alignment(
        input_trace
    )
    _write_csv(
        args.output_dir / "event_time_alignment.csv", event_alignment_rows
    )
    input_trace["signal_phase_position_seconds"] = input_trace[
        "sumo_junction_exit"
    ].map(
        lambda value: converter.signal_phase_position(
            value,
            signal_program["offset_seconds"],
            cycle_seconds=signal_program["cycle_seconds"],
            sample_offset=converter.DEFAULT_SIGNAL_STEP_LENGTH_SECONDS / 2,
        )
    )
    input_trace["signal_cycle_index"] = np.floor(
        (
            input_trace["sumo_junction_exit"]
            - signal_program["offset_seconds"]
            + converter.DEFAULT_SIGNAL_STEP_LENGTH_SECONDS / 2
        )
        / signal_program["cycle_seconds"]
    ).astype(int)
    input_trace["signal_phase"] = input_trace["signal_phase_position_seconds"].map(converter._phase_name_at)
    actual_phase_informative = input_trace["turning_direction"].isin(
        ["through", "left"]
    )
    input_trace["signal_movement_phase_match"] = pd.Series(
        pd.NA, index=input_trace.index, dtype="boolean"
    )
    input_trace.loc[
        actual_phase_informative, "signal_movement_phase_match"
    ] = input_trace.loc[actual_phase_informative].apply(
        lambda row: _signal_movement_matches(row, signal_program), axis=1
    )
    input_trace["observed_exit_proxy"] = input_trace["observed_stopline_proxy"]
    input_trace["observed_proxy_phase_position_seconds"] = input_trace[
        "observed_exit_proxy"
    ].map(
        lambda value: converter.signal_phase_position(
            value,
            signal_program["offset_seconds"],
            cycle_seconds=signal_program["cycle_seconds"],
        )
        if pd.notna(value)
        else np.nan
    )
    input_trace["observed_proxy_phase"] = input_trace[
        "observed_proxy_phase_position_seconds"
    ].map(converter._phase_name_at)
    informative = input_trace["turning_direction"].isin(["through", "left"])
    input_trace["observed_proxy_movement_phase_match"] = pd.Series(
        pd.NA, index=input_trace.index, dtype="boolean"
    )
    input_trace.loc[
        informative, "observed_proxy_movement_phase_match"
    ] = input_trace.loc[informative].apply(
        lambda row: _movement_matches_named_phase(
            row, row["observed_proxy_phase"]
        ),
        axis=1,
    )
    input_trace["event_matched_proxy_phase_position_seconds"] = input_trace[
        "event_matched_stopline_proxy"
    ].map(
        lambda value: converter.signal_phase_position(
            value,
            signal_program["offset_seconds"],
            cycle_seconds=signal_program["cycle_seconds"],
        )
        if pd.notna(value)
        else np.nan
    )
    input_trace["event_matched_proxy_phase"] = input_trace[
        "event_matched_proxy_phase_position_seconds"
    ].map(converter._phase_name_at)
    input_trace["event_matched_proxy_movement_phase_match"] = pd.Series(
        pd.NA, index=input_trace.index, dtype="boolean"
    )
    input_trace.loc[
        informative, "event_matched_proxy_movement_phase_match"
    ] = input_trace.loc[informative].apply(
        lambda row: _movement_matches_named_phase(
            row, row["event_matched_proxy_phase"]
        ),
        axis=1,
    )
    input_trace["sumo_minus_observed_exit_proxy_seconds"] = input_trace[
        "stopline_proxy_residual_seconds"
    ]
    signal_columns = [
        "event_id", "vehicle_id", "semantic_id", "approach", "turning_direction", "route", "sumo_depart",
        "departure_adjustment_seconds",
        "observed_entry_line", "sumo_entry_line", "entry_line_residual_seconds",
        "observed_exit_line", "sumo_exit_line", "exit_line_residual_seconds",
        "observed_stopline_proxy", "sumo_junction_exit",
        "sumo_minus_observed_exit_proxy_seconds", "sumo_arrival", "signal_cycle_index",
        "signal_phase_position_seconds", "signal_phase", "signal_movement_phase_match",
        "observed_proxy_phase_position_seconds", "observed_proxy_phase",
        "observed_proxy_movement_phase_match",
        "observed_alignment_rank", "observed_alignment_partition",
        "sumo_alignment_rank", "sumo_alignment_partition",
        "event_matched_exit_line", "event_matched_stopline_proxy",
        "event_matched_proxy_phase_position_seconds",
        "event_matched_proxy_phase",
        "event_matched_proxy_movement_phase_match",
    ]
    input_trace[signal_columns].to_csv(
        args.output_dir / "signal_cycle_crossings.csv", index=False, encoding="utf-8-sig"
    )
    phase_match_rate = float(
        input_trace.loc[
            actual_phase_informative, "signal_movement_phase_match"
        ].astype(float).mean()
    )
    proxy_phase_match_rate = float(
        input_trace.loc[
            informative, "observed_proxy_movement_phase_match"
        ].astype(float).mean()
    )
    matched_proxy_phase_match_rate = float(
        input_trace.loc[
            informative, "event_matched_proxy_movement_phase_match"
        ].astype(float).mean()
    )
    validation_mask = (
        informative
        & input_trace["observed_alignment_partition"].eq("validation")
    )
    validation_proxy_phase_match_rate = float(
        input_trace.loc[
            validation_mask, "event_matched_proxy_movement_phase_match"
        ].astype(float).mean()
    )
    event_alignment = pd.DataFrame(event_alignment_rows)
    validation_exit_residual = pd.to_numeric(
        event_alignment.loc[
            event_alignment["partition"] == "validation",
            "mapped_minus_sumo_exit_seconds",
        ],
        errors="coerce",
    ).dropna()
    timing_residual = input_trace[
        "sumo_minus_observed_exit_proxy_seconds"
    ].dropna()
    entry_line_alignment = _entry_alignment_metrics(input_trace)
    exit_line_alignment = _exit_line_alignment_metrics(input_trace)
    for column in (
        "signal_cycle_index",
        "signal_phase_position_seconds",
        "signal_phase",
        "signal_movement_phase_match",
        "observed_exit_proxy",
        "observed_proxy_phase_position_seconds",
        "observed_proxy_phase",
        "observed_proxy_movement_phase_match",
        "observed_alignment_rank",
        "observed_alignment_partition",
        "sumo_alignment_rank",
        "sumo_alignment_partition",
        "event_matched_exit_line",
        "event_matched_stopline_proxy",
        "event_matched_proxy_phase_position_seconds",
        "event_matched_proxy_phase",
        "event_matched_proxy_movement_phase_match",
        "sumo_minus_observed_exit_proxy_seconds",
        "observed_entry_line",
        "sumo_entry_line",
        "entry_line_residual_seconds",
        "observed_exit_line",
        "sumo_exit_line",
        "exit_line_residual_seconds",
        "observed_stopline_proxy",
        "stopline_proxy_residual_seconds",
    ):
        frame.loc[input_trace.index, column] = input_trace[column]
    frame.to_csv(args.output_dir / "vehicle_traceability.csv", index=False, encoding="utf-8-sig")
    frame[frame["extraction_status"] == "incomplete_trajectory"].to_csv(
        args.output_dir / "incomplete_trajectory_events.csv", index=False, encoding="utf-8-sig"
    )
    frame[frame["extraction_status"] == "topology_excluded"].to_csv(
        args.output_dir / "topology_excluded_events.csv", index=False, encoding="utf-8-sig"
    )

    tables = {}
    for window, suffix in ((60, "1min"), (300, "5min")):
        tables[f"video_{suffix}"] = _video_time_table(frame, window)
        tables[f"sumo_{suffix}"] = _sumo_time_table(frame, window)
        tables[f"flow_{suffix}"] = _route_flow(frame, window)
        tables[f"video_{suffix}"].to_csv(
            args.output_dir / f"time_flow_video_{suffix}.csv", index=False, encoding="utf-8-sig"
        )
        tables[f"sumo_{suffix}"].to_csv(
            args.output_dir / f"time_flow_sumo_{suffix}.csv", index=False, encoding="utf-8-sig"
        )
        tables[f"flow_{suffix}"].to_csv(
            args.output_dir / f"route_flow_{suffix}.csv", index=False, encoding="utf-8-sig"
        )

    masks = {
        "observed_with_approach": frame["approach"] != "",
        "extracted_complete": frame["extraction_status"].isin(
            ["sumo_input", "topology_excluded"]
        ),
        "sumo_input": frame["extraction_status"] == "sumo_input",
        "sumo_completed": frame["sumo_status"] == "completed",
    }
    approach = _composition(frame, "approach", masks)
    turn_masks = {key: value for key, value in masks.items() if key != "observed_with_approach"}
    turning = _composition(frame, "turning_direction", turn_masks)
    route = _composition(frame, "route", turn_masks)
    approach.to_csv(args.output_dir / "approach_composition.csv", index=False, encoding="utf-8-sig")
    turning.to_csv(args.output_dir / "turning_composition.csv", index=False, encoding="utf-8-sig")
    route.to_csv(args.output_dir / "route_composition.csv", index=False, encoding="utf-8-sig")

    totals = [
        {"stage": "real_observed_events", "count": len(frame)},
        {"stage": "extracted_complete_OD", "count": len(valid_ids)},
        {"stage": "SUMO_input", "count": len(routeable_ids)},
        {"stage": "SUMO_completed", "count": len(trips)},
    ]
    _write_csv(args.output_dir / "demand_scale.csv", totals)

    video_range = {
        "start_seconds": float(frame["observed_event_time"].astype(float).min()),
        "end_seconds": float(frame["observed_event_time"].astype(float).max()),
    }
    video_range["duration_seconds"] = video_range["end_seconds"] - video_range["start_seconds"]
    video_range["duration_hhmmss"] = "00:13:00.56"
    entry_values = pd.to_numeric(frame["entry_time"], errors="coerce")
    exit_values = pd.to_numeric(frame["exit_time"], errors="coerce")

    temporal = {}
    for suffix in ("1min", "5min"):
        table = tables[f"video_{suffix}"]
        temporal[suffix] = {
            "observed_to_extracted_tv_similarity": _tv_similarity(
                table["real_observed_count"], table["extracted_complete_count"]
            ),
            "observed_to_extracted_pearson": _pearson(
                table["real_observed_count"], table["extracted_complete_count"]
            ),
            "extracted_to_sumo_input_tv_similarity": _tv_similarity(
                table["extracted_complete_count"], table["sumo_input_count"]
            ),
            "extracted_to_sumo_input_pearson": _pearson(
                table["extracted_complete_count"], table["sumo_input_count"]
            ),
        }

    execution = {
        key: int(float(last_summary[key]))
        for key in ("loaded", "inserted", "running", "waiting", "ended", "arrived", "collisions", "teleports", "discarded")
    }
    metrics = {
        "schema_version": 1,
        "source": {
            "semantic_event_table": _display_path(args.input),
            "semantic_event_table_sha256": _sha256(args.input),
            "converter_default_copy": str(converter.DEFAULT_INPUT.relative_to(ROOT)),
            "converter_default_copy_sha256": _sha256(converter.DEFAULT_INPUT),
            "input_copies_identical": _sha256(args.input) == _sha256(converter.DEFAULT_INPUT),
            "network": _display_path(args.network),
            "signal_cycle": str(converter.DEFAULT_CYCLE.relative_to(ROOT)),
            "video_time_range": video_range,
            "entry_time_range_seconds": [float(entry_values.min()), float(entry_values.max())],
            "exit_time_range_seconds": [float(exit_values.min()), float(exit_values.max())],
            "entry_time_normalization_shift_seconds": shift,
        },
        "counts": {
            "real_observed_events": len(frame),
            "observed_with_entry_approach": int((frame["approach"] != "").sum()),
            "extracted_complete_OD": len(valid_ids),
            "incomplete_trajectory_events": len(frame) - len(valid_ids),
            "topology_excluded_complete_OD": len(valid_ids) - len(routeable_ids),
            "sumo_input_vehicles": len(routes),
            "sumo_completed_vehicles": len(trips),
        },
        "rates": {
            "complete_OD_extraction_rate": len(valid_ids) / len(frame),
            "SUMO_retention_of_complete_OD": len(routeable_ids) / len(valid_ids),
            "end_to_end_executable_share_of_observed": len(routeable_ids) / len(frame),
            "SUMO_completion_rate": len(trips) / len(routes),
        },
        "temporal_structure": temporal,
        "composition_structure": {
            "approach_extracted_to_input_tv_similarity": _composition_similarity(
                approach, "extracted_complete", "sumo_input", "approach"
            ),
            "turning_extracted_to_input_tv_similarity": _composition_similarity(
                turning, "extracted_complete", "sumo_input", "turning_direction"
            ),
            "route_extracted_to_input_tv_similarity": _composition_similarity(
                route, "extracted_complete", "sumo_input", "route"
            ),
        },
        "signal_program": signal_program,
        "signal_execution": {
            "stopline_crossing_records": int(actual_phase_informative.sum()),
            "movement_phase_match_records": int(
                input_trace.loc[
                    actual_phase_informative, "signal_movement_phase_match"
                ].astype(bool).sum()
            ),
            "movement_phase_match_rate": phase_match_rate,
            "definition": (
                "SUMO first-edge exit time is the stop-line crossing; only "
                "through and protected-left movements are phase-informative."
            ),
        },
        "video_to_sumo_alignment": {
            "status": "od_rank_event_aligned_with_held_out_validation",
            "observed_proxy": "exit_time + normalized shift at the same outbound logic line",
            "sumo_observation": "SUMO detector crossing at the same outbound logic line",
            "entry_line_records": int(input_trace["entry_line_residual_seconds"].notna().sum()),
            "exit_line_records": int(input_trace["exit_line_residual_seconds"].notna().sum()),
            "median_entry_line_residual_seconds": float(
                input_trace["entry_line_residual_seconds"].dropna().median()
            ),
            "mean_absolute_entry_line_residual_seconds": float(
                input_trace["entry_line_residual_seconds"].dropna().abs().mean()
            ),
            "p90_absolute_entry_line_residual_seconds": entry_line_alignment[
                "p90_absolute_residual_seconds"
            ],
            "median_exit_line_residual_seconds": float(
                input_trace["exit_line_residual_seconds"].dropna().median()
            ),
            "mean_absolute_exit_line_residual_seconds": float(
                input_trace["exit_line_residual_seconds"].dropna().abs().mean()
            ),
            "cumulative_exit_trend_median_residual_seconds": exit_line_alignment[
                "cumulative_trend_median_residual_seconds"
            ],
            "cumulative_exit_trend_mean_absolute_residual_seconds": exit_line_alignment[
                "cumulative_trend_mae_seconds"
            ],
            "cumulative_exit_trend_p90_absolute_residual_seconds": exit_line_alignment[
                "cumulative_trend_p90_absolute_residual_seconds"
            ],
            "stopline_proxy": "observed_exit_line - route median(SUMO exit_line - SUMO stop_line)",
            "informative_proxy_records": int(informative.sum()),
            "observed_proxy_phase_match_records": int(
                input_trace.loc[
                    informative, "observed_proxy_movement_phase_match"
                ].astype(bool).sum()
            ),
            "observed_proxy_phase_match_rate": proxy_phase_match_rate,
            "event_matching_method": (
                "minimum-L1 one-to-one matching of sorted exit-line events within "
                "each OD; alternating ranks calibrate a monotone map and the "
                "remaining ranks validate it"
            ),
            "event_matching_calibration_records": int(
                (event_alignment["partition"] == "calibration").sum()
            ),
            "event_matching_validation_records": int(
                (event_alignment["partition"] == "validation").sum()
            ),
            "event_matched_proxy_phase_match_rate": (
                matched_proxy_phase_match_rate
            ),
            "held_out_proxy_phase_match_records": int(
                input_trace.loc[
                    validation_mask,
                    "event_matched_proxy_movement_phase_match",
                ].astype(bool).sum()
            ),
            "held_out_proxy_phase_match_rate": (
                validation_proxy_phase_match_rate
            ),
            "held_out_exit_mapping_median_residual_seconds": float(
                validation_exit_residual.median()
            ),
            "held_out_exit_mapping_mean_absolute_residual_seconds": float(
                validation_exit_residual.abs().mean()
            ),
            "held_out_exit_mapping_p90_absolute_residual_seconds": float(
                validation_exit_residual.abs().quantile(0.9)
            ),
            "median_sumo_minus_observed_proxy_seconds": float(
                timing_residual.median()
            ),
            "mean_absolute_timing_residual_seconds": float(
                timing_residual.abs().mean()
            ),
            "p90_absolute_timing_residual_seconds": float(
                timing_residual.abs().quantile(0.9)
            ),
            "interpretation": (
                "The upper Fig3 panel compares raw same-location exit-line trends. "
                "The lower panel uses only held-out OD-rank events after the "
                "calibration ranks fit a monotone video-to-SUMO time map."
            ),
            "entry_line_calibration": signal_program["calibration"][
                "entry_line_calibration"
            ],
        },
        "sumo_execution": {
            **execution,
            "simulation_last_step_seconds": float(last_summary["time"]),
            "max_vehicle_arrival_seconds": float(frame["sumo_arrival"].max()),
            "seed": args.seed,
            "sumo_binary": str(sumo_path),
            "command": command,
            "stderr": run.stderr.strip(),
        },
    }
    criteria = {
        "complete_OD_to_input_retention_at_least_95pct": metrics["rates"]["SUMO_retention_of_complete_OD"] >= 0.95,
        "one_minute_temporal_similarity_at_least_95pct": temporal["1min"]["extracted_to_sumo_input_tv_similarity"] >= 0.95,
        "approach_similarity_at_least_95pct": metrics["composition_structure"]["approach_extracted_to_input_tv_similarity"] >= 0.95,
        "turning_similarity_at_least_95pct": metrics["composition_structure"]["turning_extracted_to_input_tv_similarity"] >= 0.95,
        "all_SUMO_inputs_inserted_and_completed": execution["inserted"] == len(routes) == execution["arrived"] == len(trips),
        "no_collision_teleport_or_discard": execution["collisions"] == execution["teleports"] == execution["discarded"] == 0,
        "no_SUMO_warning_or_error": not run.stderr.strip(),
        "signal_cycle_matches_source_128s": signal_program["cycle_seconds"] == 128.0,
        "all_red_phase_states_are_red": signal_program["all_red_states_valid"],
        "all_stopline_crossings_match_permitted_phase": phase_match_rate == 1.0,
    }
    metrics["claim_assessment"] = {
        "supported": all(criteria.values()),
        "criteria": criteria,
        "scope": (
            "The preservation claim applies from complete extracted OD trajectories "
            "to routeable SUMO demand. Incomplete detections and five observed U-turns "
            "without network connections are disclosed, not imputed."
        ),
        "video_to_sumo_absolute_alignment_established": (
            abs(entry_line_alignment["median_residual_seconds"]) <= 0.1
            and exit_line_alignment["cumulative_trend_mae_seconds"] <= 10.0
            and validation_proxy_phase_match_rate >= 0.8
        ),
        "alignment_limitation": (
            "Logic-line positions remain image-derived priors. Long-tail paired "
            "vehicle residuals reflect queue reordering, so the accepted phase "
            "claim is limited to held-out OD event trends rather than exact "
            "physical identity recovery."
        ),
    }
    (args.output_dir / "evidence_summary.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    metric_rows = []
    for group in ("counts", "rates", "composition_structure"):
        for metric, value in metrics[group].items():
            metric_rows.append({"group": group, "metric": metric, "value": value})
    for window, values in temporal.items():
        for metric, value in values.items():
            metric_rows.append({"group": f"temporal_{window}", "metric": metric, "value": value})
    for metric, value in execution.items():
        metric_rows.append({"group": "sumo_execution", "metric": metric, "value": value})
    for metric, value in metrics["video_to_sumo_alignment"].items():
        metric_rows.append(
            {"group": "video_to_sumo_alignment", "metric": metric, "value": value}
        )
    _write_csv(args.output_dir / "evidence_metrics.csv", metric_rows)
    _write_csv(
        args.output_dir / "validation_checks.csv",
        [
            {"check": key, "passed": value, "evidence": "evidence_summary.json"}
            for key, value in criteria.items()
        ]
        + [
            {
                "check": "video_to_sumo_absolute_alignment_established",
                "passed": metrics["claim_assessment"][
                    "video_to_sumo_absolute_alignment_established"
                ],
                "evidence": "video_to_sumo_alignment in evidence_summary.json",
            }
        ],
    )

    provenance_paths = [
        ("source_semantic_events", args.input),
        ("source_network", args.network),
        ("source_signal_cycle", converter.DEFAULT_CYCLE),
        ("conversion_code", Path(converter.__file__)),
        ("evidence_code", Path(__file__)),
    ]
    provenance_paths.extend(
        ("conversion_artifact", path)
        for path in sorted(args.conversion_dir.glob("xiasha1_sumo_*"))
        if path.is_file()
    )
    _write_csv(
        args.output_dir / "source_and_conversion_provenance.csv",
        [
            {
                "role": role,
                "path": _display_path(path),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for role, path in provenance_paths
        ],
    )

    _plot_lines(
        tables["video_1min"],
        ["real_observed_count", "extracted_complete_count", "sumo_input_count"],
        ["Real observed", "Extracted complete OD", "SUMO input"],
        "Observed demand preservation (1-minute windows)",
        args.output_dir / "figure_01_time_flow_video_1min",
    )
    _plot_lines(
        tables["video_5min"],
        ["real_observed_count", "extracted_complete_count", "sumo_input_count"],
        ["Real observed", "Extracted complete OD", "SUMO input"],
        "Observed demand preservation (5-minute windows)",
        args.output_dir / "figure_02_time_flow_video_5min",
    )
    _plot_signal_diagnostics(
        frame,
        signal_program,
        shift,
        args.output_dir / "figure_03_signal_aligned_execution",
    )
    offset_scan = pd.read_csv(
        args.conversion_dir / "xiasha1_sumo_signal_offset_scan.csv"
    )
    _plot_offset_scan(
        offset_scan,
        signal_program["offset_seconds"],
        args.output_dir / "figure_08_signal_offset_calibration",
        signal_program["calibration"].get("calibration_tied_best_min_seconds"),
        signal_program["calibration"].get("calibration_tied_best_max_seconds"),
    )
    _plot_composition(
        approach, "approach", "Approach composition", args.output_dir / "figure_04_approach_composition"
    )
    _plot_composition(
        turning, "turning_direction", "Turning composition", args.output_dir / "figure_05_turning_composition"
    )
    _plot_composition(
        route, "route", "Route composition", args.output_dir / "figure_06_route_composition"
    )
    _plot_totals(totals, args.output_dir / "figure_07_demand_scale")

    manifest_rows = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "manifest.csv":
            manifest_rows.append(
                {
                    "artifact": path.name,
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )
    _write_csv(args.output_dir / "manifest.csv", manifest_rows)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
