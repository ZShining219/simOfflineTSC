import math
from collections import Counter

from .models import MOVEMENTS, NetworkDemandScenario, SumoScenario


def _cardinal_from_points(source, center):
    dx = source[0] - center[0]
    dy = source[1] - center[1]
    if abs(dx) >= abs(dy):
        return "W" if dx < 0 else "E"
    return "S" if dy < 0 else "N"


def _network_movement(incoming_edge, outgoing_edge, signal_id, scenario):
    incoming_from, _ = scenario.edges[incoming_edge]
    _, outgoing_to = scenario.edges[outgoing_edge]
    center_x, center_y = scenario.junctions[signal_id]
    source_x, source_y = scenario.junctions[incoming_from]
    target_x, target_y = scenario.junctions[outgoing_to]
    in_x, in_y = center_x - source_x, center_y - source_y
    out_x, out_y = target_x - center_x, target_y - center_y
    in_norm = math.hypot(in_x, in_y)
    out_norm = math.hypot(out_x, out_y)
    if in_norm == 0 or out_norm == 0:
        return "unknown"
    dot = (in_x * out_x + in_y * out_y) / (in_norm * out_norm)
    cross = (in_x * out_y - in_y * out_x) / (in_norm * out_norm)
    if dot < -0.5:
        return "u_turn"
    if abs(cross) < 0.5 and dot > 0:
        return "through"
    return "left" if cross > 0 else "right"


def _temporal_profile(departures, begin, end):
    duration = end - begin
    five_minute_bins = max(1, int(math.ceil(duration / 300.0)))
    fifteen_minute_bins = max(1, int(math.ceil(duration / 900.0)))
    temporal_total_5min = [0] * five_minute_bins
    temporal_total_15min = [0] * fifteen_minute_bins
    for depart in departures:
        relative_depart = depart - begin
        five_index = min(five_minute_bins - 1, max(0, int(relative_depart // 300)))
        fifteen_index = min(
            fifteen_minute_bins - 1, max(0, int(relative_depart // 900))
        )
        temporal_total_5min[five_index] += 1
        temporal_total_15min[fifteen_index] += 1
    return temporal_total_5min, temporal_total_15min


def _common_metrics(
    total, begin, end, departures, temporal_total_5min, temporal_total_15min
):
    duration = end - begin
    hourly_volume = total * 3600.0 / duration
    peak_count = max(temporal_total_15min, default=0)
    peak_index = (
        temporal_total_15min.index(peak_count) if temporal_total_15min else 0
    )
    interval_count = max(1, len(temporal_total_15min))
    phf = total / (interval_count * peak_count) if peak_count else 0.0
    midpoint = begin + duration / 2.0
    first_half = sum(depart < midpoint for depart in departures)
    second_half = total - first_half
    half_window_change_pct = (
        (second_half - first_half) / total * 100.0 if total else 0.0
    )
    return {
        "vehicle_count": total,
        "hourly_volume_vph": hourly_volume,
        "peak_15min_vehicle_count": peak_count,
        "peak_15min_equivalent_vph": peak_count * 4.0,
        "peak_15min_start_seconds": begin + peak_index * 900.0,
        "peak_15min_end_seconds": min(end, begin + (peak_index + 1) * 900.0),
        "peak_hour_factor": phf,
    }, {
        "half_hour_change_pct": half_window_change_pct,
        "temporal_total_5min": temporal_total_5min,
        "temporal_total_15min": temporal_total_15min,
    }


def calculate_metrics(scenario: SumoScenario):
    approaches = scenario.approach_order
    extra_movements = sorted(
        {vehicle.movement for vehicle in scenario.vehicles} - set(MOVEMENTS)
    )
    movements = tuple(MOVEMENTS) + tuple(extra_movements)
    total = len(scenario.vehicles)
    departures = [vehicle.depart for vehicle in scenario.vehicles]
    temporal_total_5min, temporal_total_15min = _temporal_profile(
        departures, scenario.begin, scenario.end
    )
    temporal_approach_5min = {
        approach: [0] * len(temporal_total_5min) for approach in approaches
    }
    approach_counts = Counter({approach: 0 for approach in approaches})
    movement_counts = Counter({movement: 0 for movement in movements})
    movement_matrix = {
        approach: {movement: 0 for movement in movements}
        for approach in approaches
    }

    for vehicle in scenario.vehicles:
        relative_depart = vehicle.depart - scenario.begin
        five_index = min(
            len(temporal_total_5min) - 1, max(0, int(relative_depart // 300))
        )
        temporal_approach_5min[vehicle.approach][five_index] += 1
        approach_counts[vehicle.approach] += 1
        movement_counts[vehicle.movement] += 1
        movement_matrix[vehicle.approach][vehicle.movement] += 1

    common_standard, common_profile = _common_metrics(
        total,
        scenario.begin,
        scenario.end,
        departures,
        temporal_total_5min,
        temporal_total_15min,
    )
    approach_shares = {
        approach: approach_counts[approach] / total if total else 0.0
        for approach in approaches
    }
    movement_shares = {
        movement: movement_counts[movement] / total if total else 0.0
        for movement in movements
    }
    dominant_approach = (
        max(approaches, key=lambda item: approach_counts[item])
        if approaches
        else None
    )
    per_approach_movement_shares = {}
    for approach in approaches:
        approach_total = approach_counts[approach]
        per_approach_movement_shares[approach] = {
            movement: (
                movement_matrix[approach][movement] / approach_total
                if approach_total
                else 0.0
            )
            for movement in movements
        }

    common_standard.update(
        {
            "approach_order": list(approaches),
            "movement_order": list(movements),
            "approach_vehicle_count": dict(approach_counts),
            "approach_share": approach_shares,
            "dominant_approach": dominant_approach,
            "dominant_approach_share": (
                approach_shares[dominant_approach] if dominant_approach else 0.0
            ),
            "east_west_share": sum(
                approach_shares.get(item, 0.0) for item in ("E", "W")
            ),
            "north_south_share": sum(
                approach_shares.get(item, 0.0) for item in ("N", "S")
            ),
            "movement_vehicle_count": dict(movement_counts),
            "movement_share": movement_shares,
            "movement_matrix": movement_matrix,
            "per_approach_movement_share": per_approach_movement_shares,
        }
    )
    common_profile["temporal_approach_5min"] = temporal_approach_5min
    return {
        "schema_version": "2.0",
        "scenario": {
            "id": scenario.scenario_id,
            "package_id": scenario.package_id,
            "scope": "signal",
            "count_semantics": "route_crossing_occurrences",
            "event_time_basis": "vehicle_depart",
            "simulator": "sumo",
            "begin_seconds": scenario.begin,
            "end_seconds": scenario.end,
            "duration_seconds": scenario.duration,
            "signal_junction_id": scenario.signal_junction_id,
            "approach_edges": scenario.approach_edges,
        },
        "standard_metrics": common_standard,
        "profile_metrics": common_profile,
        "inputs": {
            "sumocfg": str(scenario.sumocfg_path),
            "net": str(scenario.net_path),
            "routes": [str(path) for path in scenario.route_paths],
            "additional": [str(path) for path in scenario.additional_paths],
            "sha256": scenario.input_sha256,
        },
    }


def calculate_network_metrics(scenario: NetworkDemandScenario):
    total = len(scenario.vehicles)
    departures = [vehicle.depart for vehicle in scenario.vehicles]
    temporal_total_5min, temporal_total_15min = _temporal_profile(
        departures, scenario.begin, scenario.end
    )
    standard, profile = _common_metrics(
        total,
        scenario.begin,
        scenario.end,
        departures,
        temporal_total_5min,
        temporal_total_15min,
    )
    center_nodes = [
        scenario.junctions[signal_id]
        for signal_id in scenario.signal_ids
        if signal_id in scenario.junctions
    ] or list(scenario.junctions.values())
    center = (
        sum(point[0] for point in center_nodes) / len(center_nodes),
        sum(point[1] for point in center_nodes) / len(center_nodes),
    )
    origin_order = ("W", "S", "E", "N")
    origin_counts = Counter({approach: 0 for approach in origin_order})
    unclassified_origins = 0
    crossing_events = []
    signal_set = set(scenario.signal_ids)
    for vehicle in scenario.vehicles:
        if vehicle.route_edges and vehicle.route_edges[0] in scenario.edges:
            source_id = scenario.edges[vehicle.route_edges[0]][0]
            if source_id in scenario.junctions:
                origin_counts[
                    _cardinal_from_points(scenario.junctions[source_id], center)
                ] += 1
            else:
                unclassified_origins += 1
        else:
            unclassified_origins += 1
        for incoming_edge, outgoing_edge in zip(
            vehicle.route_edges, vehicle.route_edges[1:]
        ):
            if incoming_edge not in scenario.edges or outgoing_edge not in scenario.edges:
                continue
            signal_id = scenario.edges[incoming_edge][1]
            if signal_id != scenario.edges[outgoing_edge][0] or signal_id not in signal_set:
                continue
            crossing_events.append(
                (
                    signal_id,
                    _network_movement(
                        incoming_edge, outgoing_edge, signal_id, scenario
                    ),
                )
            )

    extra_movements = sorted(
        {movement for _, movement in crossing_events} - set(MOVEMENTS)
    )
    movement_order = tuple(MOVEMENTS) + tuple(extra_movements)
    crossing_by_signal = Counter({signal_id: 0 for signal_id in scenario.signal_ids})
    crossing_by_movement = Counter({movement: 0 for movement in movement_order})
    crossing_matrix = {
        signal_id: {movement: 0 for movement in movement_order}
        for signal_id in scenario.signal_ids
    }
    for signal_id, movement in crossing_events:
        crossing_by_signal[signal_id] += 1
        crossing_by_movement[movement] += 1
        crossing_matrix[signal_id][movement] += 1
    crossing_total = len(crossing_events)
    standard.update(
        {
            "route_origin_approach_order": list(origin_order),
            "route_origin_approach_count": dict(origin_counts),
            "route_origin_approach_share": {
                approach: origin_counts[approach] / total if total else 0.0
                for approach in origin_order
            },
            "route_origin_unclassified_count": unclassified_origins,
            "signal_order": list(scenario.signal_ids),
            "signal_crossing_count": crossing_total,
            "signal_crossings_per_vehicle": crossing_total / total if total else 0.0,
            "signal_crossing_count_by_junction": dict(crossing_by_signal),
            "signal_crossing_movement_order": list(movement_order),
            "signal_crossing_movement_count": dict(crossing_by_movement),
            "signal_crossing_movement_share": {
                movement: (
                    crossing_by_movement[movement] / crossing_total
                    if crossing_total
                    else 0.0
                )
                for movement in movement_order
            },
            "signal_crossing_movement_matrix": crossing_matrix,
        }
    )
    return {
        "schema_version": "2.0",
        "scenario": {
            "id": scenario.scenario_id,
            "package_id": scenario.package_id,
            "scope": "network",
            "count_semantics": "unique_demand_elements",
            "event_time_basis": "vehicle_depart",
            "simulator": "sumo",
            "begin_seconds": scenario.begin,
            "end_seconds": scenario.end,
            "duration_seconds": scenario.duration,
            "signal_junction_id": None,
            "signal_junction_count": len(scenario.signal_ids),
        },
        "standard_metrics": standard,
        "profile_metrics": profile,
        "inputs": {
            "sumocfg": str(scenario.sumocfg_path),
            "net": str(scenario.net_path),
            "routes": [str(path) for path in scenario.route_paths],
            "additional": [str(path) for path in scenario.additional_paths],
            "sha256": scenario.input_sha256,
        },
    }
