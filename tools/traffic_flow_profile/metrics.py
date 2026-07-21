import math
from collections import Counter

from .models import APPROACHES, MOVEMENTS, SumoScenario


def _empty_movement_matrix():
    return {
        approach: {movement: 0 for movement in MOVEMENTS}
        for approach in APPROACHES
    }


def calculate_metrics(scenario: SumoScenario):
    total = len(scenario.vehicles)
    duration = scenario.duration
    five_minute_bins = int(math.ceil(duration / 300.0))
    fifteen_minute_bins = int(math.ceil(duration / 900.0))
    temporal_total_5min = [0] * five_minute_bins
    temporal_approach_5min = {
        approach: [0] * five_minute_bins for approach in APPROACHES
    }
    temporal_total_15min = [0] * fifteen_minute_bins
    approach_counts = Counter({approach: 0 for approach in APPROACHES})
    movement_counts = Counter({movement: 0 for movement in MOVEMENTS})
    movement_matrix = _empty_movement_matrix()
    unsupported_movements = Counter()

    for vehicle in scenario.vehicles:
        relative_depart = vehicle.depart - scenario.begin
        five_index = min(five_minute_bins - 1, int(relative_depart // 300))
        fifteen_index = min(fifteen_minute_bins - 1, int(relative_depart // 900))
        temporal_total_5min[five_index] += 1
        temporal_approach_5min[vehicle.approach][five_index] += 1
        temporal_total_15min[fifteen_index] += 1
        approach_counts[vehicle.approach] += 1
        if vehicle.movement in MOVEMENTS:
            movement_counts[vehicle.movement] += 1
            movement_matrix[vehicle.approach][vehicle.movement] += 1
        else:
            unsupported_movements[vehicle.movement] += 1

    if unsupported_movements:
        raise ValueError(f"Unsupported movements detected: {dict(unsupported_movements)}")

    peak_count = max(temporal_total_15min)
    peak_index = temporal_total_15min.index(peak_count)
    phf = total / (4.0 * peak_count)
    approach_shares = {
        approach: approach_counts[approach] / total for approach in APPROACHES
    }
    movement_shares = {
        movement: movement_counts[movement] / total for movement in MOVEMENTS
    }
    dominant_approach = max(APPROACHES, key=lambda item: approach_counts[item])
    ew_share = approach_shares["E"] + approach_shares["W"]
    ns_share = approach_shares["N"] + approach_shares["S"]

    per_approach_movement_shares = {}
    for approach in APPROACHES:
        approach_total = approach_counts[approach]
        per_approach_movement_shares[approach] = {
            movement: (
                movement_matrix[approach][movement] / approach_total
                if approach_total
                else 0.0
            )
            for movement in MOVEMENTS
        }

    first_half = sum(temporal_total_5min[: five_minute_bins // 2])
    second_half = total - first_half
    half_hour_change_pct = (second_half - first_half) / total * 100.0

    return {
        "schema_version": "1.0",
        "scenario": {
            "id": scenario.scenario_id,
            "simulator": "sumo",
            "begin_seconds": scenario.begin,
            "end_seconds": scenario.end,
            "duration_seconds": duration,
            "signal_junction_id": scenario.signal_junction_id,
            "approach_edges": scenario.approach_edges,
        },
        "standard_metrics": {
            "vehicle_count": total,
            "hourly_volume_vph": total * 3600.0 / duration,
            "peak_15min_vehicle_count": peak_count,
            "peak_15min_equivalent_vph": peak_count * 4.0,
            "peak_15min_start_seconds": scenario.begin + peak_index * 900.0,
            "peak_15min_end_seconds": scenario.begin + (peak_index + 1) * 900.0,
            "peak_hour_factor": phf,
            "approach_vehicle_count": dict(approach_counts),
            "approach_share": approach_shares,
            "dominant_approach": dominant_approach,
            "dominant_approach_share": approach_shares[dominant_approach],
            "east_west_share": ew_share,
            "north_south_share": ns_share,
            "movement_vehicle_count": dict(movement_counts),
            "movement_share": movement_shares,
            "movement_matrix": movement_matrix,
            "per_approach_movement_share": per_approach_movement_shares,
        },
        "profile_metrics": {
            "half_hour_change_pct": half_hour_change_pct,
            "temporal_total_5min": temporal_total_5min,
            "temporal_approach_5min": temporal_approach_5min,
            "temporal_total_15min": temporal_total_15min,
        },
        "inputs": {
            "sumocfg": str(scenario.sumocfg_path),
            "net": str(scenario.net_path),
            "routes": [str(path) for path in scenario.route_paths],
            "sha256": scenario.input_sha256,
        },
    }
