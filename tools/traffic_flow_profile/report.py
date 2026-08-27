import csv
import json
from pathlib import Path

def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_temporal_csv(path: Path, metrics):
    profile = metrics["profile_metrics"]
    begin = metrics["scenario"]["begin_seconds"]
    approaches = metrics["standard_metrics"].get("approach_order", [])
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["start_seconds", "end_seconds", "total", *approaches])
        for index, total in enumerate(profile["temporal_total_5min"]):
            writer.writerow(
                [
                    begin + index * 300,
                    begin + (index + 1) * 300,
                    total,
                    *[
                        profile["temporal_approach_5min"][item][index]
                        for item in approaches
                    ],
                ]
            )


def _write_movement_csv(path: Path, metrics):
    standard = metrics["standard_metrics"]
    approaches = standard["approach_order"]
    movements = standard["movement_order"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "approach",
                *[f"{item}_count" for item in movements],
                "total_count",
                *[f"{item}_share" for item in movements],
            ]
        )
        for approach in approaches:
            counts = standard["movement_matrix"][approach]
            shares = standard["per_approach_movement_share"][approach]
            writer.writerow(
                [
                    approach,
                    *[counts[item] for item in movements],
                    standard["approach_vehicle_count"][approach],
                    *[f"{shares[item]:.8f}" for item in movements],
                ]
            )


def _write_network_origin_csv(path: Path, metrics):
    standard = metrics["standard_metrics"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["origin_direction", "vehicle_count", "vehicle_share"])
        for approach in standard["route_origin_approach_order"]:
            writer.writerow(
                [
                    approach,
                    standard["route_origin_approach_count"][approach],
                    f"{standard['route_origin_approach_share'][approach]:.8f}",
                ]
            )
        writer.writerow(
            [
                "unclassified",
                standard["route_origin_unclassified_count"],
                "",
            ]
        )


def _write_network_crossing_csv(path: Path, metrics):
    standard = metrics["standard_metrics"]
    movements = standard["signal_crossing_movement_order"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            ["signal_junction", *[f"{item}_count" for item in movements], "total_count"]
        )
        for signal_id in standard["signal_order"]:
            counts = standard["signal_crossing_movement_matrix"][signal_id]
            writer.writerow(
                [
                    signal_id,
                    *[counts[item] for item in movements],
                    standard["signal_crossing_count_by_junction"][signal_id],
                ]
            )


def write_report(metrics, output_root, dpi: int = 160, plot_limits=None):
    from .plotting import render_profile

    output_dir = Path(output_root).resolve() / metrics["scenario"]["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "metrics.json",
        "temporal_profile": output_dir / "temporal_profile.csv",
        "profile_figure": output_dir / "scenario_profile.png",
    }
    if metrics["scenario"].get("scope", "signal") == "signal":
        paths["movement_matrix"] = output_dir / "movement_matrix.csv"
    else:
        paths["route_origins"] = output_dir / "route_origins.csv"
        paths["signal_crossings"] = output_dir / "signal_crossings.csv"
    _write_json(paths["metrics"], metrics)
    _write_temporal_csv(paths["temporal_profile"], metrics)
    if "movement_matrix" in paths:
        _write_movement_csv(paths["movement_matrix"], metrics)
    if "route_origins" in paths:
        _write_network_origin_csv(paths["route_origins"], metrics)
        _write_network_crossing_csv(paths["signal_crossings"], metrics)
    render_profile(metrics, paths["profile_figure"], dpi=dpi, plot_limits=plot_limits)
    return output_dir, paths
