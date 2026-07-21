import csv
import json
from pathlib import Path

from .models import APPROACHES, MOVEMENTS


def _write_json(path: Path, payload):
    with path.open("w", encoding="utf-8") as file_obj:
        json.dump(payload, file_obj, ensure_ascii=False, indent=2, sort_keys=True)
        file_obj.write("\n")


def _write_temporal_csv(path: Path, metrics):
    profile = metrics["profile_metrics"]
    begin = metrics["scenario"]["begin_seconds"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(["start_seconds", "end_seconds", "total", *APPROACHES])
        for index, total in enumerate(profile["temporal_total_5min"]):
            writer.writerow(
                [
                    begin + index * 300,
                    begin + (index + 1) * 300,
                    total,
                    *[profile["temporal_approach_5min"][item][index] for item in APPROACHES],
                ]
            )


def _write_movement_csv(path: Path, metrics):
    standard = metrics["standard_metrics"]
    with path.open("w", encoding="utf-8", newline="") as file_obj:
        writer = csv.writer(file_obj)
        writer.writerow(
            [
                "approach",
                *[f"{item}_count" for item in MOVEMENTS],
                "total_count",
                *[f"{item}_share" for item in MOVEMENTS],
            ]
        )
        for approach in APPROACHES:
            counts = standard["movement_matrix"][approach]
            shares = standard["per_approach_movement_share"][approach]
            writer.writerow(
                [
                    approach,
                    *[counts[item] for item in MOVEMENTS],
                    standard["approach_vehicle_count"][approach],
                    *[f"{shares[item]:.8f}" for item in MOVEMENTS],
                ]
            )


def write_report(metrics, output_root, dpi: int = 160, plot_limits=None):
    from .plotting import render_profile

    output_dir = Path(output_root).resolve() / metrics["scenario"]["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": output_dir / "metrics.json",
        "temporal_profile": output_dir / "temporal_profile.csv",
        "movement_matrix": output_dir / "movement_matrix.csv",
        "profile_figure": output_dir / "scenario_profile.png",
    }
    _write_json(paths["metrics"], metrics)
    _write_temporal_csv(paths["temporal_profile"], metrics)
    _write_movement_csv(paths["movement_matrix"], metrics)
    render_profile(metrics, paths["profile_figure"], dpi=dpi, plot_limits=plot_limits)
    return output_dir, paths
