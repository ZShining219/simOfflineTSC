"""Extract and plot unified S2 MaxPressure/FixedTime decision time series.

The source is the immutable Plan 1 evaluation package.  This utility does not
start SUMO or alter upstream records; it extracts the two S2 controllers,
validates the five traffic seeds and 360 ten-second decisions, and writes a
standalone derived analysis package.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_SOURCE = Path(
    "output_data/evaluations/plan1/"
    "plan1_best_checkpoint_reevaluation_v1_20260723/records.jsonl"
)
DEFAULT_OUTPUT = Path(
    "output_data/analysis/plan1/s2_maxpressure_fixedtime_timeseries_20260804"
)
CONTROLLERS = {
    "maxpressure_sumohz1x1": ("MaxPressure", "#0072B2"),
    "fixedtime_sumohz1x1": ("FixedTime", "#D55E00"),
}
SEEDS = (10000, 10001, 10002, 10003, 10004)
EXPECTED_STEPS = tuple(range(1, 361))
METRICS = (
    ("queue_network_sum", "Network queue (vehicles)", "lower"),
    ("delay_network_weighted_mean", "Weighted delay", "lower"),
    ("throughput_cumulative", "Cumulative throughput (vehicles)", "higher"),
    ("controller_reward_mean", "Controller reward", "higher"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_records(source: Path) -> list[dict]:
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON at {source}:{line_number}") from error
            if (
                record.get("record_type") == "DECISION_METRICS"
                and record.get("controller_id") in CONTROLLERS
            ):
                rows.append(record)
    return rows


def normalise(records: list[dict]) -> list[dict]:
    rows = []
    for record in records:
        controller_id = record["controller_id"]
        method, _ = CONTROLLERS[controller_id]
        required = (
            "evaluation_seed",
            "decision_step",
            "simulation_time_seconds",
            "queue_network_sum",
            "queue_network_mean",
            "delay_network_weighted_mean",
            "throughput_interval",
            "throughput_cumulative",
            "controller_reward_agents",
        )
        missing = [field for field in required if field not in record]
        if missing:
            raise ValueError(f"{controller_id} record missing {missing}")
        rewards = np.asarray(record["controller_reward_agents"], dtype=float)
        if rewards.size == 0 or not np.all(np.isfinite(rewards)):
            raise ValueError(f"{controller_id} has invalid controller reward")
        rows.append(
            {
                "method": method,
                "controller_id": controller_id,
                "network": record["network"],
                "training_seed": int(record["training_seed"]),
                "evaluation_seed": int(record["evaluation_seed"]),
                "decision_step": int(record["decision_step"]),
                "simulation_time_seconds": float(record["simulation_time_seconds"]),
                "queue_network_sum": float(record["queue_network_sum"]),
                "queue_network_mean": float(record["queue_network_mean"]),
                "delay_network_weighted_mean": float(
                    record["delay_network_weighted_mean"]
                ),
                "throughput_interval": int(record["throughput_interval"]),
                "throughput_cumulative": int(record["throughput_cumulative"]),
                "controller_reward_mean": float(rewards.mean()),
                "action": int(np.asarray(record.get("actions", [0])).reshape(-1)[0]),
            }
        )
    return sorted(rows, key=lambda row: (row["method"], row["evaluation_seed"], row["decision_step"]))


def validate(rows: list[dict]) -> None:
    if len(rows) != 2 * len(SEEDS) * len(EXPECTED_STEPS):
        raise ValueError(f"Expected 3600 decision rows, found {len(rows)}")
    for method in ("MaxPressure", "FixedTime"):
        for seed in SEEDS:
            group = [
                row
                for row in rows
                if row["method"] == method and row["evaluation_seed"] == seed
            ]
            steps = [row["decision_step"] for row in group]
            if tuple(steps) != EXPECTED_STEPS:
                raise ValueError(f"{method} seed {seed} does not contain steps 1..360")
            if any(row["network"] != "sumohz1x1" for row in group):
                raise ValueError(f"{method} seed {seed} is not S2/sumohz1x1")
            expected_times = [step * 10.0 for step in EXPECTED_STEPS]
            actual_times = [row["simulation_time_seconds"] for row in group]
            if not np.allclose(actual_times, expected_times):
                raise ValueError(f"{method} seed {seed} is not on a 10-second grid")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict]) -> list[dict]:
    output = []
    for method in ("MaxPressure", "FixedTime"):
        method_rows = [row for row in rows if row["method"] == method]
        for step in EXPECTED_STEPS:
            selected = [row for row in method_rows if row["decision_step"] == step]
            result = {
                "method": method,
                "decision_step": step,
                "simulation_time_seconds": step * 10,
                "n_seeds": len(selected),
            }
            for field, _, _ in METRICS:
                values = np.asarray([row[field] for row in selected], dtype=float)
                result[f"{field}_mean"] = float(values.mean())
                result[f"{field}_sd"] = float(values.std(ddof=1))
                result[f"{field}_min"] = float(values.min())
                result[f"{field}_max"] = float(values.max())
                result[f"{field}_ci95_low"] = float(
                    values.mean() - 1.96 * values.std(ddof=1) / math.sqrt(len(values))
                )
                result[f"{field}_ci95_high"] = float(
                    values.mean() + 1.96 * values.std(ddof=1) / math.sqrt(len(values))
                )
            output.append(result)
    return output


def final_summary(rows: list[dict]) -> list[dict]:
    output = []
    for method in ("MaxPressure", "FixedTime"):
        for seed in SEEDS:
            selected = [
                row
                for row in rows
                if row["method"] == method and row["evaluation_seed"] == seed
            ]
            final = selected[-1]
            output.append(
                {
                    "method": method,
                    "evaluation_seed": seed,
                    "final_decision_step": final["decision_step"],
                    "final_simulation_time_seconds": final["simulation_time_seconds"],
                    "final_queue_network_sum": final["queue_network_sum"],
                    "final_queue_network_mean": final["queue_network_mean"],
                    "final_delay_network_weighted_mean": final["delay_network_weighted_mean"],
                    "final_throughput_cumulative": final["throughput_cumulative"],
                    "final_controller_reward_mean": final["controller_reward_mean"],
                }
            )
    return output


def write_plot(rows: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, constrained_layout=True)
    for axis, (field, label, direction) in zip(axes.flat, METRICS):
        for method, color in (("MaxPressure", "#0072B2"), ("FixedTime", "#D55E00")):
            method_rows = [row for row in rows if row["method"] == method]
            by_seed = {seed: [row for row in method_rows if row["evaluation_seed"] == seed] for seed in SEEDS}
            time = np.asarray([row["simulation_time_seconds"] for row in by_seed[SEEDS[0]]], dtype=float)
            values = np.asarray([[row[field] for row in by_seed[seed]] for seed in SEEDS], dtype=float)
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            for seed, series in zip(SEEDS, values):
                axis.plot(time, series, color=color, alpha=0.16, linewidth=0.8)
            axis.plot(time, mean, color=color, linewidth=2.2, label=method)
            axis.fill_between(time, mean - sd, mean + sd, color=color, alpha=0.12)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.set_title(f"{label} — seed mean ± 1 SD")
        if direction == "lower":
            axis.text(0.99, 0.04, "lower is better", transform=axis.transAxes, ha="right", fontsize=9, color="#555555")
        else:
            axis.text(0.99, 0.04, "higher is better", transform=axis.transAxes, ha="right", fontsize=9, color="#555555")
    for axis in axes[1, :]:
        axis.set_xlabel("Simulation time (s)")
    axes[0, 0].legend(loc="upper left", frameon=True)
    fig.suptitle("S2 MaxPressure vs FixedTime: unified decision-level dynamics", fontsize=16)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def write_metric_panels(rows: list[dict], output_dir: Path) -> None:
    for field, label, _ in METRICS:
        fig, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
        for method, color in (("MaxPressure", "#0072B2"), ("FixedTime", "#D55E00")):
            method_rows = [row for row in rows if row["method"] == method]
            values = np.asarray([[row[field] for row in method_rows if row["evaluation_seed"] == seed] for seed in SEEDS], dtype=float)
            time = np.arange(1, 361, dtype=float) * 10.0
            mean = values.mean(axis=0)
            sd = values.std(axis=0, ddof=1)
            axis.plot(time, mean, color=color, linewidth=2.2, label=method)
            axis.fill_between(time, mean - sd, mean + sd, color=color, alpha=0.16)
        axis.set_title(f"S2 {label}: five traffic seeds")
        axis.set_xlabel("Simulation time (s)")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.25)
        axis.legend()
        stem = field.replace("_network_weighted_mean", "").replace("_network_sum", "").replace("_", "-")
        fig.savefig(output_dir / f"s2_{stem}_timeseries.png", dpi=220)
        fig.savefig(output_dir / f"s2_{stem}_timeseries.pdf")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = normalise(read_records(source))
    validate(rows)
    aggregate_rows = aggregate(rows)
    summary_rows = final_summary(rows)
    write_csv(output / "s2_baseline_decision_timeseries.csv", rows)
    write_csv(output / "s2_baseline_time_aggregate.csv", aggregate_rows)
    write_csv(output / "s2_baseline_final_seed_summary.csv", summary_rows)
    write_plot(rows, output / "s2_maxpressure_fixedtime_timeseries.png")
    write_metric_panels(rows, output)
    manifest = {
        "schema_version": 1,
        "status": "completed",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_records": str(source),
        "source_records_sha256": sha256_file(source),
        "scene": "S2",
        "network": "sumohz1x1",
        "controllers": ["MaxPressure", "FixedTime"],
        "evaluation_traffic_seeds": list(SEEDS),
        "simulation_steps": 3600,
        "decision_steps_per_seed": 360,
        "action_interval_seconds": 10,
        "decision_row_count": len(rows),
        "aggregation": "mean, sample SD, min/max and normal-approximate 95% CI over five traffic seeds at each decision step",
        "source_reuse": "Extracted from immutable Plan 1 evaluation records; no SUMO rerun",
        "files": {
            name: sha256_file(output / name)
            for name in (
                "s2_baseline_decision_timeseries.csv",
                "s2_baseline_time_aggregate.csv",
                "s2_baseline_final_seed_summary.csv",
                "s2_maxpressure_fixedtime_timeseries.png",
            )
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(output), "rows": len(rows), "source_sha256": manifest["source_records_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
