"""Audit Stage 0 resource waves and select a fixed formal episode budget."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


METRICS = ("travel_time", "queue", "delay", "throughput")
LOWER_IS_BETTER = {"travel_time", "queue", "delay"}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def relative_change(before, after):
    scale = max(abs(before), 1e-9)
    return abs(after - before) / scale


def stable_endpoint(records, endpoint, window, tolerance):
    previous = records[endpoint - 2 * window:endpoint - window]
    current = records[endpoint - window:endpoint]
    if len(previous) != window or len(current) != window:
        return False, {}
    changes = {}
    for metric in METRICS:
        left = statistics.mean(float(item[metric]) for item in previous)
        right = statistics.mean(float(item[metric]) for item in current)
        changes[metric] = relative_change(left, right)
    return all(value <= tolerance for value in changes.values()), changes


def analyze_curve(run_dir, window=20, tolerance=0.10, consecutive=3):
    run_dir = Path(run_dir)
    records = [item for item in read_jsonl(run_dir / "metrics/records.jsonl")
               if item.get("record_type") == "TRAIN"]
    records.sort(key=lambda item: int(item["episode"]))
    if not records:
        raise ValueError(f"No TRAIN records: {run_dir}")
    for item in records:
        for metric in METRICS:
            if not finite(item.get(metric)):
                raise ValueError(f"Non-finite {metric} in {run_dir}")
    endpoints = []
    for endpoint in range(2 * window, len(records) + 1):
        stable, changes = stable_endpoint(records, endpoint, window, tolerance)
        endpoints.append({
            "endpoint_episode": endpoint, "stable": stable,
            "relative_changes": changes,
        })
    stable_episode = None
    for index in range(0, len(endpoints) - consecutive + 1):
        block = endpoints[index:index + consecutive]
        if all(item["stable"] for item in block):
            stable_episode = block[-1]["endpoint_episode"]
            break
    tail = records[-window:]
    return {
        "run_dir": str(run_dir.resolve()),
        "training_records": len(records),
        "criterion": {
            "metrics": list(METRICS), "window_episodes": window,
            "relative_change_tolerance": tolerance,
            "consecutive_endpoints": consecutive,
            "definition": (
                "two adjacent rolling windows must differ by at most the tolerance "
                "for all four metrics at consecutive episode endpoints"),
        },
        "stable_episode": stable_episode,
        "tail_means": {
            metric: statistics.mean(float(item[metric]) for item in tail)
            for metric in METRICS
        },
        "endpoint_audit": endpoints,
    }


def resource_wave(state_dir):
    state_dir = Path(state_dir)
    tasks = read_json(state_dir / "run_manifest.json")["tasks"]
    samples = read_jsonl(state_dir / "resource_samples.jsonl")
    wall_times = [float(task["wall_time_seconds"]) for task in tasks]
    total_episodes = sum(int(task["episode_budget"]) for task in tasks)
    wave_wall = max(wall_times)
    return {
        "state_dir": str(state_dir.resolve()),
        "workers": len(tasks),
        "task_statuses": {task["run_id"]: task["status"] for task in tasks},
        "wall_time_seconds": {
            "min": min(wall_times), "mean": statistics.mean(wall_times),
            "max": wave_wall,
        },
        "episode_throughput_per_minute": total_episodes / wave_wall * 60.0,
        "cpu_percent_mean": statistics.mean(
            float(item["cpu_percent"]) for item in samples),
        "cpu_percent_peak": max(float(item["cpu_percent"]) for item in samples),
        "run_rss_bytes_peak": max(
            int(item.get("run_process_rss_bytes", 0)) for item in samples),
        "memory_percent_peak": max(float(item["memory_percent"]) for item in samples),
        "swap_growth_bytes_peak": max(int(item["swap_growth_bytes"]) for item in samples),
        "disk_write_bytes_per_second_peak": max(
            float(item.get("disk_write_bytes_per_second", 0)) for item in samples),
    }


def round_up_25(value):
    return int(math.ceil(value / 25.0) * 25)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resource-root",
                        default="artifacts/arterial_experiments/stage0")
    parser.add_argument("--budget-state",
                        default="artifacts/arterial_experiments/stage0/budget_state")
    parser.add_argument("--output",
                        default="artifacts/arterial_experiments/stage0/analysis.json")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--tolerance", type=float, default=0.10)
    parser.add_argument("--consecutive", type=int, default=3)
    args = parser.parse_args()

    resource_root = Path(args.resource_root)
    waves = [resource_wave(resource_root / f"profile_w{workers}_state")
             for workers in (1, 2, 4, 8, 12)]
    baseline = waves[0]["wall_time_seconds"]["max"]
    for wave in waves:
        wave["max_run_slowdown_vs_w1"] = (
            wave["wall_time_seconds"]["max"] / baseline - 1.0)
        wave["passes_25_percent_slowdown_gate"] = (
            wave["max_run_slowdown_vs_w1"] <= 0.25)
    eligible = [wave for wave in waves
                if wave["passes_25_percent_slowdown_gate"]
                and wave["swap_growth_bytes_peak"] == 0]
    selected_workers = max(eligible, key=lambda item: item["workers"])["workers"]

    budget_tasks = read_json(Path(args.budget_state) / "run_manifest.json")["tasks"]
    curves = {
        task["scene"]: analyze_curve(
            task["output_path"], args.window, args.tolerance, args.consecutive)
        for task in budget_tasks
    }
    stable = [value["stable_episode"] for value in curves.values()]
    if any(value is None for value in stable):
        formal_budget = None
        budget_reason = "At least one scene did not satisfy the frozen stability rule"
    else:
        latest = max(stable)
        formal_budget = min(400, max(200, round_up_25(1.5 * latest)))
        budget_reason = (
            "max(200, round_up_25(1.5 * latest stable episode)), capped at 400")
    output = {
        "schema_version": 1, "resource_waves": waves,
        "selected_max_workers": selected_workers, "curves": curves,
        "formal_episode_budget": formal_budget, "budget_rule": budget_reason,
        "collector_episode_budget": 400,
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        output, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
