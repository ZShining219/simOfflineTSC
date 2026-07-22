import csv
import json
import math
from pathlib import Path

import numpy as np


CORE_FIELDS = (
    "schema_version", "record_type", "episode", "simulation_step",
    "decision_step", "global_decision_step", "gradient_updates",
    "travel_time", "reward_mean", "reward_sum", "queue", "delay",
    "real_delay", "throughput", "loss_mean", "epsilon",
    "wall_time_seconds", "waiting_time", "unfinished_vehicles",
    "phase_switches", "phase_switch_frequency", "replay_size",
    "replay_capacity", "target_updates",
)
AUC_METRICS = (
    "travel_time", "queue", "delay", "real_delay", "reward_mean", "loss_mean",
)
AUC_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "metric", "episode_start", "episode_end", "point_count", "auc",
)


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def _run_columns(spec):
    return {
        "run_key": spec.run_key,
        "role": spec.role,
        "agent": spec.agent,
        "network": spec.network,
        "training_seed": spec.training_seed,
        "run_dir": str(spec.run_dir),
    }


def normalize_records(validated_runs):
    metric_rows = []
    action_rows = []
    for run in validated_runs:
        spec = run["spec"]
        identity = _run_columns(spec)
        for record in run["records"]:
            row = dict(identity)
            row.update({field: record.get(field) for field in CORE_FIELDS})
            row["action_distribution_json"] = json.dumps(
                record.get("action_distribution") or {}, sort_keys=True,
                separators=(",", ":"),
            )
            metric_rows.append(row)
            for action, fraction in sorted(
                (record.get("action_distribution") or {}).items(),
                key=lambda item: int(item[0]),
            ):
                action_rows.append({
                    **identity,
                    "record_type": record["record_type"],
                    "episode": record["episode"],
                    "action": int(action),
                    "fraction": fraction,
                })
    return metric_rows, action_rows


def _candidate_records(rows, run_key):
    records = [row for row in rows if row["run_key"] == run_key]
    evaluations = [
        row for row in records
        if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
    ]
    return evaluations or records


def build_run_summaries(validated_runs, metric_rows):
    summaries = []
    comparisons = []
    for run in validated_runs:
        spec = run["spec"]
        candidates = _candidate_records(metric_rows, spec.run_key)
        final = max(candidates, key=lambda row: (row["episode"], row["record_type"] == "FINAL_EVALUATION"))
        best = min(
            (row for row in candidates if _finite(row.get("travel_time"))),
            key=lambda row: (row["travel_time"], row["episode"]),
        )
        validation = run.get("trajectory_validation") or {}
        summaries.append({
            **_run_columns(spec),
            "record_count": len(run["records"]),
            "metric_schema_versions": ",".join(
                str(value) for value in sorted({r["schema_version"] for r in run["records"]})
            ),
            "config_sha256": run["resolved_config_sha256"],
            "trajectory_valid": validation.get("valid"),
            "trajectory_episodes": validation.get("episode_count"),
            "trajectory_transitions": validation.get("transition_count"),
            "evaluation_transition_count": validation.get("evaluation_transition_count"),
            "final_episode": final["episode"],
            "final_travel_time": final.get("travel_time"),
            "best_episode": best["episode"],
            "best_travel_time": best.get("travel_time"),
        })
        for label, row in (("final", final), ("best", best)):
            comparisons.append({
                **_run_columns(spec),
                "selection": label,
                "episode": row["episode"],
                "travel_time": row.get("travel_time"),
                "queue": row.get("queue"),
                "delay": row.get("delay"),
                "real_delay": row.get("real_delay"),
                "throughput": row.get("throughput"),
            })
    return summaries, comparisons


def calculate_first_100_auc(metric_rows):
    rows = []
    run_keys = list(dict.fromkeys(row["run_key"] for row in metric_rows))
    for run_key in run_keys:
        training = [
            row for row in metric_rows
            if row["run_key"] == run_key
            and row["record_type"] == "TRAIN"
            and row["episode"] <= 100
        ]
        if not training:
            continue
        identity = {key: training[0][key] for key in (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
        )}
        for metric in AUC_METRICS:
            points = sorted(
                (row["episode"], row.get(metric))
                for row in training if _finite(row.get(metric))
            )
            if not points:
                continue
            episodes = np.asarray([point[0] for point in points], dtype=float)
            values = np.asarray([point[1] for point in points], dtype=float)
            auc = float(np.trapz(values, episodes)) if len(points) > 1 else float(values[0])
            rows.append({
                **identity,
                "metric": metric,
                "episode_start": int(episodes[0]),
                "episode_end": int(episodes[-1]),
                "point_count": len(points),
                "auc": auc,
            })
    return rows


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path
