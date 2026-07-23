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
    "curve_source", "metric", "episode_start", "episode_end",
    "point_count", "auc",
)
LEARNING_SPEED_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "curve_source", "reference", "criterion", "threshold", "status",
    "episode",
)
EFFICIENCY_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "episode", "travel_time", "environment_transitions", "gradient_updates",
    "cumulative_wall_time_seconds", "replay_fill_fraction",
    "update_to_data_ratio",
)
AULC_SUMMARY_FIELDS = (
    "run_key", "role", "agent", "network", "training_seed", "run_dir",
    "curve", "point_count", "x_start", "x_end", "aulc",
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
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        for curve_source, record_types, minimum_episode in (
            ("TRAIN", {"TRAIN"}, 1),
            ("EVALUATION", {"EVALUATION", "FINAL_EVALUATION"}, 0),
        ):
            selected = [
                row for row in run_rows
                if row["record_type"] in record_types
                and minimum_episode <= row["episode"] <= 100
            ]
            if not selected:
                continue
            identity = {key: selected[0][key] for key in (
                "run_key", "role", "agent", "network",
                "training_seed", "run_dir",
            )}
            for metric in AUC_METRICS:
                points = sorted(
                    (row["episode"], row.get(metric))
                    for row in selected if _finite(row.get(metric))
                )
                if not points:
                    continue
                episodes = np.asarray([point[0] for point in points], dtype=float)
                values = np.asarray([point[1] for point in points], dtype=float)
                auc = (
                    float(np.trapz(values, episodes))
                    if len(points) > 1 else float(values[0])
                )
                rows.append({
                    **identity,
                    "curve_source": curve_source,
                    "metric": metric,
                    "episode_start": int(episodes[0]),
                    "episode_end": int(episodes[-1]),
                    "point_count": len(points),
                    "auc": auc,
                })
    return rows


def _first_attainment(points, threshold, consecutive):
    streak = []
    previous_episode = None
    for episode, value in points:
        if value <= threshold:
            if previous_episode is None or episode == previous_episode + 1:
                streak.append(episode)
            else:
                streak = [episode]
            if len(streak) >= consecutive:
                return streak[0]
        else:
            streak = []
        previous_episode = episode
    return None


def calculate_learning_speed(metric_rows):
    baseline_levels = {}
    for row in metric_rows:
        if (
            row["role"] == "baseline"
            and row["agent"] in {"fixedtime", "maxpressure"}
            and _finite(row.get("travel_time"))
        ):
            baseline_levels[(row["network"], row["agent"])] = row["travel_time"]

    results = []
    run_keys = list(dict.fromkeys(
        row["run_key"] for row in metric_rows
        if row["agent"] == "dqn" and row["role"] != "baseline"
    ))
    for run_key in run_keys:
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        evaluations = [
            row for row in run_rows
            if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
            and _finite(row.get("travel_time"))
        ]
        if not evaluations:
            continue
        final = max(evaluations, key=lambda row: row["episode"])
        references = [
            ("final_110_percent", final["travel_time"] * 1.10),
        ]
        for agent in ("fixedtime", "maxpressure"):
            level = baseline_levels.get((final["network"], agent))
            if level is not None:
                references.append((agent, level))
        identity = {key: final[key] for key in (
            "run_key", "role", "agent", "network",
            "training_seed", "run_dir",
        )}
        for curve_source, record_types in (
            ("TRAIN", {"TRAIN"}),
            ("EVALUATION", {"EVALUATION", "FINAL_EVALUATION"}),
        ):
            points = sorted(
                (row["episode"], row["travel_time"])
                for row in run_rows
                if row["record_type"] in record_types
                and _finite(row.get("travel_time"))
            )
            for reference, threshold in references:
                for criterion, consecutive in (("single", 1), ("consecutive_5", 5)):
                    episode = _first_attainment(points, threshold, consecutive)
                    results.append({
                        **identity,
                        "curve_source": curve_source,
                        "reference": reference,
                        "criterion": criterion,
                        "threshold": threshold,
                        "status": "reached" if episode is not None else "not_reached",
                        "episode": episode,
                    })
    return results


def build_efficiency_rows(metric_rows):
    """Align evaluation quality with cumulative interaction and compute cost."""
    rows = []
    run_keys = list(dict.fromkeys(
        row["run_key"] for row in metric_rows
        if row["agent"] == "dqn" and row["role"] != "baseline"
    ))
    for run_key in run_keys:
        run_rows = [row for row in metric_rows if row["run_key"] == run_key]
        training = sorted(
            (row for row in run_rows if row["record_type"] == "TRAIN"),
            key=lambda row: row["episode"],
        )
        cumulative_wall = {}
        elapsed = 0.0
        for row in training:
            if _finite(row.get("wall_time_seconds")):
                elapsed += float(row["wall_time_seconds"])
            cumulative_wall[row["episode"]] = elapsed
        evaluations = sorted(
            (
                row for row in run_rows
                if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
                and _finite(row.get("travel_time"))
            ),
            key=lambda row: row["episode"],
        )
        for row in evaluations:
            transitions = row.get("collected_transitions")
            if not _finite(transitions) or transitions == 0:
                transitions = row.get("global_decision_step")
            capacity = row.get("replay_capacity")
            replay_size = row.get("replay_size")
            replay_fill = (
                float(replay_size) / float(capacity)
                if _finite(replay_size) and _finite(capacity) and capacity > 0
                else None
            )
            utd = row.get("update_to_data_ratio")
            if not _finite(utd):
                updates = row.get("gradient_updates")
                utd = (
                    float(updates) / float(transitions)
                    if _finite(updates) and _finite(transitions) and transitions > 0
                    else 0.0
                )
            wall_time = cumulative_wall.get(row["episode"], 0.0)
            rows.append({
                **{key: row[key] for key in (
                    "run_key", "role", "agent", "network",
                    "training_seed", "run_dir",
                )},
                "episode": row["episode"],
                "travel_time": row["travel_time"],
                "environment_transitions": transitions,
                "gradient_updates": row.get("gradient_updates"),
                "cumulative_wall_time_seconds": wall_time,
                "replay_fill_fraction": replay_fill,
                "update_to_data_ratio": utd,
            })
    return rows


def _mean_curve_area(points):
    if not points:
        return None
    x_values = np.asarray([point[0] for point in points], dtype=float)
    y_values = np.asarray([point[1] for point in points], dtype=float)
    if len(points) == 1 or x_values[-1] == x_values[0]:
        return float(y_values[-1])
    return float(np.trapz(y_values, x_values) / (x_values[-1] - x_values[0]))


def calculate_efficiency_aulc(efficiency_rows, metric_rows):
    """Calculate raw travel-time AULC and baseline-normalized AULC."""
    baselines = {}
    for row in metric_rows:
        if (
            row["role"] == "baseline"
            and row["agent"] in {"fixedtime", "maxpressure"}
            and _finite(row.get("travel_time"))
        ):
            baselines[(row["network"], row["agent"])] = float(row["travel_time"])
    results = []
    by_run = {}
    for row in efficiency_rows:
        by_run.setdefault(row["run_key"], []).append(row)
    normalized_by_seed = {}
    for run_key, rows in by_run.items():
        rows = sorted(rows, key=lambda row: row["environment_transitions"])
        identity = {key: rows[0][key] for key in (
            "run_key", "role", "agent", "network", "training_seed", "run_dir",
        )}
        raw_points = [
            (row["environment_transitions"], row["travel_time"])
            for row in rows
            if _finite(row.get("environment_transitions"))
            and _finite(row.get("travel_time"))
        ]
        raw_aulc = _mean_curve_area(raw_points)
        if raw_aulc is not None:
            results.append({
                **identity, "curve": "raw_travel_time",
                "point_count": len(raw_points), "x_start": raw_points[0][0],
                "x_end": raw_points[-1][0], "aulc": raw_aulc,
            })
        fixed = baselines.get((identity["network"], "fixedtime"))
        pressure = baselines.get((identity["network"], "maxpressure"))
        if fixed is None or pressure is None or fixed == pressure:
            continue
        normalized_points = [
            (x_value, (fixed - travel_time) / (fixed - pressure))
            for x_value, travel_time in raw_points
        ]
        normalized_aulc = _mean_curve_area(normalized_points)
        results.append({
            **identity, "curve": "normalized_control_score",
            "point_count": len(normalized_points),
            "x_start": normalized_points[0][0], "x_end": normalized_points[-1][0],
            "aulc": normalized_aulc,
        })
        normalized_by_seed.setdefault(identity["training_seed"], []).append(
            normalized_aulc
        )
    for training_seed, values in sorted(normalized_by_seed.items()):
        results.append({
            "run_key": f"cross_scene_seed_{training_seed}", "role": "formal",
            "agent": "dqn", "network": "all_scenarios",
            "training_seed": training_seed, "run_dir": "",
            "curve": "cross_scene_normalized_control_score",
            "point_count": len(values), "x_start": None, "x_end": None,
            "aulc": float(np.mean(values)),
        })
    return results


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
