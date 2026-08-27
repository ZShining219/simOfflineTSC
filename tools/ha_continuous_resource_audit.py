#!/usr/bin/env python3
"""Audit HA-DHOA against the continuous Online-DQN control condition.

The tool is deliberately read-only with respect to experiment assets.  It
loads the stage-end frozen evaluations already committed by the HA-SODQN
runner, matches them on order/seed/stage/evaluation-scene, and writes derived
tables and figures to a separate output directory.  It does not train agents
or start a simulator.

The formal action-accuracy target is also audited here.  A Top-1 label would
require a recoverable microscopic SUMO state and an all-action counterfactual
rollout.  The existing NPZ state is only the eight-dimensional lane-count
observation, so the tool records the formal accuracy result as
``not_computable`` rather than silently substituting a lane-demand heuristic.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    # The existing mapping implementation is the project-audited source for
    # the SUMO phase/action to inbound-lane relation.
    from resource_metric_audit import LABEL, SCENES, phase_lane_mapping
except ImportError:  # pragma: no cover - supports ``python -m tools...``
    from tools.resource_metric_audit import LABEL, SCENES, phase_lane_mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HA_ROOT = ROOT / "data/output_data/ha_sodqn/formal_e7705f7_20260726"
DEFAULT_OUTPUT = ROOT / "data/output_data/resource_efficiency_ha_vs_continuous"

PROJECT_METHODS = ("P1C-DHOA-R25", "P1C-DHOA-R50")
BASELINE_METHOD = "CONT-FIFO"
RUN_PATTERNS = {
    BASELINE_METHOD: re.compile(r"^CONT-FIFO-(O[1-4])-SD([0-4])$"),
    "P1C-DHOA-R25": re.compile(r"^P1C-DHOA-R25-(O[1-4])-SD([0-4])$"),
    "P1C-DHOA-R50": re.compile(r"^P1C-DHOA-R50-(O[1-4])-SD([0-4])$"),
}
EVAL_KEY = re.compile(r"^evaluation:stage_(\d+):local_100:(.+)$")
SCENE_POSITION = {scene: i + 1 for i, scene in enumerate(SCENES)}
PLOT_COLORS = {
    BASELINE_METHOD: "#777777",
    "P1C-DHOA-R25": "#4c78a8",
    "P1C-DHOA-R50": "#f58518",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ha-root", type=Path, default=DEFAULT_HA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260803)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="allow this tool to replace files in its named output directory",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def jsonl(path: Path):
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def finite(value) -> bool:
    try:
        return value is not None and math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def number(value):
    return float(value) if finite(value) else None


def mean(values):
    clean = [float(value) for value in values if finite(value)]
    return statistics.mean(clean) if clean else None


def stdev(values):
    clean = [float(value) for value in values if finite(value)]
    return statistics.stdev(clean) if len(clean) > 1 else (0.0 if clean else None)


def bootstrap_ci(values, count: int, rng: np.random.Generator):
    clean = np.asarray([float(value) for value in values if finite(value)], dtype=float)
    if not len(clean):
        return None, None
    if len(clean) == 1 or count <= 1:
        return float(clean[0]), float(clean[0])
    draws = rng.choice(clean, (count, len(clean)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def scene_equal_bootstrap(
    values_by_scene: dict[str, list[float]],
    count: int,
    rng: np.random.Generator,
):
    """Bootstrap a scene-equal mean, not a pooled decision/seed mean."""
    clean = {
        scene: np.asarray([float(v) for v in values if finite(v)], dtype=float)
        for scene, values in values_by_scene.items()
    }
    clean = {scene: values for scene, values in clean.items() if len(values)}
    if not clean:
        return None, None
    if count <= 1:
        return mean([values.mean() for values in clean.values()]), mean(
            [values.mean() for values in clean.values()]
        )
    scene_draws = []
    for values in clean.values():
        if len(values) == 1:
            scene_draws.append(np.full(count, values[0]))
        else:
            scene_draws.append(rng.choice(values, (count, len(values)), replace=True).mean(axis=1))
    draws = np.vstack(scene_draws).mean(axis=0)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for row in rows:
            for field in row:
                if field not in fields:
                    fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_metadata(logical_dir: Path, method: str, order_id: str, seed: int):
    """Resolve the child manifest used to launch a completed logical run."""
    logical = read_json(logical_dir / "logical_run_manifest.json")
    attempts = {item.get("attempt_id"): item for item in logical.get("attempts", [])}
    attempt = attempts.get(logical.get("effective_attempt"), {})
    attempt_dir = Path(attempt.get("attempt_dir", ""))
    if not attempt_dir.is_absolute():
        attempt_dir = (logical_dir / attempt_dir).resolve()
    manifest_path = None
    command = attempt.get("command", [])
    if "--manifest" in command:
        manifest_path = Path(command[command.index("--manifest") + 1])
    child = {}
    if manifest_path is not None and manifest_path.exists():
        experiment = read_json(manifest_path)
        child = next(
            (row for row in experiment.get("children", []) if row.get("logical_run_id") == logical.get("logical_run_id")),
            {},
        )
    networks = list(child.get("networks", []))
    if not networks:
        # This fallback is only used for a damaged/migrated manifest.  Stage
        # event keys still carry the evaluation scenes and are checked below.
        networks = []
    all_attempt_dirs = []
    for item in logical.get("attempts", []):
        candidate = Path(item.get("attempt_dir", ""))
        if not candidate.is_absolute():
            candidate = (logical_dir / candidate).resolve()
        all_attempt_dirs.append(candidate)
    return {
        "logical_run_id": logical.get("logical_run_id", logical_dir.name),
        "method": method,
        "archive_mode": child.get("archive_mode"),
        "condition": child.get("condition"),
        "offline_ratio": child.get("offline_ratio"),
        "order_id": order_id,
        "training_seed": seed,
        "networks": networks,
        "stage_episodes": list(child.get("stage_episodes", [])),
        "initial_checkpoint_file_sha256": child.get("initial_checkpoint_file_sha256"),
        "protocol_id": experiment.get("protocol_id") if manifest_path and manifest_path.exists() else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "attempt_dir": str(attempt_dir),
        "all_attempt_dirs": [str(path) for path in all_attempt_dirs],
        "logical_status": logical.get("status"),
        "attempt_status": attempt.get("status"),
        "attempt_returncode": attempt.get("returncode"),
    }


def decision_aggregate(path: Path, cache: dict[Path, dict]):
    """Read only the scalar decision fields needed for queue burden and audit."""
    path = Path(path)
    if path in cache:
        return cache[path]
    queue_values = []
    interval_values = []
    actions = []
    throughput = None
    waiting_values = []
    for record in jsonl(path):
        if record.get("record_type") != "DECISION_METRICS":
            continue
        queue = record.get("queue_network_sum", record.get("queue"))
        interval = record.get("action_interval_seconds")
        if finite(queue):
            queue_values.append(float(queue))
        if finite(interval):
            interval_values.append(float(interval))
        if record.get("actions"):
            try:
                actions.append(int(record["actions"][0]))
            except (TypeError, ValueError, IndexError):
                pass
        if finite(record.get("throughput_cumulative")):
            throughput = float(record["throughput_cumulative"])
        for key in ("waiting_network_sum", "waiting_time_network_sum", "waiting"):
            if finite(record.get(key)):
                waiting_values.append(float(record[key]))
                break
    if len(queue_values) != len(interval_values) or not queue_values:
        result = {
            "queue_auc": None,
            "queue_sum": None,
            "queue_mean": None,
            "duration_seconds": None,
            "decision_steps": len(queue_values),
            "throughput_from_decisions": throughput,
            "phase_switch_count_from_decisions": sum(a != b for a, b in zip(actions, actions[1:])),
            "waiting_auc": None,
            "waiting_field_found": bool(waiting_values),
        }
    else:
        q = np.asarray(queue_values, dtype=float)
        dt = np.asarray(interval_values, dtype=float)
        result = {
            "queue_auc": float(np.sum(q * dt)),
            "queue_sum": float(q.sum()),
            "queue_mean": float(q.mean()),
            "duration_seconds": float(dt.sum()),
            "decision_steps": len(q),
            "throughput_from_decisions": throughput,
            "phase_switch_count_from_decisions": sum(a != b for a, b in zip(actions, actions[1:])),
            "waiting_auc": float(np.sum(np.asarray(waiting_values) * dt[: len(waiting_values)]))
            if len(waiting_values) == len(dt)
            else None,
            "waiting_field_found": bool(waiting_values),
        }
    cache[path] = result
    return result


def load_frozen_rows(root: Path):
    """Load every stage-end frozen evaluation cell for the three conditions."""
    rows = []
    run_catalog = []
    missing = []
    decision_cache = {}
    for method, pattern in RUN_PATTERNS.items():
        for logical_dir in sorted(root.iterdir()):
            if not logical_dir.is_dir():
                continue
            match = pattern.fullmatch(logical_dir.name)
            if not match or not (logical_dir / "logical_run_manifest.json").exists():
                continue
            order_id, seed = match.group(1), int(match.group(2))
            metadata = run_metadata(logical_dir, method, order_id, seed)
            run_catalog.append(metadata)
            attempt_dir = Path(metadata["attempt_dir"])
            # Resumed runs keep some stage-end evaluations in an earlier
            # interrupted attempt.  The effective attempt's current_state
            # carries the committed-operation index, so merge all attempts
            # instead of reading only the last events.jsonl.
            events = {}
            attempt_dirs = [Path(path) for path in metadata.get("all_attempt_dirs", [])]
            if attempt_dir not in attempt_dirs:
                attempt_dirs.append(attempt_dir)
            for candidate_dir in attempt_dirs:
                events_path = candidate_dir / "events.jsonl"
                if events_path.exists():
                    for event in jsonl(events_path):
                        if event.get("event_type") != "OPERATION_COMMITTED":
                            continue
                        key = event.get("payload", {}).get("operation_key", "")
                        parsed = EVAL_KEY.fullmatch(key)
                        if parsed:
                            stage = int(parsed.group(1))
                            scene = parsed.group(2)
                            if stage <= 4:
                                events[(stage, scene)] = event["payload"].get("artifact")
                current_state_path = candidate_dir / "current_state.json"
                if current_state_path.exists():
                    try:
                        current_state = read_json(current_state_path)
                    except (OSError, ValueError):
                        current_state = {}
                    for key, operation in current_state.get("completed_operations", {}).items():
                        parsed = EVAL_KEY.fullmatch(key)
                        if parsed and int(parsed.group(1)) <= 4 and operation.get("artifact"):
                            events[(int(parsed.group(1)), parsed.group(2))] = operation["artifact"]
            expected = []
            if metadata["networks"]:
                expected = [
                    (stage, scene)
                    for stage, _ in enumerate(metadata["networks"], start=1)
                    for scene in metadata["networks"][:stage]
                ]
            else:
                expected = sorted(events)
            for key in expected:
                if key not in events:
                    missing.append(
                        {
                            "method": method,
                            "order_id": order_id,
                            "training_seed": seed,
                            "stage_index": key[0],
                            "evaluation_network": key[1],
                            "reason": "missing_stage_end_evaluation_alias",
                        }
                    )
                    continue
                alias_path = Path(events[key])
                if not alias_path.exists():
                    missing.append(
                        {
                            "method": method,
                            "order_id": order_id,
                            "training_seed": seed,
                            "stage_index": key[0],
                            "evaluation_network": key[1],
                            "reason": "alias_path_missing",
                            "alias_path": str(alias_path),
                        }
                    )
                    continue
                alias = read_json(alias_path)
                committed_path = Path(alias["physical_committed_path"])
                committed = read_json(committed_path)
                summary_path = Path(committed["summary_path"])
                summary = read_json(summary_path)
                decisions_path = Path(committed["decisions_path"])
                aggregates = decision_aggregate(decisions_path, decision_cache)
                duration = aggregates["duration_seconds"] or (
                    float(summary.get("simulation_steps")) if finite(summary.get("simulation_steps")) else None
                )
                if duration is not None and aggregates["duration_seconds"] is None:
                    # SUMO simulation_steps is in seconds in this protocol.
                    duration = float(duration)
                queue_auc = aggregates["queue_auc"]
                if queue_auc is None and finite(summary.get("queue")) and duration is not None:
                    queue_auc = float(summary["queue"]) * duration
                    queue_auc_source = "summary_queue_times_duration_fallback"
                else:
                    queue_auc_source = "decision_metrics_queue_times_interval"
                row = {
                    "method": method,
                    "logical_run_id": metadata["logical_run_id"],
                    "archive_mode": metadata["archive_mode"],
                    "offline_ratio": metadata["offline_ratio"],
                    "order_id": order_id,
                    "training_seed": seed,
                    "stage_index": key[0],
                    "global_episode": key[0] * int(metadata["stage_episodes"][key[0] - 1])
                    if len(metadata["stage_episodes"]) >= key[0]
                    else None,
                    "training_network": metadata["networks"][key[0] - 1]
                    if len(metadata["networks"]) >= key[0]
                    else None,
                    "evaluation_network": summary.get("evaluation_network", key[1]),
                    "scene": LABEL.get(summary.get("evaluation_network", key[1]), key[1]),
                    "scene_position": SCENE_POSITION.get(summary.get("evaluation_network", key[1])),
                    "stage_budget": metadata["stage_episodes"][key[0] - 1]
                    if len(metadata["stage_episodes"]) >= key[0]
                    else None,
                    "initial_checkpoint_file_sha256": metadata.get("initial_checkpoint_file_sha256"),
                    "evaluation_protocol_digest": summary.get("evaluation_protocol_digest", committed.get("evaluation_protocol_digest")),
                    "checkpoint_digest": summary.get("checkpoint_digest", committed.get("checkpoint_digest")),
                    "decision_steps": aggregates["decision_steps"] or summary.get("decision_steps"),
                    "duration_seconds": duration,
                    "queue_mean": aggregates["queue_mean"] if aggregates["queue_mean"] is not None else summary.get("queue"),
                    "queue_sum": aggregates["queue_sum"],
                    "queue_auc": queue_auc,
                    "queue_auc_source": queue_auc_source,
                    "waiting_auc": aggregates["waiting_auc"],
                    "waiting_field_found": aggregates["waiting_field_found"],
                    "throughput": number(summary.get("throughput"))
                    if finite(summary.get("throughput"))
                    else aggregates["throughput_from_decisions"],
                    "queue": number(summary.get("queue")),
                    "real_delay": number(summary.get("real_delay")),
                    "approximate_delay": number(summary.get("delay")),
                    "waiting_time": number(summary.get("waiting_time")),
                    "phase_switch_count": number(summary.get("phase_switches"))
                    if finite(summary.get("phase_switches"))
                    else aggregates["phase_switch_count_from_decisions"],
                    "travel_time": number(summary.get("travel_time")),
                    "unfinished_vehicles": number(summary.get("unfinished_vehicles")),
                    "reward_mean": number(summary.get("reward_mean")),
                    "alias_path": str(alias_path),
                    "committed_path": str(committed_path),
                    "summary_path": str(summary_path),
                    "decisions_path": str(decisions_path),
                    "manifest_path": metadata["manifest_path"],
                }
                rows.append(row)
    return rows, run_catalog, missing, decision_cache


def ratio_change(method_value, baseline_value, higher_is_better: bool):
    if not finite(method_value) or not finite(baseline_value) or float(baseline_value) == 0:
        return None
    if higher_is_better:
        return (float(method_value) / float(baseline_value) - 1.0) * 100.0
    return (float(baseline_value) - float(method_value)) / float(baseline_value) * 100.0


def pair_final_rows(rows):
    """Match project and CONT cells on the pre-registered comparability key."""
    baseline = {
        (row["order_id"], row["training_seed"], row["stage_index"], row["evaluation_network"]): row
        for row in rows
        if row["method"] == BASELINE_METHOD
    }
    pairs = []
    unmatched = []
    for row in rows:
        if row["method"] not in PROJECT_METHODS:
            continue
        key = (row["order_id"], row["training_seed"], row["stage_index"], row["evaluation_network"])
        base = baseline.get(key)
        if base is None:
            unmatched.append({"method": row["method"], "order_id": row["order_id"], "training_seed": row["training_seed"], "stage_index": row["stage_index"], "evaluation_network": row["evaluation_network"], "reason": "no_matching_CONT_cell"})
            continue
        mismatch = []
        if base.get("evaluation_protocol_digest") != row.get("evaluation_protocol_digest"):
            mismatch.append("evaluation_protocol_digest")
        if base.get("initial_checkpoint_file_sha256") and row.get("initial_checkpoint_file_sha256") and base.get("initial_checkpoint_file_sha256") != row.get("initial_checkpoint_file_sha256"):
            mismatch.append("initial_checkpoint_file_sha256")
        if base.get("stage_budget") != row.get("stage_budget"):
            mismatch.append("stage_budget")
        if base.get("training_network") != row.get("training_network"):
            mismatch.append("training_network")
        if finite(base.get("duration_seconds")) and finite(row.get("duration_seconds")) and not math.isclose(float(base["duration_seconds"]), float(row["duration_seconds"]), rel_tol=0, abs_tol=1e-6):
            mismatch.append("duration_seconds")
        if finite(base.get("decision_steps")) and finite(row.get("decision_steps")) and int(base["decision_steps"]) != int(row["decision_steps"]):
            mismatch.append("decision_steps")
        if mismatch:
            unmatched.append({"method": row["method"], "order_id": row["order_id"], "training_seed": row["training_seed"], "stage_index": row["stage_index"], "evaluation_network": row["evaluation_network"], "reason": "comparability_mismatch", "mismatch_fields": ";".join(mismatch)})
            continue
        pair = {
            "method": row["method"],
            "order_id": row["order_id"],
            "training_seed": row["training_seed"],
            "stage_index": row["stage_index"],
            "global_episode": row["global_episode"],
            "training_network": row["training_network"],
            "evaluation_network": row["evaluation_network"],
            "scene": row["scene"],
            "scene_position": row["scene_position"],
            "stage_budget": row["stage_budget"],
            "initial_checkpoint_file_sha256": row.get("initial_checkpoint_file_sha256"),
            "evaluation_protocol_digest": row["evaluation_protocol_digest"],
            "baseline_logical_run_id": base["logical_run_id"],
            "project_logical_run_id": row["logical_run_id"],
            "baseline_summary_path": base["summary_path"],
            "project_summary_path": row["summary_path"],
            "baseline_decisions_path": base["decisions_path"],
            "project_decisions_path": row["decisions_path"],
        }
        for metric in ("throughput", "queue_auc", "queue_mean", "real_delay", "waiting_time", "phase_switch_count", "travel_time", "approximate_delay", "duration_seconds", "decision_steps"):
            pair[f"continuous_{metric}"] = base.get(metric)
            pair[f"ha_{metric}"] = row.get(metric)
        pair["throughput_abs_gain"] = number(row.get("throughput")) - number(base.get("throughput")) if finite(row.get("throughput")) and finite(base.get("throughput")) else None
        pair["throughput_improvement_pct"] = ratio_change(row.get("throughput"), base.get("throughput"), True)
        pair["queue_auc_reduction_pct"] = ratio_change(row.get("queue_auc"), base.get("queue_auc"), False)
        pair["queue_mean_reduction_pct"] = ratio_change(row.get("queue_mean"), base.get("queue_mean"), False)
        pair["real_delay_reduction_pct"] = ratio_change(row.get("real_delay"), base.get("real_delay"), False)
        pair["waiting_time_change_pct"] = ratio_change(row.get("waiting_time"), base.get("waiting_time"), True)
        pair["waiting_time_reduction_pct"] = -pair["waiting_time_change_pct"] if pair["waiting_time_change_pct"] is not None else None
        pair["phase_switch_change_pct"] = ratio_change(row.get("phase_switch_count"), base.get("phase_switch_count"), True)
        pair["travel_time_reduction_pct"] = ratio_change(row.get("travel_time"), base.get("travel_time"), False)
        pairs.append(pair)
    return pairs, unmatched


def summarize_pairs(pairs, bootstrap_resamples, rng, final_only=True):
    selected = [row for row in pairs if not final_only or row["stage_index"] == 4]
    grouped = defaultdict(list)
    for row in selected:
        grouped[(row["method"], row["evaluation_network"])].append(row)
    metrics = (
        "throughput_improvement_pct",
        "queue_auc_reduction_pct",
        "queue_mean_reduction_pct",
        "real_delay_reduction_pct",
        "waiting_time_change_pct",
        "waiting_time_reduction_pct",
        "phase_switch_change_pct",
        "travel_time_reduction_pct",
    )
    summary = []
    for (method, scene), items in sorted(grouped.items()):
        row = {
            "scope": "final_stage_scene" if final_only else "stage_scene",
            "method": method,
            "scenario": scene,
            "scene": LABEL.get(scene, scene),
            "n_pairs": len(items),
            "orders": ";".join(sorted({str(item["order_id"]) for item in items})),
            "training_seeds": ";".join(str(x) for x in sorted({int(item["training_seed"]) for item in items})),
            "stage_index": 4 if final_only else "all",
            "continuous_throughput_mean": mean(item["continuous_throughput"] for item in items),
            "ha_throughput_mean": mean(item["ha_throughput"] for item in items),
            "continuous_queue_auc_mean": mean(item["continuous_queue_auc"] for item in items),
            "ha_queue_auc_mean": mean(item["ha_queue_auc"] for item in items),
            "continuous_real_delay_mean": mean(item["continuous_real_delay"] for item in items),
            "ha_real_delay_mean": mean(item["ha_real_delay"] for item in items),
            "continuous_waiting_time_mean": mean(item["continuous_waiting_time"] for item in items),
            "ha_waiting_time_mean": mean(item["ha_waiting_time"] for item in items),
        }
        for metric in metrics:
            values = [item.get(metric) for item in items]
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = stdev(values)
            low, high = bootstrap_ci(values, bootstrap_resamples, rng)
            row[metric + "_ci95_low"] = low
            row[metric + "_ci95_high"] = high
            row[metric + "_n"] = sum(finite(value) for value in values)
        row["throughput_target_10pct_met"] = row["throughput_improvement_pct_mean"] is not None and row["throughput_improvement_pct_mean"] >= 10.0
        summary.append(row)
    return summary


def scene_equal_summary(final_summary, bootstrap_resamples, rng):
    output = []
    for method in PROJECT_METHODS:
        selected = [row for row in final_summary if row["method"] == method]
        if not selected:
            continue
        result = {
            "scope": "scene_equal_mean",
            "method": method,
            "scenario_count": len(selected),
            "scenarios": ";".join(row["scenario"] for row in selected),
            "coverage_orders": ";".join(sorted({order for row in selected for order in row["orders"].split(";") if order})),
            "coverage_training_seeds": ";".join(sorted({seed for row in selected for seed in row["training_seeds"].split(";") if seed})),
            "n_pairs_total": sum(int(row["n_pairs"]) for row in selected),
            "aggregation": "equal weight over matched evaluation scenarios; within scenario paired order/seed mean",
        }
        for metric in (
            "throughput_improvement_pct",
            "queue_auc_reduction_pct",
            "queue_mean_reduction_pct",
            "real_delay_reduction_pct",
            "waiting_time_change_pct",
            "waiting_time_reduction_pct",
            "phase_switch_change_pct",
            "travel_time_reduction_pct",
        ):
            by_scene = {
                row["scenario"]: [
                    item[metric]
                    for item in _final_pair_rows_for_scene(row["scenario"], method)
                    if finite(item.get(metric))
                ]
                for row in selected
            }
            # The selected scene summary contains only means; use its scene
            # means for the point estimate and all paired rows for the CI.
            result[metric + "_mean"] = mean(row.get(metric + "_mean") for row in selected)
            result[metric + "_scene_std"] = stdev(row.get(metric + "_mean") for row in selected)
            low, high = scene_equal_bootstrap(by_scene, bootstrap_resamples, rng)
            result[metric + "_ci95_low"] = low
            result[metric + "_ci95_high"] = high
        result["throughput_target_10pct_met"] = result["throughput_improvement_pct_mean"] is not None and result["throughput_improvement_pct_mean"] >= 10.0
        result["scenarios_meeting_throughput_target"] = sum(
            row.get("throughput_improvement_pct_mean") is not None and row["throughput_improvement_pct_mean"] >= 10.0
            for row in selected
        )
        result["scenarios_with_throughput_decrease"] = sum(
            row.get("throughput_improvement_pct_mean") is not None and row["throughput_improvement_pct_mean"] < 0
            for row in selected
        )
        output.append(result)
    return output


# ``scene_equal_summary`` needs the pair rows for bootstrap CIs.  The active
# dataset is installed by ``main`` immediately before this function is used;
# keeping it module-local avoids copying large nested tables through every call.
_FINAL_PAIRS_BY_METHOD_SCENE: dict[tuple[str, str], list[dict]] = {}


def _final_pair_rows_for_scene(scene: str, method: str):
    return _FINAL_PAIRS_BY_METHOD_SCENE.get((method, scene), [])


def forgetting_rows(frozen_rows):
    by_run = defaultdict(list)
    for row in frozen_rows:
        by_run[(row["method"], row["order_id"], row["training_seed"])].append(row)
    output = []
    missing = []
    for (method, order_id, seed), items in sorted(by_run.items()):
        networks = []
        for row in sorted(items, key=lambda x: (x["stage_index"], x["evaluation_network"])):
            if row["training_network"] and row["training_network"] not in networks:
                networks.append(row["training_network"])
        if len(networks) != 4:
            # The manifest is authoritative when available.
            network_candidates = [row["training_network"] for row in items if row["training_network"]]
            networks = list(dict.fromkeys(network_candidates))
        for scene_position, scene in enumerate(networks, start=1):
            own_stage = scene_position
            own = next((row for row in items if row["stage_index"] == own_stage and row["evaluation_network"] == scene), None)
            if own is None:
                missing.append({"method": method, "order_id": order_id, "training_seed": seed, "scenario": scene, "reason": "missing_own_stage_end_cell"})
                continue
            for evaluation_stage in range(own_stage, 5):
                current = next((row for row in items if row["stage_index"] == evaluation_stage and row["evaluation_network"] == scene), None)
                if current is None:
                    missing.append({"method": method, "order_id": order_id, "training_seed": seed, "scenario": scene, "evaluation_stage": evaluation_stage, "reason": "missing_historical_evaluation_cell"})
                    continue
                row = {
                    "method": method,
                    "order_id": order_id,
                    "training_seed": seed,
                    "scenario": scene,
                    "scene": LABEL.get(scene, scene),
                    "scene_position": scene_position,
                    "own_training_stage": own_stage,
                    "evaluation_stage": evaluation_stage,
                    "additional_scenes_trained": evaluation_stage - own_stage,
                    "training_network_at_evaluation": current["training_network"],
                    "stage_label": "own_stage_end" if evaluation_stage == own_stage else ("final_stage" if evaluation_stage == 4 else f"after_stage_{evaluation_stage}"),
                    "own_summary_path": own["summary_path"],
                    "evaluation_summary_path": current["summary_path"],
                    "own_throughput": own["throughput"],
                    "evaluation_throughput": current["throughput"],
                    "own_queue_auc": own["queue_auc"],
                    "evaluation_queue_auc": current["queue_auc"],
                    "own_real_delay": own["real_delay"],
                    "evaluation_real_delay": current["real_delay"],
                    "own_waiting_time": own["waiting_time"],
                    "evaluation_waiting_time": current["waiting_time"],
                    "own_phase_switch_count": own["phase_switch_count"],
                    "evaluation_phase_switch_count": current["phase_switch_count"],
                }
                row["throughput_retention_pct"] = (current["throughput"] / own["throughput"] * 100) if finite(current["throughput"]) and finite(own["throughput"]) and float(own["throughput"]) != 0 else None
                # Forgetting is reported as a positive degradation: a loss in
                # higher-is-better throughput is own-stage value minus later
                # value; a burden increase is later value minus own-stage
                # value for lower-is-better metrics.
                row["throughput_loss_pct"] = ((own["throughput"] - current["throughput"]) / own["throughput"] * 100) if finite(current["throughput"]) and finite(own["throughput"]) and float(own["throughput"]) != 0 else None
                row["queue_burden_ratio_pct"] = (current["queue_auc"] / own["queue_auc"] * 100) if finite(current["queue_auc"]) and finite(own["queue_auc"]) and float(own["queue_auc"]) != 0 else None
                row["queue_increase_pct"] = ((current["queue_auc"] - own["queue_auc"]) / own["queue_auc"] * 100) if finite(current["queue_auc"]) and finite(own["queue_auc"]) and float(own["queue_auc"]) != 0 else None
                row["real_delay_ratio_pct"] = (current["real_delay"] / own["real_delay"] * 100) if finite(current["real_delay"]) and finite(own["real_delay"]) and float(own["real_delay"]) != 0 else None
                row["real_delay_increase_pct"] = ((current["real_delay"] - own["real_delay"]) / own["real_delay"] * 100) if finite(current["real_delay"]) and finite(own["real_delay"]) and float(own["real_delay"]) != 0 else None
                row["waiting_time_ratio_pct"] = (current["waiting_time"] / own["waiting_time"] * 100) if finite(current["waiting_time"]) and finite(own["waiting_time"]) and float(own["waiting_time"]) != 0 else None
                row["waiting_time_increase_pct"] = ((current["waiting_time"] - own["waiting_time"]) / own["waiting_time"] * 100) if finite(current["waiting_time"]) and finite(own["waiting_time"]) and float(own["waiting_time"]) != 0 else None
                output.append(row)
    return output, missing


def summarize_forgetting(rows, bootstrap_resamples, rng):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["scenario"], row["additional_scenes_trained"])].append(row)
    metrics = ("throughput_retention_pct", "throughput_loss_pct", "queue_burden_ratio_pct", "queue_increase_pct", "real_delay_ratio_pct", "real_delay_increase_pct", "waiting_time_ratio_pct", "waiting_time_increase_pct")
    output = []
    for (method, scene, offset), items in sorted(grouped.items()):
        row = {
            "scope": "scene_transition",
            "method": method,
            "scenario": scene,
            "scene": LABEL.get(scene, scene),
            "additional_scenes_trained": offset,
            "evaluation_stage_values": ";".join(str(x) for x in sorted({item["evaluation_stage"] for item in items})),
            "n_runs": len(items),
            "orders": ";".join(sorted({item["order_id"] for item in items})),
        }
        for metric in metrics:
            values = [item.get(metric) for item in items]
            row[metric + "_mean"] = mean(values)
            row[metric + "_std"] = stdev(values)
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = bootstrap_ci(values, bootstrap_resamples, rng)
        output.append(row)
    historical_final = [row for row in rows if row["evaluation_stage"] == 4 and row["scene_position"] < 4]
    by_final = defaultdict(list)
    for row in historical_final:
        by_final[row["method"]].append(row)
    for method, items in sorted(by_final.items()):
        row = {
            "scope": "historical_final_equal_scene",
            "method": method,
            "scenario": "HISTORICAL_SCENES",
            "scene": "Historical scenes (final stage)",
            "additional_scenes_trained": "final",
            "evaluation_stage_values": "4",
            "n_runs": len(items),
            "orders": ";".join(sorted({item["order_id"] for item in items})),
            "scenarios": ";".join(sorted({item["scenario"] for item in items})),
        }
        for metric in metrics:
            by_scene = defaultdict(list)
            for item in items:
                if finite(item.get(metric)):
                    by_scene[item["scenario"]].append(item[metric])
            scene_means = [mean(values) for values in by_scene.values()]
            row[metric + "_mean"] = mean(scene_means)
            row[metric + "_std"] = stdev(scene_means)
            row[metric + "_ci95_low"], row[metric + "_ci95_high"] = scene_equal_bootstrap(by_scene, bootstrap_resamples, rng)
        output.append(row)
    return output


def pair_forgetting(cont_rows, project_rows):
    baseline = {(r["order_id"], r["training_seed"], r["scenario"], r["evaluation_stage"]): r for r in cont_rows}
    output = []
    for row in project_rows:
        key = (row["order_id"], row["training_seed"], row["scenario"], row["evaluation_stage"])
        base = baseline.get(key)
        if base is None:
            continue
        out = {
            "method": row["method"],
            "order_id": row["order_id"],
            "training_seed": row["training_seed"],
            "scenario": row["scenario"],
            "scene": row["scene"],
            "scene_position": row["scene_position"],
            "own_training_stage": row["own_training_stage"],
            "evaluation_stage": row["evaluation_stage"],
            "additional_scenes_trained": row["additional_scenes_trained"],
        }
        for metric in ("throughput_loss_pct", "queue_increase_pct", "real_delay_increase_pct", "waiting_time_increase_pct"):
            out["continuous_" + metric] = base.get(metric)
            out["ha_" + metric] = row.get(metric)
            # Positive advantage means HA has less degradation than CONT.
            out[metric + "_ha_advantage"] = (base.get(metric) - row.get(metric)) if finite(base.get(metric)) and finite(row.get(metric)) else None
        output.append(out)
    return output


def summarize_paired_forgetting(paired_rows, bootstrap_resamples, rng):
    """Summarize HA-vs-CONT forgetting advantages from matched cells."""
    grouped = defaultdict(list)
    for row in paired_rows:
        grouped[(row["method"], row["additional_scenes_trained"])].append(row)
    metrics = (
        "throughput_loss_pct",
        "queue_increase_pct",
        "real_delay_increase_pct",
        "waiting_time_increase_pct",
    )
    output = []
    for (method, offset), items in sorted(grouped.items()):
        row = {
            "scope": "paired_comparison",
            "method": method,
            "scenario": "all_matched_scenes",
            "scene": "All matched scenes",
            "additional_scenes_trained": offset,
            "n_pairs": len(items),
            "orders": ";".join(sorted({item["order_id"] for item in items})),
        }
        for metric in metrics:
            continuous_values = [item.get("continuous_" + metric) for item in items]
            ha_values = [item.get("ha_" + metric) for item in items]
            advantage_values = [item.get(metric + "_ha_advantage") for item in items]
            row["continuous_" + metric + "_mean"] = mean(continuous_values)
            row["ha_" + metric + "_mean"] = mean(ha_values)
            row[metric + "_ha_advantage_mean"] = mean(advantage_values)
            row[metric + "_ha_advantage_median"] = statistics.median([float(v) for v in advantage_values if finite(v)]) if any(finite(v) for v in advantage_values) else None
            row[metric + "_ha_advantage_ci95_low"], row[metric + "_ha_advantage_ci95_high"] = bootstrap_ci(advantage_values, bootstrap_resamples, rng)
            row[metric + "_ha_advantage_positive_rate"] = mean([float(v > 0) for v in advantage_values if finite(v)])
        output.append(row)
    historical_final = [item for item in paired_rows if item["evaluation_stage"] == 4 and int(item["scene_position"]) < 4]
    by_method = defaultdict(list)
    for item in historical_final:
        by_method[item["method"]].append(item)
    for method, items in sorted(by_method.items()):
        row = {
            "scope": "paired_final_historical",
            "method": method,
            "scenario": "historical_scenes",
            "scene": "Historical scenes (final stage)",
            "additional_scenes_trained": "final",
            "n_pairs": len(items),
            "orders": ";".join(sorted({item["order_id"] for item in items})),
        }
        for metric in metrics:
            continuous_values = [item.get("continuous_" + metric) for item in items]
            ha_values = [item.get("ha_" + metric) for item in items]
            advantage_values = [item.get(metric + "_ha_advantage") for item in items]
            row["continuous_" + metric + "_mean"] = mean(continuous_values)
            row["ha_" + metric + "_mean"] = mean(ha_values)
            row[metric + "_ha_advantage_mean"] = mean(advantage_values)
            row[metric + "_ha_advantage_median"] = statistics.median([float(v) for v in advantage_values if finite(v)]) if any(finite(v) for v in advantage_values) else None
            row[metric + "_ha_advantage_ci95_low"], row[metric + "_ha_advantage_ci95_high"] = bootstrap_ci(advantage_values, bootstrap_resamples, rng)
            row[metric + "_ha_advantage_positive_rate"] = mean([float(v > 0) for v in advantage_values if finite(v)])
        output.append(row)
    return output


def asset_accuracy_audit(root: Path, output_dir: Path, frozen_stage_cell_count: int):
    """Audit the requested Top-1 chain without claiming it exists."""
    npz_paths = list(root.glob("P1C-DHOA-*/attempts/attempt_*/trajectory/episodes/*.npz"))
    decision_paths = list(root.glob("_shared_evaluation/physical/*/attempt_*/decisions.jsonl"))
    state_files = list(ROOT.glob("data/output_data/**/saveState*")) + list(ROOT.glob("data/output_data/**/*save_state*"))
    counterfactual_files = []
    this_script = Path(__file__).resolve()
    for search_root in (ROOT / "tools", ROOT / "sequential", ROOT / "world", ROOT / "scripts"):
        if search_root.exists():
            for path in search_root.rglob("*.py"):
                try:
                    text = path.read_text(encoding="utf-8", errors="ignore")
                except OSError:
                    continue
                if path.resolve() != this_script and re.search(r"counter.?factual|saveState|loadState", text, re.I):
                    counterfactual_files.append(str(path))
    mapping_rows = []
    mapping_errors = []
    mappings = {}
    for scene in SCENES:
        try:
            mapping = phase_lane_mapping(scene)
            mappings[scene] = mapping
            for index, lane in enumerate(mapping["state_lanes"]):
                mapping_rows.append({
                    "scenario": scene,
                    "scene": LABEL[scene],
                    "state_index": index,
                    "lane_id": lane,
                    "served_by_actions": ";".join(str(action) for action, lanes in enumerate(mapping["action_service_lanes"]) if lane in lanes),
                    "green_phase_states": "|".join(mapping["green_phase_states"]),
                    "net_xml": mapping["net_xml"],
                    "net_xml_sha256": mapping["net_xml_sha256"],
                })
        except Exception as exc:  # keep audit report available if one scene is damaged
            mapping_errors.append({"scenario": scene, "error": repr(exc)})
    write_csv(output_dir / "configuration_phase_lane_mapping.csv", mapping_rows)
    audit = {
        "status": "formal_top1_not_computable",
        "definition": "Top-1 agreement between frozen-agent action and the standard-optimal action label from all legal-action short rollouts",
        "formal_result": "not_computable",
        "reason": "No recoverable microscopic SUMO state or deterministic counterfactual rollout labels were found for the frozen decision points. Existing NPZ state is the 8-dimensional lane-count observation, not a complete vehicle-position/speed/signal state.",
        "assets": {
            "frozen_agent_actions": {"found": bool(decision_paths), "stage_end_cell_count": frozen_stage_cell_count, "decision_file_count_in_formal_root": len(decision_paths), "source": "stage-end decisions.jsonl actions"},
            "frozen_npz_trajectories": {"found": bool(npz_paths), "count": len(npz_paths), "state_semantics": "8-D LaneVehicleGenerator(lane_count), not microscopic SUMO state"},
            "recoverable_sumo_save_state": {"found": bool(state_files), "count": len(state_files), "paths": [str(path) for path in state_files[:20]]},
            "all_action_counterfactual_rollout": {"found": bool(counterfactual_files), "count": len(counterfactual_files), "paths": counterfactual_files[:20]},
            "phase_lane_mapping": {"found": bool(mappings) and not mapping_errors, "scenes": sorted(mappings), "mapping_csv": str(output_dir / "configuration_phase_lane_mapping.csv"), "errors": mapping_errors},
            "unified_metrics": {"found": True, "fields": ["queue", "real_delay", "waiting_time", "throughput"], "waiting_auc": "not present in stage-end decision records"},
        },
        "retained_diagnostic": "Existing lane-demand capture/instantaneous high-demand matching outputs remain diagnostic only; they are not Top-1 optimal-action accuracy.",
        "minimum_supplement": [
            "At representative frozen decision points, save a SUMO saveState (or prove a bit-for-bit deterministic replay) containing vehicle positions, speeds, routes, signal internal state, and traffic demand state.",
            "Restore each state and run all eight legal green actions for the same short horizon and evaluation metric.",
            "Persist the best-action label, tie rule, state/checkpoint identity, and agent action; then compute Top-1 agreement with episode/seed and scenario confidence intervals.",
        ],
    }
    return audit


def render_performance_plot(output_dir: Path, final_summary: list[dict], scene_equal: list[dict]):
    metrics = [
        ("throughput_improvement_pct_mean", "Throughput improvement vs CONT (%)", True),
        ("queue_auc_reduction_pct_mean", "Queue AUC reduction vs CONT (%)", False),
        ("real_delay_reduction_pct_mean", "Real-delay reduction vs CONT (%)", False),
        ("waiting_time_change_pct_mean", "Waiting-time change vs CONT (%)", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=False)
    scenes = [scene for scene in SCENES if any(row["scenario"] == scene for row in final_summary)]
    x = np.arange(len(scenes))
    width = 0.34
    for ax, (metric, title, is_main) in zip(axes.flat, metrics):
        for idx, method in enumerate(PROJECT_METHODS):
            chosen = {row["scenario"]: row for row in final_summary if row["method"] == method}
            values = [chosen.get(scene, {}).get(metric) for scene in scenes]
            bars = ax.bar(x + (idx - 0.5) * width, [v if finite(v) else 0 for v in values], width, label=method, color=PLOT_COLORS[method])
            for bar, value in zip(bars, values):
                if finite(value):
                    y = float(value) + (0.7 if float(value) >= 0 else -1.5)
                    ax.text(bar.get_x() + bar.get_width() / 2, y, f"{float(value):.1f}", ha="center", va="bottom" if float(value) >= 0 else "top", fontsize=8)
        ax.axhline(0, color="black", linewidth=0.8)
        if is_main:
            ax.axhline(10, color="crimson", linestyle="--", linewidth=1.4, label="10% target")
        ax.set_xticks(x, [LABEL[scene] for scene in scenes])
        ax.set_title(title)
        ax.set_ylabel("percent")
        ax.grid(axis="y", alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("HA-DHOA final frozen performance relative to matched continuous Online DQN\nScene-equal reporting; R50 coverage is O2 only")
    fig.tight_layout()
    fig.savefig(output_dir / "resource_efficiency_vs_continuous.png", dpi=200)
    plt.close(fig)

    # A compact summary figure keeps the primary throughput target visible
    # without a misleading zero-valued baseline bar.
    fig, ax = plt.subplots(figsize=(10, 5.5))
    labels = ["R25\n4 orders", "R50\nO2 only"]
    values = []
    for method in PROJECT_METHODS:
        row = next((item for item in scene_equal if item["method"] == method), None)
        values.append(row.get("throughput_improvement_pct_mean") if row else None)
    x = np.arange(len(labels))
    bars = ax.bar(x, [v if finite(v) else 0 for v in values], color=[PLOT_COLORS[m] for m in PROJECT_METHODS], width=0.55)
    for bar, value in zip(bars, values):
        if finite(value):
            ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.5, f"{float(value):.2f}%", ha="center", va="bottom")
    ax.axhline(10, color="crimson", linestyle="--", linewidth=1.6, label="10% throughput target")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Throughput improvement vs CONT (%)")
    ax.set_title("Scene-equal throughput improvement\nR25 and R50 are reported with their actual coverage")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_dir / "resource_efficiency_throughput_target.png", dpi=200)
    plt.close(fig)


def render_forgetting_plot(output_dir: Path, forgetting_summary: list[dict]):
    metrics = [
        ("throughput_retention_pct_mean", "Throughput retention relative to own-stage end (%)", 100),
        ("queue_burden_ratio_pct_mean", "Queue burden relative to own-stage end (%)", 100),
        ("real_delay_ratio_pct_mean", "Real-delay ratio relative to own-stage end (%)", 100),
        ("waiting_time_ratio_pct_mean", "Waiting-time ratio relative to own-stage end (%)", 100),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    offsets = sorted({int(row["additional_scenes_trained"]) for row in forgetting_summary if row["scope"] == "scene_transition" and str(row["additional_scenes_trained"]).isdigit()})
    for ax, (metric, title, reference) in zip(axes.flat, metrics):
        for method in (BASELINE_METHOD, *PROJECT_METHODS):
            selected = [row for row in forgetting_summary if row["scope"] == "scene_transition" and row["method"] == method]
            values = []
            x_values = []
            for offset in offsets:
                matches = [row for row in selected if int(row["additional_scenes_trained"]) == offset]
                if not matches:
                    continue
                # Equal-weight the historical scenes at each offset.  A
                # pooled run mean would over-weight orders/scenes with more
                # available transitions (especially for R50's O2-only
                # coverage).
                values.append(mean(row.get(metric) for row in matches))
                x_values.append(offset)
            if values:
                ax.plot(x_values, values, marker="o", label=method, color=PLOT_COLORS[method])
        ax.axhline(reference, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("Additional scene-training stages after own stage")
        ax.set_ylabel("percent")
        ax.set_xticks(offsets)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle("Historical-scene retention across later stages\nValues are equal-weight means over historical scenes and matched runs")
    fig.tight_layout()
    fig.savefig(output_dir / "forgetting_retention_comparison.png", dpi=200)
    plt.close(fig)


def render_accuracy_status(output_dir: Path, audit: dict):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("off")
    lines = [
        "FORMAL CONFIGURATION ACCURACY: NOT COMPUTABLE",
        "",
        "Required label: standard-optimal action from all legal-action short rollouts",
        "Available: frozen actions + 8-D lane-count observations + phase/lane mapping",
        "Missing: recoverable microscopic SUMO state and saved counterfactual best-action labels",
        "",
        "The existing instantaneous lane-demand capture result remains a diagnostic, not Top-1 optimal-action accuracy.",
        "Minimum supplement: saveState/deterministic replay → enumerate 8 actions → persist best-action labels → compare frozen actions.",
    ]
    ax.text(0.02, 0.92, lines[0], fontsize=18, weight="bold", color="#9c0006", va="top")
    ax.text(0.02, 0.82, "\n".join(lines[2:]), fontsize=12, va="top", family="DejaVu Sans")
    ax.add_patch(plt.Rectangle((0.01, 0.08), 0.98, 0.84, fill=False, edgecolor="#9c0006", linewidth=2))
    ax.set_title("Top-1 action agreement audit", pad=18)
    fig.tight_layout()
    fig.savefig(output_dir / "configuration_accuracy_status.png", dpi=200)
    plt.close(fig)


def write_report(output_dir: Path, audit: dict, accuracy_audit: dict, run_catalog: list[dict], missing: list[dict], unmatched: list[dict], final_summary: list[dict], scene_equal: list[dict], forgetting_summary: list[dict]):
    lines = [
        "# HA-DHOA 与普通连续 Online DQN 的资源配置指标重组审计",
        "",
        "本报告只使用当前磁盘中已经完成的冻结评价、stage checkpoint、decisions.jsonl 与既有轨迹资产；没有重新训练，也没有重新启动批量仿真。所有百分比是冻结评价数据上的探索性比较。",
        "",
        "## 结论摘要",
        "",
        "- 正式资源配置准确率（冻结 Agent 动作与全动作短时仿真得到的标准最优动作 Top-1 一致率）当前不可计算，因此不能据此宣称达到 90%。原因是没有动作前可恢复的微观 SUMO 状态，也没有已保存的全动作反事实最优标签。",
        "- 配置效率主比较已改为同一 `(Order, training seed, stage, evaluation scenario)` 下的 `CONT-FIFO` 普通连续 Online DQN；throughput 是主要指标，queue、real delay、waiting time 是交叉验证。",
    ]
    lines += ["", "## 资产—指标—缺口盘点", "", "| 目标链路 | 当前资产 | 状态 | 说明 |", "|---|---|---|---|"]
    lines += [
        "| 冻结 Agent 动作 | stage-end `decisions.jsonl` | 可复用 | 可读取每个决策动作、throughput、queue 等结果 |",
        "| 冻结状态 | stage trajectory NPZ | 部分可复用 | state 是 8 维 lane-count，不含车辆位置、速度、信号内部状态 |",
        "| SUMO 微观 saveState/replay | 未发现可用于这些冻结决策点的快照/反事实重放链路 | 缺失 | `current_state.json` 是运行事务状态，不是 SUMO 微观状态 |",
        "| 全动作枚举/标准最优标签 | 未发现可直接复用的反事实 rollout 和标签表 | 缺失 | 现有 phase/lane 枚举是映射逻辑，不是最优动作搜索 |",
        "| action—lane 映射 | 由路网 `tlLogic`/controlled links 重建 | 可复用 | 已输出 `configuration_phase_lane_mapping.csv` |",
        "| 统一交通指标 | summary 含 throughput、queue、real_delay、waiting_time | 可复用 | stage-end decisions 未提供 waiting AUC 序列 |",
    ]
    lines += ["", "## 正式配置准确率", "", "### 计算定义", "", "目标是对同一可恢复交通状态枚举 8 个合法绿灯动作，以统一短时治理代价选出标准最优动作，再与冻结 Agent 的动作做 Top-1 比较。只使用 lane-count 需求捕获率会把‘即时需求匹配’误称为‘真实最优动作准确率’，本报告不这样处理。", "", "### 当前判定", "", "**不可计算 / 未判定是否达到 90%。** `configuration_accuracy_status.png` 和 `configuration_accuracy_audit.json` 给出缺失字段、搜索结果和最小补充验证方案。", ""]
    lines += ["## 效率比较口径", "", "- 基线：`CONT-FIFO`，manifest 中 `method=CONT`、`archive_mode=NONE`、`offline_ratio=0`。", "- 项目方法：`P1C-DHOA-R25` 与 `P1C-DHOA-R50`；项目方法不进入对照集合。", "- 配对键：`order_id + training_seed + stage_index + evaluation_network`。只有评价 protocol digest、仿真时长和决策步数一致的单元才纳入。", "- queue 主交叉指标为 `queue_auc = Σ(queue_network_sum × action_interval_seconds)`；因所有冻结评价时长一致，亦保留 queue mean。", "- throughput 主指标：`(HA throughput / CONT throughput − 1) × 100%`。queue、real delay 用负荷下降的正号约定；waiting time 同时给出 `change`（增加为正）和 `reduction`（下降为正）。", "- 场景等权汇总：先在场景内对匹配的 Order/seed 求均值，再对纳入场景等权平均；R25 与 R50 分开，不把 R50 O2 外推成跨 Order 结论。", ""]
    lines += ["## 最终 stage 冻结性能（相对普通连续 Online DQN）", "", "| 方法 | 场景 | 配对数 | throughput 提升 | queue AUC 下降 | real delay 下降 | waiting time 变化 | 10% throughput 场景判定 |", "|---|---|---:|---:|---:|---:|---:|---|"]
    for row in final_summary:
        lines.append("| {method} | {scene} | {n_pairs} | {throughput_improvement_pct_mean:.2f}% | {queue_auc_reduction_pct_mean:.2f}% | {real_delay_reduction_pct_mean:.2f}% | {waiting_time_change_pct_mean:.2f}% | {target} |".format(
            method=row["method"], scene=row["scene"], n_pairs=row["n_pairs"],
            throughput_improvement_pct_mean=row["throughput_improvement_pct_mean"] if finite(row["throughput_improvement_pct_mean"]) else float("nan"),
            queue_auc_reduction_pct_mean=row["queue_auc_reduction_pct_mean"] if finite(row["queue_auc_reduction_pct_mean"]) else float("nan"),
            real_delay_reduction_pct_mean=row["real_delay_reduction_pct_mean"] if finite(row["real_delay_reduction_pct_mean"]) else float("nan"),
            waiting_time_change_pct_mean=row["waiting_time_change_pct_mean"] if finite(row["waiting_time_change_pct_mean"]) else float("nan"),
            target="达到" if row["throughput_target_10pct_met"] else "未达到",
        ))
    lines += ["", "### 场景等权项目汇总", "", "| 方法 | 纳入场景 | 覆盖 Order | throughput 等权提升 | 95% bootstrap 区间 | 达到 10% 的场景数 | 场景等权 10% 判定 |", "|---|---:|---|---:|---:|---:|---|"]
    for row in scene_equal:
        lines.append("| {method} | {scenario_count} | {coverage_orders} | {value:.2f}% | [{low:.2f}%, {high:.2f}%] | {passed}/{count} | {status} |".format(
            method=row["method"], scenario_count=row["scenario_count"], coverage_orders=row["coverage_orders"],
            value=row["throughput_improvement_pct_mean"] if finite(row["throughput_improvement_pct_mean"]) else float("nan"),
            low=row["throughput_improvement_pct_ci95_low"] if finite(row["throughput_improvement_pct_ci95_low"]) else float("nan"),
            high=row["throughput_improvement_pct_ci95_high"] if finite(row["throughput_improvement_pct_ci95_high"]) else float("nan"),
            passed=row["scenarios_meeting_throughput_target"], count=row["scenario_count"],
            status="达到" if row["throughput_target_10pct_met"] else "未达到",
        ))
    lines += ["", "图 `resource_efficiency_vs_continuous.png` 展示每个场景的 throughput、queue AUC、real delay 和 waiting time 变化；`resource_efficiency_throughput_target.png` 单独保留 R25/R50 的场景等权 throughput 数值和 10% 参照线。", ""]
    lines += ["## Forgetting 验证", "", "对每个历史场景，记录其刚训练完成的 stage-end 冻结值，并追踪后续 stage 结束以及最终 stage 的同场景冻结值。对 throughput，损失定义为 `own_stage_end − later_stage`；对 queue、real delay、waiting time，负荷增加定义为 `later_stage − own_stage_end`。正的 HA advantage 表示 HA-DHOA 的退化更小。", "", "| 方法 | 范围 | 场景/聚合 | 额外训练 stage | throughput 保持率 | queue 负荷比 | real delay 比 | waiting 比 | n |", "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in forgetting_summary:
        if row["scope"] not in {"scene_transition", "historical_final_equal_scene"}:
            continue
        lines.append("| {method} | {scope} | {scene} | {offset} | {throughput:.2f}% | {queue:.2f}% | {delay:.2f}% | {waiting:.2f}% | {n} |".format(
            method=row["method"], scope=row["scope"], scene=row["scene"], offset=row["additional_scenes_trained"],
            throughput=row["throughput_retention_pct_mean"] if finite(row["throughput_retention_pct_mean"]) else float("nan"),
            queue=row["queue_burden_ratio_pct_mean"] if finite(row["queue_burden_ratio_pct_mean"]) else float("nan"),
            delay=row["real_delay_ratio_pct_mean"] if finite(row["real_delay_ratio_pct_mean"]) else float("nan"),
            waiting=row["waiting_time_ratio_pct_mean"] if finite(row["waiting_time_ratio_pct_mean"]) else float("nan"),
            n=row["n_runs"],
        ))
    lines += ["", "### HA-DHOA 相对 CONT 的遗忘优势（正值表示 HA 退化更小）", "", "| 方法 | 范围 | 额外训练 stage | throughput 遗忘优势 | queue 负荷遗忘优势 | real delay 遗忘优势 | waiting 遗忘优势 | n |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in forgetting_summary:
        if not str(row["scope"]).startswith("paired_"):
            continue
        lines.append("| {method} | {scope} | {offset} | {throughput:.2f}% | {queue:.2f}% | {delay:.2f}% | {waiting:.2f}% | {n} |".format(
            method=row["method"], scope=row["scope"], offset=row["additional_scenes_trained"],
            throughput=row["throughput_loss_pct_ha_advantage_mean"] if finite(row["throughput_loss_pct_ha_advantage_mean"]) else float("nan"),
            queue=row["queue_increase_pct_ha_advantage_mean"] if finite(row["queue_increase_pct_ha_advantage_mean"]) else float("nan"),
            delay=row["real_delay_increase_pct_ha_advantage_mean"] if finite(row["real_delay_increase_pct_ha_advantage_mean"]) else float("nan"),
            waiting=row["waiting_time_increase_pct_ha_advantage_mean"] if finite(row["waiting_time_increase_pct_ha_advantage_mean"]) else float("nan"),
            n=row["n_pairs"],
        ))
    lines += ["", "`forgetting_retention_comparison.png` 的 100% 虚线是各场景 own-stage-end 基准；throughput 越接近或高于 100% 越好，负荷类指标越接近或低于 100% 越好。该图用于机制解释，不把相关性直接升级为因果证明。", ""]
    lines += ["## 覆盖与限制", "", f"- 冻结 stage-end 单元：{len([row for row in run_catalog if row['method'] == BASELINE_METHOD])} 个 CONT 运行、{len([row for row in run_catalog if row['method'] == 'P1C-DHOA-R25'])} 个 R25 运行、{len([row for row in run_catalog if row['method'] == 'P1C-DHOA-R50'])} 个 R50 运行。", "- R25：O1–O4 × seed 0–4；R50：O2 × seed 0–4。", f"- 纳入匹配比较单元：{len(audit['counts']['matched_stage_cells']) if isinstance(audit.get('counts', {}).get('matched_stage_cells'), list) else audit.get('counts', {}).get('matched_stage_cells', '见 audit JSON')}；未匹配或口径不一致单元：{len(unmatched)}。", "- 当前交通需求实现和评价 protocol 来自冻结实验；bootstrap 区间描述 episode/order/seed 的已有波动，不声称覆盖新的交通随机性。", "- FixedTime、MaxPressure 和旧的 queue AUC/即时需求捕获率结果保留在历史输出中，但不是本次 HA 对 CONT 的主比较对象。", "", "## 最小补充验证", "", "1. 从少量代表性冻结决策点保存 SUMO 微观 saveState，或建立可证明的确定性 replay。", "2. 对每个状态固定短时窗口枚举 8 个合法动作，使用预注册的 queue/delay 代价和 tie 规则生成标准最优动作标签。", "3. 让同一 checkpoint 输出动作，计算 episode/seed/scenario 的 Top-1 一致率及置信区间；在这一步之前不报告 90% 达标。"]
    (output_dir / "resource_efficiency_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    options = args()
    output_dir = options.output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not options.allow_existing_output:
        raise FileExistsError(f"Output directory is not empty; use --allow-existing-output: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(options.bootstrap_seed)

    frozen_rows, run_catalog, missing_cells, _ = load_frozen_rows(options.ha_root)
    pairs, unmatched = pair_final_rows(frozen_rows)
    _FINAL_PAIRS_BY_METHOD_SCENE.clear()
    for pair in pairs:
        if pair["stage_index"] == 4:
            _FINAL_PAIRS_BY_METHOD_SCENE.setdefault((pair["method"], pair["evaluation_network"]), []).append(pair)
    final_summary = summarize_pairs(pairs, options.bootstrap_resamples, rng, final_only=True)
    scene_equal = scene_equal_summary(final_summary, options.bootstrap_resamples, rng)
    forgetting, forgetting_missing = forgetting_rows(frozen_rows)
    forgetting_summary = summarize_forgetting(forgetting, options.bootstrap_resamples, rng)
    baseline_forgetting = [row for row in forgetting if row["method"] == BASELINE_METHOD]
    paired_forgetting_rows = []
    for method in PROJECT_METHODS:
        method_forgetting = [row for row in forgetting if row["method"] == method]
        paired = pair_forgetting(baseline_forgetting, method_forgetting)
        paired_forgetting_rows.extend(paired)
        write_csv(output_dir / f"forgetting_{method.lower().replace('-', '_')}_paired.csv", paired)
    paired_forgetting_summary = summarize_paired_forgetting(paired_forgetting_rows, options.bootstrap_resamples, rng)
    accuracy_audit = asset_accuracy_audit(options.ha_root, output_dir, len(frozen_rows))

    write_csv(output_dir / "resource_efficiency_episode_level.csv", pairs)
    write_csv(output_dir / "resource_efficiency_scenario_summary.csv", final_summary)
    write_csv(output_dir / "resource_efficiency_comparison.csv", scene_equal)
    write_csv(
        output_dir / "resource_efficiency_match_audit.csv",
        unmatched + missing_cells,
        fields=["method", "order_id", "training_seed", "stage_index", "evaluation_network", "reason", "mismatch_fields", "alias_path"],
    )
    write_csv(output_dir / "forgetting_episode_level.csv", forgetting)
    write_csv(output_dir / "forgetting_scenario_summary.csv", forgetting_summary + paired_forgetting_summary)
    write_csv(output_dir / "forgetting_comparison.csv", paired_forgetting_rows)
    (output_dir / "configuration_accuracy_audit.json").write_text(json.dumps(accuracy_audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    all_methods = [BASELINE_METHOD, *PROJECT_METHODS]
    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_efficiency_with_formal_accuracy_gap",
        "scope": {
            "ha_root": str(options.ha_root),
            "methods": all_methods,
            "project_methods": list(PROJECT_METHODS),
            "baseline_method": BASELINE_METHOD,
            "stage_end_local_episode": 100,
            "stage_budgets": [100, 100, 100, 100],
        },
        "coverage": {
            method: {
                "logical_runs": len([run for run in run_catalog if run["method"] == method]),
                "orders": sorted({run["order_id"] for run in run_catalog if run["method"] == method}),
                "training_seeds": sorted({run["training_seed"] for run in run_catalog if run["method"] == method}),
            }
            for method in all_methods
        },
        "comparability_key": ["order_id", "training_seed", "stage_index", "evaluation_network"],
        "comparability_checks": ["evaluation_protocol_digest", "duration_seconds", "decision_steps"],
        "counts": {
            "frozen_stage_cells": len(frozen_rows),
            "matched_stage_cells": len(pairs),
            "matched_final_stage_cells": len([row for row in pairs if row["stage_index"] == 4]),
            "unmatched_or_inconsistent_cells": len(unmatched) + len(missing_cells),
            "forgetting_rows": len(forgetting),
            "forgetting_missing_rows": len(forgetting_missing),
        },
        "efficiency_metrics": {
            "throughput": "(HA - CONT) / CONT * 100; higher is better; 10% exploratory target",
            "queue_auc": "sum(queue_network_sum * action_interval_seconds); reduction is positive",
            "real_delay": "(CONT - HA) / CONT * 100; reduction is positive",
            "waiting_time_change": "(HA - CONT) / CONT * 100; increase is positive, reduction = negative change",
            "aggregation": "within-scenario paired mean, then equal weight over scenarios",
        },
        "accuracy": accuracy_audit,
        "forgetting": {
            "own_stage_end": "scene's evaluation at the stage in which it was trained",
            "throughput_loss": "own_stage_end - later_stage",
            "burden_increase": "later_stage - own_stage_end for queue_auc/real_delay/waiting_time",
            "advantage_sign": "positive HA advantage means less degradation than CONT",
        },
        "source_paths": {
            "ha_root": str(options.ha_root),
            "output_dir": str(output_dir),
            "existing_fixedtime_and_old_accuracy_outputs": str(ROOT / "data/output_data/resource_metric_audit_p1c_dhoa_r25_r50"),
        },
        "bootstrap": {"resamples": options.bootstrap_resamples, "seed": options.bootstrap_seed, "interpretation": "existing paired order/seed variation; not new traffic randomness"},
    }
    (output_dir / "resource_efficiency_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    render_performance_plot(output_dir, final_summary, scene_equal)
    render_forgetting_plot(output_dir, forgetting_summary)
    render_accuracy_status(output_dir, accuracy_audit)
    write_report(output_dir, audit, accuracy_audit, run_catalog, missing_cells + forgetting_missing, unmatched, final_summary, scene_equal, forgetting_summary + paired_forgetting_summary)
    print(json.dumps({"output_dir": str(output_dir), "counts": audit["counts"], "scene_equal": scene_equal, "accuracy": accuracy_audit["formal_result"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
