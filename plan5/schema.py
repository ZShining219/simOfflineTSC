"""Frozen result table schemas for Plan5 analysis outputs."""
import csv
import os
from pathlib import Path
import tempfile

from sequential.io import sha256_file

RUN_MANIFEST_COLUMNS = (
    "logical_run_id", "algorithm_id", "run_type", "scene",
    "transition_id", "source_scene", "target_scene", "training_seed",
    "config_sha256", "source_commit", "status",
    "source_checkpoint_path", "source_checkpoint_sha256",
    "model_state_digest", "optimizer_state_digest",
    "algorithm_state_digest", "rng_state_digest", "attempt_id",
    "ppo_config_path", "ppo_config_sha256",
)


TABLE_SCHEMAS = {
    "plan5_run_manifest.csv": RUN_MANIFEST_COLUMNS,
    "ppo_calibration_run_manifest.csv": (
        "logical_run_id", "algorithm_id", "run_type", "scene",
        "training_seed", "learning_rate", "entropy_coefficient",
        "config_sha256", "source_commit", "status", "attempt_id",
    ),
    "plan5_anchor_summary.csv": (
        "logical_run_id", "algorithm_id", "scene", "network",
        "training_seed", "status", "completed", "finite_metrics",
        "checkpoint_valid", "resume_valid", "evaluation_isolation_valid",
        "fixed_probe_valid", "late_metrics_present", "tt_episode0",
        "tt_late", "tt_fixedtime", "improvement_fraction", "valid",
        "episode100_checkpoint_path", "episode100_checkpoint_sha256",
        "run_summary_path", "run_summary_sha256", "attempt_id",
        "failure_class",
    ),
    "plan5_transition_summary.csv": (
        "logical_run_id", "algorithm_id", "transition_id",
        "source_scene", "target_scene", "training_seed", "status",
        "completed", "finite_metrics", "checkpoint_valid",
        "resume_valid", "evaluation_isolation_valid", "fixed_probe_valid",
        "same_start_valid", "worst_historical_degradation",
        "episode100_degradation", "recovery_gap", "specialist_tt",
        "current_adaptation_aulc", "episode100_normalized_performance",
        "time_to_reference", "time_to_reference_censored",
        "max_policy_disagreement_main", "policy_disagreement100_main",
        "max_policy_disagreement_heldout",
        "policy_disagreement100_heldout", "valid", "attempt_id",
        "failure_class",
    ),
    "plan5_historical_timeline.csv": (
        "algorithm_id", "transition_id", "training_seed", "source_scene",
        "episode", "travel_time", "degradation", "mean_queue",
        "mean_delay", "throughput", "mean_reward", "evaluation_manifest",
        "evaluation_manifest_sha256", "valid",
    ),
    "plan5_current_adaptation.csv": (
        "algorithm_id", "transition_id", "training_seed", "target_scene",
        "episode", "travel_time", "specialist_tt",
        "normalized_travel_time", "mean_queue", "mean_delay",
        "throughput", "mean_reward", "evaluation_manifest",
        "evaluation_manifest_sha256", "valid",
    ),
    "plan5_policy_displacement.csv": (
        "algorithm_id", "transition_id", "training_seed", "source_scene",
        "episode", "split", "probe_row_count", "policy_disagreement",
        "episode0_actions_digest", "episode_actions_digest", "valid",
    ),
    "plan5_internal_metrics.csv": (
        "algorithm_id", "transition_id", "training_seed", "scene",
        "episode", "split", "metric", "value", "valid",
    ),
    "plan5_same_start_H34_L32.csv": (
        "algorithm_id", "training_seed", "metric", "h34", "l32",
        "effect", "source_checkpoint_sha256", "same_start_valid",
        "row_type", "effect_mean", "effect_median",
        "direction_consistency", "ci95_lower", "ci95_upper",
        "bootstrap_resamples", "analysis_seed",
    ),
    "plan5_secondary_H43_L23.csv": (
        "algorithm_id", "training_seed", "metric", "h43", "l23",
        "effect", "same_target_scene", "valid",
    ),
    "ppo_calibration_summary.csv": (
        "learning_rate", "entropy_coefficient", "training_seed",
        "logical_run_id", "status", "tt_episode0", "tt_late_median",
        "tt_fixedtime", "improvement_fraction",
        "current_adaptation_aulc", "candidate_score", "candidate_iqr",
        "candidate_aulc_median", "candidate_rank", "selected",
        "seed_eligible", "candidate_eligible_seed_count",
        "formal_eligible", "run_summary_path", "run_summary_sha256",
    ),
    "plan5_cross_algorithm_summary.csv": (
        "algorithm_left", "algorithm_right", "metric",
        "training_seed", "effect_left", "effect_right",
        "difference", "row_type", "difference_mean",
        "difference_median", "seed_consistency_positive",
        "ci95_lower", "ci95_upper", "bootstrap_resamples",
        "analysis_seed", "valid",
    ),
    "plan5_context_distinction.csv": (
        "probe_id", "pair_weighted_distinguished_fraction",
        "unique_raw_state_weighted_distinguished_fraction",
        "matched_raw_state_groups", "matched_pairs", "valid",
    ),
}


JSON_REQUIRED_FIELDS = {
    "plan5_validation_report.json": (
        "schema_version", "phase", "phase0_engineering_ready",
        "environment_ready", "source_ready", "canonical_reference_ready",
        "fixedtime_ready", "probe_ready", "tests_passed", "smoke_ready",
        "resume_ready", "evaluation_isolation_ready",
        "resource_gate_ready", "selected_formal_concurrency",
        "formal_training", "reason",
    ),
    "plan5_resource_report.json": (
        "schema_version", "profile", "levels",
        "selected_formal_concurrency", "valid",
    ),
    "canonical_reference_manifest.json": (
        "schema_version", "plan_id", "source", "assets",
        "available_counts", "manifest_sha256",
    ),
    "plan5_fixedtime_reference.json": (
        "schema_version", "protocol", "scenes", "digest",
    ),
    "ppo_formal_config.json": (
        "schema_version", "algorithm_id", "learning_rate",
        "entropy_coefficient", "calibration_summary_sha256",
        "config_sha256", "formal_status",
    ),
    "probe_manifest.json": (
        "schema_version", "probe_id", "validation", "paths",
        "trajectories", "boundary_entry_mapping",
        "all_causal_gates_valid", "manifest_digest",
    ),
}


def validate_table_header(path, table_name=None):
    table_name=table_name or Path(path).name
    expected=TABLE_SCHEMAS.get(table_name)
    if expected is None: raise ValueError(f"Unknown Plan5 table: {table_name}")
    with open(path,newline="",encoding="utf-8") as handle:
        actual=tuple(next(csv.reader(handle),()))
    if actual != expected: raise ValueError(f"Schema mismatch for {table_name}: {actual!r}")
    return {"valid":True,"table":table_name,"columns":list(expected)}


def validate_json_fields(path, schema_name=None):
    import json
    schema_name = schema_name or Path(path).name
    expected = JSON_REQUIRED_FIELDS.get(schema_name)
    if expected is None:
        raise ValueError(f"Unknown Plan5 JSON schema: {schema_name}")
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    missing = sorted(set(expected) - set(payload))
    if missing:
        raise ValueError(
            f"JSON schema mismatch for {schema_name}: missing {missing}"
        )
    return {"valid": True, "schema": schema_name,
            "required_fields": list(expected)}


def write_table(path, table_name, rows):
    """Atomically freeze one schema-exact Plan5 result table."""
    rows = list(rows)
    columns = TABLE_SCHEMAS.get(table_name)
    if columns is None:
        raise ValueError(f'Unknown Plan5 table: {table_name}')
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f'.tmp-{table_name}-', dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, 'w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(
                handle, fieldnames=columns, extrasaction='raise',
            )
            writer.writeheader()
            for row in rows:
                missing = set(columns) - set(row)
                if missing:
                    raise ValueError(
                        f'Plan5 {table_name} row missing {sorted(missing)}'
                    )
                writer.writerow(row)
            handle.flush(); os.fsync(handle.fileno())
        if path.exists():
            if sha256_file(path) != sha256_file(temporary):
                raise FileExistsError(f'Plan5 result table is immutable: {path}')
            os.unlink(temporary); temporary = None
        else:
            os.replace(temporary, path); temporary = None
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    validate_table_header(path, table_name)
    return {
        'path': str(path.resolve()), 'sha256': sha256_file(path),
        'rows': len(rows),
    }


def write_schema_index(output_path):
    from sequential.io import atomic_json
    atomic_json(output_path, {
        "schema_version": 1,
        "tables": {
            name: list(columns) for name, columns in TABLE_SCHEMAS.items()
        },
        "json_required_fields": {
            name: list(fields)
            for name, fields in JSON_REQUIRED_FIELDS.items()
        },
    })
