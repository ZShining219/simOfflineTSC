#!/usr/bin/env python3
"""Build traceable evidence for frozen-action training-experience support.

This is a derived, read-only audit. It does not train an agent or start SUMO.
The primary metric is conditional on evaluation states for which the pooled
P1C-DHOA-R25 effective-attempt trajectories contain a valid exact match on
``(scene, phase, 8D lane-count state)``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HA_ROOT = ROOT / "data/output_data/ha_sodqn/formal_e7705f7_20260726"
DEFAULT_ALLOCATION_ROOT = ROOT / "data/output_data/allocation_accuracy"
DEFAULT_RESOURCE_ROOT = (
    ROOT / "data/output_data/resource_metric_audit_p1c_dhoa_r25_r50"
)
DEFAULT_OUTPUT = DEFAULT_ALLOCATION_ROOT / "experience_traceability_support"
METHOD = "P1C-DHOA-R25"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ha-root", type=Path, default=DEFAULT_HA_ROOT)
    parser.add_argument(
        "--allocation-root", type=Path, default=DEFAULT_ALLOCATION_ROOT
    )
    parser.add_argument("--resource-root", type=Path, default=DEFAULT_RESOURCE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--allow-existing-output",
        action="store_true",
        help="replace only this audit's named files in an existing directory",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def state_key(scene: str, phase: int, state) -> tuple:
    vector = tuple(float(value) for value in np.asarray(state).reshape(-1))
    return scene, int(phase), vector


def completed_effective_attempt(logical_dir: Path) -> tuple[dict, str, Path]:
    manifest_path = logical_dir / "logical_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise ValueError(f"{logical_dir.name}: logical run is not completed")
    effective_attempt = manifest["effective_attempt"]
    attempt = next(
        item
        for item in manifest["attempts"]
        if item["attempt_id"] == effective_attempt
    )
    if attempt.get("status") != "completed":
        raise ValueError(
            f"{logical_dir.name}/{effective_attempt}: attempt is not completed"
        )
    attempt_dir = Path(attempt["attempt_dir"])
    if not attempt_dir.is_absolute():
        attempt_dir = (logical_dir / attempt_dir).resolve()
    return manifest, effective_attempt, attempt_dir


def witness(
    logical_run_id: str,
    effective_attempt: str,
    npz_path: Path,
    transition_id: str,
    decision_index: int,
    action: int,
    reward: float,
) -> dict:
    return {
        "logical_run_id": logical_run_id,
        "effective_attempt": effective_attempt,
        "trajectory_npz": relative(npz_path),
        "transition_id": transition_id,
        "decision_index": int(decision_index),
        "action": int(action),
        "reward": float(reward),
    }


def main() -> None:
    args = arguments()
    if args.output_dir.exists() and not args.allow_existing_output:
        raise FileExistsError(
            f"Output exists; use --allow-existing-output: {args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    state_manifest_path = args.allocation_root / "allocation_accuracy_state_manifest.csv"
    predictions_path = (
        args.allocation_root / "allocation_accuracy_agent_predictions.csv"
    )
    state_rows = {row["state_id"]: row for row in read_csv(state_manifest_path)}
    predictions = [
        row
        for row in read_csv(predictions_path)
        if row["method"] == METHOD and row.get("predicted_action", "").strip()
    ]

    targets = []
    target_keys = set()
    for prediction in predictions:
        state_row = state_rows[prediction["state_id"]]
        vector = json.loads(state_row["state_vector"])
        key = state_key(state_row["scene"], state_row["phase"], vector)
        targets.append((prediction, state_row, key))
        target_keys.add(key)

    observations = defaultdict(
        lambda: {
            "exact_count": 0,
            "valid_count": 0,
            "empty_demand_count": 0,
            "illegal_action_count": 0,
            "actions": defaultdict(int),
            "action_reward_sum": defaultdict(float),
            "action_witness": {},
            "valid_witness": None,
            "any_witness": None,
            "source_files": set(),
            "logical_runs": set(),
        }
    )
    source_rows = []
    total_npz_files = 0
    total_transitions = 0
    matched_exact_transitions = 0
    matched_valid_transitions = 0

    for logical_dir in sorted(args.ha_root.iterdir()):
        if not logical_dir.name.startswith(METHOD + "-"):
            continue
        manifest_path = logical_dir / "logical_run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest, effective_attempt, attempt_dir = completed_effective_attempt(
            logical_dir
        )
        npz_paths = sorted(attempt_dir.glob("trajectory/episodes/*.npz"))
        run_transition_count = 0
        run_exact_count = 0
        run_valid_count = 0
        for npz_path in npz_paths:
            total_npz_files += 1
            with np.load(npz_path, allow_pickle=False) as data:
                states = np.asarray(data["state"]).reshape(-1, 8)
                phases = np.asarray(data["phase"]).reshape(-1)
                actions = np.asarray(data["action"]).reshape(-1)
                rewards = np.asarray(data["reward"]).reshape(-1)
                transition_ids = np.asarray(data["transition_id"]).reshape(-1)
                decision_indices = np.asarray(data["decision_index"]).reshape(-1)
                run_transition_count += len(actions)
                total_transitions += len(actions)
                for state, phase, action, reward, transition_id, decision_index in zip(
                    states,
                    phases,
                    actions,
                    rewards,
                    transition_ids,
                    decision_indices,
                ):
                    scene = str(transition_id).split(":", 1)[0]
                    key = state_key(scene, phase, state)
                    if key not in target_keys:
                        continue
                    record = observations[key]
                    record["exact_count"] += 1
                    run_exact_count += 1
                    matched_exact_transitions += 1
                    action = int(action)
                    current_witness = witness(
                        logical_dir.name,
                        effective_attempt,
                        npz_path,
                        str(transition_id),
                        int(decision_index),
                        action,
                        float(reward),
                    )
                    if record["any_witness"] is None:
                        record["any_witness"] = current_witness
                    if not 0 <= action < 8:
                        record["illegal_action_count"] += 1
                        continue
                    if float(np.asarray(state, dtype=float).sum()) <= 0.0:
                        record["empty_demand_count"] += 1
                        continue
                    record["valid_count"] += 1
                    record["actions"][action] += 1
                    record["action_reward_sum"][action] += float(reward)
                    record["source_files"].add(relative(npz_path))
                    record["logical_runs"].add(logical_dir.name)
                    record["action_witness"].setdefault(action, current_witness)
                    if record["valid_witness"] is None:
                        record["valid_witness"] = current_witness
                    run_valid_count += 1
                    matched_valid_transitions += 1
        source_rows.append(
            {
                "method": METHOD,
                "logical_run_id": logical_dir.name,
                "logical_status": manifest.get("status"),
                "effective_attempt": effective_attempt,
                "effective_attempt_status": "completed",
                "logical_manifest": relative(manifest_path),
                "effective_attempt_dir": relative(attempt_dir),
                "trajectory_npz_count": len(npz_paths),
                "trajectory_transition_count": run_transition_count,
                "matched_exact_key_transition_count": run_exact_count,
                "matched_valid_reference_transition_count": run_valid_count,
            }
        )

    detail_rows = []
    digest_cache = {}
    for prediction, state_row, key in targets:
        record = observations.get(key)
        exact_seen = bool(record and record["exact_count"])
        covered = bool(record and record["valid_count"])
        predicted_action = int(prediction["predicted_action"])
        predicted_count = record["actions"].get(predicted_action, 0) if covered else 0
        supported = predicted_count > 0
        if not exact_seen:
            exclusion_reason = "exact_training_key_not_observed"
        elif not covered and record["empty_demand_count"]:
            exclusion_reason = "only_empty_demand_training_observations"
        elif not covered and record["illegal_action_count"]:
            exclusion_reason = "only_illegal_action_training_observations"
        elif not covered:
            exclusion_reason = "no_valid_training_observation"
        else:
            exclusion_reason = ""
        if supported:
            chosen_witness = record["action_witness"][predicted_action]
            witness_role = "exact_state_phase_and_predicted_action"
        elif covered:
            chosen_witness = record["valid_witness"]
            witness_role = "exact_state_phase_only_no_predicted_action_match"
        elif exact_seen:
            chosen_witness = record["any_witness"]
            witness_role = "excluded_exact_key_observation"
        else:
            chosen_witness = None
            witness_role = ""
        witness_path = chosen_witness["trajectory_npz"] if chosen_witness else ""
        if witness_path:
            digest_cache.setdefault(witness_path, sha256(ROOT / witness_path))
        reward_mean = (
            record["action_reward_sum"][predicted_action] / predicted_count
            if supported
            else None
        )
        detail_rows.append(
            {
                "method": METHOD,
                "state_id": prediction["state_id"],
                "scene": state_row["scene"],
                "phase": int(state_row["phase"]),
                "state_vector": state_row["state_vector"],
                "predicted_action": predicted_action,
                "final_checkpoint": relative(Path(prediction["checkpoint"])),
                "final_checkpoint_sha256": prediction["checkpoint_digest"],
                "exact_training_key_seen": exact_seen,
                "training_reference_covered": covered,
                "coverage_exclusion_reason": exclusion_reason,
                "training_exact_state_phase_observation_count": (
                    record["exact_count"] if exact_seen else 0
                ),
                "training_valid_observation_count": (
                    record["valid_count"] if exact_seen else 0
                ),
                "training_invalid_empty_demand_count": (
                    record["empty_demand_count"] if exact_seen else 0
                ),
                "training_invalid_illegal_action_count": (
                    record["illegal_action_count"] if exact_seen else 0
                ),
                "training_observed_actions": json.dumps(
                    sorted(record["actions"]) if covered else []
                ),
                "training_observed_action_counts": json.dumps(
                    {
                        str(action): record["actions"][action]
                        for action in sorted(record["actions"])
                    }
                    if covered
                    else {},
                    sort_keys=True,
                ),
                "predicted_action_training_observation_count": predicted_count,
                "predicted_action_training_reward_mean": reward_mean,
                "experience_supported": supported,
                "training_source_npz_count": (
                    len(record["source_files"]) if covered else 0
                ),
                "training_source_logical_run_count": (
                    len(record["logical_runs"]) if covered else 0
                ),
                "training_source_logical_runs": json.dumps(
                    sorted(record["logical_runs"]) if covered else []
                ),
                "witness_role": witness_role,
                "witness_trajectory_npz": witness_path,
                "witness_trajectory_npz_sha256": digest_cache.get(witness_path, ""),
                "witness_logical_run_id": (
                    chosen_witness["logical_run_id"] if chosen_witness else ""
                ),
                "witness_effective_attempt": (
                    chosen_witness["effective_attempt"] if chosen_witness else ""
                ),
                "witness_transition_id": (
                    chosen_witness["transition_id"] if chosen_witness else ""
                ),
                "witness_decision_index": (
                    chosen_witness["decision_index"] if chosen_witness else ""
                ),
                "witness_action": (
                    chosen_witness["action"] if chosen_witness else ""
                ),
                "witness_reward": (
                    chosen_witness["reward"] if chosen_witness else ""
                ),
            }
        )

    evaluation_states = len(detail_rows)
    exact_seen_states = sum(row["exact_training_key_seen"] for row in detail_rows)
    covered_states = sum(row["training_reference_covered"] for row in detail_rows)
    supported_states = sum(row["experience_supported"] for row in detail_rows)
    unsupported_covered = sum(
        row["training_reference_covered"] and not row["experience_supported"]
        for row in detail_rows
    )
    uncovered_states = evaluation_states - covered_states
    conditional_pct = 100.0 * supported_states / covered_states
    overall_pct = 100.0 * supported_states / evaluation_states
    summary_rows = [
        {
            "method": METHOD,
            "evaluation_states": evaluation_states,
            "exact_training_key_seen_states_before_validity_filter": exact_seen_states,
            "training_reference_covered_states": covered_states,
            "training_reference_coverage_pct": (
                100.0 * covered_states / evaluation_states
            ),
            "experience_supported_states": supported_states,
            "experience_unsupported_covered_states": unsupported_covered,
            "training_reference_uncovered_states": uncovered_states,
            "experience_traceability_pct_on_covered_states": conditional_pct,
            "experience_support_pct_on_all_evaluation_states": overall_pct,
            "primary_fraction": f"{supported_states}/{covered_states}",
            "all_state_fraction": f"{supported_states}/{evaluation_states}",
        }
    ]

    write_csv(args.output_dir / "experience_traceability_summary.csv", summary_rows)
    write_csv(
        args.output_dir / "experience_traceability_state_evidence.csv", detail_rows
    )
    write_csv(
        args.output_dir / "experience_traceability_training_sources.csv", source_rows
    )

    effectiveness_path = args.resource_root / "resource_metric_comparison.csv"
    exploration_quality_path = (
        args.resource_root / "resource_accuracy_quantitative_evidence.csv"
    )
    evidence_catalog = [
        {
            "evidence_role": "training_exploration_state_phase_action_reward",
            "scope": "P1C-DHOA-R25 completed effective attempts",
            "path": relative(args.ha_root),
            "key_fields_or_metrics": "state;phase;action;reward;transition_id",
            "relationship_to_93_48_pct": "direct numerator/denominator source",
        },
        {
            "evidence_role": "frozen_agent_action_output",
            "scope": "256 allocation-accuracy evaluation states",
            "path": relative(predictions_path),
            "key_fields_or_metrics": (
                "state_id;predicted_action;checkpoint;checkpoint_digest"
            ),
            "relationship_to_93_48_pct": "direct frozen-action source",
        },
        {
            "evidence_role": "evaluation_state_identity",
            "scope": "256 allocation-accuracy evaluation states",
            "path": relative(state_manifest_path),
            "key_fields_or_metrics": "state_id;scene;phase;state_vector",
            "relationship_to_93_48_pct": "direct exact-match key source",
        },
        {
            "evidence_role": "policy_level_traffic_effectiveness",
            "scope": "frozen R25/R50 and traditional controls",
            "path": relative(effectiveness_path),
            "key_fields_or_metrics": (
                "queue;real_delay;throughput;service_efficiency"
            ),
            "relationship_to_93_48_pct": "indirect policy-level value support",
        },
        {
            "evidence_role": "training_exploration_quality_diagnostic",
            "scope": "stage-end training trajectories",
            "path": relative(exploration_quality_path),
            "key_fields_or_metrics": "Acc@0.9;valid_decisions;CI95",
            "relationship_to_93_48_pct": (
                "indirect exploration-quality support"
            ),
        },
    ]
    write_csv(
        args.output_dir / "experience_traceability_evidence_catalog.csv",
        evidence_catalog,
    )

    audit = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed_read_only_source_audit",
        "metric_name": "configuration_decision_experience_traceability",
        "method": METHOD,
        "definition": (
            "Among evaluation states with a valid exact training reference match "
            "on (scene, phase, 8D lane-count state), the fraction whose frozen "
            "final action was observed in training for the same exact key."
        ),
        "matching_key": ["scene", "phase", "8D_lane_count_state"],
        "valid_training_reference": (
            "legal action AND sum(8D lane-count state) > 0"
        ),
        "numerator": supported_states,
        "denominator": covered_states,
        "value_pct": conditional_pct,
        "all_evaluation_state_context": {
            "numerator": supported_states,
            "denominator": evaluation_states,
            "value_pct": overall_pct,
        },
        "counts": {
            "evaluation_states": evaluation_states,
            "exact_key_seen_states_before_validity_filter": exact_seen_states,
            "covered_states": covered_states,
            "supported_states": supported_states,
            "covered_but_unsupported_states": unsupported_covered,
            "uncovered_states": uncovered_states,
            "logical_runs": len(source_rows),
            "effective_attempt_trajectory_npz_files": total_npz_files,
            "effective_attempt_trajectory_transitions": total_transitions,
            "matched_exact_key_transitions": matched_exact_transitions,
            "matched_valid_reference_transitions": matched_valid_transitions,
        },
        "source_policy": {
            "logical_run_status_required": "completed",
            "attempt_selection": (
                "logical_run_manifest.json -> effective_attempt"
            ),
            "effective_attempt_status_required": "completed",
            "training_scope": (
                "all persisted trajectory/episodes/*.npz in each effective attempt"
            ),
            "evaluation_scope": (
                "non-empty frozen P1C-DHOA-R25 predictions in "
                "allocation_accuracy_agent_predictions.csv"
            ),
        },
        "source_paths": {
            "training_root": relative(args.ha_root),
            "evaluation_predictions": relative(predictions_path),
            "evaluation_state_manifest": relative(state_manifest_path),
            "checkpoint_and_prediction_module": (
                "tools/allocation_accuracy_counterfactual.py"
            ),
            "trajectory_schema_and_validity_module": (
                "tools/resource_metric_audit.py"
            ),
            "traceability_calculation_module": (
                "tools/allocation_experience_traceability.py"
            ),
            "traffic_effectiveness_summary": relative(effectiveness_path),
            "exploration_quality_summary": relative(exploration_quality_path),
        },
        "source_digests": {
            "evaluation_predictions_sha256": sha256(predictions_path),
            "evaluation_state_manifest_sha256": sha256(state_manifest_path),
            "traffic_effectiveness_summary_sha256": sha256(effectiveness_path),
            "exploration_quality_summary_sha256": sha256(
                exploration_quality_path
            ),
        },
        "calculation_pseudocode": [
            "training_index[(scene, phase, state)][action] += 1",
            "covered = training_index[key] has any valid observation",
            "supported = covered and frozen_action in training_index[key]",
            "traceability = sum(supported) / sum(covered)",
        ],
        "interpretation_limits": [
            (
                "93.48% is conditional on 184 covered evaluation states; "
                "it is 67.19% over all 256 evaluation states."
            ),
            (
                "The metric proves action lineage/support in recorded exploration "
                "experience, not that each matched action was individually optimal."
            ),
            (
                "Throughput, queue, delay and reward evidence supports policy-level "
                "effectiveness; it does not convert every recorded training action "
                "into an individually validated expert label."
            ),
            (
                "Exact matching uses the stored 8D lane-count observation and phase, "
                "not a complete microscopic SUMO state."
            ),
        ],
        "no_new_training": True,
        "no_new_sumo_simulation": True,
    }
    (args.output_dir / "experience_traceability_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output_dir": relative(args.output_dir),
                "summary": summary_rows[0],
                "counts": audit["counts"],
                "witness_npz_files_hashed": len(digest_cache),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
