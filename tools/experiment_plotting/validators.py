import hashlib
import json
from pathlib import Path

import torch
import yaml

from utils.logger import StructuredMetricLogger, verify_config_archive
from utils.run_config_compare import compare_runs

from .loaders import load_json, load_metric_records


DQN_EVIDENCE_ROLES = {"pilot", "formal"}
ALLOWED_ROLES = {"baseline", "pilot", "formal", "smoke"}


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_trajectory_files(run_dir, spec, validation):
    trajectory_dir = run_dir / "trajectory"
    manifest = load_json(trajectory_dir / "manifest.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("storage") != "episode_npz"
        or manifest.get("network") != spec.network
        or manifest.get("behavior_training_seed") != spec.training_seed
    ):
        raise ValueError(f"Invalid trajectory manifest: {trajectory_dir / 'manifest.json'}")
    entries = []
    index_path = trajectory_dir / "index.jsonl"
    with index_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid trajectory index at {index_path}:{line_number}"
                ) from error
            entries.append(entry)
    total = 0
    previous_last_step = 0
    for expected_episode, entry in enumerate(entries, start=1):
        if entry.get("schema_version") != 1 or entry.get("episode_id") != expected_episode:
            raise ValueError(f"Non-contiguous trajectory index: {index_path}")
        relative_path = Path(entry.get("file", ""))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"Unsafe trajectory shard path in {index_path}")
        shard_path = trajectory_dir / relative_path
        if not shard_path.is_file() or _sha256(shard_path) != entry.get("sha256"):
            raise ValueError(f"Trajectory shard hash mismatch: {shard_path}")
        transition_count = entry.get("transition_count")
        first_step = entry.get("first_global_step")
        last_step = entry.get("last_global_step")
        if (
            not isinstance(transition_count, int) or transition_count <= 0
            or first_step != previous_last_step + 1
            or last_step != first_step + transition_count - 1
        ):
            raise ValueError(f"Invalid trajectory count/continuity index: {index_path}")
        total += transition_count
        previous_last_step = last_step
    if (
        len(entries) != validation.get("episode_count")
        or total != validation.get("transition_count")
    ):
        raise ValueError(f"Trajectory validation/index count mismatch: {trajectory_dir}")


def _checkpoint_episodes(directory, checkpoint_type, config_hash):
    files = sorted(directory.glob("episode_*.pt"))
    episodes = []
    for path in files:
        try:
            payload = torch.load(path, map_location="cpu")
        except Exception as error:
            raise IOError(f"Cannot load checkpoint: {path}") from error
        if (
            not isinstance(payload, dict)
            or payload.get("schema_version") != 1
            or payload.get("checkpoint_type") != checkpoint_type
            or payload.get("config_hash") != config_hash
            or not isinstance(payload.get("episode"), int)
        ):
            raise ValueError(f"Invalid {checkpoint_type} checkpoint: {path}")
        expected_name = f"episode_{payload['episode']:04d}.pt"
        if path.name != expected_name:
            raise ValueError(f"Checkpoint filename/episode mismatch: {path}")
        episodes.append(payload["episode"])
    return episodes


def _validate_evaluation_protocol_v2(
    run_dir, records, summary, resolved_config, config_hash, trajectory_validation,
):
    trainer = resolved_config.get("trainer") or {}
    episodes = trainer.get("episodes")
    evaluation_episodes = trainer.get("evaluation_episodes")
    resumable_episodes = trainer.get("resumable_checkpoint_episodes")
    expected_evaluations = list(range(episodes + 1)) if isinstance(episodes, int) else None
    if evaluation_episodes != expected_evaluations:
        raise ValueError(f"New protocol requires evaluation episodes 0..{episodes}: {run_dir}")
    if (
        not isinstance(resumable_episodes, list)
        or not resumable_episodes
        or resumable_episodes[0] != 0
        or resumable_episodes[-1] != episodes
        or resumable_episodes != sorted(set(resumable_episodes))
        or any(item not in evaluation_episodes for item in resumable_episodes)
    ):
        raise ValueError(f"Invalid resumable checkpoint schedule: {run_dir}")

    training = [row for row in records if row["record_type"] == "TRAIN"]
    evaluations = [
        row for row in records
        if row["record_type"] in {"EVALUATION", "FINAL_EVALUATION"}
    ]
    if [row["episode"] for row in training] != list(range(1, episodes + 1)):
        raise ValueError(f"TRAIN episodes are incomplete or non-contiguous: {run_dir}")
    if [row["episode"] for row in evaluations] != expected_evaluations:
        raise ValueError(f"Evaluation episodes are incomplete or non-contiguous: {run_dir}")
    if (
        any(row["record_type"] != "EVALUATION" for row in evaluations[:-1])
        or evaluations[-1]["record_type"] != "FINAL_EVALUATION"
    ):
        raise ValueError(f"Invalid final evaluation record placement: {run_dir}")

    if (
        summary.get("evaluation_episodes") != evaluation_episodes
        or summary.get("resumable_checkpoint_episodes") != resumable_episodes
        or summary.get("evaluation_checkpoint_count") != len(evaluation_episodes)
        or summary.get("resumable_checkpoint_count") != len(resumable_episodes)
        or summary.get("evaluation_isolation_check_count") != len(evaluation_episodes)
        or summary.get("evaluation_rng_unchanged") is not True
        or [item.get("episode") for item in summary.get("evaluations", [])]
        != evaluation_episodes
    ):
        raise ValueError(f"Evaluation summary does not match the new protocol: {run_dir}")

    evaluation_checkpoints = _checkpoint_episodes(
        run_dir / "checkpoints" / "evaluation", "evaluation", config_hash
    )
    resumable_checkpoints = _checkpoint_episodes(
        run_dir / "checkpoints" / "resumable", "resumable", config_hash
    )
    if evaluation_checkpoints != evaluation_episodes:
        raise ValueError(f"Evaluation checkpoint schedule mismatch: {run_dir}")
    if resumable_checkpoints != resumable_episodes:
        raise ValueError(f"Resumable checkpoint schedule mismatch: {run_dir}")

    steps = trainer.get("steps")
    action_interval = trainer.get("action_interval")
    expected_transitions = (
        episodes * ((steps + action_interval - 1) // action_interval)
        if isinstance(steps, int) and isinstance(action_interval, int)
        else None
    )
    if (
        trajectory_validation.get("episode_count") != episodes
        or trajectory_validation.get("transition_count") != expected_transitions
        or trajectory_validation.get("evaluation_transition_count") != 0
    ):
        raise ValueError(f"Trajectory counts do not match the new protocol: {run_dir}")


def validate_run(spec):
    run_dir = spec.run_dir
    if spec.role not in ALLOWED_ROLES:
        raise ValueError(
            f"Unsupported role {spec.role!r} for {run_dir}; "
            f"expected one of {sorted(ALLOWED_ROLES)}"
        )
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    manifest = load_json(run_dir / "run_manifest.json")
    status = load_json(run_dir / "run_status.json")
    if manifest.get("schema_version") != 1 or status.get("schema_version") != 1:
        raise ValueError(f"Unsupported run schema for {run_dir}")
    if status.get("status") != "已完成" or status.get("exit_code") != 0:
        raise ValueError(f"Run did not complete successfully: {run_dir}")
    expected_identity = {
        "agent": spec.agent,
        "network": spec.network,
        "training_seed": spec.training_seed,
    }
    mismatches = [
        f"{field}: run={manifest.get(field)!r}, list={expected!r}"
        for field, expected in expected_identity.items()
        if manifest.get(field) != expected
    ]
    if mismatches:
        raise ValueError(
            f"Run-list identity mismatch for {run_dir}: " + "; ".join(mismatches)
        )

    config_dir = run_dir / "config"
    verify_config_archive(str(config_dir))
    resolved_path = config_dir / "resolved_config.yaml"
    with resolved_path.open(encoding="utf-8") as handle:
        resolved_config = yaml.safe_load(handle)
    resolved_hash = _sha256(resolved_path)
    if manifest.get("config_hash") != resolved_hash:
        raise ValueError(
            f"Run manifest config hash mismatch for {run_dir}: "
            f"{manifest.get('config_hash')} != {resolved_hash}"
        )

    StructuredMetricLogger(str(run_dir)).validate(require_records=True)
    records = load_metric_records(run_dir)
    for record in records:
        if record["agent"] != spec.agent or record["network"] != spec.network:
            raise ValueError(f"Metric identity mismatch in {run_dir}")
        if record["training_seed"] != spec.training_seed:
            raise ValueError(f"Metric training_seed mismatch in {run_dir}")

    evaluation_summary = None
    summary_path = run_dir / "evaluation" / "summary.json"
    if summary_path.is_file():
        evaluation_summary = load_json(summary_path)
        if evaluation_summary.get("schema_version") not in {1, 2}:
            raise ValueError(f"Unsupported evaluation summary schema: {summary_path}")
        if spec.agent == "dqn" and spec.role == "formal" and (
            evaluation_summary.get("schema_version") != 2
        ):
            raise ValueError(
                f"Formal DQN run requires evaluation summary schema v2: {summary_path}"
            )
    elif spec.agent == "dqn" and spec.role in DQN_EVIDENCE_ROLES:
        raise FileNotFoundError(f"DQN {spec.role} evaluation summary is missing: {summary_path}")

    trajectory_validation = None
    if spec.agent == "dqn" and spec.role in DQN_EVIDENCE_ROLES:
        trajectory_path = run_dir / "trajectory" / "validation.json"
        trajectory_validation = load_json(trajectory_path)
        if (
            trajectory_validation.get("schema_version") != 1
            or trajectory_validation.get("valid") is not True
            or trajectory_validation.get("evaluation_transition_count") != 0
        ):
            raise ValueError(f"Invalid DQN trajectory evidence: {trajectory_path}")
        _validate_trajectory_files(run_dir, spec, trajectory_validation)
        if evaluation_summary.get("schema_version") == 2:
            _validate_evaluation_protocol_v2(
                run_dir, records, evaluation_summary, resolved_config,
                manifest["config_hash"], trajectory_validation,
            )

    return {
        "spec": spec,
        "manifest": manifest,
        "status": status,
        "records": records,
        "evaluation_summary": evaluation_summary,
        "trajectory_validation": trajectory_validation,
        "resolved_config_sha256": resolved_hash,
    }


def compare_dqn_run_configs(validated_runs):
    dqn_paths = [
        str(run["spec"].run_dir)
        for run in validated_runs
        if run["spec"].agent == "dqn"
        and run["spec"].role in DQN_EVIDENCE_ROLES
    ]
    if len(dqn_paths) < 2:
        return {
            "reference": dqn_paths[0] if dqn_paths else None,
            "compatible": True,
            "comparisons": [],
            "note": "Fewer than two DQN Pilot/formal runs; no cross-run comparison needed.",
        }
    result = compare_runs(dqn_paths)
    if not result["compatible"]:
        differences = []
        for comparison in result["comparisons"]:
            differences.extend(comparison["differences"])
        raise ValueError(
            "DQN run configurations are incompatible: " + "; ".join(differences)
        )
    return result
