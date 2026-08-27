"""Stream-audit independent-online archives before building the frozen HOA."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


SCENES = ("300_0.3", "300_0.6", "700_0.3", "700_0.6")
COLLECTOR_SEEDS = (0, 1, 2, 3, 4)
INTERSECTIONS = tuple(f"intersection_{index}_1" for index in range(1, 7))
EXPECTED_EPISODES = 400
EXPECTED_DECISIONS = 400 * 360
EXPECTED_TRANSITIONS = EXPECTED_DECISIONS * 6


def sha256_bytes(content):
    return hashlib.sha256(content).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schema_hash(value):
    return sha256_bytes(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8"))


def finite_transition(record):
    arrays = (record.state, record.phase, record.next_state, record.next_phase)
    return (all(np.isfinite(np.asarray(value)).all() for value in arrays)
            and math.isfinite(float(record.reward)))


def audit_run(task):
    root = Path(task["output_path"])
    archive = root / "history_archive"
    errors = []
    try:
        manifest = json.loads((archive / "manifest.json").read_text(encoding="utf-8"))
        status = json.loads((root / "run_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        return {"run_id": task["run_id"], "valid": False,
                "errors": [f"metadata read failure: {error}"]}
    scene = str(task["scene"]); seed = int(task["training_seed"])
    if status.get("status") != "已完成" or status.get("exit_code") != 0:
        errors.append("run status is not completed with exit code 0")
    checks = {
        "scene_id": (str(manifest.get("scene_id")), scene),
        "training_seed": (int(manifest.get("training_seed", -1)), seed),
        "num_episodes": (int(manifest.get("num_episodes", -1)), EXPECTED_EPISODES),
        "num_decision_steps": (int(manifest.get("num_decision_steps", -1)),
                               EXPECTED_DECISIONS),
        "num_transitions": (int(manifest.get("num_transitions", -1)),
                             EXPECTED_TRANSITIONS),
        "intersection_ids": (tuple(manifest.get("intersection_ids", ())),
                              INTERSECTIONS),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            errors.append(f"{name}: expected {expected!r}, got {actual!r}")
    files = manifest.get("transition_files", [])
    expected_files = [f"episode_{episode:04d}.pt"
                      for episode in range(1, EXPECTED_EPISODES + 1)]
    if files != expected_files:
        errors.append("transition_files do not exactly cover episodes 1-400")
    intersections = Counter(); episodes = Counter(); collector_seeds = Counter()
    transition_count = 0; duplicate_count = 0; nonfinite_count = 0
    seen = set(); file_hashes = {}
    for episode, name in enumerate(files, start=1):
        path = archive / name
        try:
            records = torch.load(path, weights_only=False)
            file_hashes[name] = sha256_file(path)
        except (OSError, ValueError, RuntimeError) as error:
            errors.append(f"cannot load {name}: {error}")
            continue
        if len(records) != 360 * 6:
            errors.append(f"{name} has {len(records)} transitions, expected 2160")
        episode_decisions = set()
        for record in records:
            transition_count += 1
            metadata = record.metadata
            if metadata.transition_id in seen:
                duplicate_count += 1
            else:
                seen.add(metadata.transition_id)
            if str(metadata.scene_id) != scene:
                errors.append(f"wrong scene_id in {name}")
                break
            if int(metadata.episode_id) != episode:
                errors.append(f"wrong episode_id in {name}")
                break
            if metadata.source != "offline_history":
                errors.append(f"wrong source in {name}")
                break
            if metadata.collector_training_seed != seed:
                errors.append(f"wrong collector_training_seed in {name}")
                break
            intersections[str(metadata.intersection_id)] += 1
            episodes[int(metadata.episode_id)] += 1
            collector_seeds[int(metadata.collector_training_seed)] += 1
            episode_decisions.add(int(metadata.decision_step))
            if not finite_transition(record):
                nonfinite_count += 1
        if len(episode_decisions) != 360:
            errors.append(f"{name} has {len(episode_decisions)} decisions, expected 360")
    if transition_count != EXPECTED_TRANSITIONS:
        errors.append(
            f"stream transition count {transition_count}, expected {EXPECTED_TRANSITIONS}")
    if duplicate_count:
        errors.append(f"duplicate transition IDs: {duplicate_count}")
    if nonfinite_count:
        errors.append(f"non-finite transitions: {nonfinite_count}")
    expected_per_intersection = EXPECTED_EPISODES * 360
    if intersections != Counter({key: expected_per_intersection for key in INTERSECTIONS}):
        errors.append("intersection distribution is incomplete or unbalanced")
    archive_digest = hashlib.sha256()
    archive_digest.update((archive / "manifest.json").read_bytes())
    for name in files:
        if name in file_hashes:
            archive_digest.update(name.encode("utf-8"))
            archive_digest.update(file_hashes[name].encode("ascii"))
    return {
        "run_id": task["run_id"], "scene_id": scene,
        "collector_seed": seed, "episode_count": len(episodes),
        "decision_count": int(manifest.get("num_decision_steps", -1)),
        "transition_count": transition_count,
        "intersection_distribution": dict(sorted(intersections.items())),
        "episode_distribution": dict(sorted(episodes.items())),
        "collector_seed_distribution": dict(sorted(collector_seeds.items())),
        "state_schema_hash": schema_hash(manifest.get("state_schema")),
        "action_schema_hash": schema_hash(manifest.get("action_schema")),
        "reward_schema_hash": schema_hash(manifest.get("reward_schema")),
        "config_hash": task.get("config_hash"),
        "git_commit": task.get("git_commit"),
        "archive_hash": archive_digest.hexdigest(),
        "duplicate_transition_count": duplicate_count,
        "nonfinite_transition_count": nonfinite_count,
        "file_hashes": file_hashes, "valid": not errors, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue-state",
                        default="artifacts/arterial_experiments/stage1/run_state/run_manifest.json")
    parser.add_argument("--output",
                        default="artifacts/arterial_experiments/historical_archive_audit.json")
    args = parser.parse_args()
    tasks = json.loads(Path(args.queue_state).read_text(encoding="utf-8"))["tasks"]
    if {(task["scene"], int(task["training_seed"])) for task in tasks} != {
            (scene, seed) for scene in SCENES for seed in COLLECTOR_SEEDS}:
        raise ValueError("Stage 1 queue state does not contain the frozen 20-run matrix")
    runs = [audit_run(task) for task in tasks]
    schema_sets = {name: {run.get(name) for run in runs}
                   for name in ("state_schema_hash", "action_schema_hash",
                                "reward_schema_hash")}
    schema_consistent = all(len(values) == 1 for values in schema_sets.values())
    valid = all(run["valid"] for run in runs) and schema_consistent
    payload = {
        "schema_version": 1, "valid": valid, "runs": runs,
        "schema_consistent": schema_consistent,
        "schema_hashes": {name: sorted(values) for name, values in schema_sets.items()},
        "expected_scenes": list(SCENES),
        "collector_training_seeds": list(COLLECTOR_SEEDS),
        "sequential_training_seeds": [1000, 1001, 1002, 1003, 1004],
    }
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(destination)
    raise SystemExit(0 if valid else 1)


if __name__ == "__main__":
    main()
