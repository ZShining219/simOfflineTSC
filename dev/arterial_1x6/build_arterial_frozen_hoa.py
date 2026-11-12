"""Build immutable-by-contract HOA indexes from a valid Stage 1 audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


SCENES = ("300_0.3", "300_0.6", "700_0.3", "700_0.6")
COLLECTOR_SEEDS = (0, 1, 2, 3, 4)


def canonical_hash(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build(audit_path, queue_state_path, output_dir):
    audit_path = Path(audit_path).resolve()
    queue_state_path = Path(queue_state_path).resolve()
    output_dir = Path(output_dir).resolve()
    audit = _load(audit_path)
    if audit.get("valid") is not True or audit.get("schema_consistent") is not True:
        raise ValueError("Frozen HOA requires a valid, schema-consistent Stage 1 audit")
    tasks = _load(queue_state_path)["tasks"]
    task_by_id = {task["run_id"]: task for task in tasks}
    runs = audit.get("runs", [])
    expected = {(scene, seed) for scene in SCENES for seed in COLLECTOR_SEEDS}
    actual = {(str(run.get("scene_id")), int(run.get("collector_seed", -1)))
              for run in runs if run.get("valid") is True}
    if actual != expected or len(runs) != len(expected):
        raise ValueError("Audit does not contain the exact frozen 20-run matrix")

    schema_hashes = {name: values[0] for name, values in audit["schema_hashes"].items()}
    scene_indexes = []
    for scene in SCENES:
        scene_runs = []
        for run in sorted(
                (item for item in runs if str(item["scene_id"]) == scene),
                key=lambda item: int(item["collector_seed"])):
            task = task_by_id.get(run["run_id"])
            if task is None:
                raise ValueError(f"Audit run missing from queue state: {run['run_id']}")
            archive_path = (Path(task["output_path"]) / "history_archive").resolve()
            scene_runs.append({
                "run_id": run["run_id"],
                "collector_training_seed": int(run["collector_seed"]),
                "archive_path": str(archive_path),
                "archive_hash": run["archive_hash"],
                "num_episodes": int(run["episode_count"]),
                "num_decisions": int(run["decision_count"]),
                "num_transitions": int(run["transition_count"]),
                "config_hash": run.get("config_hash"),
                "git_commit": run.get("git_commit"),
            })
        scene_core = {
            "schema_version": 1, "roadnet_id": "arterial_1x6",
            "scene_id": scene,
            "collector_training_seeds": list(COLLECTOR_SEEDS),
            "episode_range": [1, 400],
            "intersection_ids": [f"intersection_{index}_1" for index in range(1, 7)],
            "num_runs": 5, "num_episodes": 2000,
            "num_decisions": sum(item["num_decisions"] for item in scene_runs),
            "num_transitions": sum(item["num_transitions"] for item in scene_runs),
            "source_policy": "independent_online_shared_dqn",
            "source_archives": scene_runs, "schema_hashes": schema_hashes,
        }
        scene_hash = canonical_hash(scene_core)
        scene_payload = {**scene_core, "archive_hash": scene_hash}
        destination = output_dir / scene.replace(".", "_") / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(
            scene_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        scene_indexes.append({
            "scene_id": scene, "manifest_path": str(destination),
            "archive_hash": scene_hash,
            "history_paths": [item["archive_path"] for item in scene_runs],
            "num_transitions": scene_payload["num_transitions"],
        })

    global_core = {
        "schema_version": 1, "archive_version": "arterial_1x6_hoa_v1",
        "roadnet_id": "arterial_1x6", "scenes": scene_indexes,
        "collector_training_seeds": list(COLLECTOR_SEEDS),
        "episode_range": [1, 400], "num_runs": 20,
        "num_episodes": 8000,
        "num_transitions": sum(item["num_transitions"] for item in scene_indexes),
        "schema_hashes": schema_hashes,
        "source_audit_path": str(audit_path),
        "source_audit_hash": canonical_hash(audit),
        "source_queue_state_path": str(queue_state_path),
    }
    payload = {
        **global_core, "archive_hash": canonical_hash(global_core),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "immutability_contract": (
            "read_only; no append, overwrite, filtering, or semi-run transitions"),
    }
    destination = output_dir / "manifest.json"
    destination.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return destination, payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", default=
                        "artifacts/arterial_experiments/historical_archive_audit.json")
    parser.add_argument("--queue-state", default=
                        "artifacts/arterial_experiments/stage1/run_state/run_manifest.json")
    parser.add_argument("--output-dir", default=
                        "artifacts/arterial_experiments/historical_archive")
    args = parser.parse_args()
    destination, payload = build(args.audit, args.queue_state, args.output_dir)
    print(json.dumps({"manifest": str(destination),
                      "archive_hash": payload["archive_hash"]}))


if __name__ == "__main__":
    main()
