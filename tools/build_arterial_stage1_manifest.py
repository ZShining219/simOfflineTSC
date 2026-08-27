"""Build the frozen 20-run independent-online collector manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arterial.experiment import ARTERIAL_SCENES, validate_experiment_config
from tools.build_arterial_stage0_manifest import THREAD_ENV


COLLECTOR_SEEDS = (0, 1, 2, 3, 4)
SCENES = ("300_0.3", "300_0.6", "700_0.3", "700_0.6")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arterial/independent_online.yml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--profile", default="artifacts/system_profile/profile.json")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = {key: value for key, value in raw.items()
                  if key not in {"model", "trainer", "world", "logger"}}
    base = validate_experiment_config(experiment)
    if int(base["episode_budget"]) != 400:
        raise ValueError("Formal collector budget is frozen at 400 episodes")
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    profile_path = str(Path(args.profile).resolve())
    profile_hash = sha256(args.profile)
    overlay_dir = Path(args.overlay_dir).resolve()
    tasks = []
    # Interleave scenes so the queue does not launch only high-load runs together.
    for seed in COLLECTOR_SEEDS:
        for scene in SCENES:
            network = ARTERIAL_SCENES[scene]
            run_id = f"stage1_collector_{scene.replace('.', '')}_seed{seed}"
            prefix = run_id
            overlay = overlay_dir / f"{run_id}.yml"
            run_output = (Path.cwd() / "data/output_data/tsc/sumo_shared_dqn" /
                          network / prefix).resolve()
            tasks.append({
                "run_id": run_id, "plan": "stage1_independent_online",
                "method": "independent_online_archive", "scene": scene,
                "scene_order": scene, "offline_ratio": 0.0,
                "history_mode": "causal",
                "epsilon_mode": base["epsilon_mode"],
                "training_seed": seed, "episode_budget": 400,
                "config_path": str(config_path), "output_path": str(run_output),
                "command": [
                    sys.executable, "arterial_run.py", "--config", str(config_path),
                    "--stage", "0", "--training-seed", str(seed),
                    "--scene", scene,
                    "--episode-budget", "400", "--prefix", prefix,
                    "--system-profile-path", profile_path,
                    "--system-profile-hash", profile_hash,
                    "--output-overlay", str(overlay), "--execute",
                ],
                "cwd": str(Path.cwd().resolve()), "env": THREAD_ENV,
                "completion": {
                    "path": str(run_output / "run_status.json"),
                    "json_field": "status", "equals": "已完成",
                },
                "checkpoint_path": str(
                    run_output / "checkpoints/resumable/episode_0400.pt"),
                "config_hash": sha256(config_path), "git_commit": git_commit,
            })
    payload = {
        "schema_version": 1, "plan": "stage1_independent_online",
        "collector_training_seeds": list(COLLECTOR_SEEDS),
        "sequential_training_seeds": [1000, 1001, 1002, 1003, 1004],
        "sumo_seed_policy": "fixed simulator config seed 0",
        "episode_range": [1, 400], "expected_tasks": 20,
        "system_profile_path": profile_path,
        "system_profile_hash": profile_hash, "tasks": tasks,
    }
    destination = Path(args.output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
