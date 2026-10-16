"""Build the two-scene Stage 0 training-budget pilot manifest."""

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


CONFIGS = (
    "configs/arterial/stage0_budget_300_06.yml",
    "configs/arterial/stage0_budget_700_06.yml",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--profile", default="artifacts/system_profile/profile.json")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    overlay_dir = Path(args.overlay_dir).resolve()
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    tasks = []
    for config_name in CONFIGS:
        config_path = Path(config_name).resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        experiment = {key: value for key, value in raw.items()
                      if key not in {"model", "trainer", "world", "logger"}}
        config = validate_experiment_config(experiment)
        if len(config["scene_order"]) != 1:
            raise ValueError("Budget pilot configs must contain exactly one scene")
        scene = config["scene_order"][0]
        seed = int(config["training_seed"])
        budget = int(config["episode_budget"])
        network = ARTERIAL_SCENES[scene]
        run_id = f"stage0_budget_{scene.replace('.', '')}_seed{seed}"
        prefix = run_id
        overlay = overlay_dir / f"{run_id}.yml"
        run_output = (Path.cwd() / "data/output_data/tsc/sumo_shared_dqn" /
                      network / prefix).resolve()
        tasks.append({
            "run_id": run_id, "plan": "stage0_budget_pilot",
            "method": "independent_online_budget_pilot", "scene": scene,
            "scene_order": scene, "offline_ratio": 0.0,
            "history_mode": "causal", "epsilon_mode": config["epsilon_mode"],
            "training_seed": seed, "episode_budget": budget,
            "config_path": str(config_path), "output_path": str(run_output),
            "command": [
                sys.executable, "arterial_run.py", "--config", str(config_path),
                "--stage", "0", "--prefix", prefix,
                "--output-overlay", str(overlay), "--execute",
            ],
            "cwd": str(Path.cwd().resolve()), "env": THREAD_ENV,
            "completion": {
                "path": str(run_output / "run_status.json"),
                "json_field": "status", "equals": "已完成",
            },
            "checkpoint_path": str(
                run_output / "checkpoints/resumable" / f"episode_{budget:04d}.pt"),
            "config_hash": sha256(config_path), "git_commit": git_commit,
        })
    payload = {
        "schema_version": 1, "plan": "stage0_budget_pilot",
        "system_profile_path": str(Path(args.profile).resolve()),
        "system_profile_hash": sha256(args.profile), "tasks": tasks,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
