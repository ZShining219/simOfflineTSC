"""Build a reproducible Stage 0 concurrency-profile queue manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from arterial.experiment import ARTERIAL_SCENES, validate_experiment_config


THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "LD_LIBRARY_PATH": "/tmp/sumo-runtime-libs/root/usr/lib/x86_64-linux-gnu",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/arterial/stage0_resource_profile.yml")
    parser.add_argument("--workers", required=True, type=int)
    parser.add_argument("--seed-start", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--overlay-dir", required=True)
    parser.add_argument("--profile", default="artifacts/system_profile/profile.json")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")

    config_path = Path(args.config).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    experiment = {key: value for key, value in raw.items()
                  if key not in {"model", "trainer", "world", "logger"}}
    config = validate_experiment_config(experiment)
    if len(config["scene_order"]) != 1:
        raise ValueError("Stage 0 resource profile config must contain one scene")
    scene = config["scene_order"][0]
    network = ARTERIAL_SCENES[scene]
    overlay_dir = Path(args.overlay_dir).resolve()
    output_path = Path(args.output).resolve()
    tasks = []
    git_commit = __import__("subprocess").check_output(
        ["git", "rev-parse", "HEAD"], text=True).strip()
    for index in range(args.workers):
        seed = args.seed_start + index
        run_id = f"stage0_w{args.workers}_{scene.replace('.', '')}_seed{seed}"
        prefix = run_id
        overlay = overlay_dir / f"{run_id}.yml"
        run_output = (Path.cwd() / "data/output_data/tsc/sumo_shared_dqn" /
                      network / prefix).resolve()
        command = [
            sys.executable, "arterial_run.py", "--config", str(config_path),
            "--stage", "0", "--training-seed", str(seed),
            "--prefix", prefix, "--output-overlay", str(overlay), "--execute",
        ]
        tasks.append({
            "run_id": run_id,
            "plan": "stage0_resource_profile",
            "method": "independent_online_profile",
            "scene": scene,
            "scene_order": scene,
            "offline_ratio": 0.0,
            "history_mode": "causal",
            "epsilon_mode": config["epsilon_mode"],
            "training_seed": seed,
            "episode_budget": config["episode_budget"],
            "config_path": str(config_path),
            "output_path": str(run_output),
            "command": command,
            "cwd": str(Path.cwd().resolve()),
            "env": THREAD_ENV,
            "completion": {
                "path": str(run_output / "run_status.json"),
                "json_field": "status", "equals": "已完成",
            },
            "checkpoint_path": str(
                run_output / "checkpoints/resumable" /
                f"episode_{int(config['episode_budget']):04d}.pt"),
            "config_hash": sha256(config_path),
            "git_commit": git_commit,
        })
    payload = {
        "schema_version": 1,
        "plan": "stage0_resource_profile",
        "workers_under_test": args.workers,
        "system_profile_path": str(Path(args.profile).resolve()),
        "system_profile_hash": sha256(args.profile),
        "tasks": tasks,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
