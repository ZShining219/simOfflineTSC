#!/usr/bin/env python3
"""Generate the xiasha1_online_v2 queue (288 train runs).

Campaign v2 re-runs the 8 off-policy arms on all 12 demand packs x 3 seeds
with full-horizon exploration:

* DQN family / PressLight / FRAP:
  epsilon 1.0 -> 0.05, decay 0.99995, learning_start 5000
  (epsilon_min reached ~ep180);
* SAC: native Categorical-policy exploration (temperature 0.2), no epsilon;
* MPLight: native pfrl linear epsilon 1.0 -> 0.0 over ~160 episodes.

Unified protocol: legacy pseudo_step, 200 train episodes, greedy eval every
episode (ep0-200), resumable ckpt subset, saveReplay=false.

Usage (colight environment):
    python dev/xiasha1_online_v2/build_online_v2_queue.py

Then:
    python tools/run_experiment_queue.py \
        --queue artifacts/xiasha1_online_v2/run_queue.json \
        --state artifacts/xiasha1_online_v2/run_state \
        --parallel 12 --stagger 2
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_online_v2"
PACK_FILE = ROOT / "data" / "raw_data" / "xiasha1_demand_packs.json"
SEEDS = (0, 1, 2)
ARMS = {
    "dqn": {"agent": "dqn", "bucket": "sumo_dqn",
            "overlay": "configs/tsc/xiasha1_online_v2/dqn.yml"},
    "ddqn": {"agent": "dqn", "bucket": "sumo_dqn",
             "overlay": "configs/tsc/xiasha1_online_v2/ddqn.yml"},
    "dueling": {"agent": "dqn", "bucket": "sumo_dqn",
                "overlay": "configs/tsc/xiasha1_online_v2/dueling.yml"},
    "d3qn": {"agent": "dqn", "bucket": "sumo_dqn",
             "overlay": "configs/tsc/xiasha1_online_v2/d3qn.yml"},
    "presslight": {"agent": "presslight", "bucket": "sumo_presslight",
                   "overlay": "configs/tsc/xiasha1_online_v2/presslight.yml"},
    "frap": {"agent": "frap", "bucket": "sumo_frap",
             "overlay": "configs/tsc/xiasha1_online_v2/frap.yml"},
    "mplight": {"agent": "mplight", "bucket": "sumo_mplight",
                "overlay": "configs/tsc/xiasha1_online_v2/mplight.yml"},
    "sac": {"agent": "sac", "bucket": "sumo_sac",
            "overlay": "configs/tsc/xiasha1_online_v2/sac.yml"},
}


def main():
    if Path(sys.prefix).name != "colight":
        raise SystemExit(
            "Run this generator with the colight environment so the queue "
            "commands use the correct Python interpreter."
        )
    ld_preload = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not ld_preload.is_file():
        raise SystemExit(f"Missing LD_PRELOAD library: {ld_preload}")

    packs = [item["pack"] for item in
             json.loads(PACK_FILE.read_text(encoding="utf-8"))["packs"]]
    env = {
        "LD_PRELOAD": str(ld_preload),
        "CUDA_VISIBLE_DEVICES": "-1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
    }
    tasks = []
    # Pack-major order interleaves the 8 arms, so every arm is exercised
    # early instead of waiting for the previous arm's 36 runs to finish.
    for pack in packs:
        for seed in SEEDS:
            for arm, meta in ARMS.items():
                prefix = f"xi2on_{arm}_{pack}_s{seed}"
                run_dir = (ROOT / "data" / "output_data" / "tsc"
                           / meta["bucket"] / f"xiasha1_{pack}" / prefix)
                tasks.append({
                    "run_id": prefix,
                    "arm": arm,
                    "pack": pack,
                    "seed": seed,
                    "command": [
                        sys.executable, "run.py",
                        "-w", "sumo", "-a", meta["agent"],
                        "-n", f"xiasha1_{pack}",
                        "--seed", str(seed),
                        "--interface", "libsumo",
                        "--prefix", prefix,
                        "--experiment-config", meta["overlay"],
                    ],
                    "cwd": str(ROOT),
                    "env": env,
                    "output_path": str(run_dir),
                    "completion": {
                        "path": str(run_dir / "run_status.json"),
                        "json_field": "status",
                        "equals": "已完成",
                    },
                })

    OUT.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "plan": "xiasha1_online_v2",
        "expected_tasks": len(tasks),
        "arms": list(ARMS),
        "packs": packs,
        "seeds": list(SEEDS),
        "exploration": {
            "dqn_family_presslight_frap": {
                "epsilon": 1.0,
                "epsilon_decay": 0.99995,
                "epsilon_min": 0.05,
                "learning_start": 5000,
                "epsilon_min_episode_approx": 180,
            },
            "sac": "native Categorical policy, sac_temperature 0.2",
            "mplight": "native pfrl linear epsilon 1.0 -> 0.0 over ~160 eps",
        },
        "note": "v2 sufficient-exploration online baseline: 8 arms x 12 packs "
                "x 3 seeds = 288 run; 200ep; greedy eval every episode; "
                "legacy pseudo_step; saveReplay=false.",
        "tasks": tasks,
    }
    queue_path = OUT / "run_queue.json"
    queue_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"tasks={len(tasks)} -> {queue_path.relative_to(ROOT)}")
    print(
        "run with: python tools/run_experiment_queue.py "
        f"--queue {queue_path.relative_to(ROOT)} "
        f"--state {(OUT / 'run_state').relative_to(ROOT)} "
        "--parallel 12 --stagger 2"
    )


if __name__ == "__main__":
    main()
