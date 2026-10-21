#!/usr/bin/env python3
"""Generate the xiasha1 sufficient-exploration queue (12 runs).

Compares three epsilon schedules on the same 200-episode budget:

* native: existing ``xi1on_dqn_<pack>_s<seed>`` runs
  (epsilon 0.1 -> 0.01, decay 0.995, learning_start 1000; min ~ep4);
* slow:   epsilon 1.0 -> 0.05, decay 0.99992, learning_start 5000
  (min ~ep118);
* full:   epsilon 1.0 -> 0.05, decay 0.99995, learning_start 5000
  (min ~ep180).

New runs: 2 packs (asym_nonly, asym_ns) x 3 seeds x 2 schedules = 12.
The native reference runs already exist under
``data/output_data/tsc/sumo_dqn/xiasha1_<pack>/xi1on_dqn_<pack>_s<seed>``.

Usage (colight environment):
    python dev/xiasha1_online/build_schedule_pilot_queue.py

Then:
    python tools/run_experiment_queue.py \
        --queue artifacts/xiasha1_explore_v1/run_queue.json \
        --state artifacts/xiasha1_explore_v1/run_state \
        --parallel 6 --stagger 2

Analyze after completion:
    python dev/xiasha1_online/analyze_schedule_pilot.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_explore_v1"
PACKS = ("asym_nonly", "asym_ns")
SEEDS = (0, 1, 2)
SCHEDULES = {
    "slow": {
        "overlay": "configs/tsc/xiasha1_online/dqn_slow.yml",
        "prefix": "xi1exp_dqn_slow_{pack}_s{seed}",
        "epsilon_decay": 0.99992,
        "epsilon_min_episode_approx": 118,
    },
    "full": {
        "overlay": "configs/tsc/xiasha1_online/dqn_full.yml",
        "prefix": "xi1exp_dqn_full_{pack}_s{seed}",
        "epsilon_decay": 0.99995,
        "epsilon_min_episode_approx": 180,
    },
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
    for schedule, meta in SCHEDULES.items():
        for pack in PACKS:
            for seed in SEEDS:
                prefix = meta["prefix"].format(pack=pack, seed=seed)
                run_dir = (ROOT / "data" / "output_data" / "tsc" / "sumo_dqn"
                           / f"xiasha1_{pack}" / prefix)
                tasks.append({
                    "run_id": prefix,
                    "command": [
                        sys.executable, "run.py",
                        "-w", "sumo", "-a", "dqn",
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
        "plan": "xiasha1_explore_v1",
        "expected_tasks": len(tasks),
        "schedules": {
            "native": {
                "note": "existing xi1on_dqn_* reference runs",
                "epsilon": 0.1,
                "epsilon_decay": 0.995,
                "epsilon_min": 0.01,
                "learning_start": 1000,
                "epsilon_min_episode_approx": 4,
            },
            **{name: {
                "note": "new runs in this queue",
                "overlay": meta["overlay"],
                "epsilon": 1.0,
                "epsilon_decay": meta["epsilon_decay"],
                "epsilon_min": 0.05,
                "learning_start": 5000,
                "epsilon_min_episode_approx":
                    meta["epsilon_min_episode_approx"],
            } for name, meta in SCHEDULES.items()},
        },
        "note": "Sufficient-exploration validation: native vs slow vs full; "
                "2 packs x 3 seeds x 2 new schedules; 200ep; eval every "
                "episode; legacy pseudo_step; saveReplay=false.",
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
        "--parallel 6 --stagger 2"
    )


if __name__ == "__main__":
    main()
