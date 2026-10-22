#!/usr/bin/env python3
"""Generate the xiasha1 cross-scenario best-checkpoint eval queue.

For each v2 arm and each source pack in ``PACKS``, take the best checkpoint
of every training seed and evaluate it on every target pack in ``PACKS``
(including the source pack as the in-domain control).

Pilot scope:
    arms      = 8
    sources   = asym_nonly / asym_ns / dense_extreme / fluct_burst
    targets   = same 4 packs (full 4x4 matrix)
    train seeds = 0/1/2
    SUMO seeds  = 10000
    => 8 * 4 * 3 * 4 = 384 eval runs

Usage (colight environment):
    python dev/xiasha1_online_v2/build_cross_scenario_eval_queue.py

Then:
    python tools/run_experiment_queue.py \
        --queue artifacts/xiasha1_cross_eval_v1/run_queue.json \
        --state artifacts/xiasha1_cross_eval_v1/run_state \
        --parallel 24 --stagger 1
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_cross_eval_v1"
V2_QUEUE = ROOT / "artifacts" / "xiasha1_online_v2" / "run_queue.json"
PACKS = ("asym_nonly", "asym_ns", "dense_extreme", "fluct_burst")
TRAIN_SEEDS = (0, 1, 2)
SUMO_SEEDS = (10000,)
ARMS = {
    "dqn": ("dqn", "sumo_dqn"),
    "ddqn": ("dqn", "sumo_dqn"),
    "dueling": ("dqn", "sumo_dqn"),
    "d3qn": ("dqn", "sumo_dqn"),
    "presslight": ("presslight", "sumo_presslight"),
    "frap": ("frap", "sumo_frap"),
    "mplight": ("mplight", "sumo_mplight"),
    "sac": ("sac", "sumo_sac"),
}
EVAL_CONFIG = "configs/tsc/eval_baseline_checkpoint.yml"


def main():
    if Path(sys.prefix).name != "colight":
        raise SystemExit(
            "Run this generator with the colight environment so the queue "
            "commands use the correct Python interpreter."
        )
    ld_preload = Path(sys.prefix) / "lib" / "libstdc++.so.6"
    if not ld_preload.is_file():
        raise SystemExit(f"Missing LD_PRELOAD library: {ld_preload}")

    v2 = json.loads(V2_QUEUE.read_text(encoding="utf-8"))
    run_dirs = {(task["arm"], task["pack"], int(task["seed"])):
                Path(task["output_path"]) for task in v2["tasks"]}
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
    for source in PACKS:
        for arm, (agent, bucket) in ARMS.items():
            for train_seed in TRAIN_SEEDS:
                run_dir = run_dirs[(arm, source, train_seed)]
                summary_path = run_dir / "evaluation" / "summary.json"
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                best = summary.get("best_checkpoint")
                if not best:
                    raise SystemExit(
                        f"Missing best_checkpoint in {summary_path}")
                checkpoint = (run_dir / best).resolve()
                if not checkpoint.is_file():
                    raise SystemExit(f"Missing checkpoint: {checkpoint}")
                for target in PACKS:
                    for sumo_seed in SUMO_SEEDS:
                        prefix = (f"xi2x_{arm}_tr{source}_s{train_seed}"
                                  f"_te{target}_ss{sumo_seed}")
                        output_dir = (ROOT / "data" / "output_data" / "tsc"
                                      / bucket / f"xiasha1_{target}" / prefix)
                        tasks.append({
                            "run_id": prefix,
                            "arm": arm,
                            "source_pack": source,
                            "target_pack": target,
                            "train_seed": train_seed,
                            "sumo_seed": sumo_seed,
                            "checkpoint": str(checkpoint),
                            "best_episode": summary.get("best_episode"),
                            "command": [
                                sys.executable,
                                "tools/eval_baseline_checkpoint.py",
                                "--agent", agent,
                                "--network", f"xiasha1_{target}",
                                "--config", EVAL_CONFIG,
                                "--checkpoint", str(checkpoint),
                                "--seed", str(train_seed),
                                "--sumo_seed", str(sumo_seed),
                                "--prefix", prefix,
                            ],
                            "cwd": str(ROOT),
                            "env": env,
                            "output_path": str(output_dir),
                            "completion": {
                                "path": str(output_dir / "run_status.json"),
                                "json_field": "status",
                                "equals": "已完成",
                            },
                        })

    OUT.mkdir(parents=True, exist_ok=True)
    spec = {
        "schema_version": 1,
        "plan": "xiasha1_cross_eval_v1",
        "expected_tasks": len(tasks),
        "sources": list(PACKS),
        "targets": list(PACKS),
        "arms": list(ARMS),
        "train_seeds": list(TRAIN_SEEDS),
        "sumo_seeds": list(SUMO_SEEDS),
        "eval_config": EVAL_CONFIG,
        "note": "Cross-scenario best-checkpoint eval: 8 arms x 4 source packs "
                "x 3 train seeds x 4 target packs x 1 SUMO seed = 384 eval "
                "runs; source pack is the in-domain control.",
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
        "--parallel 24 --stagger 1"
    )


if __name__ == "__main__":
    main()
