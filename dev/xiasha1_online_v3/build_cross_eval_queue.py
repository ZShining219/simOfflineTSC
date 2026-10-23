#!/usr/bin/env python3
"""Generate the xiasha1 v3 cross-scenario best-checkpoint eval queue.

与 xiasha1_cross_eval_v1 同构：每 (arm, source_pack, train_seed) 取 v3
训练 run 的 best checkpoint，在 4 个目标包上评估（含源包作域内对照）。
源 ckpt 来自 xiasha1_online_v3（280m 修复几何），被评路网同为修复几何。

    arms      = 8
    sources   = asym_nonly / asym_ns / dense_extreme / fluct_burst
    targets   = same 4 packs (full 4x4 matrix)
    train seeds = 0/1/2
    SUMO seeds  = 10000
    => 8 * 4 * 3 * 4 = 384 eval runs

用法（colight 环境）:
    python dev/xiasha1_online_v3/build_cross_eval_queue.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_v3_cross_eval_v1"
V3_QUEUE = ROOT / "artifacts" / "xiasha1_online_v3" / "run_queue.json"
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
EVAL_CONFIGS = {
    "dqn": EVAL_CONFIG,
    "ddqn": "configs/tsc/eval_baseline_checkpoint_ddqn.yml",
    "dueling": "configs/tsc/eval_baseline_checkpoint_dueling.yml",
    "d3qn": "configs/tsc/eval_baseline_checkpoint_d3qn.yml",
    "presslight": EVAL_CONFIG,
    "frap": EVAL_CONFIG,
    "mplight": EVAL_CONFIG,
    "sac": EVAL_CONFIG,
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

    v3 = json.loads(V3_QUEUE.read_text(encoding="utf-8"))
    run_dirs = {(task["arm"], task["pack"], int(task["seed"])):
                Path(task["output_path"]) for task in v3["tasks"]}
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
                        prefix = (f"xi3x_{arm}_tr{source}_s{train_seed}"
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
                                "--config", EVAL_CONFIGS[arm],
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
        "plan": "xiasha1_v3_cross_eval_v1",
        "expected_tasks": len(tasks),
        "sources": list(PACKS),
        "targets": list(PACKS),
        "arms": list(ARMS),
        "train_seeds": list(TRAIN_SEEDS),
        "sumo_seeds": list(SUMO_SEEDS),
        "eval_config": EVAL_CONFIGS,
        "note": "v3 几何修复版 cross-scenario best-checkpoint eval（与 v1 "
                "同构，源 ckpt 为 xiasha1_online_v3 best_checkpoint）："
                "8 arms x 4 source x 3 seeds x 4 target x 1 SUMO seed = 384。",
        "tasks": tasks,
    }
    queue_path = OUT / "run_queue.json"
    queue_path.write_text(
        json.dumps(spec, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"tasks={len(tasks)} -> {queue_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
