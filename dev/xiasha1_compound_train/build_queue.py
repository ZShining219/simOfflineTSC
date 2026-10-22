#!/usr/bin/env python3
"""Generate the xiasha1_compound_train_v1 queue (30 train runs).

Hypothesis probe: was the ~ep30 early convergence in xiasha1_online_v2 an
artifact of the single-axis demand packs?  Train 5 arms on the two
multi-axis compound packs under the *identical* v2 protocol (same
overlays, full-horizon epsilon for DQN-family/PressLight/FRAP, native
exploration for SAC/MPLight, 200 ep, per-episode greedy eval, legacy
pseudo_step, saveReplay=false), then compare convergence episode and
final-20 metrics against the v2 saturated packs.

Matrix: 5 arms {d3qn, mplight, frap, presslight, sac}
        x 2 packs {compound_a, compound_b} x 3 seeds {0,1,2} = 30 runs.
Prefix: xi3on_<arm>_<pack>_s<seed>.

Usage (colight environment):
    python dev/xiasha1_compound_train/build_queue.py

Then:
    python tools/run_experiment_queue.py \
        --queue artifacts/xiasha1_compound_train/run_queue.json \
        --state artifacts/xiasha1_compound_train/run_state \
        --parallel 12 --stagger 2
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts" / "xiasha1_compound_train"
PACKS = ("compound_a", "compound_b")
SEEDS = (0, 1, 2)
ARMS = {
    "d3qn": {"agent": "dqn", "bucket": "sumo_dqn",
             "overlay": "configs/tsc/xiasha1_online_v2/d3qn.yml"},
    "mplight": {"agent": "mplight", "bucket": "sumo_mplight",
                "overlay": "configs/tsc/xiasha1_online_v2/mplight.yml"},
    "frap": {"agent": "frap", "bucket": "sumo_frap",
             "overlay": "configs/tsc/xiasha1_online_v2/frap.yml"},
    "presslight": {"agent": "presslight", "bucket": "sumo_presslight",
                   "overlay": "configs/tsc/xiasha1_online_v2/presslight.yml"},
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
    for pack in PACKS:
        for seed in SEEDS:
            for arm, meta in ARMS.items():
                prefix = f"xi3on_{arm}_{pack}_s{seed}"
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
        "plan": "xiasha1_compound_train_v1",
        "expected_tasks": len(tasks),
        "arms": list(ARMS),
        "packs": list(PACKS),
        "seeds": list(SEEDS),
        "exploration": "identical to xiasha1_online_v2 overlays "
                       "(DQN-family/PressLight/FRAP epsilon 1.0->0.05 "
                       "decay 0.99995; SAC native; MPLight pfrl linear)",
        "note": "compound-pack train probe: does multi-axis demand prevent "
                "the ~ep30 early convergence seen on single-axis packs? "
                "5 arms x 2 packs x 3 seeds = 30 run; 200ep; greedy eval "
                "every episode; legacy pseudo_step; saveReplay=false.",
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
