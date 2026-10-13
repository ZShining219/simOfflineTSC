#!/usr/bin/env python3
"""Greedy-policy SUMO traffic evaluation for rank-aux trained checkpoints.

Runs full 3600 s episodes of a checkpoint in world_sumo and reports the
bench three metrics plus queue/delay diagnostics per future seed:
throughput (completed vehicles), mean travel time of completed vehicles,
unfinished vehicles in the network at episode end, queue AUC, real delay.

Each episode runs in a FRESH World so --seed reseeds SUMO demand properly.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "allocation_accuracy_counterfactual",
    REPO_ROOT / "tools" / "allocation_accuracy_counterfactual.py",
)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)  # noqa: SLF001

from common import interface as registry_interface
from common.registry import Registry
import world.world_sumo as world_sumo

ACTION_INTERVAL = cf.ACTION_INTERVAL
EPISODE_END = 3600.0

SCENE_CFG = {
    s: REPO_ROOT / "configs" / "sim" / f"{s}.cfg" for s in cf.SCENES
}


def run_episode(sim_cfg: Path, checkpoint: Path, seed: Optional[int],
                log_dir: Path) -> Dict[str, Any]:
    registry_interface.Command_Setting_Interface({"command": {"sumo_seed": seed}})
    Registry.mapping["logger_mapping"]["path"].path = str(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    world = world_sumo.World(str(sim_cfg), 1, interface="libsumo")
    started = time.time()
    try:
        world.reset()
        agent = cf.FrozenDQN(world, checkpoint)
        queue_sec = 0.0
        phase_switches = 0
        prev = None
        while world.get_current_time() < EPISODE_END:
            obs = agent.observation()
            phase = agent.phase()
            action = agent.action(obs, phase)
            if prev is not None and action != prev:
                phase_switches += 1
            prev = action
            for _ in range(ACTION_INTERVAL):
                if world.get_current_time() >= EPISODE_END:
                    break
                world.step([action])
                queue_sec += float(sum(world.get_lane_waiting_vehicle_count().values()))
        completed = list(world.vehicles.values())
        return {
            "seed": seed if seed is not None else "default",
            "safety_valid": True,
            "throughput": int(len(world.vehicles)),
            "avg_travel_time": float(np.mean(completed)) if completed else float("nan"),
            "unfinished": int(world.get_unfinished_vehicle_count()),
            "queue_auc": float(queue_sec),
            "real_delay": float(world.get_real_delay()),
            "avg_waiting": float(world.get_average_waiting_time()),
            "phase_switches": int(phase_switches),
            "sim_seconds": float(world.get_current_time()),
            "wall_seconds": round(time.time() - started, 1),
        }
    except Exception as exc:
        return {"seed": seed, "safety_valid": False,
                "error": f"{type(exc).__name__}: {exc}"}
    finally:
        world.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--seeds", type=str, default="default,101,202,303,404")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--tag", type=str, default="")
    args = p.parse_args()

    seeds: List[Optional[int]] = [
        None if t in ("default", "none", "") else int(t)
        for t in (s.strip() for s in args.seeds.split(","))
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"episodes_{args.tag or args.checkpoint.stem}.jsonl"
    rows = []
    with out.open("w", encoding="utf-8") as h:
        for seed in seeds:
            rec = run_episode(SCENE_CFG[args.scene], args.checkpoint, seed,
                              args.out / "world_logs" / args.scene)
            rec["checkpoint"] = str(args.checkpoint)
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
            h.flush()
            rows.append(rec)
            print(json.dumps({k: v for k, v in rec.items() if k != "checkpoint"}), flush=True)
    valid = [r for r in rows if r["safety_valid"]]
    if valid:
        summary = {k: float(np.mean([r[k] for r in valid]))
                   for k in ("throughput", "avg_travel_time", "unfinished",
                             "queue_auc", "real_delay")}
        summary["n_episodes"] = len(valid)
        print("SUMMARY", json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
