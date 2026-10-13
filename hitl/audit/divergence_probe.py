#!/usr/bin/env python3
"""Closed-loop action-divergence probe between two checkpoints.

Runs a full greedy episode where --driver controls the intersection while
--shadow is evaluated on the SAME (obs, phase) at every decision point.
Wherever the two argmax disagree, a SUMO saveState snapshot (+ python state)
is stored for later counterfactual comparison.  Per-decision rows record the
state features needed to characterize where divergence happens.
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
SCENE_CFG = {s: REPO_ROOT / "configs" / "sim" / f"{s}.cfg" for s in cf.SCENES}


def run_episode(sim_cfg: Path, driver_ckpt: Path, shadow_ckpt: Path,
                seed: Optional[int], out_dir: Path, tag: str) -> Dict[str, Any]:
    registry_interface.Command_Setting_Interface({"command": {"sumo_seed": seed}})
    log_dir = out_dir / "world_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    Registry.mapping["logger_mapping"]["path"].path = str(log_dir)
    world = world_sumo.World(str(sim_cfg), 1, interface="libsumo")
    seed_tag = seed if seed is not None else "default"
    decisions_path = out_dir / f"decisions_{tag}_{seed_tag}.jsonl"
    snap_dir = out_dir / "snapshots" / tag / str(seed_tag)
    n_div = 0
    try:
        world.reset()
        driver = cf.FrozenDQN(world, driver_ckpt)
        shadow = cf.FrozenDQN(world, shadow_ckpt)
        decision = 0
        rows = 0
        with decisions_path.open("w", encoding="utf-8") as out:
            while world.get_current_time() < EPISODE_END:
                obs = driver.observation()
                phase = driver.phase()
                q_d = driver.q_values(obs, phase)
                q_s = shadow.q_values(obs, phase)
                a_d = int(np.argmax(q_d))
                a_s = int(np.argmax(q_s))
                waits = world.get_lane_waiting_vehicle_count()
                lane_waits = {k: float(v) for k, v in waits.items()}
                rec: Dict[str, Any] = {
                    "seed": seed_tag, "decision": decision,
                    "sim_time": float(world.get_current_time()),
                    "phase": int(phase),
                    "obs": [float(v) for v in obs.reshape(-1)],
                    "queue_incoming": float(sum(lane_waits.get(l, 0.0) for l in driver.lane_ids)),
                    "queue_total": float(sum(lane_waits.values())),
                    "lane_waits": lane_waits,
                    "vehicles_in_net": int(world.get_unfinished_vehicle_count()),
                    "throughput_so_far": int(len(world.vehicles)),
                    "action_driver": a_d, "action_shadow": a_s,
                    "diverged": bool(a_d != a_s),
                    "q_driver": [float(v) for v in q_d],
                    "q_shadow": [float(v) for v in q_s],
                }
                if a_d != a_s:
                    n_div += 1
                    sid = f"{tag}_{seed_tag}_d{decision:04d}"
                    xml = snap_dir / f"{sid}.xml"
                    pkl = snap_dir / f"{sid}.pkl"
                    cf.save_snapshot(world, xml, pkl)
                    rec["snapshot_id"] = sid
                    rec["snapshot_xml"] = str(xml)
                    rec["snapshot_pkl"] = str(pkl)
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                rows += 1
                for _ in range(ACTION_INTERVAL):
                    if world.get_current_time() >= EPISODE_END:
                        break
                    world.step([a_d])
                decision += 1
        completed = list(world.vehicles.values())
        return {
            "tag": tag, "seed": seed_tag, "decisions": rows, "diverged": n_div,
            "divergence_rate": n_div / max(1, rows),
            "throughput": int(len(world.vehicles)),
            "avg_travel_time": float(np.mean(completed)) if completed else float("nan"),
            "unfinished": int(world.get_unfinished_vehicle_count()),
        }
    finally:
        world.close()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--driver", type=Path, required=True)
    p.add_argument("--shadow", type=Path, required=True)
    p.add_argument("--scene", type=str, required=True)
    p.add_argument("--seeds", type=str, default="default,101,202")
    p.add_argument("--tag", type=str, required=True, help="e.g. B_vs_C")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    seeds: List[Optional[int]] = [
        None if t in ("default", "none", "") else int(t)
        for t in (s.strip() for s in args.seeds.split(","))
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    summary = []
    for seed in seeds:
        t0 = time.time()
        rec = run_episode(SCENE_CFG[args.scene], args.driver, args.shadow,
                          seed, args.out, args.tag)
        rec["wall_seconds"] = round(time.time() - t0, 1)
        summary.append(rec)
        print(json.dumps(rec, ensure_ascii=False), flush=True)
    (args.out / f"summary_{args.tag}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
