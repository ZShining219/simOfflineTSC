#!/usr/bin/env python3
"""Counterfactual check on closed-loop divergence states.

For every snapshot where driver and shadow argmax disagreed, run two
branches (force a_driver / a_shadow) under a shared continuation policy and
shared future seed, then compare discounted 60 s returns.

Continuations:
  'cont'  -> the trajectory owner's checkpoint (B trajectories continue as B)
  'indep' -> scene indep seed-0 DQN (the neutral oracle evaluator)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "counterfactual_v2",
    REPO_ROOT / "dev" / "action_ranking_audit" / "counterfactual_v2.py",
)
cfv2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cfv2)  # noqa: SLF001

SCENE_CFG = {s: REPO_ROOT / "configs" / "sim" / f"{s}.cfg" for s in cfv2.SCENE_LIST}


def build_states(out_dir: Path, driver_map: Dict[str, str]) -> List[Dict[str, Any]]:
    states = []
    for f in sorted(out_dir.glob("decisions_*.jsonl")):
        tag = f.stem.replace("decisions_", "").rsplit("_", 1)[0]
        for line in f.open("r", encoding="utf-8"):
            r = json.loads(line)
            if not r.get("diverged"):
                continue
            owner = tag.split("_")[0]  # 'B' or 'C'
            # run_branch always loads source_checkpoint as the feature agent
            # (obs/phase helpers only); use the driver's checkpoint for that.
            feature_ckpt = driver_map.get(tag, "")
            states.append({
                "scene": r["scene"] if "scene" in r else "sumohz1x1_config2",
                "state_id": r["snapshot_id"],
                "snapshot_tag": tag,
                "episode_seed": r["seed"],
                "decision": r["decision"],
                "simulation_time_seconds": r["sim_time"],
                "phase": r["phase"],
                "queue_incoming": r["queue_incoming"],
                "a_driver": r["action_driver"],
                "a_shadow": r["action_shadow"],
                "state_path": str(REPO_ROOT / r["snapshot_xml"])
                    if not os.path.isabs(r["snapshot_xml"]) else r["snapshot_xml"],
                "python_state_path": str(REPO_ROOT / r["snapshot_pkl"])
                    if not os.path.isabs(r["snapshot_pkl"]) else r["snapshot_pkl"],
                "simulator_config": str(SCENE_CFG[r.get("scene", "sumohz1x1_config2")])
                    if "scene" in r else str(SCENE_CFG["sumohz1x1_config2"]),
                "training_seed": 0,
                "source_run_id": tag,
                "source_checkpoint": feature_ckpt,
            })
    return states


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, required=True,
                   help="divergence probe output dir (decisions_*.jsonl)")
    p.add_argument("--scene", type=str, default="sumohz1x1_config2")
    p.add_argument("--driver-ckpt", type=str, default="",
                   help="json map tag-prefix->checkpoint path for 'cont' continuation")
    p.add_argument("--seeds", type=str, default="default,101,202")
    p.add_argument("--rollout-seconds", type=int, default=60)
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--nshards", type=int, default=1)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--only-continuation", type=str, default="",
                   help="restrict to 'cont' or 'indep' tasks")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    driver_map = json.loads(args.driver_ckpt) if args.driver_ckpt else {}
    seeds: List[Optional[int]] = [
        None if t in ("default", "none", "") else int(t)
        for t in (s.strip() for s in args.seeds.split(","))
    ]

    states = build_states(args.dir, driver_map)
    tasks = []
    for st in states:
        for cont_label, ckpt in (("cont", driver_map.get(st["snapshot_tag"])),
                                 ("indep", None)):
            if args.only_continuation and cont_label != args.only_continuation:
                continue
            if cont_label == "cont" and not ckpt:
                continue
            st2 = dict(st)
            for action in sorted({st["a_driver"], st["a_shadow"]}):
                for seed in seeds:
                    tasks.append({"state": st2, "continuation": cont_label,
                                  "action": action, "future_seed": seed})
    tasks = [t for i, t in enumerate(tasks) if i % args.nshards == args.shard]
    if args.limit:
        tasks = tasks[: args.limit]

    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"divcf_shard{args.shard:02d}.jsonl"
    n_ok = n_err = 0
    with out_path.open("w", encoding="utf-8") as h:
        for i, t in enumerate(tasks):
            rec = cfv2.run_branch(
                t["state"], t["continuation"], t["action"], t["future_seed"],
                args.rollout_seconds, args.out / "world_logs")
            rec["snapshot_tag"] = t["state"]["snapshot_tag"]
            rec["a_driver"] = t["state"]["a_driver"]
            rec["a_shadow"] = t["state"]["a_shadow"]
            rec["episode_seed"] = t["state"]["episode_seed"]
            rec["decision"] = t["state"]["decision"]
            rec["queue_incoming"] = t["state"]["queue_incoming"]
            h.write(json.dumps(rec, ensure_ascii=False) + "\n")
            h.flush()
            n_ok += int(rec["safety_valid"])
            n_err += int(not rec["safety_valid"])
            if i % 200 == 0:
                print(f"[shard{args.shard}] {i}/{len(tasks)} ok={n_ok} err={n_err}", flush=True)
    print(f"[done] {len(tasks)} branches ok={n_ok} err={n_err} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
