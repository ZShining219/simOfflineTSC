#!/usr/bin/env python3
"""Counterfactual v2 rollouts for the action-ranking audit.

Extends tools/allocation_accuracy_counterfactual.py in three ways:
  1. Records the ORIGINAL DQN training reward per decision cycle
     (r = -12 * mean lane_waiting_count, averaged over the 10 sim steps of a
     cycle, i.e. -0.15 * incoming queue-seconds) and the discounted return
     G_H = sum(gamma^k * r_k) with gamma=0.95 and the training decision
     cadence of 10 s.
  2. Lets the continuation policy be chosen per branch: the scene's
     independent Plan-1 DQN ('indep', as in the original audit) or the state's
     own CONT-DQN source checkpoint ('cont').
  3. Runs each branch in a FRESH World. saveState files do not carry RNG
     state, so a fresh world with a fixed --seed gives every action branch
     of a (state, future_seed) cell the same demand realization as closely as
     SUMO allows -- strictly better "identical subsequent conditions" than
     sequential branches inside one process.

Outputs one JSONL shard per (mode, shard) under --out.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "allocation_accuracy_counterfactual",
    REPO_ROOT / "tools" / "allocation_accuracy_counterfactual.py",
)
cf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cf)  # noqa: SLF001 - reuse frozen audit helpers

import numpy as np

from common import interface as registry_interface
from common.registry import Registry
import world.world_sumo as world_sumo


ACTION_COUNT = cf.ACTION_COUNT
ACTION_INTERVAL = cf.ACTION_INTERVAL
GAMMA = 0.95
REWARD_SCALE = 12.0
SCENE_LIST = cf.SCENES
DEFAULT_MANIFEST = (
    REPO_ROOT
    / "data/output_data"
    / "allocation_accuracy_formal_state"
    / "allocation_accuracy_state_manifest.csv"
)
INDEP_ROOT = REPO_ROOT / "data/output_data" / "tsc" / "sumo_dqn"
DEFAULT_OUT = REPO_ROOT / "data/output_data" / "action_ranking_audit" / "v2_rollouts"


def remap_path(value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value.replace("/data/output_data/", "/data/output_data/")


def independent_checkpoint(scene: str, seed: int) -> Optional[Path]:
    candidates = sorted(
        (INDEP_ROOT / scene).glob(f"p1_formal_dqn_{scene}_seed{seed}_400ep_*/model/400_0.pt")
    )
    return candidates[-1] if candidates else None


def load_manifest(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            row = dict(record)
            row["state"] = json.loads(row.get("state_vector", "[]"))
            row["lane_ids"] = json.loads(row.get("lane_ids", "[]"))
            row["lane_counts"] = json.loads(row.get("lane_counts", "{}"))
            for key in ("decision_index", "phase", "training_seed"):
                if row.get(key, "") != "":
                    row[key] = int(float(row[key]))
            for key in ("simulation_time_seconds", "total_vehicle_count", "direction_imbalance"):
                if row.get(key, "") != "":
                    row[key] = float(row[key])
            for key in ("state_path", "python_state_path", "source_checkpoint", "simulator_config"):
                row[key] = remap_path(row.get(key))
            rows.append(row)
    return rows


def queue_sum(world: Any, lane_ids: Sequence[str]) -> float:
    counts = world.get_lane_waiting_vehicle_count()
    return float(sum(float(counts.get(lane, 0.0)) for lane in lane_ids))


def run_branch(
    state: Mapping[str, Any],
    continuation: str,
    candidate_action: int,
    future_seed: Optional[int],
    rollout_seconds: int,
    log_root: Path,
    capture_boundary: bool = False,
    episode_end: Optional[float] = None,
) -> Dict[str, Any]:
    """One fresh-world branch: loadState -> first cycle action -> continuation.

    capture_boundary records (obs, phase, Q vectors of both checkpoints) at the
    30/60/90 s decision points.  episode_end keeps stepping until the SUMO
    simulation clock reaches that value (full-episode return approximation).
    """
    scene = str(state["scene"])
    registry_interface.Command_Setting_Interface({"command": {"sumo_seed": future_seed}})
    log_dir = log_root / scene / str(state["state_id"])
    log_dir.mkdir(parents=True, exist_ok=True)
    Registry.mapping["logger_mapping"]["path"].path = str(log_dir)
    record: Dict[str, Any] = {
        "state_id": str(state["state_id"]),
        "scene": scene,
        "source_run_id": state.get("source_run_id", ""),
        "order": state.get("order", ""),
        "training_seed": state.get("training_seed"),
        "continuation": continuation,
        "future_seed": future_seed if future_seed is not None else "default",
        "candidate_action": int(candidate_action),
        "rollout_seconds": int(rollout_seconds),
    }
    if continuation == "cont":
        cont_ckpt = Path(str(state["source_checkpoint"]))
    else:
        cont_ckpt = independent_checkpoint(scene, int(state["training_seed"]))
    if cont_ckpt is None or not cont_ckpt.is_file():
        record.update({"safety_valid": False, "error": f"missing continuation checkpoint: {cont_ckpt}"})
        return record
    world = None
    try:
        world = world_sumo.World(str(state["simulator_config"]), 1, interface="libsumo")
        world.reset()
        feature_agent = cf.FrozenDQN(world, Path(str(state["source_checkpoint"])))
        cont_agent = feature_agent if continuation == "cont" else cf.FrozenDQN(world, cont_ckpt)
        lane_ids = feature_agent.lane_ids
        cf.load_snapshot(world, Path(state["state_path"]), Path(state["python_state_path"]))
        baseline_departures = len(world.vehicles)
        previous_action = int(state["phase"])
        phase_switches = int(candidate_action != previous_action)
        decisions = max(1, int(np.ceil(rollout_seconds / ACTION_INTERVAL)))
        if episode_end is not None:
            remaining = max(0.0, float(episode_end) - float(state["simulation_time_seconds"]))
            decisions = max(1, int(np.ceil(remaining / ACTION_INTERVAL)))
        cycle_queue_sec: List[float] = []
        cycle_reward: List[float] = []
        actions: List[int] = []
        real_delay_at: Dict[int, float] = {}
        throughput_at: Dict[int, float] = {}
        step_counter = 0
        current_action = int(candidate_action)
        terminated = False
        step_error: Optional[str] = None
        for decision in range(decisions):
            if terminated:
                break
            if decision > 0:
                phase = feature_agent.phase()
                obs = feature_agent.observation()
                if capture_boundary and decision * ACTION_INTERVAL in (30, 60, 90):
                    bkey = f"boundary_{decision * ACTION_INTERVAL}"
                    record[f"{bkey}_phase"] = int(phase)
                    record[f"{bkey}_obs"] = json.dumps(
                        [float(v) for v in obs.reshape(-1)]
                    )
                    for tag, agent in (("feature", feature_agent), ("cont", cont_agent)):
                        try:
                            record[f"{bkey}_q_{tag}"] = json.dumps(
                                [float(v) for v in agent.q_values(obs, phase)]
                            )
                        except Exception as exc:
                            record[f"{bkey}_q_{tag}_error"] = f"{type(exc).__name__}: {exc}"
                current_action = cont_agent.action(obs, phase)
                if current_action != previous_action:
                    phase_switches += 1
            previous_action = int(current_action)
            actions.append(int(current_action))
            queue_sec = 0.0
            steps_this_cycle = 0
            for _ in range(ACTION_INTERVAL):
                if episode_end is not None and world.get_current_time() >= episode_end:
                    terminated = True
                    break
                try:
                    world.step([current_action])
                except Exception as exc:
                    terminated = True
                    step_error = f"{type(exc).__name__}: {exc}"
                    break
                step_counter += 1
                steps_this_cycle += 1
                queue_sec += queue_sum(world, lane_ids)
                if step_counter in (30, 60, 90):
                    try:
                        real_delay_at[step_counter] = float(world.get_real_delay())
                    except Exception:
                        real_delay_at[step_counter] = float("nan")
                    throughput_at[step_counter] = float(len(world.vehicles) - baseline_departures)
            cycle_queue_sec.append(queue_sec)
            cycle_reward.append(
                -REWARD_SCALE * queue_sec / (len(lane_ids) * max(1, steps_this_cycle))
            )
        result: Dict[str, Any] = {
            "actions_after_first_cycle": json.dumps(actions[1:]),
            "phase_switch_count": int(phase_switches),
            "cycle_queue_sec": json.dumps(cycle_queue_sec),
            "cycle_reward": json.dumps(cycle_reward),
            "safety_valid": True,
            "error": None,
        }
        if episode_end is not None or step_error is not None:
            result["n_cycles"] = len(cycle_reward)
            result["final_sim_time"] = float(world.get_current_time())
            result["terminated_at_episode_end"] = bool(
                episode_end is not None and float(world.get_current_time()) >= episode_end - 1e-6
            )
            result["step_error"] = step_error
            result["dqn_return_full"] = float(
                np.sum([GAMMA ** k * r for k, r in enumerate(cycle_reward)])
            )
            result["undiscounted_return_full"] = float(np.sum(cycle_reward))
        for horizon, cycles in ((30, 3), (60, 6), (90, 9)):
            used = cycle_queue_sec[:cycles]
            rewards = cycle_reward[:cycles]
            result[f"queue_auc_{horizon}"] = float(np.sum(used))
            result[f"real_delay_{horizon}"] = real_delay_at.get(horizon, float("nan"))
            result[f"throughput_{horizon}"] = throughput_at.get(horizon, float("nan"))
            result[f"dqn_return_{horizon}"] = float(
                np.sum([GAMMA ** k * r for k, r in enumerate(rewards)])
            )
            result[f"undiscounted_return_{horizon}"] = float(np.sum(rewards))
        record.update(result)
    except Exception as exc:
        record.update({"safety_valid": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if world is not None:
            world.close()
    return record


def continuation_tasks(states: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    tasks = []
    for state in states:
        for continuation in ("indep", "cont"):
            for action in range(ACTION_COUNT):
                tasks.append({"state": state, "continuation": continuation,
                              "action": action, "future_seed": None})
    return tasks


def multiseed_tasks(
    states: Sequence[Mapping[str, Any]],
    pairs: Mapping[str, Tuple[int, int]],
    seeds: Sequence[Optional[int]],
) -> List[Dict[str, Any]]:
    tasks = []
    for state in states:
        pair = pairs.get(str(state["state_id"]))
        if pair is None:
            continue
        for seed in seeds:
            for action in sorted(set(pair)):
                tasks.append({"state": state, "continuation": "indep",
                              "action": action, "future_seed": seed})
    return tasks


def derive_error_pairs(
    rollouts_csv: Path,
    predictions_csv: Path,
    min_gap: float,
    max_states: int,
) -> Dict[str, Tuple[int, int]]:
    """Pairs (argmax-Q, oracle) where the true 60 s gap is at least min_gap."""
    import pandas as pd

    pred = pd.read_csv(predictions_csv)
    pred = pred[pred["method"] == "CONT-DQN"]
    pred = pred[pred["prediction_error"].isna() | pred["prediction_error"].astype(str).str.strip().eq("")]
    rolls = pd.read_csv(rollouts_csv)
    rolls = rolls[rolls["safety_valid"].astype(str).str.lower().eq("true")]
    costs = (
        rolls.pivot_table(index=["scene", "state_id"], columns="candidate_action", values="cost_60")
    )
    pairs: Dict[str, Tuple[int, int]] = {}
    selected = 0
    for _, row in pred.iterrows():
        key = (row["scene"], row["state_id"])
        if key not in costs.index:
            continue
        cost = costs.loc[key]
        if cost.isna().any():
            continue
        predicted = int(row["predicted_action"])
        oracle = int(cost.idxmin())
        gap = float(cost[predicted] - cost[oracle])
        if predicted == oracle or gap < min_gap:
            continue
        pairs[str(row["state_id"])] = (predicted, oracle)
        selected += 1
        if selected >= max_states:
            break
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--mode", choices=("continuations", "multiseed", "boundary", "episodes"), required=True)
    parser.add_argument("--episode-end", type=float, default=3600.0,
                        help="episodes mode: simulate until this SUMO clock time")
    parser.add_argument("--seeds", type=str, default="101,202,303,404,505",
                        help="comma list of future seeds; 'default' = no --seed flag")
    parser.add_argument("--rollout-seconds", type=int, default=90)
    parser.add_argument("--states", type=str, default="", help="comma state_id subset")
    parser.add_argument("--min-gap", type=float, default=0.30)
    parser.add_argument("--max-pair-states", type=int, default=80)
    parser.add_argument("--rollouts-csv", type=Path,
                        default=REPO_ROOT / "data/output_data" / "allocation_accuracy_formal_cf" / "allocation_accuracy_all_action_rollouts.csv")
    parser.add_argument("--predictions-csv", type=Path,
                        default=REPO_ROOT / "data/output_data" / "allocation_accuracy_formal_cf" / "allocation_accuracy_agent_predictions.csv")
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0, help="cap tasks for smoke runs")
    args = parser.parse_args()

    states = load_manifest(args.manifest)
    if args.states:
        keep = set(args.states.split(","))
        states = [row for row in states if str(row["state_id"]) in keep]

    if args.mode in ("continuations", "boundary", "episodes"):
        tasks = continuation_tasks(states)
        tag = args.mode
    else:
        seeds: List[Optional[int]] = []
        for token in args.seeds.split(","):
            token = token.strip()
            seeds.append(None if token in ("", "default", "none") else int(token))
        pairs = derive_error_pairs(args.rollouts_csv, args.predictions_csv, args.min_gap, args.max_pair_states)
        states = [row for row in states if str(row["state_id"]) in pairs]
        tasks = multiseed_tasks(states, pairs, seeds)
        tag = "multiseed"
        pair_path = args.out / "multiseed_pairs.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        pair_path.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

    tasks = [task for i, task in enumerate(tasks) if i % args.nshards == args.shard]
    if args.limit:
        tasks = tasks[: args.limit]

    out_dir = args.out / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"branches_shard{args.shard:02d}.jsonl"
    started = time.time()
    with out_path.open("w", encoding="utf-8") as handle:
        for index, task in enumerate(tasks):
            record = run_branch(
                task["state"], task["continuation"], task["action"],
                task["future_seed"], args.rollout_seconds, args.out / "world_logs",
                capture_boundary=(args.mode == "boundary"),
                episode_end=(args.episode_end if args.mode == "episodes" else None),
            )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 32 == 0:
                elapsed = time.time() - started
                rate = elapsed / max(1, index)
                print(
                    f"[{tag} shard{args.shard}] {index}/{len(tasks)} "
                    f"elapsed={elapsed:.0f}s eta={rate * len(tasks):.0f}s",
                    flush=True,
                )
    print(f"[done] {len(tasks)} branches -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
