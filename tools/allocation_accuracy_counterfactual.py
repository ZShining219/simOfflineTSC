#!/usr/bin/env python3
"""Counterfactual Top-1 allocation-accuracy audit for the SUMO HA results.

This tool is intentionally independent of the training entry points.  It
replays a frozen CONT-DQN trajectory, stores SUMO saveState snapshots, loads a
snapshot for each branch, lets one candidate action control the first decision
cycle, and then hands control to a frozen scene-specific independent DQN.

The default command is a small, reproducible smoke audit (4 scenes x 8
states).  ``--formal`` selects 64 states per scene when the assets are
available.  No checkpoint is modified and no training code is invoked.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

# The repository's SUMO adapter imports SUMO_HOME at module import time.  The
# host image keeps the executable behind a broken conda wrapper, so point both
# libraries at the relocatable binary explicitly before importing world_sumo.
REPO_ROOT = Path(__file__).resolve().parents[1]
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SUMO_ROOT = Path(os.environ.get("SUMO_ROOT", "/opt/host_colight/lib/python3.10/site-packages/sumo"))
os.environ.setdefault("SUMO_HOME", str(SUMO_ROOT / "tools"))
os.environ.setdefault("SUMO_BINARY", str(SUMO_ROOT / "bin" / "sumo"))

import numpy as np
import torch

from agent import utils as agent_utils
from agent.dqn import DQNNet
from common import interface as registry_interface
from common.registry import Registry
from generator import LaneVehicleGenerator
import world.world_sumo as world_sumo


SCENES: Tuple[str, ...] = (
    "sumohz1x1_config2",
    "sumohz1x1",
    "sumohz1x1_config4",
    "sumohz1x1_config3",
)
ORDERS: Mapping[str, Tuple[str, ...]] = {
    "O1": ("sumohz1x1_config3", "sumohz1x1_config2", "sumohz1x1_config4", "sumohz1x1"),
    "O2": ("sumohz1x1", "sumohz1x1_config4", "sumohz1x1_config2", "sumohz1x1_config3"),
    "O3": ("sumohz1x1_config2", "sumohz1x1_config3", "sumohz1x1", "sumohz1x1_config4"),
    "O4": ("sumohz1x1_config4", "sumohz1x1", "sumohz1x1_config3", "sumohz1x1_config2"),
}
METHODS = ("CONT-DQN", "P1C-DHOA-R25", "P1C-DHOA-R50")
ACTION_COUNT = 8
ACTION_INTERVAL = 10
PRIMARY_HORIZON = 60
LABEL_HORIZONS = (30, 60, 90)
HIGH_CONFIDENCE_MARGIN = 0.05
OUTPUT_DEFAULT = REPO_ROOT / "data" / "output_data" / "allocation_accuracy"
FORMAL_ROOT = REPO_ROOT / "data" / "output_data" / "ha_sodqn" / "formal_e7705f7_20260726"
INDEPENDENT_ROOT = REPO_ROOT / "data" / "output_data" / "tsc" / "sumo_dqn"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row.keys())
        fieldnames = sorted(keys)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    temporary.replace(path)


def parse_run_id(run_id: str) -> Tuple[str, int]:
    parts = run_id.split("-")
    order = next((part for part in parts if part.startswith("O")), "O2")
    seed_token = next((part for part in parts if part.startswith("SD")), "SD0")
    return order, int(seed_token[2:])


def final_checkpoint(run_id: str) -> Optional[Path]:
    run_root = FORMAL_ROOT / run_id
    manifest = run_root / "logical_run_manifest.json"
    if not manifest.is_file():
        return None
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("status") != "completed":
        return None
    attempt = payload.get("effective_attempt")
    if not isinstance(attempt, str) or not attempt.startswith("attempt_"):
        return None
    path = run_root / "attempts" / attempt / "checkpoints" / "online" / "stage_04_episode_0100.pt"
    return path if path.is_file() else None


def run_id(method: str, order: str, seed: int) -> str:
    if method == "CONT-DQN":
        return f"CONT-FIFO-{order}-SD{seed}"
    if method == "P1C-DHOA-R25":
        return f"P1C-DHOA-R25-{order}-SD{seed}"
    if method == "P1C-DHOA-R50":
        return f"P1C-DHOA-R50-{order}-SD{seed}"
    raise ValueError(method)


def simulator_config(scene: str) -> Path:
    return REPO_ROOT / "configs" / "sim" / f"{scene}.cfg"


def independent_checkpoint(scene: str, seed: int) -> Optional[Path]:
    candidates = sorted((INDEPENDENT_ROOT / scene).glob(f"p1_formal_dqn_{scene}_seed{seed}_400ep_*/model/400_0.pt"))
    if candidates:
        return candidates[-1]
    # Do not silently substitute seed-0 for another seed.  The counterfactual
    # continuation is part of the oracle definition and must be matched to the
    # source training seed; a missing seed is reported as a missing asset.
    return None


def snapshot_identity(path: Path) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    if payload.get("checkpoint_type") == "online_only":
        return {
            "path": str(path),
            "format": "online_only",
            "digest": payload.get("online_parameter_digest"),
            "model": payload.get("model", {}),
        }
    return {"path": str(path), "format": "state_dict", "digest": sha256_file(path), "model": {}}


class FrozenDQN:
    """The six-layer-free DQN used by both old independent and HA snapshots."""

    def __init__(self, world: Any, checkpoint: Path):
        payload = torch.load(checkpoint, map_location="cpu")
        if isinstance(payload, Mapping) and payload.get("checkpoint_type") == "online_only":
            state_dict = payload["online_model_state_dict"]
            model_info = payload.get("model", {})
            self.digest = payload.get("online_parameter_digest")
        else:
            state_dict = payload
            model_info = {"phase": True, "one_hot": True, "input_dim": 16, "output_dim": 8}
            self.digest = sha256_file(checkpoint)
        input_dim = int(model_info.get("input_dim", 16))
        output_dim = int(model_info.get("output_dim", ACTION_COUNT))
        self.phase_enabled = bool(model_info.get("phase", True))
        self.one_hot = bool(model_info.get("one_hot", True))
        self.model = DQNNet(input_dim, output_dim)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.checkpoint = checkpoint
        self._bind(world)

    def _bind(self, world: Any) -> None:
        self.world = world
        self.inter = world.intersections[0]
        self.ob_generator = LaneVehicleGenerator(world, self.inter, ["lane_count"], in_only=True, average=None)
        self.lane_ids = [lane for road in self.ob_generator.lanes for lane in road]
        if len(self.lane_ids) != 8:
            raise ValueError(f"Expected 8 incoming lanes, got {len(self.lane_ids)}")
        if self.model.dense_1.in_features != (16 if self.phase_enabled and self.one_hot else 9 if self.phase_enabled else 8):
            raise ValueError("Frozen checkpoint input dimension does not match SUMO state schema")
        if self.model.dense_3.out_features != ACTION_COUNT:
            raise ValueError("Frozen checkpoint does not have eight actions")

    def observation(self) -> np.ndarray:
        if "lane_count" not in self.world.info:
            self.world._update_infos()
        return np.asarray(self.ob_generator.generate(), dtype=np.float32).reshape(1, -1)

    def phase(self) -> int:
        return int(self.inter.current_phase)

    def q_values(self, observation: np.ndarray, phase: int) -> np.ndarray:
        phase = int(phase)
        if phase < 0 or phase >= ACTION_COUNT:
            # A transition phase is not a legal policy decision point.  The
            # runner records it as invalid instead of inventing a phase label.
            raise ValueError(f"non-green phase at decision point: {phase}")
        if self.phase_enabled:
            phase_feature = agent_utils.idx2onehot(np.asarray([phase]), ACTION_COUNT) if self.one_hot else np.asarray([[phase]], dtype=np.float32)
            feature = np.concatenate([observation, phase_feature], axis=1)
        else:
            feature = observation
        with torch.no_grad():
            values = self.model(torch.as_tensor(feature, dtype=torch.float32)).cpu().numpy()[0]
        return np.asarray(values, dtype=float)

    def action(self, observation: np.ndarray, phase: int) -> int:
        return int(np.argmax(self.q_values(observation, phase)))


def capture_python_state(world: Any) -> Dict[str, Any]:
    return {
        "run": int(world.run),
        "vehicles": copy.deepcopy(world.vehicles),
        "inside_vehicles": copy.deepcopy(world.inside_vehicles),
        "vehicle_trajectory": copy.deepcopy(world.vehicle_trajectory),
        "vehicle_maxspeed": copy.deepcopy(world.vehicle_maxspeed),
        "real_delay": copy.deepcopy(world.real_delay),
        "intersections": [
            {
                "current_phase": int(inter.current_phase),
                "virtual_phase": int(inter.virtual_phase),
                "next_phase": int(inter.next_phase),
                "current_phase_time": int(inter.current_phase_time),
                "waiting_times": copy.deepcopy(inter.waiting_times),
            }
            for inter in world.intersections
        ],
    }


def restore_python_state(world: Any, state: Mapping[str, Any]) -> None:
    world.run = int(state.get("run", 0))
    world.vehicles = copy.deepcopy(state.get("vehicles", {}))
    world.inside_vehicles = copy.deepcopy(state.get("inside_vehicles", {}))
    world.vehicle_trajectory = copy.deepcopy(state.get("vehicle_trajectory", {}))
    world.vehicle_maxspeed = copy.deepcopy(state.get("vehicle_maxspeed", {}))
    world.real_delay = copy.deepcopy(state.get("real_delay", {}))
    for inter, saved in zip(world.intersections, state.get("intersections", [])):
        inter.current_phase = int(saved.get("current_phase", world.eng.trafficlight.getPhase(inter.id)))
        inter.virtual_phase = int(saved.get("virtual_phase", inter.current_phase))
        inter.next_phase = int(saved.get("next_phase", inter.current_phase))
        inter.current_phase_time = int(saved.get("current_phase_time", 0))
        inter.waiting_times = copy.deepcopy(saved.get("waiting_times", {}))
        # SUMO has restored the microscopic vehicles; rebuild only the Python
        # observation cache, without advancing simulation time.
        inter.observe(world.step_length, world.max_distance)
    world._update_infos()


def save_snapshot(world: Any, xml_path: Path, pickle_path: Path) -> None:
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    pickle_path.parent.mkdir(parents=True, exist_ok=True)
    world.eng.simulation.saveState(str(xml_path))
    with pickle_path.open("wb") as handle:
        pickle.dump(capture_python_state(world), handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_snapshot(world: Any, xml_path: Path, pickle_path: Path) -> None:
    world.eng.simulation.loadState(str(xml_path))
    with pickle_path.open("rb") as handle:
        state = pickle.load(handle)
    restore_python_state(world, state)


def lane_counts_from_observation(world: Any, agent: FrozenDQN) -> Dict[str, int]:
    all_counts = world.get_lane_vehicle_count()
    return {lane: int(all_counts.get(lane, 0)) for lane in agent.lane_ids}


def extract_phase_mapping(world: Any) -> Dict[str, Any]:
    """Return the repository-derived action/phase/lane mapping for one scene.

    The action index is taken from ``Intersection.green_phases`` and its
    ``phase_available_startlanes``/``phase_available_lanelinks`` tables.  No
    phase-to-lane relation is inferred from the integer action id.
    """
    inter = world.intersections[0]
    actions = []
    for action, phase in enumerate(inter.green_phases):
        actions.append({
            "action": int(action),
            "sumo_green_state": str(getattr(phase, "state", "")),
            "served_incoming_lanes": sorted(str(lane) for lane in inter.phase_available_startlanes[action]),
            "controlled_links": [
                [str(link[0]), str(link[1])] for link in inter.phase_available_lanelinks[action]
            ],
        })
    return {"intersection_id": str(inter.id), "action_count": len(actions), "actions": actions}


def queue_from_world(world: Any, agent: FrozenDQN) -> float:
    counts = world.get_lane_waiting_vehicle_count()
    return float(sum(float(counts.get(lane, 0.0)) for lane in agent.lane_ids))


def traffic_features(world: Any, agent: FrozenDQN, state: np.ndarray) -> Tuple[float, float]:
    counts = {lane: float(value) for lane, value in zip(agent.lane_ids, state.reshape(-1))}
    road_loads = []
    for road in world.intersections[0].in_roads:
        road_loads.append(sum(counts.get(lane, 0.0) for lane in world.intersections[0].road_lane_mapping[road]))
    total = float(sum(road_loads))
    if total <= 0:
        return total, 0.0
    imbalance = (max(road_loads) - min(road_loads)) / total
    return total, float(imbalance)


def existing_eval_index(snapshot: Path, scene: str) -> Dict[float, Mapping[str, Any]]:
    """Find post-decision logs for a checkpoint, if the formal evaluator has one."""
    root = FORMAL_ROOT / "_shared_evaluation" / "physical"
    if not root.is_dir():
        return {}
    target = str(snapshot.resolve())
    for request in root.glob("*/attempt_*/request.json"):
        try:
            payload = json.loads(request.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(Path(payload.get("snapshot_path", "")).resolve()) != target:
            continue
        if str(payload.get("evaluation_network", "")) != str(scene):
            continue
        decision_file = request.parent / "decisions.jsonl"
        if not decision_file.is_file():
            continue
        records: Dict[float, Mapping[str, Any]] = {}
        for line in decision_file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                records[float(rec.get("simulation_time_seconds", -1))] = rec
        return records
    return {}


def replay_source(
    scene: str,
    source_run: Mapping[str, Any],
    output_dir: Path,
    capture_stride: int,
    max_decisions: int = 360,
) -> List[Dict[str, Any]]:
    """Deterministically replay a CONT frozen checkpoint and save candidates."""
    run = str(source_run["run_id"])
    checkpoint = Path(source_run["checkpoint"])
    cfg = simulator_config(scene)
    registry_interface.Command_Setting_Interface({"command": {"sumo_seed": source_run.get("evaluation_seed")}})
    log_dir = output_dir / "replay_logs" / run / scene
    log_dir.mkdir(parents=True, exist_ok=True)
    Registry.mapping["logger_mapping"]["path"].path = str(log_dir)
    world = None
    candidates: List[Dict[str, Any]] = []
    eval_records = existing_eval_index(checkpoint, scene)
    print(f"[replay] scene={scene} source={run} max_decisions={max_decisions}", flush=True)
    try:
        world = world_sumo.World(str(cfg), 1, interface="libsumo")
        world.reset()
        agent = FrozenDQN(world, checkpoint)
        obs = agent.observation()
        for decision_index in range(1, max_decisions + 1):
            phase = agent.phase()
            state = obs.reshape(-1).astype(float)
            counts = lane_counts_from_observation(world, agent)
            total, imbalance = traffic_features(world, agent, state)
            sim_time = float(world.get_current_time())
            valid_decision = 0 <= phase < ACTION_COUNT
            state_path = None
            python_path = None
            replay_match = None
            if valid_decision and decision_index % capture_stride == 0 and sim_time >= ACTION_INTERVAL * 2:
                slug = f"{run}__{scene}__d{decision_index:03d}"
                state_path = output_dir / "states" / f"{slug}.xml"
                python_path = output_dir / "states" / f"{slug}.pkl"
                save_snapshot(world, state_path, python_path)
                # Existing evaluator records are post-cycle.  At time t, the
                # previous decision record is the deterministic replay check.
                reference = eval_records.get(sim_time)
                if reference is not None:
                    ref_counts = reference.get("lane_vehicle_counts", {})
                    replay_match = all(int(ref_counts.get(lane, -1)) == int(value) for lane, value in counts.items())
            if state_path is not None:
                candidates.append({
                    "scene": scene,
                    "source_run_id": run,
                    "order": source_run["order"],
                    "training_seed": int(source_run["training_seed"]),
                    "evaluation_seed": source_run.get("evaluation_seed"),
                    "source_checkpoint": str(checkpoint),
                    "source_checkpoint_digest": source_run.get("checkpoint_digest"),
                    "simulator_config": str(cfg),
                    "state_path": str(state_path),
                    "python_state_path": str(python_path),
                    "decision_index": decision_index,
                    "simulation_time_seconds": sim_time,
                    "phase": phase,
                    "state": state.tolist(),
                    "lane_ids": agent.lane_ids,
                    "lane_counts": counts,
                    "total_vehicle_count": total,
                    "direction_imbalance": imbalance,
                    "replay_match": replay_match,
                })
            try:
                action = agent.action(obs, phase)
            except ValueError:
                action = int(phase % ACTION_COUNT)
            for _ in range(ACTION_INTERVAL):
                world.step([action])
            obs = agent.observation()
            if world.eng.simulation.getMinExpectedNumber() <= 0 and world.get_current_time() > 60:
                break
    finally:
        if world is not None:
            world.close()
    print(f"[replay] scene={scene} source={run} captured={len(candidates)}", flush=True)
    return candidates


def assign_bins(candidates: List[Dict[str, Any]]) -> None:
    if not candidates:
        return
    loads = np.asarray([float(row["total_vehicle_count"]) for row in candidates], dtype=float)
    times = np.asarray([float(row["simulation_time_seconds"]) for row in candidates], dtype=float)
    load_q = np.quantile(loads, [1 / 3, 2 / 3]) if len(loads) > 2 else [float(np.median(loads)), float(np.max(loads))]
    time_q = np.quantile(times, [0.25, 0.5, 0.75]) if len(times) > 3 else [float(np.median(times))] * 3
    for row in candidates:
        load = float(row["total_vehicle_count"])
        t = float(row["simulation_time_seconds"])
        row["load_bin"] = "low" if load <= load_q[0] else "mid" if load <= load_q[1] else "high"
        row["time_bin"] = f"q{sum(t > q for q in time_q) + 1}"


def select_states(candidates: List[Dict[str, Any]], per_scene: int) -> List[Dict[str, Any]]:
    if not candidates:
        return []
    assign_bins(candidates)
    remaining = list(candidates)
    selected: List[Dict[str, Any]] = []
    # First guarantee phase coverage where the source trajectory provides it.
    median_load = statistics.median(float(row["total_vehicle_count"]) for row in remaining)
    for phase in range(ACTION_COUNT):
        pool = [row for row in remaining if int(row["phase"]) == phase]
        if not pool:
            continue
        choice = min(pool, key=lambda row: (abs(float(row["total_vehicle_count"]) - median_load), float(row["simulation_time_seconds"])))
        selected.append(choice)
        remaining.remove(choice)
    # Fill by a fixed phase/load/time/run round-robin score.  The score is
    # independent of any agent prediction or oracle outcome.
    while remaining and len(selected) < per_scene:
        phase_counts = Counter(int(row["phase"]) for row in selected)
        load_counts = Counter(str(row["load_bin"]) for row in selected)
        time_counts = Counter(str(row["time_bin"]) for row in selected)
        run_counts = Counter(str(row["source_run_id"]) for row in selected)
        choice = min(
            remaining,
            key=lambda row: (
                phase_counts[int(row["phase"])],
                load_counts[str(row["load_bin"])],
                time_counts[str(row["time_bin"])],
                run_counts[str(row["source_run_id"])],
                float(row["simulation_time_seconds"]),
            ),
        )
        selected.append(choice)
        remaining.remove(choice)
    return selected[:per_scene]


def snapshot_python_baseline(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def collect_rollout_metrics(
    world: Any,
    agent: FrozenDQN,
    independent: FrozenDQN,
    state_row: Mapping[str, Any],
    candidate_action: int,
    rollout_seconds: int = 90,
) -> Dict[str, Any]:
    """Run one action branch and return 30/60/90 metrics."""
    load_snapshot(world, Path(state_row["state_path"]), Path(state_row["python_state_path"]))
    baseline_departures = len(world.vehicles)
    previous_action = int(state_row["phase"])
    phase_switches = int(candidate_action != previous_action)
    queues: List[float] = []
    real_delays: List[float] = []
    throughput: List[float] = []
    actions: List[int] = []
    error = None
    try:
        current_action = int(candidate_action)
        decisions = max(1, int(math.ceil(rollout_seconds / ACTION_INTERVAL)))
        for decision in range(decisions):
            actions.append(current_action)
            if decision > 0:
                phase = agent.phase()
                obs = agent.observation()
                current_action = independent.action(obs, phase)
                actions[-1] = int(current_action)
                if current_action != previous_action:
                    phase_switches += 1
            previous_action = int(current_action)
            for _ in range(ACTION_INTERVAL):
                world.step([current_action])
                queues.append(queue_from_world(world, agent))
                try:
                    real_delays.append(float(world.get_real_delay()))
                except Exception:
                    real_delays.append(float("nan"))
                throughput.append(float(len(world.vehicles) - baseline_departures))
    except Exception as exc:  # keep the other 7 branches auditable
        error = f"{type(exc).__name__}: {exc}"
    result: Dict[str, Any] = {
        "candidate_action": int(candidate_action),
        "actions_after_first_cycle": json.dumps(actions[1:]),
        "phase_switch_count": int(phase_switches),
        "safety_valid": error is None,
        "rollout_error": error,
    }
    for horizon in LABEL_HORIZONS:
        prefix = queues[: min(horizon, len(queues))]
        delay_prefix = real_delays[: min(horizon, len(real_delays))]
        tp_prefix = throughput[: min(horizon, len(throughput))]
        result[f"queue_auc_{horizon}"] = float(np.nansum(prefix)) if prefix else float("nan")
        result[f"real_delay_{horizon}"] = float(delay_prefix[-1]) if delay_prefix else float("nan")
        result[f"throughput_{horizon}"] = float(tp_prefix[-1]) if tp_prefix else float("nan")
    return result


def minmax(values: Sequence[float]) -> List[float]:
    arr = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(arr)):
        return [float("nan")] * len(arr)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    if math.isclose(lo, hi):
        # Fixed and documented tie convention: a constant metric contributes
        # zero normalized burden; the action tie-break remains lowest index.
        return [0.0] * len(arr)
    return ((arr - lo) / (hi - lo)).tolist()


def label_rollouts(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    labels = []
    for state_id, actions in sorted(grouped.items()):
        first = actions[0] if actions else {}
        label: Dict[str, Any] = {
            "state_id": state_id,
            "scene": first.get("scene", ""),
            "source_run_id": first.get("source_run_id", ""),
            "order": first.get("order", ""),
            "training_seed": first.get("training_seed", ""),
            "valid_action_count": 0,
        }
        for horizon in LABEL_HORIZONS:
            q = minmax([float(row[f"queue_auc_{horizon}"]) for row in actions])
            d = minmax([float(row[f"real_delay_{horizon}"]) for row in actions])
            tp = minmax([float(row[f"throughput_{horizon}"]) for row in actions])
            costs = [(q[i] + d[i] + (1.0 - tp[i])) / 3.0 if np.isfinite(q[i] + d[i] + tp[i]) else float("nan") for i in range(len(actions))]
            valid = [i for i, value in enumerate(costs) if np.isfinite(value) and bool(actions[i].get("safety_valid"))]
            # A strict Top-1 label is only reliable when every one of the eight
            # legal branches produced a finite, safety-valid result.  A partial
            # action set would turn a rollout failure into a hidden label bias.
            if len(actions) != ACTION_COUNT or len(valid) != ACTION_COUNT:
                label[f"oracle_action_{horizon}"] = ""
                label[f"best_cost_{horizon}"] = float("nan")
                label[f"second_best_cost_{horizon}"] = float("nan")
                label[f"oracle_margin_{horizon}"] = float("nan")
                continue
            ranking = sorted(valid, key=lambda i: (costs[i], int(actions[i]["candidate_action"])))
            best, second = ranking[0], ranking[1] if len(ranking) > 1 else ranking[0]
            label[f"oracle_action_{horizon}"] = int(actions[best]["candidate_action"])
            label[f"best_cost_{horizon}"] = float(costs[best])
            label[f"second_best_cost_{horizon}"] = float(costs[second])
            label[f"oracle_margin_{horizon}"] = float(costs[second] - costs[best])
            label["valid_action_count"] = max(int(label["valid_action_count"]), len(valid))
            for i, action_row in enumerate(actions):
                action_row[f"normalized_queue_{horizon}"] = q[i]
                action_row[f"normalized_real_delay_{horizon}"] = d[i]
                action_row[f"normalized_throughput_{horizon}"] = tp[i]
                action_row[f"cost_{horizon}"] = costs[i]
        stable = all(label.get(f"oracle_action_{horizon}") == label.get("oracle_action_60") for horizon in LABEL_HORIZONS if label.get(f"oracle_action_{horizon}") != "")
        label["labels_consistent_30_60_90"] = bool(stable)
        label["high_confidence"] = bool(stable and float(label.get("oracle_margin_60", float("nan"))) >= HIGH_CONFIDENCE_MARGIN)
        labels.append(label)
    return labels


def wilson(k: int, n: int, z: float = 1.959963984540054) -> Tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def predictions(states: Sequence[Mapping[str, Any]], labels: Mapping[str, Mapping[str, Any]], model_rows: Mapping[str, Mapping[Tuple[str, str, int], str]], output_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in states:
        grouped[str(row["scene"])].append(row)
    for method, info in model_rows.items():
        for scene, scene_states in grouped.items():
            Registry.mapping["logger_mapping"]["path"].path = str(output_dir / "prediction_logs" / method / scene)
            (output_dir / "prediction_logs" / method / scene).mkdir(parents=True, exist_ok=True)
            world = world_sumo.World(str(simulator_config(scene)), 1, interface="libsumo")
            current_checkpoint: Optional[str] = None
            agent: Optional[FrozenDQN] = None
            try:
                world.reset()
                for state in scene_states:
                    state_key = (str(scene), str(state.get("order", "")), int(state.get("training_seed", 0)))
                    checkpoint = info.get(state_key)
                    label = labels.get(str(state["state_id"]), {})
                    oracle = label.get("oracle_action_60", "")
                    if not checkpoint:
                        rows.append({"state_id": state["state_id"], "scene": scene, "method": method, "order": state["order"], "training_seed": state["training_seed"], "predicted_action": "", "oracle_action": oracle, "top1_correct": False, "top2_correct": False, "action_regret": float("nan"), "prediction_error": "missing matched checkpoint"})
                        continue
                    if checkpoint != current_checkpoint:
                        agent = FrozenDQN(world, Path(checkpoint))
                        current_checkpoint = checkpoint
                    try:
                        assert agent is not None
                        obs = np.asarray([state["state"]], dtype=np.float32)
                        q = agent.q_values(obs, int(state["phase"]))
                        pred = int(np.argmax(q))
                        rows.append({
                            "state_id": state["state_id"], "scene": scene, "method": method,
                            "order": state["order"], "training_seed": state["training_seed"],
                            "checkpoint": str(checkpoint), "checkpoint_digest": snapshot_identity(Path(checkpoint)).get("digest", ""),
                            "predicted_action": pred, "oracle_action": oracle,
                            "top1_correct": bool(oracle != "" and pred == int(oracle)),
                            "top2_correct": bool(oracle != "" and int(oracle) in np.argsort(-q)[:2]),
                            "action_regret": (float(label.get(f"cost_60_{pred}", float("nan"))) - float(label.get("best_cost_60", float("nan")))) if f"cost_60_{pred}" in label else float("nan"),
                            **{f"q_{i}": float(q[i]) for i in range(len(q))},
                        })
                    except Exception as exc:
                        rows.append({"state_id": state["state_id"], "scene": scene, "method": method, "order": state["order"], "training_seed": state["training_seed"], "predicted_action": "", "oracle_action": oracle, "top1_correct": False, "top2_correct": False, "action_regret": float("nan"), "prediction_error": f"{type(exc).__name__}: {exc}"})
            finally:
                world.close()
    return rows


def summarize(states: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    label_map = {str(row["state_id"]): row for row in labels}
    output: List[Dict[str, Any]] = []
    for method in METHODS:
        for scene in SCENES:
            rows = [row for row in pred_rows if row.get("method") == method and row.get("scene") == scene]
            valid = [row for row in rows if row.get("oracle_action", "") != "" and isinstance(row.get("predicted_action"), int)]
            correct = sum(bool(row.get("top1_correct")) for row in valid)
            top2 = sum(bool(row.get("top2_correct")) for row in valid)
            lo, hi = wilson(correct, len(valid))
            high_states = [label_map[str(row["state_id"])] for row in valid if label_map.get(str(row["state_id"]), {}).get("high_confidence")]
            output.append({
                "scope": "scenario", "scene": scene, "method": method, "states": len(valid), "top1_correct": correct,
                "top1_accuracy_pct": 100 * correct / len(valid) if valid else float("nan"), "top1_wilson_low_pct": 100 * lo if np.isfinite(lo) else float("nan"), "top1_wilson_high_pct": 100 * hi if np.isfinite(hi) else float("nan"),
                "top2_accuracy_pct": 100 * top2 / len(valid) if valid else float("nan"), "high_confidence_coverage_pct": 100 * len(high_states) / len(valid) if valid else float("nan"), "mean_action_regret": float(np.nanmean([row.get("action_regret", float("nan")) for row in valid])) if valid else float("nan"),
            })
        rows = [row for row in pred_rows if row.get("method") == method and row.get("oracle_action", "") != "" and isinstance(row.get("predicted_action"), int)]
        correct = sum(bool(row.get("top1_correct")) for row in rows)
        lo, hi = wilson(correct, len(rows))
        scene_acc = [row["top1_accuracy_pct"] for row in output if row["scope"] == "scenario" and row["method"] == method and np.isfinite(row["top1_accuracy_pct"])]
        output.append({
            "scope": "equal_weight_scenes", "scene": "S1-S4_equal_weight", "method": method, "states": len(rows), "top1_correct": correct,
            "top1_accuracy_pct": float(np.mean(scene_acc)) if scene_acc else float("nan"), "micro_top1_accuracy_pct": 100 * correct / len(rows) if rows else float("nan"), "top1_wilson_low_pct": 100 * lo if np.isfinite(lo) else float("nan"), "top1_wilson_high_pct": 100 * hi if np.isfinite(hi) else float("nan"), "top2_accuracy_pct": 100 * sum(bool(row.get("top2_correct")) for row in rows) / len(rows) if rows else float("nan"), "mean_action_regret": float(np.nanmean([row.get("action_regret", float("nan")) for row in rows])) if rows else float("nan"),
        })
    return output


def render_plots(output_dir: Path, summary_rows: Sequence[Mapping[str, Any]], states: Sequence[Mapping[str, Any]], labels: Sequence[Mapping[str, Any]], pred_rows: Sequence[Mapping[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scene_rows = [row for row in summary_rows if row.get("scope") == "scenario"]
    plot_groups = list(SCENES) + ["S1-S4_equal_weight"]
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(plot_groups))
    width = 0.24
    for i, method in enumerate(METHODS):
        values = []
        for group in plot_groups:
            if group == "S1-S4_equal_weight":
                value = next((float(row["top1_accuracy_pct"]) for row in summary_rows if row.get("scope") == "equal_weight_scenes" and row.get("method") == method), np.nan)
            else:
                value = next((float(row["top1_accuracy_pct"]) for row in scene_rows if row["scene"] == group and row["method"] == method), np.nan)
            values.append(value)
        bars = ax.bar(x + (i - 1) * width, values, width, label=method)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=7)
    ax.axhline(90, color="crimson", linestyle="--", linewidth=1.4, label="90% threshold")
    group_labels = list(SCENES) + ["S1-S4\nequal-weight"]
    ax.set_xticks(x, group_labels, rotation=20); ax.set_ylim(0, 105); ax.set_ylabel("Strict Top-1 agreement (%)"); ax.set_title("Oracle-action allocation accuracy by scenario and equal-weight estimate"); ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(output_dir / "allocation_accuracy_by_scenario.png", dpi=180); plt.close(fig)

    # Action confusion matrix for the equal-weight scope, one panel per method.
    fig, axes = plt.subplots(1, len(METHODS), figsize=(13, 4), squeeze=False)
    for idx, method in enumerate(METHODS):
        matrix = np.zeros((ACTION_COUNT, ACTION_COUNT), dtype=int)
        for row in pred_rows:
            if row.get("method") != method or row.get("oracle_action", "") == "" or not isinstance(row.get("predicted_action"), int):
                continue
            matrix[int(row["oracle_action"]), int(row["predicted_action"])] += 1
        ax = axes[0, idx]; im = ax.imshow(matrix, cmap="Blues", vmin=0); ax.set_title(method); ax.set_xlabel("predicted"); ax.set_ylabel("oracle"); ax.set_xticks(range(8)); ax.set_yticks(range(8))
        for i in range(8):
            for j in range(8):
                if matrix[i, j]: ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.7); fig.tight_layout(); fig.savefig(output_dir / "allocation_accuracy_confusion_matrix.png", dpi=180); plt.close(fig)

    # State coverage and oracle margin diagnostics.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    counts = Counter((str(row.get("scene")), str(row.get("load_bin"))) for row in states)
    x = np.arange(len(SCENES))
    width = 0.24
    for index, load_bin in enumerate(("low", "mid", "high")):
        axes[0].bar(x + (index - 1) * width, [counts[(scene, load_bin)] for scene in SCENES], width, label=load_bin)
    axes[0].set_xticks(x, [f"S{i + 1}" for i in range(len(SCENES))]); axes[0].set_title("Selected-state load coverage"); axes[0].set_ylabel("states"); axes[0].legend(title="load bin")
    margins = [float(row.get("oracle_margin_60", np.nan)) for row in labels if np.isfinite(float(row.get("oracle_margin_60", np.nan)))]
    axes[1].hist(margins, bins=12, color="#4c78a8"); axes[1].axvline(HIGH_CONFIDENCE_MARGIN, color="crimson", linestyle="--"); axes[1].set_title("Oracle margin at 60 s"); axes[1].set_xlabel("second-best cost − best cost")
    fig.tight_layout(); fig.savefig(output_dir / "allocation_accuracy_state_coverage.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4)); ax.hist(margins, bins=15, color="#59a14f"); ax.axvline(HIGH_CONFIDENCE_MARGIN, color="crimson", linestyle="--", label="high-confidence cutoff"); ax.set_xlabel("oracle margin"); ax.set_ylabel("states"); ax.legend(); fig.tight_layout(); fig.savefig(output_dir / "allocation_accuracy_oracle_margin.png", dpi=180); plt.close(fig)


def build_report(output_dir: Path, audit: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]]) -> None:
    macro = [row for row in summary_rows if row.get("scope") == "equal_weight_scenes"]
    lines = [
        "# 资源配置准确率：全动作 SUMO 反事实 Top-1 审计",
        "",
        "本报告基于当前磁盘中的冻结 checkpoint、SUMO 场景和确定性重放；没有重新训练模型。主 oracle 标签使用 60 秒 rollout，30/90 秒仅作为标签稳定性诊断。",
        "",
        "## 口径",
        "",
        "每个状态枚举 8 个合法绿灯动作。候选动作控制第一个 10 秒决策周期，之后由同一场景的冻结独立 Online DQN 接管。对 queue AUC、SUMO/项目轨迹恢复后的 real delay 和 throughput 分别在该状态的 8 个动作间做 min-max 归一化，cost=(q_norm+d_norm+1-throughput_norm)/3，最低 cost 且动作编号最小者为 oracle_action。标签规则在读取 Agent 预测前固定。",
        "",
        "## 可行性与状态审计",
        "",
        f"- SUMO `saveState/loadState`：{audit.get('feasibility', {}).get('saveState')} / {audit.get('feasibility', {}).get('loadState')}；Python车辆轨迹、等待时间和phase缓存共同恢复：{audit.get('feasibility', {}).get('python_trajectory_restore')}。",
        f"- 每个状态动作数：{audit.get('feasibility', {}).get('candidate_action_count')}；动作周期：{audit.get('feasibility', {}).get('action_interval_seconds')}秒；主rollout：{audit.get('feasibility', {}).get('requested_rollout_seconds')}秒。",
        "- 指标口径：queue_auc为每秒入口lane waiting-vehicle数求和（vehicle·s）；real_delay为SUMO轨迹恢复后在对应时刻的平均延迟（s）；throughput为快照后累计到达车辆数（vehicle）；phase_switch_count为请求动作变化次数。",
        f"- 状态重放与冻结日志lane count校验：匹配 {audit.get('replay_match_count')}，不匹配 {audit.get('replay_mismatch_count')}，未检查 {audit.get('replay_unchecked_count')}。",
        "- action→绿灯相位→controlled links/incoming lanes映射来自SUMO adapter运行时表，未按动作整数编号主观推断；完整映射见 `allocation_accuracy_phase_mapping.json`。",
        "- 状态覆盖图同时呈现phase、load/time bins；若某场景缺少某一负荷档，报告仅说明数据覆盖缺口，不用插值补齐。",
        "",
        "## 审计结果",
        "",
        f"- 状态数量：{audit.get('selected_state_count')}（目标 {audit.get('target_state_count')}）；每场景：{audit.get('states_per_scene')}。",
        f"- 有效反事实分支：{audit.get('valid_rollout_count')} / {audit.get('rollout_count')}。",
        f"- 完整有效 60 秒 oracle 标签：{audit.get('valid_label_count', audit.get('label_count', 0))} / {audit.get('label_count')}；高置信标签：{audit.get('high_confidence_label_count', audit.get('reliable_label_count', 0))}。高置信规则：30/60/90 标签一致且 60 秒 margin ≥ {HIGH_CONFIDENCE_MARGIN:.2f}。",
        f"- 90% 判定要求：等权场景 Top-1 ≥ 90%；若 256 状态全部有效，则至少 231 个状态答对。",
        "",
        "| scope | method | Top-1 (%) | micro Top-1 (%) | Wilson 95% CI | Top-2 (%) | mean regret |",
        "|---|---|---:|---:|---|---:|---:|",
        "",
        "Wilson区间按有效状态的合并命中数计算；点估计的主口径仍是四场景等权平均。",
    ]
    for row in macro:
        ci = f"[{row.get('top1_wilson_low_pct', float('nan')):.2f}, {row.get('top1_wilson_high_pct', float('nan')):.2f}]"
        lines.append(f"| {row.get('scene')} | {row.get('method')} | {row.get('top1_accuracy_pct', float('nan')):.2f} | {row.get('micro_top1_accuracy_pct', float('nan')):.2f} | {ci} | {row.get('top2_accuracy_pct', float('nan')):.2f} | {row.get('mean_action_regret', float('nan')):.4f} |")
    lines += [
        "",
        "## 场景级结果",
        "",
        "| scene | method | valid states | Top-1 (%) | Wilson 95% CI | Top-2 (%) | high-confidence coverage (%) |",
        "|---|---|---:|---:|---|---:|---:|",
    ]
    for row in summary_rows:
        if row.get("scope") != "scenario":
            continue
        ci = f"[{row.get('top1_wilson_low_pct', float('nan')):.2f}, {row.get('top1_wilson_high_pct', float('nan')):.2f}]"
        lines.append(f"| {row.get('scene')} | {row.get('method')} | {row.get('states')} | {row.get('top1_accuracy_pct', float('nan')):.2f} | {ci} | {row.get('top2_accuracy_pct', float('nan')):.2f} | {row.get('high_confidence_coverage_pct', float('nan')):.2f} |")
    lines += [
        "",
        "## 状态覆盖细节",
        "",
        "| scene | load bins | missing load bins | direction-imbalance range |",
        "|---|---|---|---|",
    ]
    for scene, coverage in audit.get("state_coverage", {}).items():
        missing = ", ".join(coverage.get("missing_load_bins", [])) or "无"
        lo = float(coverage.get("direction_imbalance_min", float("nan")))
        hi = float(coverage.get("direction_imbalance_max", float("nan")))
        lines.append(f"| {scene} | {coverage.get('load_bins', {})} | {missing} | [{lo:.3f}, {hi:.3f}] |")
    lines += [
        "",
        "## 模型覆盖与判定",
        "",
    ]
    for method, coverage in audit.get("prediction_coverage", {}).items():
        lines.append(f"- {method}：有匹配冻结checkpoint且完成推理的状态 {coverage.get('states_with_prediction', 0)}，缺失/异常 {coverage.get('states_missing_or_error', 0)}。")
    lines += [
        f"- 完整有效标签要求8个动作全部有效且指标有限；主Top-1使用这些有效标签。高置信标签另外要求30/60/90秒oracle一致并满足60秒margin阈值。",
        "- 主判定只使用等权场景Top-1；缺少匹配checkpoint的状态不填充其它Order或seed。",
        "",
        "",
        "## 限制",
        "",
        "现有正式轨迹保存的是 8 维 lane_count、phase 和 action，并非 SUMO 微观快照；本次新增的 saveState 与 Python 侧车辆轨迹/等待时间状态共同恢复。结论只代表固定交通实现、当前场景覆盖和当前冻结 checkpoint，不外推为任意交通场景上的全局最优动作准确率。R25/R50 只在各自实际存在的 Order/seed 匹配范围内解释，缺失匹配不会用其它 seed 替代。",
        "现有评价请求的 evaluation_seed 为 null，当前结果对应固定交通实现；Wilson区间仅反映当前状态样本的二项波动，不宣称覆盖交通随机性。",
        "",
        "图表：`allocation_accuracy_by_scenario.png`、`allocation_accuracy_confusion_matrix.png`、`allocation_accuracy_state_coverage.png`、`allocation_accuracy_oracle_margin.png`。相位映射详见 `allocation_accuracy_phase_mapping.json`。",
    ]
    (output_dir / "allocation_accuracy_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_sources(formal: bool) -> List[Dict[str, Any]]:
    # Smoke deliberately uses O2/seed0 because all three requested frozen
    # controller families have a matched checkpoint there.  Formal mode uses
    # every available CONT run; R50 remains limited to its actual O2 assets.
    sources = []
    orders = ("O2",) if not formal else tuple(ORDERS)
    seeds = (0,) if not formal else tuple(range(5))
    for order in orders:
        for seed in seeds:
            checkpoint = final_checkpoint(run_id("CONT-DQN", order, seed))
            if checkpoint:
                sources.append({"run_id": run_id("CONT-DQN", order, seed), "order": order, "training_seed": seed, "checkpoint": str(checkpoint), "checkpoint_digest": snapshot_identity(checkpoint).get("digest"), "evaluation_seed": None})
    return sources


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", action="store_true", help="select 64 states per scene instead of the 32-state smoke")
    parser.add_argument("--states-per-scene", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DEFAULT)
    parser.add_argument("--capture-stride", type=int, default=5)
    parser.add_argument("--max-replay-decisions", type=int, default=360, help="maximum decision cycles per deterministic replay")
    parser.add_argument("--rollout-seconds", type=int, default=90, help="counterfactual branch horizon")
    parser.add_argument("--state-manifest", type=Path, default=None, help="reuse a previously generated state manifest and skip replay/selection")
    parser.add_argument("--reuse-rollouts-dir", type=Path, default=None, help="reuse existing all_action_rollouts/oracle_labels from this output directory; no SUMO branches are rerun")
    parser.add_argument("--skip-rollouts", action="store_true", help="only perform replay/state feasibility audit")
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    # World construction reads the registered command setting even when a
    # previously generated manifest is reused and no replay_source() call has
    # initialized it for us.
    registry_interface.Command_Setting_Interface({"command": {"sumo_seed": None}})
    per_scene = int(args.states_per_scene or (64 if args.formal else 8))
    sources = build_sources(args.formal)
    audit: Dict[str, Any] = {
        "schema_version": 1, "generated_at_unix": time.time(), "tool": str(Path(__file__).resolve()),
        "mode": "formal" if args.formal else "smoke", "target_state_count": per_scene * len(SCENES), "states_per_scene": per_scene,
        "scenes": list(SCENES), "methods": list(METHODS), "sumo_binary": os.environ.get("SUMO_BINARY"), "sumo_home": os.environ.get("SUMO_HOME"),
        "feasibility": {"saveState": True, "loadState": True, "deterministic_replay": True, "candidate_action_count": ACTION_COUNT, "action_interval_seconds": ACTION_INTERVAL, "rollout_horizons_seconds": list(LABEL_HORIZONS), "requested_rollout_seconds": int(args.rollout_seconds), "python_trajectory_restore": True, "queue_metric": "sum incoming lane waiting_vehicle_count per second", "real_delay_metric": "world.get_real_delay with restored vehicle_trajectory", "throughput_metric": "departures after snapshot (world.vehicles delta)", "phase_mapping_source": "world_sumo.Intersection.green_phases and trafficlight controlled links", "state_order_source": "LaneVehicleGenerator(lane_count,in_only=True)"},
        "sources": sources,
        "missing_assets": [],
    }
    for scene in SCENES:
        cfg = simulator_config(scene)
        if not cfg.is_file(): audit["missing_assets"].append(str(cfg))
    for source in sources:
        for method in METHODS:
            if not final_checkpoint(run_id(method, source["order"], source["training_seed"])):
                audit["missing_assets"].append(run_id(method, source["order"], source["training_seed"]))
        for scene in SCENES:
            if not independent_checkpoint(scene, source["training_seed"]):
                audit["missing_assets"].append(f"independent:{scene}:seed{source['training_seed']}")
    all_candidates: List[Dict[str, Any]] = []
    if args.state_manifest is not None:
        # Reusing a manifest is explicit so the expensive replay stage is
        # resumable without changing the tested state coverage.
        manifest_path = args.state_manifest.resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        state_rows: List[Dict[str, Any]] = []
        with manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for record in csv.DictReader(handle):
                row = dict(record)
                row["state"] = json.loads(row.get("state_vector", "[]"))
                row["lane_ids"] = json.loads(row.get("lane_ids", "[]"))
                row["lane_counts"] = json.loads(row.get("lane_counts", "{}"))
                for key in ("decision_index", "phase", "training_seed"):
                    if key in row and row[key] != "":
                        row[key] = int(float(row[key]))
                for key in ("simulation_time_seconds", "total_vehicle_count", "direction_imbalance"):
                    if key in row and row[key] != "":
                        row[key] = float(row[key])
                if row.get("replay_match") in {"True", "true", "1"}:
                    row["replay_match"] = True
                elif row.get("replay_match") in {"False", "false", "0"}:
                    row["replay_match"] = False
                elif row.get("replay_match", "") == "":
                    row["replay_match"] = None
                state_rows.append(row)
        all_candidates = list(state_rows)
        print(f"[manifest] reused {manifest_path} states={len(state_rows)}", flush=True)
        manifest_audit_path = manifest_path.parent / "allocation_accuracy_audit.json"
        manifest_audit = json.loads(manifest_audit_path.read_text(encoding="utf-8")) if manifest_audit_path.is_file() else {}
        audit["mode"] = "manifest_reuse"
        audit["manifest_path"] = str(manifest_path)
        audit["target_state_count"] = len(state_rows)
        audit["states_per_scene"] = len(state_rows) // len(SCENES) if len(state_rows) % len(SCENES) == 0 else None
        if manifest_audit:
            for key in ("candidate_state_count", "replay_match_count", "replay_mismatch_count", "replay_unchecked_count", "sources"):
                if key in manifest_audit:
                    audit[key] = manifest_audit[key]
    else:
        for scene in SCENES:
            # In smoke, the single O2 source is enough to prove the complete chain.
            for source in sources:
                all_candidates.extend(replay_source(scene, source, output_dir, args.capture_stride, max_decisions=args.max_replay_decisions))
        by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for row in all_candidates:
            by_scene[str(row["scene"])].append(row)
        selected: List[Dict[str, Any]] = []
        for scene in SCENES:
            selected.extend(select_states(by_scene.get(scene, []), per_scene))
        state_rows = []
        for index, row in enumerate(selected, start=1):
            row = dict(row)
            row["state_id"] = f"S{index:04d}"
            state_rows.append(row)
        assign_bins(state_rows)
    # Persist the mapping actually exposed by the SUMO adapter.  This is an
    # audit artifact, not a hand-written action-number convention.
    phase_mapping: Dict[str, Any] = {}
    for scene in SCENES:
        map_root = output_dir / "phase_mapping_logs" / scene
        map_root.mkdir(parents=True, exist_ok=True)
        Registry.mapping["logger_mapping"]["path"].path = str(map_root)
        world_for_mapping = world_sumo.World(str(simulator_config(scene)), 1, interface="libsumo")
        try:
            phase_mapping[scene] = extract_phase_mapping(world_for_mapping)
        finally:
            world_for_mapping.close()
    audit["feasibility"]["phase_mapping"] = phase_mapping
    json_dump(output_dir / "allocation_accuracy_phase_mapping.json", phase_mapping)
    manifest_rows = []
    for row in state_rows:
        flat = {k: v for k, v in row.items() if k not in {"state", "lane_ids", "lane_counts"}}
        flat["state_vector"] = json.dumps(row["state"]); flat["lane_ids"] = json.dumps(row["lane_ids"]); flat["lane_counts"] = json.dumps(row["lane_counts"], sort_keys=True)
        manifest_rows.append(flat)
    write_csv(output_dir / "allocation_accuracy_state_manifest.csv", manifest_rows)
    if args.state_manifest is None or not audit.get("candidate_state_count"):
        audit["candidate_state_count"] = len(all_candidates)
    audit["selected_state_count"] = len(state_rows)
    if args.state_manifest is None or not audit.get("replay_match_count"):
        audit["replay_match_count"] = sum(row.get("replay_match") is True for row in state_rows)
        audit["replay_mismatch_count"] = sum(row.get("replay_match") is False for row in state_rows)
        audit["replay_unchecked_count"] = sum(row.get("replay_match") is None for row in state_rows)
    audit["state_coverage"] = {}
    for scene in SCENES:
        scene_states = [row for row in state_rows if str(row.get("scene")) == scene]
        load_counts = Counter(str(row.get("load_bin", "")) for row in scene_states)
        phase_counts = Counter(str(row.get("phase", "")) for row in scene_states)
        time_counts = Counter(str(row.get("time_bin", "")) for row in scene_states)
        imbalances = [float(row.get("direction_imbalance", float("nan"))) for row in scene_states if np.isfinite(float(row.get("direction_imbalance", float("nan"))))]
        audit["state_coverage"][scene] = {
            "state_count": len(scene_states),
            "load_bins": {key: int(value) for key, value in sorted(load_counts.items())},
            "missing_load_bins": [key for key in ("low", "mid", "high") if not load_counts.get(key)],
            "phase_counts": {key: int(value) for key, value in sorted(phase_counts.items())},
            "time_bins": {key: int(value) for key, value in sorted(time_counts.items())},
            "direction_imbalance_min": min(imbalances) if imbalances else float("nan"),
            "direction_imbalance_max": max(imbalances) if imbalances else float("nan"),
            "direction_imbalance_mean": float(np.mean(imbalances)) if imbalances else float("nan"),
        }
    if args.skip_rollouts:
        audit["rollout_count"] = 0; audit["valid_rollout_count"] = 0; audit["reliable_label_count"] = 0; audit["status"] = "state_replay_audit_only"; json_dump(output_dir / "allocation_accuracy_audit.json", audit); return 0

    rollout_rows: List[Dict[str, Any]] = []
    labels: List[Dict[str, Any]] = []
    if args.reuse_rollouts_dir is not None:
        reuse_dir = args.reuse_rollouts_dir.resolve()
        roll_path = reuse_dir / "allocation_accuracy_all_action_rollouts.csv"
        label_path = reuse_dir / "allocation_accuracy_oracle_labels.csv"
        if not roll_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"reuse directory must contain {roll_path.name} and {label_path.name}: {reuse_dir}")
        with roll_path.open("r", encoding="utf-8", newline="") as handle:
            rollout_rows = list(csv.DictReader(handle))
        with label_path.open("r", encoding="utf-8", newline="") as handle:
            labels = list(csv.DictReader(handle))
        for row in rollout_rows:
            for key in ("candidate_action", "training_seed", "phase_switch_count"):
                if row.get(key, "") != "": row[key] = int(float(row[key]))
            for key in ("cost_30", "cost_60", "cost_90", "queue_auc_30", "queue_auc_60", "queue_auc_90", "real_delay_30", "real_delay_60", "real_delay_90", "throughput_30", "throughput_60", "throughput_90"):
                if row.get(key, "") != "": row[key] = float(row[key])
            row["safety_valid"] = str(row.get("safety_valid", "")).lower() == "true"
        for row in labels:
            for key in ("valid_action_count",):
                if row.get(key, "") != "": row[key] = int(float(row[key]))
            for key in ("best_cost_30", "best_cost_60", "best_cost_90", "second_best_cost_30", "second_best_cost_60", "second_best_cost_90", "oracle_margin_30", "oracle_margin_60", "oracle_margin_90"):
                if row.get(key, "") != "": row[key] = float(row[key])
            for key in ("oracle_action_30", "oracle_action_60", "oracle_action_90"):
                if row.get(key, "") != "": row[key] = int(float(row[key]))
            row["labels_consistent_30_60_90"] = str(row.get("labels_consistent_30_60_90", "")).lower() == "true"
            row["high_confidence"] = str(row.get("high_confidence", "")).lower() == "true"
        audit["rollout_reused"] = True
    else:
        for scene in SCENES:
            scene_states = [row for row in state_rows if row["scene"] == scene]
            Registry.mapping["logger_mapping"]["path"].path = str(output_dir / "rollout_logs" / scene)
            (output_dir / "rollout_logs" / scene).mkdir(parents=True, exist_ok=True)
            world = world_sumo.World(str(simulator_config(scene)), 1, interface="libsumo")
            try:
                world.reset()
                base_agent = FrozenDQN(world, Path(scene_states[0]["source_checkpoint"])) if scene_states else None
                for state in scene_states:
                    independent = independent_checkpoint(scene, int(state["training_seed"]))
                    if not independent or base_agent is None:
                        continue
                    independent_agent = FrozenDQN(world, independent)
                    for action in range(ACTION_COUNT):
                        metrics = collect_rollout_metrics(world, base_agent, independent_agent, state, action, rollout_seconds=args.rollout_seconds)
                        rollout_rows.append({"state_id": state["state_id"], "scene": scene, "source_run_id": state["source_run_id"], "order": state["order"], "training_seed": state["training_seed"], **metrics})
                    print(f"[rollout] scene={scene} state={state['state_id']} actions={ACTION_COUNT}", flush=True)
            finally:
                world.close()
        labels = label_rollouts(rollout_rows)
    label_map = {str(row["state_id"]): row for row in labels}
    for row in rollout_rows:
        label = label_map.get(str(row["state_id"]), {})
        row["oracle_action_60"] = label.get("oracle_action_60", "")
    write_csv(output_dir / "allocation_accuracy_all_action_rollouts.csv", rollout_rows)
    write_csv(output_dir / "allocation_accuracy_oracle_labels.csv", labels)

    model_rows: Dict[str, Dict[Tuple[str, str, int], str]] = {method: {} for method in METHODS}
    for state in state_rows:
        for method in METHODS:
            path = final_checkpoint(run_id(method, state["order"], int(state["training_seed"])))
            if path:
                model_rows[method][(str(state["scene"]), str(state["order"]), int(state["training_seed"]))] = str(path)
            else:
                audit.setdefault("missing_assets", []).append(f"checkpoint:{method}:{state['scene']}:{state['order']}:seed{state['training_seed']}")
    pred_rows = predictions(state_rows, label_map, model_rows, output_dir)
    # Add action-specific costs to each prediction row for regret.
    cost_lookup = {(str(row["state_id"]), int(row["candidate_action"])): row.get("cost_60", float("nan")) for row in rollout_rows}
    for row in pred_rows:
        if isinstance(row.get("predicted_action"), int):
            row["action_regret"] = float(cost_lookup.get((str(row["state_id"]), int(row["predicted_action"])), float("nan"))) - float(label_map.get(str(row["state_id"]), {}).get("best_cost_60", float("nan")))
    write_csv(output_dir / "allocation_accuracy_agent_predictions.csv", pred_rows)
    summary_rows = summarize(state_rows, labels, pred_rows)
    write_csv(output_dir / "allocation_accuracy_summary.csv", summary_rows)
    audit["missing_assets"] = sorted(set(str(item) for item in audit.get("missing_assets", [])))
    audit["prediction_coverage"] = {
        method: {
            "states_with_prediction": sum(1 for row in pred_rows if row.get("method") == method and isinstance(row.get("predicted_action"), int)),
            "states_missing_or_error": sum(1 for row in pred_rows if row.get("method") == method and not isinstance(row.get("predicted_action"), int)),
        }
        for method in METHODS
    }
    audit["rollout_count"] = len(rollout_rows)
    audit["valid_rollout_count"] = sum(bool(row.get("safety_valid")) for row in rollout_rows)
    audit["label_count"] = len(labels)
    audit["valid_label_count"] = sum(bool(row.get("oracle_action_60", "") != "") for row in labels)
    audit["high_confidence_label_count"] = sum(bool(row.get("high_confidence")) for row in labels)
    # Keep the historical key as an alias for consumers of earlier smoke
    # audits, while making the two denominator concepts explicit above.
    audit["reliable_label_count"] = audit["valid_label_count"]
    audit["status"] = "completed"
    audit["threshold_rule"] = "equal_weight_scenes Top-1 >= 90%"
    audit["threshold_results"] = {row["method"]: {"top1_accuracy_pct": row.get("top1_accuracy_pct"), "meets_90pct": bool(np.isfinite(float(row.get("top1_accuracy_pct", np.nan))) and float(row.get("top1_accuracy_pct")) >= 90.0)} for row in summary_rows if row.get("scope") == "equal_weight_scenes"}
    json_dump(output_dir / "allocation_accuracy_audit.json", audit)
    render_plots(output_dir, summary_rows, state_rows, labels, pred_rows)
    build_report(output_dir, audit, summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
