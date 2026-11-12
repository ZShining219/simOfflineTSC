"""Frozen Plan5 configuration and identity validation."""
from dataclasses import dataclass
import copy
import hashlib
import json
from pathlib import Path

import yaml

SCENES = {
    "S1": "sumohz1x1_config2",
    "S2": "sumohz1x1",
    "S3": "sumohz1x1_config4",
    "S4": "sumohz1x1_config3",
}
TRANSITIONS = {
    "H34": ("S3", "S4"), "H43": ("S4", "S3"),
    "L23": ("S2", "S3"), "L32": ("S3", "S2"),
}
ALGORITHMS = ("DDQN", "CTXDDQN", "PPO")
SEEDS = (0, 1, 2, 3, 4)
HISTORICAL_SCHEDULE = (0, 1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 100)


@dataclass(frozen=True)
class Plan5Config:
    plan_id: str
    output_root: str
    simulation_duration: int
    sumo_step: int
    decision_interval: int
    training_episodes: int
    scenes: dict
    transitions: dict
    algorithms: tuple
    training_seeds: tuple
    observation_dim: int
    context_dim: int
    action_dim: int
    reward_scale: float
    bootstrap_truncated: bool
    analysis_seed: int
    bootstrap_resamples: int
    resolved: dict

    @property
    def decisions_per_episode(self):
        return self.simulation_duration // self.decision_interval

    def as_dict(self):
        return copy.deepcopy(self.resolved)

    def sha256(self):
        encoded = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


def _required(data, key):
    if key not in data:
        raise ValueError(f"Plan5 config missing {key}")
    return data[key]


def validate_config(data):
    if data.get("schema_version") != 1 or data.get("plan_id") != "Plan5":
        raise ValueError("Plan5 config requires schema_version=1 and plan_id=Plan5")
    scenes = _required(data, "scenes")
    if scenes != SCENES:
        raise ValueError(f"Scene mapping is not frozen: {scenes!r}")
    transitions = {k: tuple(v) for k, v in _required(data, "transitions").items()}
    if transitions != TRANSITIONS:
        raise ValueError("Transition mapping is not frozen")
    if tuple(_required(data, "algorithms")) != ALGORITHMS:
        raise ValueError("Only DDQN, CTXDDQN and PPO are authorized in Phase1")
    if tuple(_required(data, "training_seeds")) != SEEDS:
        raise ValueError("Training seeds must be 0..4")
    protocol = _required(data, "protocol")
    expected = {
        "simulation_duration": 3600, "sumo_step": 1, "decision_interval": 10,
        "training_episodes": 100, "observation_dim": 16, "context_dim": 4,
        "action_dim": 8, "reward_scale": 12.0,
    }
    for key, value in expected.items():
        if protocol.get(key) != value:
            raise ValueError(f"Frozen protocol mismatch for {key}: {protocol.get(key)!r}")
    if protocol.get("device") != "cpu" \
            or protocol.get("sumo_seed") is not None \
            or protocol.get("sumo_random") is not False \
            or tuple(protocol.get("historical_evaluation_episodes", ())) != HISTORICAL_SCHEDULE \
            or protocol.get("bootstrap_truncated") is not True:
        raise ValueError("Plan5 runtime/evaluation protocol changed")
    analysis = _required(data, "analysis")
    if analysis.get("seed") != 20260827 or analysis.get("bootstrap_resamples") != 10000:
        raise ValueError("Analysis seed/resample count is not frozen")
    environment = _required(data, "environment")
    if environment != {
        "conda_environment": "colight", "python": "3.10.18",
        "torch": "1.13.1+cu116", "gym": "0.26.2",
        "numpy": "1.26.4", "pfrl": "0.4.0", "sumo": "1.27.1",
    }:
        raise ValueError("Plan5 formal environment lock changed")
    ddqn = _required(data, "ddqn")
    ddqn_expected = {
        "network": [20, 20], "optimizer": "RMSprop",
        "learning_rate": 0.001, "gamma": 0.95, "batch_size": 64,
        "replay_capacity": 5000, "learning_start": 1000,
        "gradient_clip": 5.0, "target_sync_successful_updates": 10,
        "epsilon_initial": 1.0, "epsilon_decay": 0.995,
        "epsilon_min": 0.01,
    }
    if ddqn != ddqn_expected:
        raise ValueError("Plan5 DDQN parameters changed")
    ppo = _required(data, "ppo")
    ppo_expected = {
        "shared_trunk": [64, 64], "optimizer": "Adam",
        "optimizer_eps": 1e-5, "gamma": 0.95, "gae_lambda": 0.95,
        "clip_ratio": 0.2, "value_coefficient": 1.0,
        "value_clip": 0.2, "standardize_advantages": True,
        "max_grad_norm": 0.5, "observation_normalizer": None,
        "rollout_length": 360, "minibatch_size": 90, "epochs": 4,
        "calibration": {
            "scene": "S2", "seeds": [100, 101, 102],
            "learning_rates": [1e-4, 2.5e-4, 5e-4],
            "entropy_coefficients": [0.001, 0.01],
        },
    }
    if ppo != ppo_expected:
        raise ValueError("Plan5 PPO parameters changed")
    probe = _required(data, "probe")
    if probe != {
        "probe_id": "plan5_fixed_probe_v1",
        "decisions_per_trajectory": 360,
        "main_source_seeds": [0, 1, 2, 3],
        "heldout_source_seeds": [4],
    }:
        raise ValueError("Plan5 fixed probe contract changed")
    return True


def load_config(path="configs/sequential/plan5_cross_algorithm_b100.yml"):
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    validate_config(data)
    protocol, analysis = data["protocol"], data["analysis"]
    return Plan5Config(
        plan_id=data["plan_id"], output_root=data["output_root"],
        simulation_duration=protocol["simulation_duration"], sumo_step=protocol["sumo_step"],
        decision_interval=protocol["decision_interval"], training_episodes=protocol["training_episodes"],
        scenes=copy.deepcopy(data["scenes"]), transitions={k: tuple(v) for k, v in data["transitions"].items()},
        algorithms=tuple(data["algorithms"]), training_seeds=tuple(data["training_seeds"]),
        observation_dim=protocol["observation_dim"], context_dim=protocol["context_dim"],
        action_dim=protocol["action_dim"], reward_scale=protocol["reward_scale"],
        bootstrap_truncated=protocol.get("bootstrap_truncated", True),
        analysis_seed=analysis["seed"], bootstrap_resamples=analysis["bootstrap_resamples"],
        resolved=copy.deepcopy(data),
    )
