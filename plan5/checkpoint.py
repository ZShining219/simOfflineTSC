"""Full resumable checkpoints and evaluation-isolation helpers."""
import copy
import os
import random
import tempfile

import numpy as np
import torch

from sequential.core import canonical_digest

SCHEMA_VERSION = 1


def rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state.get("torch_cuda"):
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def training_state_digest(agent, extra=None):
    payload = {"agent": agent.full_state_dict(), "rng": rng_state()}
    if extra is not None: payload["extra"] = extra
    return canonical_digest(payload)


def build_checkpoint(agent, identity, *, episode, stage, extra_state=None):
    state = agent.full_state_dict()
    payload = {
        "schema_version": SCHEMA_VERSION, "checkpoint_type": "full_resumable",
        "identity": dict(identity), "episode": int(episode), "stage": int(stage),
        "agent_state": state, "rng_state": rng_state(),
        "extra_state": {} if extra_state is None else copy.deepcopy(extra_state),
    }
    payload["canonical_state_digest"] = canonical_digest({"agent": state, "rng": payload["rng_state"], "extra": payload["extra_state"]})
    return payload


def atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=".plan5-checkpoint-", suffix=".pt", dir=directory,
    )
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def save_checkpoint(agent, path, identity, *, episode, stage, extra_state=None):
    payload = build_checkpoint(agent, identity, episode=episode, stage=stage, extra_state=extra_state)
    atomic_torch_save(payload, path)
    return payload


def load_checkpoint(path, expected_identity=None):
    payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("checkpoint_type") != "full_resumable":
        raise ValueError("Invalid Plan5 full checkpoint schema")
    if expected_identity:
        for key, value in expected_identity.items():
            if payload.get("identity", {}).get(key) != value:
                raise ValueError(f"Checkpoint identity mismatch for {key}")
    expected = canonical_digest({"agent": payload["agent_state"], "rng": payload["rng_state"], "extra": payload["extra_state"]})
    if payload.get("canonical_state_digest") != expected: raise ValueError("Checkpoint canonical digest mismatch")
    _validate_checkpoint_state(payload)
    return payload


def _validate_checkpoint_state(payload):
    state = payload.get('agent_state', {})
    identity = payload.get('identity', {})
    algorithm_id = identity.get('algorithm_id', state.get('algorithm_id'))
    common_rng = {'python', 'numpy', 'torch_cpu', 'torch_cuda'}
    if common_rng - set(payload.get('rng_state', {})):
        raise ValueError('Plan5 checkpoint RNG state is incomplete')
    if algorithm_id in {'DDQN', 'CTXDDQN'}:
        required = {
            'online_model_state_dict', 'target_model_state_dict',
            'optimizer_state_dict', 'epsilon', 'replay_state', 'counters',
            'target_scheduler', 'rng_state',
        }
        if required - set(state):
            raise ValueError('Plan5 DQN checkpoint state is incomplete')
        if algorithm_id == 'CTXDDQN' and (
                'context_state' not in state or 'last_context' not in state):
            raise ValueError('Plan5 CTXDDQN checkpoint context is incomplete')
    elif algorithm_id == 'PPO':
        required = {
            'model_state_dict', 'optimizer_state_dict', 'memory',
            'last_episode', 'last_state', 'last_action',
            'batch_last_episode', 'batch_last_state', 'batch_last_action',
            'n_updates', 'explained_variance', 'statistics',
            'recurrent_state', 'counters', 'stage', 'rng_state',
        }
        if required - set(state):
            raise ValueError('Plan5 PPO checkpoint state is incomplete')
        pending = sum(len(episode) for episode in state['memory']) \
            + len(state['last_episode']) \
            + (0 if state['batch_last_episode'] is None else sum(
                len(episode) for episode in state['batch_last_episode']
            ))
        batch_pending = any(
            value is not None
            for field in ('batch_last_state', 'batch_last_action')
            for value in ([] if state[field] is None else state[field])
        )
        if pending or state['last_state'] is not None \
                or state['last_action'] is not None or batch_pending:
            raise ValueError('Plan5 PPO full checkpoint rollout is not empty')
    elif algorithm_id is not None:
        raise ValueError('Plan5 checkpoint algorithm identity is invalid')


def restore_checkpoint(agent, path, expected_identity=None):
    payload = load_checkpoint(path, expected_identity)
    agent.load_full_state_dict(payload["agent_state"])
    restore_rng_state(payload["rng_state"])
    return payload


def evaluate_isolated(agent, evaluator, snapshot=None):
    """Evaluate a copied agent and prove parent training state did not mutate."""
    before = training_state_digest(agent)
    candidate = copy.deepcopy(agent)
    result = evaluator(candidate, snapshot)
    after = training_state_digest(agent)
    if before != after: raise AssertionError("Frozen evaluation mutated training state")
    return result
