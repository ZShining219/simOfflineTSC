"""Separate-process deterministic evaluator with full state isolation proof."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import gym
import numpy as np
import torch

from agent import utils as agent_utils
from agent.dqn import DQNNet
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator
from sequential.core import canonical_digest
from sequential.io import atomic_json, read_json, sha256_file

from .checkpoint import atomic_torch_save, training_state_digest
from .sumo_runtime import evaluate_policy


def save_evaluation_snapshot(agent, path, identity):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Plan5 evaluation snapshot is immutable: {path}')
    algorithm_id = agent.algorithm_id
    model = agent.model
    payload = {
        'schema_version': 1, 'checkpoint_type': 'evaluation_snapshot',
        'algorithm_id': algorithm_id, 'identity': dict(identity),
        'model_state_dict': model.state_dict(),
        'input_dim': int(model.dense_1.in_features)
        if algorithm_id in {'DDQN', 'CTXDDQN'} else 16,
        'action_dim': 8,
    }
    payload['model_state_digest'] = canonical_digest(
        payload['model_state_dict']
    )
    atomic_torch_save(payload, path)
    return payload


class SnapshotDQNPolicy:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def bind(self, world):
        self.world = world
        self.intersection = world.intersections[0]
        self.ob_generator = LaneVehicleGenerator(
            world, self.intersection, ['lane_count'], in_only=True, average=None,
        )
        self.phase_generator = IntersectionPhaseGenerator(
            world, self.intersection, ['phase'], targets=['cur_phase'],
            negative=False,
        )
        self.action_space = gym.spaces.Discrete(len(self.intersection.phases))
        self.model = DQNNet(self.snapshot['input_dim'], 8)
        self.model.load_state_dict(self.snapshot['model_state_dict'])
        self.model.eval()

    def get_ob(self):
        return np.asarray([self.ob_generator.generate()], dtype=np.float32)

    def get_phase(self):
        return np.concatenate([self.phase_generator.generate()]).astype(np.int8)

    def action(self, observation, phase):
        one_hot = agent_utils.idx2onehot(phase, 8).astype(np.float32)
        raw = np.concatenate([observation, one_hot], axis=1)
        if self.snapshot['algorithm_id'] == 'CTXDDQN':
            context = self.world.get_plan5_arrival_context()['rates']
            raw = np.concatenate([
                raw, np.asarray([context], dtype=np.float32),
            ], axis=1)
        with torch.no_grad():
            return torch.argmax(
                self.model(torch.as_tensor(raw, dtype=torch.float32)), dim=1,
            ).cpu().numpy()


class SnapshotPPOPolicy:
    def __init__(self, snapshot):
        self.snapshot = snapshot

    def bind(self, world):
        from .ppo_adapter import Plan5PPOAdapter
        self.adapter = Plan5PPOAdapter(
            world, 0, 2.5e-4, 0.001, training_seed=0,
        )
        self.adapter.model.load_state_dict(self.snapshot['model_state_dict'])
        self.adapter.model.eval()
        self.intersection = self.adapter.inter
        self.action_space = self.adapter.action_space

    def get_ob(self): return self.adapter.get_ob()
    def get_phase(self): return self.adapter.get_phase()
    def action(self, observation, phase):
        return self.adapter.get_action(observation, phase, test=True)


def _atomic_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tmp-plan5-eval-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            for row in rows:
                handle.write(json.dumps(row, sort_keys=True, separators=(',', ':')))
                handle.write('\n')
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary): os.unlink(temporary)
        raise


def evaluation_worker(request_path):
    request = read_json(request_path)
    snapshot = torch.load(request['snapshot_path'], map_location='cpu')
    if snapshot.get('schema_version') != 1 \
            or snapshot.get('checkpoint_type') != 'evaluation_snapshot':
        raise ValueError('Invalid Plan5 evaluation snapshot')
    if canonical_digest(snapshot['model_state_dict']) != snapshot[
        'model_state_digest'
    ]:
        raise ValueError('Plan5 evaluation snapshot model digest mismatch')
    policy = (
        SnapshotPPOPolicy(snapshot) if snapshot['algorithm_id'] == 'PPO'
        else SnapshotDQNPolicy(snapshot)
    )
    evaluation = evaluate_policy(
        request['network'], request['attempt_dir'], policy,
        interface=request.get('interface', 'traci'), collect_probe=False,
    )
    decisions_path = Path(request['attempt_dir']) / 'decisions.jsonl'
    _atomic_jsonl(decisions_path, evaluation['decisions'])
    summary = dict(evaluation['summary'])
    summary.update({
        'schema_version': 1, 'algorithm_id': snapshot['algorithm_id'],
        'action_selection': 'deterministic_argmax',
        'snapshot_path': request['snapshot_path'],
        'snapshot_sha256': sha256_file(request['snapshot_path']),
        'worker_pid': os.getpid(), 'decisions_path': str(decisions_path.resolve()),
        'decisions_sha256': sha256_file(decisions_path),
    })
    atomic_json(Path(request['attempt_dir']) / 'summary.json', summary)
    return summary


class Plan5FrozenEvaluator:
    def __init__(self, python_executable=None, timeout_seconds=600):
        self.python_executable = python_executable or os.sys.executable
        self.timeout_seconds = int(timeout_seconds)

    def evaluate(self, agent, snapshot_path, network, attempt_dir,
                 *, interface='traci', parent_checkpoint=None):
        snapshot_path = os.path.abspath(snapshot_path)
        attempt_dir = os.path.abspath(attempt_dir)
        os.makedirs(attempt_dir, exist_ok=False)
        before = training_state_digest(agent)
        snapshot_sha = sha256_file(snapshot_path)
        parent_sha = (
            None if parent_checkpoint is None
            else sha256_file(parent_checkpoint)
        )
        request = {
            'snapshot_path': snapshot_path, 'network': network,
            'attempt_dir': attempt_dir, 'interface': interface,
            'parent_pid': os.getpid(),
        }
        request_path = os.path.join(attempt_dir, 'request.json')
        atomic_json(request_path, request)
        completed = subprocess.run(
            [self.python_executable, '-m', 'plan5.frozen_evaluator',
             '--worker', request_path],
            cwd=os.getcwd(), capture_output=True, text=True,
            timeout=self.timeout_seconds, check=False,
            env={**os.environ, 'PYTHONNOUSERSITE': '1'},
        )
        if completed.returncode != 0:
            raise RuntimeError(
                'Plan5 evaluator failed: ' + completed.stderr[-4000:]
            )
        after = training_state_digest(agent)
        summary = read_json(os.path.join(attempt_dir, 'summary.json'))
        checks = {
            'separate_process': summary['worker_pid'] != os.getpid(),
            'training_state_unchanged': before == after,
            'snapshot_unchanged': snapshot_sha == sha256_file(snapshot_path),
            'parent_checkpoint_unchanged': parent_checkpoint is None
            or parent_sha == sha256_file(parent_checkpoint),
            'deterministic_action': (
                summary.get('action_selection') == 'deterministic_argmax'
            ),
        }
        manifest = {
            'schema_version': 1, 'summary': summary,
            'training_state_digest_before': before,
            'training_state_digest_after': after,
            'snapshot_sha256_before': snapshot_sha,
            'snapshot_sha256_after': sha256_file(snapshot_path),
            'parent_checkpoint_sha256_before': parent_sha,
            'parent_checkpoint_sha256_after': (
                None if parent_checkpoint is None
                else sha256_file(parent_checkpoint)
            ),
            'checks': checks, 'valid': all(checks.values()),
            'worker_stdout': completed.stdout,
        }
        if not manifest['valid']:
            raise RuntimeError('Plan5 frozen evaluation isolation failed')
        atomic_json(os.path.join(attempt_dir, 'evaluation_manifest.json'), manifest)
        return manifest


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', required=True)
    args = parser.parse_args(argv)
    summary = evaluation_worker(args.worker)
    print(json.dumps(summary, sort_keys=True))


if __name__ == '__main__':
    main()
