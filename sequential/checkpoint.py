import os
import tempfile

import torch

from .core import canonical_digest


CHECKPOINT_SCHEMA_VERSION = 1


def atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-checkpoint-', suffix='.pt', dir=directory,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def build_full_checkpoint(agent, checkpoint_type, identity):
    agent_state = agent.full_state_dict()
    return {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'checkpoint_type': checkpoint_type,
        'identity': dict(identity),
        'agent_state': agent_state,
        'canonical_state_digest': canonical_digest(agent_state),
    }


def validate_full_checkpoint(payload, expected_type=None):
    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('Unsupported Sequential checkpoint schema')
    if expected_type is not None and payload.get('checkpoint_type') != expected_type:
        raise ValueError('Sequential checkpoint type mismatch')
    if canonical_digest(payload['agent_state']) != payload.get('canonical_state_digest'):
        raise ValueError('Sequential checkpoint canonical digest mismatch')
    return payload


def load_full_checkpoint(path, expected_type=None):
    try:
        payload = torch.load(path, map_location='cpu')
    except Exception as error:
        raise IOError(f'Cannot load Sequential checkpoint: {path}') from error
    return validate_full_checkpoint(payload, expected_type=expected_type)


class RollingRecoveryManager:
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        self.latest_path = os.path.join(self.directory, 'latest.pt')
        self.previous_path = os.path.join(self.directory, 'previous.pt')
        os.makedirs(self.directory, exist_ok=True)

    def save_episode_start(self, agent, identity):
        payload = build_full_checkpoint(agent, 'episode_start_recovery', identity)
        temporary_latest = self.latest_path + '.new'
        atomic_torch_save(payload, temporary_latest)
        if os.path.isfile(self.latest_path):
            os.replace(self.latest_path, self.previous_path)
        os.replace(temporary_latest, self.latest_path)
        return self.latest_path

    def latest_valid(self):
        errors = []
        for path in (self.latest_path, self.previous_path):
            if not os.path.isfile(path):
                continue
            try:
                return path, load_full_checkpoint(
                    path, expected_type='episode_start_recovery'
                )
            except Exception as error:
                errors.append(f'{path}: {error}')
        if errors:
            raise IOError('No valid recovery checkpoint: ' + '; '.join(errors))
        raise FileNotFoundError('No recovery checkpoint exists')

    def restore_latest(self, agent):
        path, payload = self.latest_valid()
        agent.load_full_state_dict(payload['agent_state'])
        return path, payload


def save_stage_checkpoint(agent, path, identity):
    payload = build_full_checkpoint(agent, 'stage_boundary', identity)
    atomic_torch_save(payload, path)
    return payload
