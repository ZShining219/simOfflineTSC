import os
import tempfile

import torch

from .core import canonical_digest
from .io import sha256_file


CHECKPOINT_SCHEMA_VERSION = 1


def build_resume_validation(resume_path, resume_state_path, payload,
                            loaded_agent_state):
    expected = payload['canonical_state_digest']
    loaded = canonical_digest(loaded_agent_state)
    if loaded != expected:
        raise ValueError('Loaded resume state canonical digest mismatch')
    component_keys = (
        'online_model_state_dict', 'target_model_state_dict',
        'optimizer_state_dict', 'replay_state', 'rng_state', 'counters',
        'ha_sodqn',
    )
    return {
        'schema_version': 1,
        'valid': True,
        'resume_checkpoint': os.path.abspath(resume_path),
        'resume_checkpoint_sha256': sha256_file(resume_path),
        'resume_state': (
            None if resume_state_path is None
            else os.path.abspath(resume_state_path)
        ),
        'checkpoint_identity': payload.get('identity'),
        'expected_canonical_state_digest': expected,
        'loaded_canonical_state_digest': loaded,
        'component_digests': {
            key: canonical_digest(payload['agent_state'][key])
            for key in component_keys if key in payload['agent_state']
        },
    }


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


def build_full_checkpoint(agent, checkpoint_type, identity, extra_state=None):
    agent_state = agent.full_state_dict()
    payload = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'checkpoint_type': checkpoint_type,
        'identity': dict(identity),
        'agent_state': agent_state,
        'canonical_state_digest': canonical_digest(agent_state),
    }
    if extra_state is not None:
        payload['extra_state'] = extra_state
        payload['extra_state_digest'] = canonical_digest(extra_state)
    return payload


def validate_full_checkpoint(payload, expected_type=None):
    if payload.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError('Unsupported Sequential checkpoint schema')
    if expected_type is not None and payload.get('checkpoint_type') != expected_type:
        raise ValueError('Sequential checkpoint type mismatch')
    if canonical_digest(payload['agent_state']) != payload.get('canonical_state_digest'):
        raise ValueError('Sequential checkpoint canonical digest mismatch')
    if 'extra_state' in payload and canonical_digest(
        payload['extra_state']
    ) != payload.get('extra_state_digest'):
        raise ValueError('Sequential checkpoint extra-state digest mismatch')
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

    def save_episode_start(self, agent, identity, extra_state=None):
        payload = build_full_checkpoint(
            agent, 'episode_start_recovery', identity, extra_state=extra_state,
        )
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


def save_stage_checkpoint(agent, path, identity, extra_state=None):
    payload = build_full_checkpoint(
        agent, 'stage_boundary', identity, extra_state=extra_state,
    )
    atomic_torch_save(payload, path)
    return payload
