import os
import re
import tempfile

import numpy as np
import torch

from .core import canonical_digest
from .io import sha256_file


CHECKPOINT_SCHEMA_VERSION = 1

# Wire encoding for the HA-SODQN historical-sampler reuse counter.  The
# in-memory sampler state keeps schema_version 1 (a plain
# {transition_id: count} dict) so ``canonical_state_digest`` keeps covering
# the logical state; only the serialized form is column-packed.  The
# wire-level schema_version 2 marker makes pre-change readers fail loudly
# instead of silently interpreting the packed structure.
_SAMPLER_WIRE_SCHEMA_VERSION = 2
_REUSE_COLUMNS_ENCODING = 'reuse_columns_v1'
_TRANSITION_ID_TAIL = re.compile(r'\A(.*):episode(\d+):decision(\d+)\Z')


def _pack_reuse_counts(counts):
    structured = []
    literal_ids = []
    literal_counts = []
    for key, count in counts.items():
        match = _TRANSITION_ID_TAIL.match(key)
        rebuilt = None if match is None else (
            f'{match.group(1)}:episode{int(match.group(2)):04d}'
            f':decision{int(match.group(3)):04d}'
        )
        if rebuilt == key:
            structured.append((
                match.group(1), int(match.group(2)), int(match.group(3)),
                int(count),
            ))
        else:
            literal_ids.append(str(key))
            literal_counts.append(int(count))
    prefixes = sorted({row[0] for row in structured})
    codes = {prefix: index for index, prefix in enumerate(prefixes)}
    return {
        'encoding': _REUSE_COLUMNS_ENCODING,
        'prefixes': prefixes,
        'prefix_index': np.asarray(
            [codes[row[0]] for row in structured], dtype=np.int32,
        ),
        'episode': np.asarray(
            [row[1] for row in structured], dtype=np.int32,
        ),
        'decision': np.asarray(
            [row[2] for row in structured], dtype=np.int32,
        ),
        'count': np.asarray(
            [row[3] for row in structured], dtype=np.int64,
        ),
        'literal_ids': literal_ids,
        'literal_counts': np.asarray(literal_counts, dtype=np.int64),
    }


def _unpack_reuse_counts(packed):
    if packed.get('encoding') != _REUSE_COLUMNS_ENCODING:
        raise ValueError('Unsupported reuse_counts wire encoding')
    prefixes = packed['prefixes']
    counts = {
        f'{prefixes[int(code)]}:episode{int(episode):04d}'
        f':decision{int(decision):04d}': int(count)
        for code, episode, decision, count in zip(
            packed['prefix_index'], packed['episode'],
            packed['decision'], packed['count'],
        )
    }
    counts.update(
        (str(key), int(count))
        for key, count in zip(
            packed['literal_ids'], packed['literal_counts'],
        )
    )
    return counts


def encode_checkpoint_payload(payload):
    """Return the on-disk form of a full-checkpoint payload.

    ``agent_state.ha_sodqn.historical_sampler_state.reuse_counts`` is stored
    as packed columns instead of a ``{transition_id: count}`` dict.  The
    canonical digest is computed over the logical state before encoding, so
    digests stay comparable with un-encoded checkpoints.
    """
    agent_state = payload.get('agent_state')
    if not isinstance(agent_state, dict):
        return payload
    ha_sodqn = agent_state.get('ha_sodqn')
    if not isinstance(ha_sodqn, dict):
        return payload
    sampler = ha_sodqn.get('historical_sampler_state')
    if not isinstance(sampler, dict) \
            or not isinstance(sampler.get('reuse_counts'), dict):
        return payload
    packed = _pack_reuse_counts(sampler['reuse_counts'])
    packed['sampler_schema_version'] = sampler.get('schema_version')
    wire_sampler = dict(sampler)
    wire_sampler['schema_version'] = _SAMPLER_WIRE_SCHEMA_VERSION
    wire_sampler['reuse_counts'] = packed
    encoded = dict(payload)
    encoded['agent_state'] = {
        **agent_state,
        'ha_sodqn': {
            **ha_sodqn,
            'historical_sampler_state': wire_sampler,
        },
    }
    return encoded


def decode_checkpoint_payload(payload):
    """Restore the logical form of a payload written by atomic_torch_save.

    Payloads without the sampler wire marker are returned unchanged, so this
    is also the normalization point for direct ``torch.load`` consumers that
    need ``agent_state`` semantics.
    """
    agent_state = payload.get('agent_state')
    if not isinstance(agent_state, dict):
        return payload
    ha_sodqn = agent_state.get('ha_sodqn')
    if not isinstance(ha_sodqn, dict):
        return payload
    sampler = ha_sodqn.get('historical_sampler_state')
    if not isinstance(sampler, dict) \
            or sampler.get('schema_version') != _SAMPLER_WIRE_SCHEMA_VERSION:
        return payload
    packed = sampler.get('reuse_counts')
    if not isinstance(packed, dict):
        return payload
    wire_sampler = dict(sampler)
    wire_sampler['schema_version'] = packed.get('sampler_schema_version', 1)
    wire_sampler['reuse_counts'] = _unpack_reuse_counts(packed)
    decoded = dict(payload)
    decoded['agent_state'] = {
        **agent_state,
        'ha_sodqn': {
            **ha_sodqn,
            'historical_sampler_state': wire_sampler,
        },
    }
    return decoded


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
        torch.save(encode_checkpoint_payload(payload), temporary)
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
    return validate_full_checkpoint(
        decode_checkpoint_payload(payload), expected_type=expected_type,
    )


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
