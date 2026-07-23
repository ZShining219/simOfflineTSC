import dataclasses
import hashlib
import json
import math
import random
from collections import deque

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class TrainingPayload:
    state: object
    phase: object
    action: object
    reward: object
    next_state: object
    next_phase: object
    terminated: bool = False
    truncated: bool = False

    @classmethod
    def from_legacy(cls, payload):
        if not isinstance(payload, (tuple, list)) or len(payload) != 6:
            raise ValueError('Legacy DQN replay payload must contain six fields')
        values = []
        for value in payload:
            array = np.array(value, copy=True)
            array.setflags(write=False)
            values.append(array)
        return cls(*values)

    def legacy_tuple(self):
        return (self.state, self.phase, self.action, self.reward,
                self.next_state, self.next_phase)


@dataclasses.dataclass(frozen=True)
class ReplayMetadata:
    transition_id: str
    source_network: str
    stage_index: int
    local_episode: int
    decision_index: int
    written_global_step: int


@dataclasses.dataclass(frozen=True)
class ReplayRecord:
    payload: TrainingPayload
    metadata: ReplayMetadata


@dataclasses.dataclass
class TargetUpdateScheduler:
    interval: int
    next_update: int

    def __post_init__(self):
        if self.interval <= 0 or self.next_update <= 0:
            raise ValueError('Target scheduler values must be positive')

    def after_successful_gradient(self, completed_updates):
        if completed_updates < self.next_update:
            return False
        if completed_updates != self.next_update:
            raise ValueError(
                f'Gradient update skipped target boundary {self.next_update}'
            )
        self.next_update += self.interval
        return True

    @classmethod
    def from_plan1_parent(cls, gradient_updates, target_updates, interval):
        """Restore Plan 1's post-update target phase without re-aligning it."""
        expected_targets = gradient_updates // interval
        if target_updates != expected_targets:
            raise ValueError(
                'Parent target update count is incompatible with Plan 1: '
                f'{target_updates} != {expected_targets}'
            )
        # Plan 1 synchronised when the zero-based decision/update index was a
        # multiple of interval.  The next child sync therefore occurs after
        # interval-1 additional successful gradients.
        return cls(interval=interval, next_update=gradient_updates + interval - 1)


class SequentialReplay:
    def __init__(self, capacity, records=()):
        if capacity <= 0:
            raise ValueError('Replay capacity must be positive')
        self.records = deque(records, maxlen=capacity)
        self.capacity = capacity
        self.stage_insertions = 0

    def append(self, record):
        if not isinstance(record, ReplayRecord):
            raise TypeError('Sequential replay accepts ReplayRecord only')
        self.records.append(record)
        self.stage_insertions += 1

    def begin_stage(self, policy):
        if policy not in {'clear', 'fifo', 'fifo_matched_wait'}:
            raise ValueError(f'Unknown replay policy: {policy}')
        if policy == 'clear':
            self.records.clear()
        self.stage_insertions = 0

    def is_update_ready(self, policy, learning_start, batch_size):
        enough_batch = len(self.records) >= batch_size
        if policy == 'fifo':
            return enough_batch and len(self.records) > learning_start
        if policy == 'clear':
            return enough_batch and len(self.records) > learning_start
        if policy == 'fifo_matched_wait':
            return enough_batch and self.stage_insertions > learning_start
        raise ValueError(f'Unknown replay policy: {policy}')

    def sample(self, batch_size, random_module):
        sample = random_module.sample(list(self.records), batch_size)
        return sample

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'records': list(self.records),
            'stage_insertions': self.stage_insertions,
        }

    @classmethod
    def from_state_dict(cls, state):
        replay = cls(int(state['capacity']), state['records'])
        replay.stage_insertions = int(state['stage_insertions'])
        return replay

    def composition(self):
        counts = {}
        for record in self.records:
            network = record.metadata.source_network
            counts[network] = counts.get(network, 0) + 1
        total = len(self.records)
        return {
            'size': total, 'capacity': self.capacity, 'count_by_scene': counts,
            'ratio_by_scene': {key: value / total for key, value in counts.items()}
            if total else {},
        }


def _digest_update(digest, value, path='$'):
    def tagged(tag, payload=b''):
        digest.update(tag.encode('ascii') + len(payload).to_bytes(8, 'big') + payload)

    if value is None:
        tagged('none')
    elif isinstance(value, bool):
        tagged('bool', b'1' if value else b'0')
    elif isinstance(value, (int, np.integer)):
        tagged('int', str(int(value)).encode())
    elif isinstance(value, (float, np.floating)):
        numeric = float(value)
        if math.isnan(numeric):
            tagged('float', b'nan')
        else:
            tagged('float', np.float64(numeric).tobytes())
    elif isinstance(value, str):
        tagged('str', value.encode('utf-8'))
    elif isinstance(value, bytes):
        tagged('bytes', value)
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        tagged('field-path', path.encode('utf-8'))
        tagged('tensor-dtype', str(tensor.dtype).encode())
        tagged('tensor-shape', json.dumps(list(tensor.shape)).encode())
        tagged('tensor-data', tensor.numpy().tobytes(order='C'))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        tagged('field-path', path.encode('utf-8'))
        tagged('array-dtype', str(array.dtype).encode())
        tagged('array-shape', json.dumps(list(array.shape)).encode())
        tagged('array-data', array.tobytes(order='C'))
    elif dataclasses.is_dataclass(value):
        tagged('dataclass', value.__class__.__qualname__.encode())
        for field in dataclasses.fields(value):
            _digest_update(digest, field.name, path)
            _digest_update(digest, getattr(value, field.name), f'{path}.{field.name}')
    elif isinstance(value, dict):
        tagged('dict')
        encoded_keys = [(canonical_digest(key), key) for key in value]
        for _, key in sorted(encoded_keys, key=lambda item: item[0]):
            _digest_update(digest, key, path)
            _digest_update(digest, value[key], f'{path}[{key!r}]')
    elif isinstance(value, (list, tuple, deque)):
        tagged(value.__class__.__name__)
        for index, item in enumerate(value):
            _digest_update(digest, item, f'{path}[{index}]')
    else:
        raise TypeError(f'Unsupported canonical digest type: {type(value)!r}')


def canonical_digest(value):
    digest = hashlib.sha256()
    _digest_update(digest, value)
    return digest.hexdigest()


def online_parameter_digest(state):
    return canonical_digest(state)


def target_parameter_digest(state):
    return canonical_digest(state)


def optimizer_state_digest(state):
    return canonical_digest(state)


def rng_state_digest(python_state, numpy_state, torch_cpu_state, torch_cuda_states):
    return canonical_digest({
        'python': python_state, 'numpy': numpy_state,
        'torch_cpu': torch_cpu_state, 'torch_cuda': torch_cuda_states,
    })


def replay_content_digest(records):
    return canonical_digest([record.payload for record in records])


def replay_metadata_digest(records):
    return canonical_digest([record.metadata for record in records])


def canonical_transition_digest(transition):
    """Digest only fields that can affect training semantics."""
    fields = (
        'stage_index', 'local_episode', 'global_episode', 'decision_index',
        'global_decision_step', 'state', 'phase', 'action', 'reward',
        'next_state', 'next_phase', 'terminated', 'truncated',
    )
    if dataclasses.is_dataclass(transition):
        source = dataclasses.asdict(transition)
    else:
        source = dict(transition)
    return canonical_digest({key: source.get(key) for key in fields})


def environment_signature_digest(signature):
    return canonical_digest(signature)


def capture_rng_state():
    return {
        'python_random_state': random.getstate(),
        'numpy_random_state': np.random.get_state(),
        'torch_cpu_rng_state': torch.get_rng_state(),
        'torch_cuda_rng_states': (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }
