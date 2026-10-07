"""Owned, immutable reward contracts for new paper-baseline runs.

Legacy model rewards remain unchanged. These contracts reproduce the selected
author-code reward algebra with an explicit SUMO stopped-vehicle adapter, not
the papers' entire training protocol. Environment aggregation occurs once per
simulation step; the existing trainer averages over the action interval.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path

import yaml


REQUIREMENTS = {
    'frap': ('queue_sum', 'phase_movements'),
    'presslight': ('absolute_queue_pressure', 'all_incoming_outgoing'),
    'colight': ('queue_sum', 'all_incoming'),
}


@dataclass(frozen=True)
class RewardProfile:
    schema_version: str
    profile_id: str
    owner: str
    statistic: str
    lane_scope: str
    weight: float
    normal_factor: float
    stopped_speed_mps: float
    temporal_aggregation: str
    references: tuple
    qualification: str

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True).encode()).hexdigest()

    def calculate(self, incoming, outgoing=()):
        incoming, outgoing = tuple(incoming), tuple(outgoing)
        values = incoming + outgoing
        if not incoming or any(not math.isfinite(float(v)) or v < 0 for v in values):
            raise ValueError('Reward requires finite nonnegative queue counts and incoming lanes')
        statistic = sum(incoming)
        if self.statistic == 'absolute_queue_pressure':
            if not outgoing:
                raise ValueError('Pressure reward requires outgoing lanes')
            statistic = abs(statistic - sum(outgoing))
        raw = self.weight * statistic
        return {'statistic': float(statistic), 'raw_reward': float(raw),
                'reward': float(raw / self.normal_factor)}


def load_profile(path, owner):
    """Reject cross-model selection and unknown/contradictory reward fields."""
    data = yaml.safe_load(Path(path).read_text())
    expected = set(RewardProfile.__dataclass_fields__)
    if not isinstance(data, dict) or set(data) != expected:
        raise ValueError('Reward profile fields must exactly match the versioned schema')
    if data['schema_version'] != 'paper-reward-v1' or owner not in REQUIREMENTS:
        raise ValueError('Unsupported reward schema or model')
    if data['owner'] != owner or not str(data['profile_id']).startswith(owner + '_'):
        raise ValueError(f'Reward ownership mismatch: {owner} cannot use {data["profile_id"]}')
    if (data['statistic'], data['lane_scope']) != REQUIREMENTS[owner]:
        raise ValueError(f'Reward statistic/scope does not match the {owner} contract')
    for key in ('weight', 'normal_factor', 'stopped_speed_mps'):
        if isinstance(data[key], bool) or not isinstance(data[key], (int, float)) or not math.isfinite(data[key]):
            raise ValueError(f'Invalid reward parameter: {key}')
    if data['weight'] >= 0 or data['normal_factor'] <= 0 or data['stopped_speed_mps'] <= 0:
        raise ValueError('Expected negative penalty weight and positive normalization/threshold')
    if data['temporal_aggregation'] != 'action_interval_mean':
        raise ValueError('This trainer supports action_interval_mean only')
    if not isinstance(data['references'], list) or not data['references']:
        raise ValueError('Reward provenance is required')
    data['references'] = tuple(data['references'])
    return RewardProfile(**data)
