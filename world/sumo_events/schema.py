"""Versioned, deterministic event schedules; times are SUMO seconds.

Intervals are [begin, end). Reports contain observed facts only. Schedules
and audit output include future events and must never be policy inputs.
"""
from dataclasses import asdict, dataclass
import math
from pathlib import Path
import re
from typing import Optional, Tuple

import yaml


KINDS = ('lane_blockage', 'road_closure', 'global_rain')


@dataclass(frozen=True)
class Event:
    event_id: str
    kind: str
    begin: float
    end: float
    lane_id: Optional[str] = None
    edge_id: Optional[str] = None
    position: Optional[float] = None
    position_tolerance: float = 0.0
    speed_factor: Optional[float] = None

    def __post_init__(self):
        if not isinstance(self.event_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', self.event_id):
            raise ValueError('event_id must be a nonempty ASCII identifier')
        if self.kind not in KINDS:
            raise ValueError(f'Unknown event kind: {self.kind}')
        for key in ('begin', 'end', 'position_tolerance'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f'{key} must be finite seconds/metres')
        if self.begin < 0 or self.end <= self.begin or self.position_tolerance < 0:
            raise ValueError('Require 0 <= begin < end and nonnegative position_tolerance')
        for key in ('lane_id', 'edge_id'):
            value = getattr(self, key)
            if value is not None and (not isinstance(value, str) or not value or any(c.isspace() for c in value)):
                raise ValueError(f'{key} must be a nonempty SUMO identifier')
        if self.kind == 'lane_blockage':
            if not self.lane_id or self.edge_id is not None or self.speed_factor is not None:
                raise ValueError('lane_blockage requires lane_id only as its target')
            if (isinstance(self.position, bool) or not isinstance(self.position, (int, float))
                    or not math.isfinite(self.position) or self.position < 5):
                raise ValueError('position must be a finite front-bumper position >= 5 metres')
        elif self.kind == 'road_closure':
            if (not self.edge_id or self.lane_id is not None or self.position is not None
                    or self.speed_factor is not None or self.position_tolerance):
                raise ValueError('road_closure requires one directed edge_id, with no lane/rain parameters')
        else:
            if self.lane_id is not None or self.edge_id is not None or self.position is not None or self.position_tolerance:
                raise ValueError('global_rain has network-wide scope, not a local target')
            if (isinstance(self.speed_factor, bool) or not isinstance(self.speed_factor, (int, float))
                    or not math.isfinite(self.speed_factor) or not 0 < self.speed_factor < 1):
                raise ValueError('Rain speed_factor must be in (0, 1)')


@dataclass(frozen=True)
class Schedule:
    events: Tuple[Event, ...]
    schema_version: str = 'sumo-events-v1'

    def __post_init__(self):
        object.__setattr__(self, 'events', tuple(self.events))
        if self.schema_version != 'sumo-events-v1':
            raise ValueError(f'Unsupported event schema: {self.schema_version}')
        if not all(isinstance(event, Event) for event in self.events):
            raise TypeError('Schedule.events must contain Event objects')
        if len({e.event_id for e in self.events}) != len(self.events):
            raise ValueError('Duplicate event_id')

    def to_dict(self):
        return {'schema_version': self.schema_version, 'events': [asdict(e) for e in self.events]}


def load_schedule(path):
    """Load a strict YAML/JSON schedule. No RNG or simulation is invoked."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or set(data) != {'schema_version', 'events'}:
        raise ValueError('Expected exactly schema_version and events')
    if not isinstance(data['events'], list):
        raise ValueError('events must be a list')
    return Schedule(tuple(Event(**row) for row in data['events']), data['schema_version'])


@dataclass(frozen=True)
class Report:
    """Public message: no future end time, privileged target mask, or action advice."""
    event_id: str
    status: str
    updated_at: float
    text: str
