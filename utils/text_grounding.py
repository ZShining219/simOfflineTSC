"""Deterministic mention-to-catalog linking, independent of simulator state.

Design references (ideas, not reproductions of pretrained models):
* BLINK, Wu et al., EMNLP 2020: https://aclanthology.org/2020.emnlp-main.519/
  separates text mentions from a catalog of candidate entities.
* PURE, Zhong and Chen, NAACL 2021: https://arxiv.org/abs/2010.12812
  separates entity extraction from relation extraction.
* EMMA, Hanjie et al., ICML 2021:
  https://proceedings.mlr.press/v139/hanjie21a.html motivates retaining
  entity/report associations for downstream entity-conditioned attention.

This controlled-language baseline uses exact IDs and static relation checks.
It neither learns attention nor infers control actions or congestion propagation.
"""
from dataclasses import asdict, dataclass, replace
from copy import deepcopy
import math
from typing import Optional, Tuple

import numpy as np


class GroundingError(ValueError):
    """A public report cannot be grounded without guessing."""

    def __init__(self, code, message):
        self.code = code
        super().__init__(f'{code}: {message}')


@dataclass(frozen=True)
class Gnode:
    """Stable observation grounding kept beside, but outside, scene vectors.

    ``node_mask`` is the aggregate node-level grounding used by later policy
    code.  The report- and feature-level masks remain available for auditing
    and future SGA variants; none of these arrays are folded into ``z_scene``.
    """

    node_mask: np.ndarray
    node_report_mask: np.ndarray
    feature_report_mask: np.ndarray
    valid_feature_mask: np.ndarray
    lane_report_mask: np.ndarray
    report_ids: Tuple[str, ...] = ()
    scope: str = 'none'
    # Optional route relation channels: direct target, upstream/from-junction,
    # and downstream/to-junction.  Kept outside Zscene and read-only.
    node_event_relation: Optional[np.ndarray] = None
    lane_ids: Tuple[str, ...] = ()
    node_ids: Tuple[str, ...] = ()

    def __post_init__(self):
        arrays = {
            'node_mask': self.node_mask,
            'node_report_mask': self.node_report_mask,
            'feature_report_mask': self.feature_report_mask,
            'valid_feature_mask': self.valid_feature_mask,
            'lane_report_mask': self.lane_report_mask,
        }
        for name, value in arrays.items():
            array = np.asarray(value, dtype=bool).copy()
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if self.node_event_relation is None:
            relation = np.zeros((*self.node_report_mask.shape, 3), dtype=bool)
            relation[..., 0] = self.node_report_mask
        else:
            relation = np.asarray(self.node_event_relation, dtype=bool).copy()
        relation.setflags(write=False)
        object.__setattr__(self, 'node_event_relation', relation)
        if self.node_mask.ndim != 1:
            raise ValueError('Gnode.node_mask must have shape [num_nodes]')
        if self.node_report_mask.ndim != 2:
            raise ValueError('Gnode.node_report_mask must have shape [num_nodes, reports]')
        if self.node_report_mask.shape[0] != self.node_mask.shape[0]:
            raise ValueError('Gnode node masks disagree on num_nodes')
        if self.feature_report_mask.ndim != 3:
            raise ValueError('Gnode.feature_report_mask must have shape [nodes, features, reports]')
        if self.valid_feature_mask.shape != self.feature_report_mask.shape[:2]:
            raise ValueError('Gnode valid_feature_mask does not match feature mask')
        if self.feature_report_mask.shape[0] != self.node_mask.shape[0]:
            raise ValueError('Gnode feature mask disagrees on num_nodes')
        if self.lane_report_mask.ndim != 2:
            raise ValueError('Gnode.lane_report_mask must have shape [lanes, reports]')
        report_count = self.node_report_mask.shape[1]
        if (self.feature_report_mask.shape[2] != report_count
                or self.lane_report_mask.shape[1] != report_count
                or len(self.report_ids) != report_count):
            raise ValueError('Gnode masks and report_ids disagree on report count')
        if self.node_event_relation.shape != (self.node_mask.shape[0], report_count, 3):
            raise ValueError('Gnode.node_event_relation must have shape [nodes, reports, 3]')
        if self.lane_ids and len(self.lane_ids) != self.lane_report_mask.shape[0]:
            raise ValueError('Gnode.lane_ids disagree with lane_report_mask')
        if self.node_ids and len(self.node_ids) != self.node_mask.shape[0]:
            raise ValueError('Gnode.node_ids disagree with node masks')
        if self.scope not in {'none', 'local', 'global', 'mixed'}:
            raise ValueError('Gnode.scope must be none, local, global, or mixed')


@dataclass(frozen=True)
class Mention:
    role: str
    entity_type: str
    entity_id: str
    start: int
    end: int  # Exclusive character offset in the original, unnormalized text.


@dataclass(frozen=True)
class Relation:
    subject: str
    predicate: str
    object: str


@dataclass(frozen=True)
class GroundedReport:
    """Immutable report snapshot; parsed semantics are diagnostic by default.

    Keep ``text`` as the text encoder input. Do not silently replace it with
    categorical event features. A cleared report retains its target associations;
    associations describe a report's referent, not a currently active restriction.
    Numeric values are as reported (possibly rounded), not simulator ground truth.
    """
    report_id: str
    updated_at: float
    text: str
    event_kind: Optional[str] = None
    report_status: Optional[str] = None
    scope: Optional[str] = None
    mentions: Tuple[Mention, ...] = ()
    relations: Tuple[Relation, ...] = ()
    direction: Optional[str] = None
    movements: Tuple[str, ...] = ()
    position_m: Optional[float] = None
    speed_percent: Optional[float] = None
    target_lane_ids: Tuple[str, ...] = ()
    mapping_status: str = 'parsed'
    diagnostics: Tuple[str, ...] = ()

    def to_dict(self):
        """JSON-serializable evidence, including mention offsets and raw text."""
        return asdict(self)


class EntityCatalog:
    """Snapshot a lane catalog; never retain a World/runtime/schedule reference."""

    def __init__(self, lanes):
        self._lanes = deepcopy(dict(lanes))
        self._edges = {}
        self._junctions = set()
        required = {'lane_id', 'edge_id', 'from_junction', 'to_junction',
                    'direction', 'movements', 'internal', 'motor_vehicle_lane', 'length'}
        for key, lane in self._lanes.items():
            if not isinstance(lane, dict) or not required <= lane.keys() or lane['lane_id'] != key:
                raise ValueError(f'Incomplete or inconsistent catalog lane: {key}')
            if any(not isinstance(lane[k], str) or not lane[k] for k in
                   ('lane_id', 'edge_id', 'from_junction', 'to_junction', 'direction')):
                raise ValueError(f'Invalid catalog identifiers: {key}')
            if (not isinstance(lane['internal'], bool)
                    or not isinstance(lane['motor_vehicle_lane'], bool)
                    or isinstance(lane['length'], bool)
                    or not isinstance(lane['length'], (int, float))
                    or not math.isfinite(lane['length']) or lane['length'] <= 0):
                raise ValueError(f'Invalid catalog lane properties: {key}')
            if (not isinstance(lane['movements'], (list, tuple))
                    or any(not isinstance(m, str) for m in lane['movements'])):
                raise ValueError(f'Invalid catalog movements: {key}')
            self._edges.setdefault(lane['edge_id'], []).append(key)
            self._junctions.update((lane['from_junction'], lane['to_junction']))
        for edge, ids in self._edges.items():
            endpoints = {(self._lanes[i]['from_junction'], self._lanes[i]['to_junction']) for i in ids}
            if len(endpoints) != 1:
                raise ValueError(f'Inconsistent endpoints for edge {edge}')
            self._edges[edge] = tuple(sorted(ids))
        self._lane_ids = tuple(sorted(self._lanes))

    @property
    def lane_ids(self):
        return self._lane_ids

    def lane(self, lane_id):
        """Return a copy so callers cannot mutate the linker catalog."""
        return deepcopy(self._lanes[lane_id])

    def link(self, report):
        """Resolve explicit mentions, check their relationships, expand scope."""
        by_role = {m.role: m.entity_id for m in report.mentions}
        for mention in report.mentions:
            candidates = {'lane': self._lanes, 'edge': self._edges, 'junction': self._junctions}
            if mention.entity_type not in candidates or mention.entity_id not in candidates[mention.entity_type]:
                raise GroundingError('unknown_entity', f'{mention.entity_type} {mention.entity_id}')
        relations = []
        if report.scope == 'lane':
            lane_id = by_role['target_lane']
            lane = self._lanes[lane_id]
            if lane['internal'] or not lane['motor_vehicle_lane']:
                raise GroundingError('invalid_target', 'Blockage requires an external motor-vehicle lane')
            expected = (lane['edge_id'], lane['to_junction'], lane['direction'], set(lane['movements']))
            actual = (by_role['road'], by_role['approaching'], report.direction, set(report.movements))
            if expected != actual:
                raise GroundingError('relation_conflict', f'Lane attributes disagree with catalog: {lane_id}')
            # Public positions are rounded to centimetres. A value at the end
            # can represent a valid position just before it; do not infer hidden
            # precision or reject a report on that basis.
            if report.position_m is not None and not 5 <= report.position_m <= lane['length'] + 0.005:
                raise GroundingError('invalid_value', 'Obstacle position outside supported lane range')
            targets = (lane_id,)
            relations.extend((Relation(lane_id, 'part_of', lane['edge_id']),
                              Relation(lane_id, 'approaches', lane['to_junction']),
                              Relation(lane['edge_id'], 'from_junction', lane['from_junction']),
                              Relation(lane['edge_id'], 'to_junction', lane['to_junction'])))
        elif report.scope == 'edge':
            edge_id = by_role['target_edge']
            targets = self._edges[edge_id]
            first = self._lanes[targets[0]]
            if any(self._lanes[i]['internal'] for i in targets):
                raise GroundingError('invalid_target', 'Closure requires an external directed road')
            if (first['from_junction'], first['to_junction']) != (by_role['from'], by_role['to']):
                raise GroundingError('relation_conflict', f'Road endpoints disagree with catalog: {edge_id}')
            relations.extend(Relation(i, 'part_of', edge_id) for i in targets)
            relations.extend((Relation(edge_id, 'from_junction', first['from_junction']),
                              Relation(edge_id, 'to_junction', first['to_junction'])))
        elif report.scope == 'network':
            targets = tuple(i for i in self.lane_ids if self._lanes[i]['motor_vehicle_lane'])
            if not targets:
                raise GroundingError('invalid_target', 'Network has no motor-vehicle lanes')
        else:
            raise GroundingError('unsupported_scope', str(report.scope))
        return replace(report, relations=tuple(relations), target_lane_ids=targets,
                       mapping_status='mapped')
