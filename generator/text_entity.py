"""Bind grounded reports to existing lane_count observations, without encoding.

Usage after world.reset() and generator construction::

    from world.sumo_events.grounding import ReportGrounder
    mapper = ReportGrounder(runtime.network_catalog())
    binder = TextEntityBinder(mapper.catalog, world.intersection_ids, generators)
    batch = binder.bind(mapper.map_reports(runtime.reports()))

``batch.texts`` is ordered by report, not broadcast/pooled over intersections.
``batch.lane_report_mask`` and ``batch.feature_report_mask`` describe referents,
not active restrictions, learned attention, or upstream congestion effects.
Recreate the binder after reset because World replaces intersection objects.
"""
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from utils.text_grounding import Gnode, GroundedReport
from utils.scene_context import scene_context_at as build_scene_context


@dataclass(frozen=True)
class FeatureBinding:
    lane_id: str
    intersection_id: str
    intersection_index: int
    feature_index: int
    feature_name: str = 'lane_count'


@dataclass(frozen=True)
class BoundReport:
    report: GroundedReport
    feature_bindings: Tuple[FeatureBinding, ...]
    unobserved_lane_ids: Tuple[str, ...]


@dataclass(frozen=True)
class GroundingBatch:
    reports: Tuple[BoundReport, ...]
    lane_ids: Tuple[str, ...]
    intersection_ids: Tuple[str, ...]
    feature_sizes: Tuple[int, ...]
    lane_report_mask: np.ndarray  # [catalog lanes, reports]
    feature_report_mask: np.ndarray  # [intersections, max feature size, reports]
    valid_feature_mask: np.ndarray  # [intersections, max feature size]; excludes padding

    @property
    def texts(self):
        return tuple(item.report.text for item in self.reports)

    def scene_context_at(self, observed_at):
        """Build the current control scene from this synchronized snapshot.

        ``GroundingBatch`` may retain cleared reports for audit/replay, but
        ``scene_context_at`` promotes only reports whose public status is
        currently ``active``.  The timestamp is explicit so callers cannot
        accidentally reuse a context from another control step.
        """
        return build_scene_context(
            observed_at, (item.report for item in self.reports))

    def to_gnode(self, active_only=True):
        """Return the stable node-level grounding contract for this snapshot.

        Policy-facing G defaults to currently active reports.  Passing
        ``active_only=False`` is available to audit consumers that need the
        complete active/cleared report snapshot.
        """
        selected_with_indices = sorted(
            ((index, item) for index, item in enumerate(self.reports)
             if not active_only or item.report.report_status == 'active'),
            key=lambda pair: pair[1].report.report_id)
        report_indices = tuple(index for index, _ in selected_with_indices)
        selected = tuple(item for _, item in selected_with_indices)
        report_scopes = tuple(item.report.scope for item in selected)
        feature_report_mask = self.feature_report_mask[:, :, report_indices]
        lane_report_mask = self.lane_report_mask[:, report_indices]
        node_report_mask = np.any(feature_report_mask, axis=1)
        global_columns = np.asarray(
            [scope in ('global', 'network') for scope in report_scopes], dtype=bool)
        if np.any(global_columns):
            # A global event applies to every valid policy node, even if a
            # particular node has no directly observed lane binding.
            global_nodes = np.any(self.valid_feature_mask, axis=1)
            node_report_mask[:, global_columns] = global_nodes[:, None]
        node_relation = np.zeros((*node_report_mask.shape, 3), dtype=bool)
        node_relation[..., 0] = node_report_mask
        node_index = {node: index for index, node in enumerate(self.intersection_ids)}
        for report_index, item in enumerate(selected):
            # The catalog relation is public grounding, not a learned guess.
            # Preserve from/to junctions as topology hints beside direct masks.
            for relation in item.report.relations:
                target = node_index.get(relation.object)
                if target is None:
                    continue
                if relation.predicate == 'from_junction':
                    node_relation[target, report_index, 1] = True
                elif relation.predicate == 'to_junction':
                    node_relation[target, report_index, 2] = True
        node_mask = np.any(node_report_mask, axis=1)
        if not report_scopes:
            scope = 'none'
        elif np.any(global_columns) and any(
                s not in (None, 'global', 'network') for s in report_scopes):
            scope = 'mixed'
        elif np.any(global_columns):
            scope = 'global'
        else:
            scope = 'local'
        return Gnode(
            node_mask=node_mask,
            node_report_mask=node_report_mask,
            feature_report_mask=feature_report_mask,
            valid_feature_mask=self.valid_feature_mask,
            lane_report_mask=lane_report_mask,
            report_ids=tuple(item.report.report_id for item in selected),
            scope=scope,
            node_event_relation=node_relation,
            lane_ids=tuple(self.lane_ids),
            node_ids=tuple(self.intersection_ids),
        )


class TextEntityBinder:
    """Exact feature layout for unaggregated incoming lane_count generators.

    Reads the generator's actual lane order, never a guessed ID suffix order.
    Unsupported aggregation/features are rejected. Phase/text concatenation is
    outside this adapter; masks apply to the original lane observation only.
    """

    def __init__(self, catalog, intersection_ids, generators):
        self.catalog = catalog
        self.intersection_ids = tuple(intersection_ids)
        self._generators = tuple(generators)
        if (len(set(self.intersection_ids)) != len(self.intersection_ids)
                or len(self.intersection_ids) != len(self._generators) or not self._generators):
            raise ValueError('Supply one generator per distinct intersection, in policy order')
        self._layout = self._read_layout()
        self.feature_sizes = tuple(row[2] for row in self._layout)
        self._bindings = {}
        for node_index, (node_id, lanes, _) in enumerate(self._layout):
            for feature_index, lane_id in enumerate(lanes):
                self._bindings.setdefault(lane_id, []).append(
                    FeatureBinding(lane_id, node_id, node_index, feature_index))

    def _read_layout(self):
        rows = []
        for node_id, gen in zip(self.intersection_ids, self._generators):
            if tuple(gen.fns) != ('lane_count',) or gen.average is not None or gen.negative:
                raise ValueError('Only positive, unaggregated lane_count is supported')
            if gen.I.id != node_id or gen.world.id2intersection[node_id] is not gen.I:
                raise ValueError('Generator order is wrong or generator is stale after reset')
            lanes = tuple(lane for road in gen.lanes for lane in road)
            if not lanes or len(set(lanes)) != len(lanes):
                raise ValueError('Expected distinct incoming observation lanes')
            for lane_id in lanes:
                if lane_id not in self.catalog.lane_ids:
                    raise ValueError(f'Observation lane absent from catalog: {lane_id}')
                lane = self.catalog.lane(lane_id)
                if lane['to_junction'] != node_id or lane['internal']:
                    raise ValueError('Only external incoming lane observations are supported')
            # LaneVehicleGenerator pads two or three scalar values to four.
            size = 4 if len(lanes) in (2, 3) else len(lanes)
            expected_declared = 4 if len(lanes) == 3 else len(lanes)
            if gen.ob_length != expected_declared:
                raise ValueError('Generator feature length disagrees with actual lane layout')
            rows.append((node_id, lanes, size))
        return tuple(rows)

    def bind(self, reports):
        """Build read-only masks; no state or report history is retained."""
        if self._read_layout() != self._layout:
            raise ValueError('Observation layout changed; rebuild the binder')
        reports = tuple(reports)
        if any(r.mapping_status != 'mapped' for r in reports):
            raise ValueError('Resolve or explicitly handle failed reports before policy binding')
        if len({r.report_id for r in reports}) != len(reports):
            raise ValueError('Duplicate report IDs in snapshot')
        lane_ids = self.catalog.lane_ids
        lane_index = {lane: i for i, lane in enumerate(lane_ids)}
        lane_mask = np.zeros((len(lane_ids), len(reports)), dtype=bool)
        feature_mask = np.zeros((len(self.intersection_ids), max(self.feature_sizes), len(reports)), dtype=bool)
        valid = np.zeros(feature_mask.shape[:2], dtype=bool)
        for i, (_, lanes, _) in enumerate(self._layout):
            valid[i, :len(lanes)] = True
        bound = []
        for j, report in enumerate(reports):
            bindings, unobserved = [], []
            for lane in report.target_lane_ids:
                lane_mask[lane_index[lane], j] = True
                current = self._bindings.get(lane, ())
                bindings.extend(current)
                if not current:
                    unobserved.append(lane)
                for binding in current:
                    feature_mask[binding.intersection_index, binding.feature_index, j] = True
            bound.append(BoundReport(report, tuple(bindings), tuple(unobserved)))
        for array in (lane_mask, feature_mask, valid):
            array.setflags(write=False)
        return GroundingBatch(tuple(bound), lane_ids, self.intersection_ids, self.feature_sizes,
                              lane_mask, feature_mask, valid)
