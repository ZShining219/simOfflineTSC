"""Featurization for the scene-guided dispatch router.

Lane-level state tokens + structured event-report fields -> fixed-size
numpy inputs for ``dispatch_model``.  Shared by training, evaluation and
the live-broadcast adapter so all three see identical features.

Text channel = structured event report (broadcast-delayed), rendered to a
Chinese sentence for logging/UI while the model reads its fields:

    {kind, edge, active, recovering, onset_age_s, severity,
     est_remaining_s}

The ``edge`` field binds to lane tokens via an indicator over the edge's
lanes — that binding is the aux-alignment target for the attention.
"""

from __future__ import annotations

import numpy as np

from tools.sumo_live_broadcast.injects import (
    EVENT_KINDS,
    active_event,
    event_specs,
)

LANE_FEATS = 4          # queue, count, wait_time_count, event_flag
TEXT_FIELDS = None      # computed from n_lanes at build time


class FeatureBuilder:
    """World+event -> model inputs.  Lane order fixed at construction."""

    def __init__(self, world, injects, report_delay=30.0):
        self.world = world
        self.events = event_specs(injects)
        self.report_delay = float(report_delay)
        self.lanes = sorted(world.get_info('lane_waiting_count') or {})
        self.edge_to_laneidx = {}
        for idx, lane in enumerate(self.lanes):
            edge = lane.rsplit('_', 1)[0]
            self.edge_to_laneidx.setdefault(edge, []).append(idx)
        self.n_lanes = len(self.lanes)
        self.n_kinds = len(EVENT_KINDS)
        # text vec: kind onehot + lane indicator + scalars(+objective)
        self.text_dim = self.n_kinds + self.n_lanes + 6
        self.global_dim = 8 + 3     # phase onehot + t_norm + totals
        self.active_edge_idx = None

    def event_fields(self, now):
        fields, sentence = active_event(
            self.events, now, report_delay=self.report_delay)
        return fields, sentence

    def _report_fields(self, now, override=None):
        """Event fields as reported to the model (ablation overrides applied).

        The same resolved fields drive both the text vector and the lane-token
        event flag, so a 'no_text' report removes report-derived location info
        everywhere and a 'wrong_location' report moves the flag consistently.
        """
        fields, _ = self.event_fields(now)
        if override:
            fields = dict(fields)
            fields.update(override)
            if override.get('kind') == 'none':
                fields.update({
                    'active': False, 'recovering': False, 'severity': 0,
                    'est_remaining_s': 0.0, 'onset_age_s': 0.0,
                    'edge': '', 'objective': 'global',
                })
        return fields

    def lane_tokens(self, now, override=None):
        queue = self.world.get_info('lane_waiting_count') or {}
        count = self.world.get_info('lane_count') or {}
        wait = self.world.get_info('lane_waiting_time_count') or {}
        fields = self._report_fields(now, override)
        flagged = set()
        if fields.get('edge') and fields.get('active') is not False:
            flagged = set(self.edge_to_laneidx.get(fields['edge'], []))
        tokens = np.zeros((self.n_lanes, LANE_FEATS), dtype=np.float32)
        for idx, lane in enumerate(self.lanes):
            tokens[idx, 0] = queue.get(lane, 0) / 10.0
            tokens[idx, 1] = count.get(lane, 0) / 10.0
            tokens[idx, 2] = wait.get(lane, 0) / 100.0
            tokens[idx, 3] = 1.0 if idx in flagged else 0.0
        self.active_edge_idx = sorted(flagged)
        return tokens

    def text_vec(self, now, override=None):
        """override: dict-like mutations for ablation
        ({'kind':'none'} no-text, {'edge': 'road_x'} wrong-location).
        """
        fields = self._report_fields(now, override)
        vec = np.zeros(self.text_dim, dtype=np.float32)
        kind = fields.get('kind', 'none')
        if kind in EVENT_KINDS:
            vec[EVENT_KINDS.index(kind)] = 1.0
        off = self.n_kinds
        edge = fields.get('edge')
        if edge:
            for idx in self.edge_to_laneidx.get(edge, []):
                vec[off + idx] = 1.0
        off += self.n_lanes
        vec[off + 0] = float(fields.get('active')) * 1.0
        vec[off + 1] = float(fields.get('recovering')) * 1.0
        vec[off + 2] = min(fields.get('onset_age_s', 0.0), 3600) / 3600.0
        vec[off + 3] = fields.get('severity', 0) / 3.0
        vec[off + 4] = min(fields.get('est_remaining_s', 0.0), 3600) / 3600.0
        vec[off + 5] = 1.0 if fields.get('objective') == 'focus' else 0.0
        return vec

    def global_vec(self, now):
        phase = 0
        try:
            inter = self.world.intersections[0]
            phase = int(inter.virtual_phase) % 8
        except Exception:
            pass
        queue = self.world.get_info('lane_waiting_count') or {}
        vec = np.zeros(self.global_dim, dtype=np.float32)
        vec[phase] = 1.0
        vec[8] = min(now, 3600.0) / 3600.0
        vec[9] = sum(queue.values()) / 50.0
        vec[10] = (max(queue.values()) if queue else 0) / 25.0
        return vec

    def snapshot(self, now, text_override=None):
        """One router-input sample at a decision boundary.

        ``event_lane_idx`` is the TRUE event-lane set (aux-loss target and
        eval reference); it ignores ablation overrides that perturb only what
        the model sees.
        """
        lanes = self.lane_tokens(now, override=text_override)
        true_fields = self._report_fields(now, None)
        true_flagged = set()
        if true_fields.get('edge') and true_fields.get('active') is not False:
            true_flagged = set(self.edge_to_laneidx.get(true_fields['edge'], []))
        return {
            'lanes': lanes,
            'text': self.text_vec(now, override=text_override),
            'global': self.global_vec(now),
            'event_lane_idx': sorted(true_flagged),
        }
