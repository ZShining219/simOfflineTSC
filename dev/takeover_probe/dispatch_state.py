"""Featurization for the scene-guided dispatch router.

Lane-level state tokens + structured event-report fields -> fixed-size
numpy inputs for ``dispatch_model``.  Shared by training, evaluation and
the live-broadcast adapter so all three see identical features.

Text channel = structured event report (broadcast-delayed), rendered to a
Chinese sentence for logging/UI while the model reads its fields:

    {kind, edge, active, recovering, onset_age_s, severity,
     est_remaining_s}

Feature layouts (``layout`` arg, recorded in the router ckpt meta):

  'identity' (default): lane tokens carry their edge identity as a
    one-hot over edges; the report's ``edge`` field is encoded as a
    one-hot in the SAME edge index space.  Binding report->lanes then
    reduces to an identity match that ``SceneAttention`` can score
    directly, so the text channel is the only source of event location.
  'flag' (legacy): lane tokens append a report-derived ``event_flag``
    and the report's edge field is a per-lane indicator — the channel
    duplication that made text redundant in the obj3 experiments.

``include_event_flag`` re-appends the flag under 'identity' for the
redundant-channel ablation; under 'flag' it is always on (it IS the
layout).  The ``edge`` report field binds to lane tokens through the
identity block — that binding is the aux-alignment target for the
attention.
"""

from __future__ import annotations

import numpy as np

from tools.sumo_live_broadcast.injects import (
    EVENT_KIND_ALIAS,
    EVENT_KINDS,
    active_event,
    event_specs,
)

LANE_PHYS = 3           # queue, count, wait_time_count
LANE_FEATS = 4          # legacy 'flag' layout: LANE_PHYS + event_flag
TEXT_FIELDS = None      # computed from n_lanes/n_edges at build time
SCALARS = 6             # active, recovering, onset_age, severity,
                        # est_remaining, objective


def fb_kwargs_from_meta(meta):
    """FeatureBuilder kwargs matching a router checkpoint's saved meta.

    Routers trained before the identity layout carry no feature keys and
    resolve to the legacy 'flag' layout their dims were built with.
    """
    meta = meta or {}
    kw = {'layout': meta.get('feature_layout', 'flag')}
    if kw['layout'] == 'identity':
        kw['include_event_flag'] = bool(meta.get('event_flag', False))
    kw['phys'] = meta.get('phys', 'full')
    kw['include_forecast'] = bool(meta.get('include_forecast', True))
    return kw


class FeatureBuilder:
    """World+event -> model inputs.  Lane/edge order fixed at construction."""

    def __init__(self, world, injects, report_delay=30.0,
                 layout='identity', include_event_flag=None,
                 phys='full', include_forecast=True):
        self.world = world
        self.events = event_specs(injects)
        self.report_delay = float(report_delay)
        # 'queue' restricts the lane physical channel to queue counts —
        # the conventional-observation baseline level (what a plain DQN
        # sees); 'full' keeps queue+count+wait for the scene module.
        self.phys = str(phys)
        if self.phys not in ('full', 'queue'):
            raise ValueError(f'unknown phys mode {phys!r}')
        self.n_phys = LANE_PHYS if self.phys == 'full' else 1
        # forecast fields (est_remaining_s) are supervision targets, not
        # report content — a reporter cannot know the future
        self.include_forecast = bool(include_forecast)
        self.lanes = sorted(world.get_info('lane_waiting_count') or {})
        self.edge_to_laneidx = {}
        for idx, lane in enumerate(self.lanes):
            edge = lane.rsplit('_', 1)[0]
            self.edge_to_laneidx.setdefault(edge, []).append(idx)
        self.n_lanes = len(self.lanes)
        self.n_kinds = len(EVENT_KINDS)
        # shared identity space: edge index = position in sorted edges
        self.edges = sorted(self.edge_to_laneidx)
        self.edge_to_idx = {e: i for i, e in enumerate(self.edges)}
        self.n_edges = len(self.edges)
        self.lane_edge_idx = np.array(
            [self.edge_to_idx[l.rsplit('_', 1)[0]] for l in self.lanes])
        self.layout = layout
        self.lane_id_slice = None
        self.text_id_slice = None
        if layout == 'flag':
            self.include_event_flag = True
            self.lane_dim = self.n_phys + 1
            self._text_edge_dim = self.n_lanes   # per-lane indicator
        elif layout == 'identity':
            self.include_event_flag = bool(include_event_flag)
            self.lane_dim = (self.n_phys + self.n_edges
                             + (1 if self.include_event_flag else 0))
            self.lane_id_slice = slice(self.n_phys,
                                       self.n_phys + self.n_edges)
            self.text_id_slice = slice(self.n_kinds,
                                       self.n_kinds + self.n_edges)
            self._text_edge_dim = self.n_edges   # per-edge indicator
        else:
            raise ValueError(f'unknown feature layout {layout!r}')
        # text vec: kind onehot + edge indicator block + scalars
        self.text_dim = self.n_kinds + self._text_edge_dim + SCALARS
        self.global_dim = 8 + 3     # phase onehot + t_norm + totals
        self.active_edge_idx = None

    def model_geom(self):
        """Shared identity-subspace descriptor for SceneAttention binding.

        None under the legacy 'flag' layout (no identity dims to match).
        """
        if self.layout != 'identity':
            return None
        return {'phys_dim': self.n_phys,
                'lane_id': [self.lane_id_slice.start,
                            self.lane_id_slice.stop],
                'text_id': [self.text_id_slice.start,
                            self.text_id_slice.stop]}

    def feature_meta(self):
        """Ckpt-meta fragment recording how inputs were featurized."""
        return {'feature_layout': self.layout,
                'event_flag': self.include_event_flag,
                'phys': self.phys,
                'include_forecast': self.include_forecast,
                'geom': self.model_geom()}

    def event_fields(self, now):
        fields, sentence = active_event(
            self.events, now, report_delay=self.report_delay)
        return fields, sentence

    def _report_fields(self, now, override=None):
        """Event fields as reported to the model (ablation overrides applied).

        Under 'identity' the resolved fields drive ONLY the text vector —
        the lane tokens carry no report-derived flag, so ablation overrides
        move text location without touching state.  Under 'flag' the same
        fields drive both the text vector and the lane-token event flag
        (the legacy channel duplication).
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
        tokens = np.zeros((self.n_lanes, self.lane_dim), dtype=np.float32)
        for idx, lane in enumerate(self.lanes):
            tokens[idx, 0] = queue.get(lane, 0) / 10.0
            if self.n_phys > 1:
                tokens[idx, 1] = count.get(lane, 0) / 10.0
                tokens[idx, 2] = wait.get(lane, 0) / 100.0
        if self.layout == 'identity':
            tokens[np.arange(self.n_lanes),
                   self.n_phys + self.lane_edge_idx] = 1.0
            flag_col = self.lane_dim - 1 if self.include_event_flag \
                else None
        else:
            flag_col = self.n_phys
        if flag_col is not None:
            for idx in flagged:
                tokens[idx, flag_col] = 1.0
        self.active_edge_idx = sorted(flagged)
        return tokens

    def text_vec(self, now, override=None):
        """override: dict-like mutations for ablation
        ({'kind':'none'} no-text, {'edge': 'road_x'} wrong-location).
        """
        fields = self._report_fields(now, override)
        vec = np.zeros(self.text_dim, dtype=np.float32)
        kind = fields.get('kind', 'none')
        kind = EVENT_KIND_ALIAS.get(kind, kind)
        if kind in EVENT_KINDS:
            vec[EVENT_KINDS.index(kind)] = 1.0
        off = self.n_kinds
        edge = fields.get('edge')
        if edge:
            if self.layout == 'identity':
                eidx = self.edge_to_idx.get(edge)
                if eidx is not None:
                    vec[off + eidx] = 1.0
            else:
                for idx in self.edge_to_laneidx.get(edge, []):
                    vec[off + idx] = 1.0
        off += self._text_edge_dim
        vec[off + 0] = float(fields.get('active')) * 1.0
        vec[off + 1] = float(fields.get('recovering')) * 1.0
        vec[off + 2] = min(fields.get('onset_age_s', 0.0), 3600) / 3600.0
        vec[off + 3] = fields.get('severity', 0) / 3.0
        if self.include_forecast:
            vec[off + 4] = min(fields.get('est_remaining_s', 0.0),
                               3600) / 3600.0
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
