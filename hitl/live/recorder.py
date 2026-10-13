"""Per-decision transition recorder for live sessions (HITL data plane).

Opt-in instrumentation for ``BroadcastEngine``.  When attached, every
decision interval of a live session is recorded as one transition row per
controlled junction — the same per-decision shape the sequential trainer's
``trajectory_sink`` produces — plus proposal/intervention audit rows.

Session layout (created lazily on first write)::

    <root>/<session_id>/
        manifest.json        session + feature-schema provenance
        transitions.jsonl    executed transitions (state -> next_state)
        proposals.jsonl      per-decision proposals vs accepted actions
        interventions.jsonl  intervention event stream (pins/releases/
                             gate calls once the HITL gate exists)

Transitions pair the *executed* action with the reward it produced; the
student/expert/manual candidates stay in ``hitl`` provenance fields and in
``proposals.jsonl``.  Observation/reward features reuse the same generator
recipe as the DQN training/inference path so recorded rows are directly
consumable by the sequential replay tooling (NPZ packaging happens
downstream, e.g. under ``hitl/live/``).
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

STATE_DESCRIPTOR = 'LaneVehicleGenerator(lane_count,in_only=True,average=None)'
PHASE_DESCRIPTOR = (
    "IntersectionPhaseGenerator('phase')->cur_phase "
    '(program index; green phases occupy 0..action_dim-1, appended yellows '
    'may exceed action_dim-1 mid-transition)'
)
REWARD_DESCRIPTOR = (
    'mean over interval steps of '
    'LaneVehicleGenerator(lane_waiting_count,in_only=True,average=all,'
    'negative=True) * 12'
)


def _rank_generators(world, rank):
    """Build the observation/phase/reward generators for one junction.

    Same recipe as ``sequential.agent`` / ``controllers._bind_generators``
    so recorded features match the training-time schema bit-for-bit.
    Imported lazily so this module stays importable without SUMO deps.
    """
    from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

    inter_id = world.intersection_ids[rank]
    inter = world.id2intersection[inter_id]
    return {
        'junction': inter_id,
        'ob': LaneVehicleGenerator(
            world, inter, ['lane_count'], in_only=True, average=None,
        ),
        'phase': IntersectionPhaseGenerator(
            world, inter, ['phase'], targets=['cur_phase'], negative=False,
        ),
        'reward': LaneVehicleGenerator(
            world, inter, ['lane_waiting_count'],
            in_only=True, average='all', negative=True,
        ),
        'queue': LaneVehicleGenerator(
            world, inter, ['lane_waiting_count'],
            in_only=True, average=None, negative=False,
        ),
        'delay': LaneVehicleGenerator(
            world, inter, ['lane_delay'],
            in_only=True, average='all', negative=False,
        ),
    }


def _jsonify(value):
    """Convert numpy scalars/arrays and Paths into plain JSON types."""
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonify(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


class HitlRecorder:
    """Session recorder bound to a live ``BroadcastEngine``.

    Engine-facing hooks (all no-ops before ``begin_episode``):

    * ``bind(world, ranks)`` — (re)build generators; required after
      ``world.reset()`` since Intersection objects are recreated.
    * ``begin_episode(meta=None)`` — open a new episode (reset/restart).
    * ``on_decision(now, actions, ...)`` — at a committed decision
      boundary: closes the previous interval as a transition and opens a
      new one for each rank present in ``actions``.
    * ``on_step(now=None)`` — after each ``world.step``; accumulates the
      per-step reward of open intervals.
    * ``on_intervention(kind, ...)`` — manual pin/release (and later
      gate callout/answer/timeout) events.
    * ``end_episode(reason)`` — closes dangling intervals as truncated.
    * ``close()`` — flushes manifest + file handles.
    """

    def __init__(self, root, meta=None, generator_factory=None):
        self.root = Path(root).expanduser()
        self.meta = dict(meta or {})
        self._factory = generator_factory or _rank_generators
        self.session_id = self.meta.get('session_id') or (
            time.strftime('%Y%m%d_%H%M%S')
            + '_' + str(self.meta.get('scene', 'session'))
        )
        self.session_dir = self.root / self.session_id
        self._dir_ready = False
        self._handles = {}
        self._world = None
        self._gens = {}
        self._action_dims = {}
        self._open = {}          # rank -> open interval dict
        self.episode_id = 0
        self._decision_step = 0  # per episode
        self._global_step = 0    # across the whole session
        self._episodes = []
        self._episode_meta = None
        self._counts = Counter()
        self._closed = False

    # ------------------------------------------------------------- internals

    def _ensure_dir(self):
        if not self._dir_ready:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._dir_ready = True

    def _write_row(self, name, row):
        self._ensure_dir()
        handle = self._handles.get(name)
        if handle is None:
            handle = open(
                self.session_dir / f'{name}.jsonl', 'a', encoding='utf-8')
            self._handles[name] = handle
        handle.write(json.dumps(_jsonify(row)) + '\n')
        handle.flush()
        self._counts[name] += 1

    def _ob(self, rank):
        return np.asarray(
            self._gens[rank]['ob'].generate(), dtype=np.float32
        ).reshape(-1)

    def _phase(self, rank):
        return int(np.asarray(
            self._gens[rank]['phase'].generate(), dtype=np.int64
        ).reshape(-1)[0])

    def _reward_step(self, rank):
        return float(np.squeeze(np.asarray(
            self._gens[rank]['reward'].generate()))) * 12.0

    def _metrics(self):
        """Point-in-time network/junction metrics at interval close.

        Mirrors what ``tsc_trainer`` writes into Plan 1 trajectory rows at
        a decision boundary: queue/approximate_delay from per-intersection
        generators, real_delay/throughput/waiting_time from the world.
        Every field is independently guarded — a torn-down world yields
        ``None`` instead of killing the recording.
        """
        world = self._world

        def _sample(callable_):
            try:
                return callable_()
            except Exception:
                return None

        junction_metrics = {}
        for rank, gens in self._gens.items():
            junction_metrics[gens['junction']] = {
                'queue': _sample(
                    lambda: float(np.sum(gens['queue'].generate()))),
                'queue_by_lane': _sample(
                    lambda: np.asarray(
                        gens['queue'].generate()).reshape(-1).tolist()),
                'approximate_delay': _sample(
                    lambda: float(np.mean(gens['delay'].generate()))),
            }
        return {
            'junctions': junction_metrics,
            'real_delay': _sample(
                lambda: float(world.get_real_delay()) if world else None),
            'throughput': _sample(
                lambda: float(world.get_cur_throughput()) if world else None),
            'waiting_time': _sample(
                lambda: float(world.get_average_waiting_time())
                if world else None),
        }

    def _flush_manifest(self):
        if not self._dir_ready:
            return
        manifest = {
            'schema_version': SCHEMA_VERSION,
            'session_id': self.session_id,
            'created_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
            **self.meta,
            'controlled_ranks': sorted(self._gens),
            'junctions': {
                str(rank): gens['junction']
                for rank, gens in sorted(self._gens.items())
            },
            'action_dims': dict(self._action_dims),
            'feature_schema': {
                'state': STATE_DESCRIPTOR,
                'phase': PHASE_DESCRIPTOR,
                'reward': REWARD_DESCRIPTOR,
            },
            'episodes': list(self._episodes),
            'counts': dict(self._counts),
        }
        path = self.session_dir / 'manifest.json'
        tmp = path.with_suffix('.json.tmp')
        tmp.write_text(
            json.dumps(_jsonify(manifest), indent=2), encoding='utf-8')
        tmp.replace(path)

    # ------------------------------------------------------------------ API

    def bind(self, world, ranks):
        """(Re)build per-rank generators; call whenever the world's
        Intersection objects are (re)created (session start, reset)."""
        self._world = world
        self._gens = {}
        self._action_dims = {}
        for rank in ranks:
            self._gens[rank] = self._factory(world, rank)
            inter = world.id2intersection[world.intersection_ids[rank]]
            self._action_dims[str(rank)] = len(getattr(inter, 'phases', []))
        self._open = {}
        return self

    def begin_episode(self, meta=None):
        if self._open:
            self.end_episode('superseded')
        self.episode_id += 1
        self._decision_step = 0
        self._episode_meta = dict(meta or {})
        self._episodes.append({
            'episode_id': self.episode_id,
            **self._episode_meta,
            'decisions': 0,
            'transitions': 0,
            't0': None, 't1': None, 'end_reason': None,
        })

    def on_decision(self, now, actions, source=None, proposals=None,
                    pending=None, manual=None, expert=None, note=None,
                    errors=None):
        """Commit-time hook.

        ``now`` is sim time at the decision boundary; ``actions`` is the
        committed {rank: phase} vector.  ``proposals``/``pending``/
        ``manual`` mirror the engine's decision state for provenance.
        ``expert`` carries the expert seat outcome for this boundary:
        ``{'id': ..., 'actions': {rank: phase}, 'queries': {rank:
        reason}}`` — present only when the gate is armed and fired.
        """
        if self._closed or not self._gens:
            return
        if not self._episodes or self._episodes[-1]['end_reason'] is not None:
            self.begin_episode()
        episode = self._episodes[-1]
        now = float(now)
        if episode['t0'] is None:
            episode['t0'] = now
        proposals = dict(proposals or {})
        pending = dict(pending or {})
        manual = dict(manual or {})
        expert = dict(expert or {})
        expert_id = expert.get('id')
        expert_actions = dict(expert.get('actions') or {})
        expert_queries = dict(expert.get('queries') or {})

        # Close the previously open interval: the observation right now is
        # its next_state (episode state-chain stays contiguous).
        for rank, interval in self._open.items():
            row = {
                'schema_version': SCHEMA_VERSION,
                'episode_id': self.episode_id,
                'decision_step': interval['decision_step'],
                'global_step': interval['global_step'],
                'rank': rank,
                'junction': self._gens[rank]['junction'],
                't0': interval['t0'],
                't1': now,
                'state': interval['state'],
                'current_phase': interval['phase'],
                'action': interval['action'],
                'reward': (
                    interval['reward_sum'] / interval['reward_steps']
                    if interval['reward_steps'] else None
                ),
                'reward_steps': interval['reward_steps'],
                'reward_error': bool(interval.get('reward_error')),
                'next_state': self._ob(rank),
                'next_phase': self._phase(rank),
                'terminated': False,
                'truncated': False,
                'mid_interval_override': interval['mid_override'],
                'hitl': interval['hitl'],
                'metrics': self._metrics(),
            }
            self._write_row('transitions', row)
            episode['transitions'] += 1
        self._open = {}

        # Per-decision audit row: student proposals vs accepted actions.
        junction_of = {rank: self._gens[rank]['junction'] for rank in self._gens}
        self._write_row('proposals', {
            'episode_id': self.episode_id,
            'decision_step': self._decision_step,
            'global_step': self._global_step,
            't': now,
            'source': source,
            'expert_id': expert_id,
            'proposals': {
                junction_of.get(r, str(r)): int(p)
                for r, p in proposals.items()
            },
            'pending': {
                junction_of.get(r, str(r)): int(p)
                for r, p in pending.items()
            },
            'manual_overrides': {
                junction_of.get(r, str(r)): int(p)
                for r, p in manual.items()
            },
            'expert_actions': {
                junction_of.get(r, str(r)): int(p)
                for r, p in expert_actions.items()
            },
            'queries': {
                junction_of.get(r, str(r)): str(q)
                for r, q in expert_queries.items()
            },
            'accepted': {
                junction_of.get(r, str(r)): int(p)
                for r, p in actions.items()
            },
            'note': note,
            'errors': dict(errors) if errors else None,
        })

        # Each gate-fired query is an intervention-ledger event — even
        # when the expert's answer equals the student's proposal (the
        # budget was spent either way).
        for rank, reason in expert_queries.items():
            self._write_row('interventions', {
                'episode_id': self.episode_id,
                'decision_step': self._decision_step,
                'global_step': self._global_step,
                't': now,
                'kind': 'gate_query',
                'rank': rank,
                'junction': junction_of.get(rank, str(rank)),
                'phase': None,
                'detail': {
                    'reason': reason,
                    'expert_id': expert_id,
                    'expert_action': (
                        int(expert_actions[rank])
                        if expert_actions.get(rank) is not None else None
                    ),
                    'student_action': (
                        int(proposals[rank])
                        if proposals.get(rank) is not None
                        and rank not in manual
                        else None
                    ),
                    'accepted_action': (
                        int(actions[rank]) if rank in actions else None
                    ),
                    'intervened': bool(
                        rank in actions and (
                            rank in manual
                            or proposals.get(rank) != actions[rank]
                        )
                    ),
                },
            })

        # Open the new interval for every rank that got a committed action.
        for rank, action in actions.items():
            if rank not in self._gens:
                continue
            accepted = int(action)
            proposed = proposals.get(rank)
            pinned = manual.get(rank)
            queried = rank in expert_queries
            expert_action = expert_actions.get(rank)
            self._open[rank] = {
                't0': now,
                'state': self._ob(rank),
                'phase': self._phase(rank),
                'action': accepted,
                'reward_sum': 0.0,
                'reward_steps': 0,
                'mid_override': False,
                'decision_step': self._decision_step,
                'global_step': self._global_step,
                'hitl': {
                    'student_id': source,
                    'student_action': (
                        int(proposed)
                        if proposed is not None and rank not in manual
                        else None
                    ),
                    'queried': bool(queried),
                    'query_reason': expert_queries.get(rank),
                    'expert_id': (
                        expert_id
                        if queried or expert_action is not None
                        else 'operator' if rank in manual else None
                    ),
                    'expert_action': (
                        int(expert_action)
                        if expert_action is not None
                        else int(pinned) if pinned is not None else None
                    ),
                    'accepted_action': accepted,
                    'intervened': bool(
                        rank in manual
                        or (proposed is not None and proposed != accepted)
                    ),
                },
            }

        episode['decisions'] += 1
        self._decision_step += 1
        self._global_step += 1

    def on_step(self, now=None):
        """Reward accumulation; call once after every ``world.step``."""
        if self._closed:
            return
        for rank, interval in self._open.items():
            try:
                interval['reward_sum'] += self._reward_step(rank)
                interval['reward_steps'] += 1
            except Exception:
                # Reward unreadable this step (e.g. torn-down world):
                # keep accumulating what we can.
                interval['reward_error'] = True

    def on_intervention(self, kind, now=None, rank=None, phase=None,
                        detail=None, mark_interval=True):
        """Intervention event (manual pin/release today; gate callout /
        answer / timeout once the HITL gate lands).

        ``mark_interval=False`` for events applied at a decision boundary
        (e.g. an awaiting-resolve pin): they are recorded but did not
        alter an in-flight interval.
        """
        if self._closed:
            return
        junction = None
        if rank is not None and rank in self._gens:
            junction = self._gens[rank]['junction']
        self._write_row('interventions', {
            'episode_id': self.episode_id or None,
            't': float(now) if now is not None else None,
            'kind': str(kind),
            'rank': rank,
            'junction': junction,
            'phase': int(phase) if phase is not None else None,
            'detail': dict(detail) if detail else None,
        })
        if (mark_interval
                and kind in ('manual_pin', 'manual_release')
                and rank in self._open):
            self._open[rank]['mid_override'] = True

    def end_episode(self, reason):
        """Close dangling intervals as truncated and seal the episode."""
        if self._closed:
            return
        now = None
        try:
            if self._world is not None:
                now = float(self._world.get_current_time())
        except Exception:
            now = None
        for rank, interval in list(self._open.items()):
            try:
                next_state = self._ob(rank)
                next_phase = self._phase(rank)
            except Exception:
                next_state, next_phase = None, None
            row = {
                'schema_version': SCHEMA_VERSION,
                'episode_id': self.episode_id,
                'decision_step': interval['decision_step'],
                'global_step': interval['global_step'],
                'rank': rank,
                'junction': self._gens[rank]['junction'],
                't0': interval['t0'],
                't1': now,
                'state': interval['state'],
                'current_phase': interval['phase'],
                'action': interval['action'],
                'reward': (
                    interval['reward_sum'] / interval['reward_steps']
                    if interval['reward_steps'] else None
                ),
                'reward_steps': interval['reward_steps'],
                'reward_error': bool(interval.get('reward_error')),
                'next_state': next_state,
                'next_phase': next_phase,
                'terminated': False,
                'truncated': True,
                'mid_interval_override': interval['mid_override'],
                'hitl': interval['hitl'],
                'metrics': self._metrics(),
            }
            self._write_row('transitions', row)
            if self._episodes:
                self._episodes[-1]['transitions'] += 1
        self._open = {}
        if self._episodes and self._episodes[-1]['end_reason'] is None:
            self._episodes[-1]['t1'] = now
            self._episodes[-1]['end_reason'] = str(reason)
        self._flush_manifest()

    def close(self):
        if self._closed:
            return
        self.end_episode('closed')
        self._flush_manifest()
        for handle in self._handles.values():
            try:
                handle.close()
            except Exception:
                pass
        self._handles = {}
        self._closed = True

    # context-manager sugar for headless collectors
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
