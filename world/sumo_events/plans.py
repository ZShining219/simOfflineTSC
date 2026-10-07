"""Frozen per-episode event plans over the fixed-schedule event adapter.

A plan is an explicit list of immutable Schedules addressed by episode index.
Unlike ``episodes.EpisodeEvents`` (which owns a full physical-gate/timeline
contract for the paper-experiment trainer), this adapter is intentionally
thin: the trainer declares ``select_train(episode)`` or ``select_eval()``
before each ``world.reset()``, and the wrapped reset swaps the selected
schedule onto the single installed runtime before ``bind``.  Selection is
explicit and index-addressed, so evaluation resets never consume training
episodes and crash-resume stays aligned by absolute episode number.
"""
import hashlib
import json
from pathlib import Path

import yaml

from .integration import install_events
from .schema import Event, Schedule


PLAN_SCHEMA = 'sumo-episode-plan-v1'


def _schedule_sha256(schedule):
    return hashlib.sha256(
        json.dumps(schedule.to_dict(), sort_keys=True).encode()).hexdigest()


def _row_schedule(row):
    if not isinstance(row, dict) or 'events' not in row:
        raise ValueError('Each plan row requires an events list')
    events = row['events']
    if not isinstance(events, list):
        raise ValueError('Plan row events must be a list')
    return Schedule(tuple(Event(**e) for e in events))


class EpisodePlan:
    """Immutable plan: per-episode schedules plus one fixed eval schedule."""

    def __init__(self, plan_id, episodes, eval_schedule):
        self.plan_id = str(plan_id)
        self.episodes = tuple(episodes)
        self.eval_schedule = eval_schedule
        if not all(isinstance(item, Schedule) for item in self.episodes):
            raise TypeError('episodes must contain Schedule objects')
        if not isinstance(eval_schedule, Schedule):
            raise TypeError('eval_schedule must be a Schedule')
        self.episode_sha256 = tuple(_schedule_sha256(item) for item in self.episodes)
        payload = {'schema_version': PLAN_SCHEMA, 'plan_id': self.plan_id,
                   'episodes': [item.to_dict() for item in self.episodes],
                   'eval': self.eval_schedule.to_dict()}
        self.plan_sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def schedule_for(self, episode):
        if not isinstance(episode, int) or isinstance(episode, bool):
            raise TypeError('episode index must be an int')
        if episode < 0 or episode >= len(self.episodes):
            raise IndexError(f'Episode {episode} outside plan of {len(self.episodes)}')
        return self.episodes[episode]

    def to_dict(self):
        return {'schema_version': PLAN_SCHEMA, 'plan_id': self.plan_id,
                'episodes': [item.to_dict() for item in self.episodes],
                'eval': self.eval_schedule.to_dict()}


def load_plan(path):
    """Load a strict YAML/JSON episode plan. No RNG or simulation is invoked."""
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict) or data.get('schema_version') != PLAN_SCHEMA:
        raise ValueError(f'Expected schema_version: {PLAN_SCHEMA}')
    for key in ('plan_id', 'episodes', 'eval'):
        if key not in data:
            raise ValueError(f'Episode plan requires {key}')
    rows = data['episodes']
    if not isinstance(rows, list) or not rows:
        raise ValueError('episodes must be a nonempty list')
    schedules = []
    for index, row in enumerate(rows):
        if row.get('episode', index) != index:
            raise ValueError('Plan rows must be contiguous episodes starting at 0')
        schedules.append(_row_schedule(row))
    eval_part = data['eval']
    if isinstance(eval_part, dict) and 'events' in eval_part:
        eval_schedule = _row_schedule(eval_part)
    else:
        raise ValueError('eval must be a mapping with an events list')
    return EpisodePlan(data['plan_id'], schedules, eval_schedule)


class PlanController:
    """Selects which frozen schedule each ``world.reset()`` binds.

    ``select_train(episode)`` arms the indexed plan row; ``select_eval()``
    arms the fixed eval schedule.  A reset without a pending selection fails
    loudly so no code path can silently consume or replay the wrong episode.
    """

    def __init__(self, world, plan, runtime, log_path=None):
        self.world = world
        self.plan = plan
        self.runtime = runtime
        self.log_path = None if log_path is None else Path(log_path)
        self._pending = None
        self.last_selection = None
        base_reset = world.reset

        def reset(*args, **kwargs):
            if self._pending is None:
                raise RuntimeError(
                    'Episode plan requires select_train/select_eval before reset')
            role, schedule, episode = self._pending
            self._pending = None
            # Close first: the wrapped close detaches the previous engine so
            # the runtime is unbound and legally accepts the next schedule.
            world.close()
            runtime.set_schedule(schedule)
            result = base_reset(*args, **kwargs)
            self.last_selection = {'role': role, 'episode': episode,
                               'schedule_sha256': runtime.schedule_sha256}
            self._log(self.last_selection, schedule)
            return result

        world.reset = reset

    def select_train(self, episode):
        self._pending = ('train', self.plan.schedule_for(episode), episode)

    def select_eval(self):
        self._pending = ('eval', self.plan.eval_schedule, None)

    def _log(self, selection, schedule):
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        row = dict(selection)
        row['plan_sha256'] = self.plan.plan_sha256
        row['plan_id'] = self.plan.plan_id
        row['events'] = [
            {'event_id': e.event_id, 'kind': e.kind, 'begin': e.begin, 'end': e.end}
            for e in schedule.events]
        with self.log_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(row, sort_keys=True) + '\n')


def install_event_plan(world, plan, log_path=None):
    """Attach a frozen episode plan to a closed World before its first reset.

    The existing ``install_events`` machinery keeps ownership of the wrapped
    ``reset/step_sim/close`` chain and the traffic-filtered connection view;
    this layer only swaps the runtime's schedule inside that wrapped reset.
    """
    if getattr(world, '_connection_open', False):
        raise ValueError('Install the episode plan on a closed World before reset')
    if getattr(world, '_sumo_event_runtime', None) is not None:
        raise ValueError('An event runtime is already installed')
    runtime = install_events(world, plan.eval_schedule)
    controller = PlanController(world, plan, runtime, log_path)
    world._sumo_event_plan = controller
    return controller
