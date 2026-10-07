"""Contract tests for per-episode event plans (sumo-episode-plan-v1)."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

sumolib = pytest.importorskip('sumolib')
yaml = pytest.importorskip('yaml')

from world.sumo_events import (EpisodePlan, Event, Schedule, SumoEventRuntime,
                               install_event_plan, load_plan)
from world.sumo_events.plans import PLAN_SCHEMA, PlanController, _schedule_sha256

ROOT = Path(__file__).resolve().parents[1]
NET = ROOT / 'data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml'
PLAN_FILE = ROOT / 'configs/events/plans/hz4x4_random_v1.yml'


def event(kind='lane_blockage', lane='road_1_1_0_2'):
    if kind == 'lane_blockage':
        return Event(event_id='e1', kind=kind, begin=600, end=800,
                     lane_id=lane, position=50.0, position_tolerance=20.0)
    if kind == 'road_closure':
        return Event(event_id='e1', kind=kind, begin=600, end=800,
                     edge_id='road_1_1_0')
    return Event(event_id='e1', kind=kind, begin=600, end=800, speed_factor=0.8)


def small_plan():
    return EpisodePlan('test-plan', [Schedule(()), Schedule((event(),))],
                       Schedule((event('global_rain'),)))


def test_load_plan_roundtrip_and_hashes(tmp_path):
    plan = small_plan()
    path = tmp_path / 'plan.yml'
    path.write_text(yaml.safe_dump(plan.to_dict(), sort_keys=False))
    loaded = load_plan(path)
    assert loaded.plan_id == 'test-plan'
    assert len(loaded.episodes) == 2
    assert loaded.plan_sha256 == plan.plan_sha256
    assert loaded.episode_sha256 == plan.episode_sha256
    assert loaded.eval_schedule.events[0].kind == 'global_rain'


@pytest.mark.parametrize('mutate', [
    lambda d: d.update(schema_version='bogus'),
    lambda d: d['episodes'][1].update(episode=5),
    lambda d: d.pop('eval'),
])
def test_load_plan_rejects_malformed(tmp_path, mutate):
    data = small_plan().to_dict()
    for i, row in enumerate(data['episodes']):
        row['episode'] = i
    mutate(data)
    path = tmp_path / 'bad.yml'
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError):
        load_plan(path)


def test_set_schedule_rejects_while_bound_accepts_after_close():
    runtime = SumoEventRuntime(NET, Schedule(()))
    replacement = Schedule((event(),))
    runtime._engine = object()  # simulate a bound episode
    with pytest.raises(RuntimeError):
        runtime.set_schedule(replacement)
    runtime._engine = None  # unbound state after close()
    runtime.set_schedule(replacement)
    assert runtime.schedule == replacement
    assert runtime.schedule_sha256 == _schedule_sha256(replacement)


class _StubWorld:
    def __init__(self):
        self.reset_calls = 0
        self.close_calls = 0
        self._connection_open = False

    def reset(self):
        self.reset_calls += 1
        return 'obs'

    def close(self):
        self.close_calls += 1


class _StubRuntime:
    def __init__(self):
        self.schedule = None
        self.schedule_sha256 = ''

    def set_schedule(self, schedule):
        self.schedule = schedule
        self.schedule_sha256 = _schedule_sha256(schedule)


def test_plan_controller_requires_selection_and_swaps(tmp_path):
    world, runtime, plan = _StubWorld(), _StubRuntime(), small_plan()
    controller = PlanController(world, plan, runtime, tmp_path / 'log.jsonl')
    with pytest.raises(RuntimeError):
        world.reset()
    controller.select_train(1)
    assert world.reset() == 'obs'
    assert runtime.schedule is plan.episodes[1]
    assert controller.last_selection['episode'] == 1
    assert controller.last_selection['schedule_sha256'] == plan.episode_sha256[1]
    with pytest.raises(RuntimeError):
        world.reset()
    controller.select_eval()
    world.reset()
    assert runtime.schedule is plan.eval_schedule
    assert controller.last_selection['role'] == 'eval'
    lines = (tmp_path / 'log.jsonl').read_text().strip().split('\n')
    assert len(lines) == 2
    row = json.loads(lines[0])
    assert row['plan_sha256'] == plan.plan_sha256
    assert row['events'][0]['kind'] == 'lane_blockage'


def test_committed_plan_is_valid_and_matches_protocol():
    plan = load_plan(PLAN_FILE)
    assert len(plan.episodes) == 200
    labels = {'empty': 0, 'single': 0, 'multi': 0}
    for schedule in plan.episodes:
        assert len(schedule.events) <= 2
        ids = [e.event_id for e in schedule.events]
        assert len(set(ids)) == len(ids)
        if not schedule.events:
            labels['empty'] += 1
        elif len(schedule.events) == 1:
            labels['single'] += 1
        else:
            labels['multi'] += 1
        for e in schedule.events:
            assert 600 <= e.begin <= 1800
            assert e.end <= 3000  # horizon 3600 minus >=600s recovery
            assert e.end - e.begin >= 100
            if e.kind == 'lane_blockage':
                assert e.lane_id and e.position >= 5
            elif e.kind == 'road_closure':
                assert e.edge_id and not e.edge_id.startswith(':')
            else:
                assert 0 < e.speed_factor < 1
    # Roughly 30/40/30 by construction; allow generous sampling slack.
    assert 30 <= labels['empty'] <= 90
    assert 60 <= labels['single'] <= 110
    assert 40 <= labels['multi'] <= 90


def test_generator_is_deterministic():
    from tools.build_random_event_plan import build_plan
    args = SimpleNamespace(
        net=str(NET), eval=str(ROOT / 'configs/events/hz4x4_comparison.yml'),
        episodes=8, seed=7, plan_id='det-test', horizon=3600, recovery=600,
        normal=0.3, single=0.4, begin_min=600, begin_max=1800,
        dur_min=180, dur_max=600, rain_min=0.7, rain_max=0.9,
        position_min_ratio=0.25, position_tolerance=20.0)
    first, _ = build_plan(args)
    second, _ = build_plan(args)
    args.seed = 8
    third, _ = build_plan(args)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert json.dumps(first, sort_keys=True) != json.dumps(third, sort_keys=True)


def test_episode_index_contract():
    plan = small_plan()
    assert plan.schedule_for(0) is plan.episodes[0]
    assert plan.schedule_for(1) is plan.episodes[1]
    with pytest.raises(IndexError):
        plan.schedule_for(2)
    with pytest.raises((IndexError, TypeError)):
        plan.schedule_for(-1)
