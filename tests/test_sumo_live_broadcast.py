import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tools.sumo_live_broadcast.cli import (
    _parse_model_arg, build_controllers, build_parser,
)
from tools.sumo_live_broadcast.controllers import (
    ManualController, _state_dict_dims, build_builtin_controllers,
    build_snapshot_controller,
)
from tools.sumo_live_broadcast.page import build_page
from tools.sumo_live_broadcast.recorder import HitlRecorder


def _fake_world(virtual_phase=2, phase_count=4, junction_count=1):
    inters = [
        SimpleNamespace(
            id=f'J{i}', virtual_phase=virtual_phase,
            phases=list(range(phase_count)),
            green_phases=[], yellow_dict={}, yellow_phase_time=5,
        )
        for i in range(junction_count)
    ]
    ids = [inter.id for inter in inters]
    return SimpleNamespace(
        intersection_ids=ids,
        id2intersection=dict(zip(ids, inters)),
        intersections=inters,
    )


class _FakeWorld:
    """Minimal world for recorder tests: sim time is the only signal."""

    def __init__(self):
        self.t = 0.0
        self.intersection_ids = ['J0']
        self.id2intersection = {
            'J0': SimpleNamespace(
                phases=list(range(8)), current_phase=0),
        }

    def get_current_time(self):
        return self.t

    def step(self, actions=None):
        self.t += 1.0


class _FakeGen:
    def __init__(self, fn):
        self._fn = fn

    def generate(self):
        return self._fn()


def _fake_gens(world, rank):
    """Generator factory for tests: state/reward both expose sim time."""
    tl = world.intersection_ids[rank]
    inter = world.id2intersection[tl]
    return {
        'junction': tl,
        'ob': _FakeGen(
            lambda: np.full(4, float(world.t), dtype=np.float32)),
        'phase': _FakeGen(lambda: [int(inter.current_phase)]),
        'reward': _FakeGen(
            lambda: np.array([-float(world.t)], dtype=np.float32)),
    }


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]


class SumoLiveBroadcastTest(unittest.TestCase):
    def test_builtin_controller_menu_has_manual_last(self):
        controllers = build_builtin_controllers()
        ids = [c.id for c in controllers]
        self.assertEqual(ids[:3], ['fixedtime', 'maxpressure', 'sotl'])
        self.assertEqual(ids[-1], 'manual')

    def test_manual_controller_holds_phase_without_request(self):
        controller = ManualController().bind(_fake_world(virtual_phase=3))
        self.assertEqual(controller.decide(0), 3)
        controller.request_phase(0, 1)
        self.assertEqual(controller.decide(0), 1)

    def test_manual_controller_requests_are_per_junction(self):
        controller = ManualController().bind(
            _fake_world(virtual_phase=2, junction_count=3),
            ranks=[0, 1, 2],
        )
        controller.request_phase(1, 4)
        self.assertEqual(controller.decide(0), 2)
        self.assertEqual(controller.decide(1), 4)
        self.assertEqual(controller.decide(2), 2)
        controller.release(1)
        self.assertEqual(controller.decide(1), 2)
        controller.request_phase(0, 5)
        controller.request_phase(2, 6)
        controller.release()
        self.assertEqual(controller.decide(0), 2)
        self.assertEqual(controller.decide(2), 2)

    def test_parse_model_arg_requires_name_equals_path(self):
        name, path = _parse_model_arg('dhoa=/tmp/x.pt')
        self.assertEqual(name, 'dhoa')
        self.assertTrue(str(path).endswith('x.pt'))
        with self.assertRaises(Exception):
            _parse_model_arg('no-separator')

    def test_build_controllers_orders_manual_last_and_accepts_models(self):
        args = SimpleNamespace(
            models_dir='/nonexistent-dir',
            model=[('dqn_a', Path('/tmp/a.pt'))],
        )
        controllers = build_controllers(args, 'S2')
        self.assertEqual(list(controllers)[-1], 'manual')
        self.assertIn('dqn_a', controllers)

    def test_state_dict_dims_for_plain_and_dueling(self):
        plain = {
            'dense_1.weight': np.zeros((20, 13)),
            'dense_3.weight': np.zeros((4, 20)),
        }
        self.assertEqual(_state_dict_dims(plain, 'dqn_mlp'), (13, 4))
        dueling = {
            'feature.0.weight': np.zeros((20, 13)),
            'advantage.weight': np.zeros((8, 20)),
        }
        self.assertEqual(_state_dict_dims(dueling, 'dueling_mlp'), (13, 8))

    def test_snapshot_controller_defers_weight_loading(self):
        controller = build_snapshot_controller('x', 'X', '/tmp/none.pt')
        self.assertEqual(controller.kind, 'snapshot')
        self.assertEqual(controller.id, 'x')

    def test_page_contains_stream_and_phase_panel(self):
        page = build_page()
        self.assertIn('phaseGrid', page)
        self.assertIn('mapHost', page)
        self.assertIn('/static/js/app.js', page)
        self.assertIn('/static/js/renderer', page)
        app_js = (Path(__file__).parent.parent /
                  'tools/sumo_live_broadcast/static/js/app.js').read_text()
        self.assertIn("EventSource('/api/stream')", app_js)
        self.assertIn('api/control', app_js)

    def test_page_recovers_live_state_from_frames(self):
        # The server only emits 'status' messages while not live; a frame is
        # the sole signal that building finished.  The page must infer the
        # live transition inside onFrame or the overlay stays in 'building'.
        app_js = (Path(__file__).parent.parent /
                  'tools/sumo_live_broadcast/static/js/app.js').read_text()
        onframe = app_js.split('function onFrame(f) {', 1)[1]
        self.assertIn("sessionState !== 'live'", onframe[:800])

    def test_map_renderers_implement_the_same_contract(self):
        js_dir = (Path(__file__).parent.parent /
                  'tools/sumo_live_broadcast/static/js')
        for name in ('renderer_canvas.js', 'renderer_deckgl.js'):
            src = (js_dir / name).read_text()
            for method in ('setNetwork', 'pushFrame', 'setFocus',
                           'fit', 'resize', 'dispose'):
                self.assertIn(method, src, f'{name} missing {method}')

    def test_parser_defaults(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.scene, 'S2')
        self.assertEqual(args.action_interval, 10)
        self.assertEqual(args.speed, 1.0)
        self.assertIsNone(args.record)

    def test_recorder_writes_contiguous_transitions(self):
        world = _FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            rec = HitlRecorder(
                tmp, meta={'scene': 'sumohz1x1'},
                generator_factory=_fake_gens,
            )
            rec.bind(world, [0]).begin_episode({'scene_key': 'S2'})
            rec.on_decision(0.0, {0: 2}, source='snap_x',
                            proposals={0: 2})
            for _ in range(10):
                world.step()
                rec.on_step()
            rec.on_decision(10.0, {0: 3}, source='snap_x',
                            proposals={0: 3})
            for _ in range(10):
                world.step()
                rec.on_step()
            rec.end_episode('stop')
            rec.close()

            session = Path(tmp) / rec.session_id
            rows = _read_jsonl(session / 'transitions.jsonl')
            self.assertEqual(len(rows), 2)
            first, second = rows
            # interval 0: decided at t=0, closed at t=10
            self.assertEqual((first['t0'], first['t1']), (0.0, 10.0))
            self.assertEqual(first['action'], 2)
            self.assertEqual(first['reward_steps'], 10)
            # reward = mean of -12*t over steps t=1..10 = -66
            self.assertAlmostEqual(first['reward'], -66.0)
            self.assertFalse(first['truncated'])
            self.assertFalse(first['terminated'])
            # state chain stays contiguous inside the episode
            self.assertEqual(first['next_state'], second['state'])
            self.assertEqual(first['next_phase'], second['current_phase'])
            self.assertTrue(second['truncated'])
            # executed action is what gets paired with the reward
            self.assertEqual(first['hitl']['accepted_action'], 2)
            self.assertEqual(second['hitl']['accepted_action'], 3)

            proposals = _read_jsonl(session / 'proposals.jsonl')
            self.assertEqual(len(proposals), 2)
            self.assertEqual(proposals[0]['proposals'], {'J0': 2})
            self.assertEqual(proposals[1]['accepted'], {'J0': 3})

            manifest = json.loads(
                (session / 'manifest.json').read_text(encoding='utf-8'))
            self.assertEqual(manifest['schema_version'], 1)
            self.assertEqual(manifest['junctions'], {'0': 'J0'})
            self.assertEqual(manifest['action_dims'], {'0': 8})
            self.assertEqual(manifest['counts']['transitions'], 2)
            self.assertEqual(
                manifest['episodes'][0]['end_reason'], 'stop')

    def test_recorder_marks_manual_intervention(self):
        world = _FakeWorld()
        with tempfile.TemporaryDirectory() as tmp:
            rec = HitlRecorder(
                tmp, meta={}, generator_factory=_fake_gens)
            rec.bind(world, [0]).begin_episode()
            rec.on_decision(0.0, {0: 2}, source='snap_x',
                            proposals={0: 2})
            for _ in range(3):
                world.step()
                rec.on_step()
            # operator pins phase 5 mid-interval
            rec.on_intervention('manual_pin', now=3.0, rank=0, phase=5)
            for _ in range(7):
                world.step()
                rec.on_step()
            # next boundary: pinned rank is still under manual control;
            # proposals echo the pin, not the student's own choice
            rec.on_decision(10.0, {0: 5}, source='snap_x',
                            proposals={0: 5}, manual={0: 5})
            rec.close()

            session = Path(tmp) / rec.session_id
            rows = _read_jsonl(session / 'transitions.jsonl')
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]['mid_interval_override'])
            self.assertEqual(rows[0]['hitl']['accepted_action'], 2)
            # pinned interval: no real student proposal exists
            self.assertIsNone(rows[1]['hitl']['student_action'])
            self.assertEqual(rows[1]['hitl']['expert_id'], 'operator')
            self.assertEqual(rows[1]['hitl']['expert_action'], 5)
            self.assertTrue(rows[1]['hitl']['intervened'])

            events = _read_jsonl(session / 'interventions.jsonl')
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0]['kind'], 'manual_pin')
            self.assertEqual(events[0]['phase'], 5)

    def test_recorder_never_writes_without_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            rec = HitlRecorder(tmp, generator_factory=_fake_gens)
            rec.close()  # nothing bound, nothing recorded
            self.assertFalse((Path(tmp) / rec.session_id).exists())


if __name__ == '__main__':
    unittest.main()
