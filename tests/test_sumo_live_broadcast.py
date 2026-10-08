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


if __name__ == '__main__':
    unittest.main()
