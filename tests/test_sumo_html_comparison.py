import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.sumo_html_comparison import (
    _common_states, _presentation_payload, append_html_method, build_html,
    canonical_scene, load_network_geometry,
)


class SumoHtmlComparisonTest(unittest.TestCase):
    def test_scene_aliases_and_single_method_state_bundles(self):
        self.assertEqual(canonical_scene("S2"), "sumohz1x1")
        bundles = {"fixedtime": {"states": [{"time": 10}, {"time": 20}], "intersections": {}}}
        self.assertEqual(_common_states(bundles), [10, 20])

    def test_load_network_geometry_extracts_drawable_lanes_and_bounds(self):
        geometry = load_network_geometry(
            "configs/sim/sumohz1x1_config3.cfg"
        )
        self.assertGreater(len(geometry["lanes"]), 0)
        self.assertGreater(len(geometry["junctions"]), 0)
        self.assertLess(geometry["bounds"]["min_x"], geometry["bounds"]["max_x"])

    def test_build_html_embeds_payload_and_no_template_tokens(self):
        payload = {
            "scene": "scene",
            "methods": ["online_dqn", "hadhoa", "fixedtime"],
            "method_labels": {"online_dqn": "Online DQN"},
            "colors": {},
            "evaluation_seed": 10000,
            "action_interval_seconds": 10,
            "times": [1, 2],
            "network": {"bounds": {}, "lanes": [], "junctions": []},
            "states": {"online_dqn": [], "hadhoa": [], "fixedtime": []},
        }
        html = build_html(payload)
        self.assertIn("SUMO-style traffic control comparison", html)
        self.assertIn("Focus intersection", html)
        self.assertIn("Focused intersection halting vehicles", html)
        self.assertIn("const PAYLOAD =", html)
        self.assertNotIn("__PAYLOAD__", html)
        self.assertNotIn("__SCENE__", html)

    def test_presentation_order_and_titles(self):
        payload = {
            "methods": ["hadhoa", "cont_o2_final", "fixedtime", "online_dqn"],
            "method_labels": {
                "hadhoa": "long HADHOA label",
                "cont_o2_final": "long CONT label",
            },
            "states": {
                method: [method] for method in (
                    "hadhoa", "cont_o2_final", "fixedtime", "online_dqn"
                )
            },
        }
        prepared = _presentation_payload(payload)
        self.assertEqual(
            prepared["methods"],
            ["online_dqn", "fixedtime", "hadhoa", "cont_o2_final"],
        )
        self.assertEqual(
            prepared["method_labels"],
            {
                "online_dqn": "online DQN",
                "fixedtime": "fixedtime",
                "hadhoa": "DHOA",
                "cont_o2_final": "CONT DQN",
            },
        )

    def test_append_method_keeps_existing_state_arrays(self):
        base_payload = {
            "schema_version": 2,
            "scene": "sumohz1x1",
            "evaluation_seed": 10000,
            "source_config": "configs/sim/sumohz1x1.cfg",
            "methods": ["online_dqn", "hadhoa", "fixedtime"],
            "method_labels": {
                "online_dqn": "Online DQN",
                "hadhoa": "HADHOA",
                "fixedtime": "FixedTime",
            },
            "colors": {
                "online_dqn": "#1976d2",
                "hadhoa": "#d32f2f",
                "fixedtime": "#388e3c",
            },
            "action_interval_seconds": 10,
            "sample_every_seconds": 1,
            "times": [1, 2],
            "network": {"network_file": "network.net.xml", "bounds": {},
                         "lanes": [], "junctions": []},
            "states": {
                method: [
                    {"time": 1, "queue_network_sum": index},
                    {"time": 2, "queue_network_sum": index + 1},
                ]
                for index, method in enumerate(
                    ["online_dqn", "hadhoa", "fixedtime"]
                )
            },
            "intersections": {},
            "tracking": {},
            "replay_mode": "recorded_actions_exact_interval_headless_sumo",
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            base_path = directory / "base.json"
            base_path.write_text(json.dumps(base_payload), encoding="utf-8")
            captured = {
                "states": [
                    {"time": 1, "queue_network_sum": 9},
                    {"time": 2, "queue_network_sum": 10},
                ],
                "intersections": {},
            }
            with patch(
                "tools.sumo_html_comparison._record_actions",
                return_value=(
                    {"cont_o2_final": [{"actions": [0]}]}, 10
                ),
            ), patch("tools.sumo_html_comparison._ensure_sumo_home"), patch(
                "tools.sumo_html_comparison._capture_method",
                return_value=captured,
            ):
                manifest = append_html_method(
                    base_path,
                    "cont_o2_final",
                    directory / "decisions.jsonl",
                    directory / "out",
                )
            result = json.loads(
                (directory / "out" / "comparison_states.json").read_text(
                    encoding="utf-8"
                )
            )
        self.assertEqual(manifest["methods"][-1], "cont_o2_final")
        for method in base_payload["methods"]:
            self.assertEqual(base_payload["states"][method], result["states"][method])
        self.assertEqual(result["states"]["cont_o2_final"], captured["states"])
        self.assertIsNone(result["method_evaluation_seeds"]["cont_o2_final"])


if __name__ == "__main__":
    unittest.main()
