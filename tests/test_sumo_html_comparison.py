import json
import tempfile
import unittest
from pathlib import Path

from tools.sumo_html_comparison import build_html, load_network_geometry


class SumoHtmlComparisonTest(unittest.TestCase):
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
        self.assertIn("const PAYLOAD =", html)
        self.assertNotIn("__PAYLOAD__", html)
        self.assertNotIn("__SCENE__", html)


if __name__ == "__main__":
    unittest.main()
