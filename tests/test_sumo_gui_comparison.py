import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tools.sumo_gui_comparison import _compose_frame, _load_replay


class SumoGuiComparisonTest(unittest.TestCase):
    def test_load_replay_requires_contiguous_action_timestamps(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "decisions.jsonl"
            records = []
            for index, time_seconds in enumerate((10, 20, 30), start=1):
                records.append({
                    "record_type": "DECISION_METRICS",
                    "controller_id": "controller",
                    "network": "scene",
                    "decision_step": index,
                    "simulation_time_seconds": time_seconds,
                    "action_interval_seconds": 10,
                    "actions": [index],
                })
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            replay, interval = _load_replay(
                path, "online_dqn", "scene", controller_id="controller"
            )
            self.assertEqual(interval, 10)
            self.assertEqual([item["time"] for item in replay], [10.0, 20.0, 30.0])
            records[1]["simulation_time_seconds"] = 25
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                _load_replay(path, "online_dqn", "scene", controller_id="controller")

    def test_compose_frame_adds_method_labels_and_keeps_three_panels(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            inputs = []
            for index in range(3):
                path = root / f"input_{index}.png"
                Image.new("RGB", (40, 30), (index * 80, 0, 0)).save(path)
                inputs.append(path)
            output = root / "panel.png"
            size = _compose_frame(inputs, output, 600)
            self.assertEqual(size, (120, 64))
            self.assertTrue(output.is_file())


if __name__ == "__main__":
    unittest.main()
