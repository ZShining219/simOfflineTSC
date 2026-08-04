import json
import tempfile
import unittest
from pathlib import Path

from tools.traffic_visualization import align_three_methods, render_comparison


def _write(path, method, network="scene", times=(10, 20, 30)):
    with path.open("w", encoding="utf-8") as handle:
        for index, time_seconds in enumerate(times, start=1):
            handle.write(json.dumps({
                "record_type": "DECISION_METRICS",
                "controller_id": method,
                "network": network,
                "decision_step": index,
                "simulation_time_seconds": time_seconds,
                "queue_network_sum": index * (1 if method == "hadhoa" else 2),
                "throughput_cumulative": index * 3,
            }) + "\n")


class TrafficVisualizationTest(unittest.TestCase):
    def test_align_three_methods_uses_common_timestamps_without_interpolation(self):
        with tempfile.TemporaryDirectory() as root:
            tmp_path = Path(root)
            paths = {}
            for method in ("online_dqn", "hadhoa", "fixedtime"):
                paths[method] = tmp_path / f"{method}.jsonl"
                _write(paths[method], method, times=(10, 20, 30))

            loaded, rows = align_three_methods(
                paths, "scene", selectors={
                    method: {"controller_id": method} for method in paths
                },
            )

            self.assertEqual(set(loaded), {"online_dqn", "hadhoa", "fixedtime"})
            self.assertEqual(
                [row["simulation_time_seconds"] for row in rows], [10, 20, 30]
            )
            self.assertEqual(rows[1]["hadhoa_queue_network_sum"], 2)
            self.assertEqual(rows[-1]["fixedtime_throughput_cumulative"], 9)


    def test_align_three_methods_rejects_mismatched_timestamps(self):
        with tempfile.TemporaryDirectory() as root:
            tmp_path = Path(root)
            paths = {}
            for method in ("online_dqn", "hadhoa", "fixedtime"):
                paths[method] = tmp_path / f"{method}.jsonl"
                _write(paths[method], method, times=(10, 20, 30))
            _write(paths["fixedtime"], "fixedtime", times=(10, 25, 35))

            with self.assertRaisesRegex(ValueError, "At least two common"):
                align_three_methods(paths, "scene")


    def test_render_comparison_writes_aligned_csv_and_manifest(self):
        with tempfile.TemporaryDirectory() as root:
            tmp_path = Path(root)
            rows = [
                {
                    "simulation_time_seconds": 10,
                    "online_dqn_queue_network_sum": 1,
                    "hadhoa_queue_network_sum": 2,
                    "fixedtime_queue_network_sum": 3,
                    "online_dqn_throughput_cumulative": 1,
                    "hadhoa_throughput_cumulative": 1,
                    "fixedtime_throughput_cumulative": 0,
                },
                {
                    "simulation_time_seconds": 20,
                    "online_dqn_queue_network_sum": 2,
                    "hadhoa_queue_network_sum": 1,
                    "fixedtime_queue_network_sum": 4,
                    "online_dqn_throughput_cumulative": 3,
                    "hadhoa_throughput_cumulative": 4,
                    "fixedtime_throughput_cumulative": 2,
                },
            ]

            manifest = render_comparison(
                rows, tmp_path, "scene", mode="line", fps=30, dpi=30
            )

            self.assertFalse(manifest["interpolation"])
            self.assertTrue((tmp_path / "aligned_metrics.csv").is_file())
            self.assertTrue((tmp_path / "manifest.json").is_file())
            self.assertTrue(
                any(path.endswith((".mp4", ".gif")) for path in manifest["outputs"])
            )
