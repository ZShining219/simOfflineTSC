import json
import tempfile
import unittest
from collections import deque
from pathlib import Path

from agent.dqn import DQNAgent
from tools.experiment_plotting.plotting import render_evaluation_timeseries
from tools.experiment_plotting.profiles import get_profile
from utils.logger import EvaluationPackageWriter, validate_evaluation_package


def decision_record(controller, agent, training_seed, evaluation_seed, step):
    queue = float(step % 3)
    return {
        "schema_version": 1,
        "record_type": "DECISION_METRICS",
        "controller_id": controller,
        "agent": agent,
        "network": "sumohz1x1_config2",
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "checkpoint_episode": 10 if agent == "dqn" else None,
        "checkpoint_path": "checkpoints/evaluation/episode_0010.pt" if agent == "dqn" else None,
        "checkpoint_sha256": "a" * 64 if agent == "dqn" else None,
        "simulation_time_seconds": float(step * 10),
        "decision_step": step,
        "action_interval_seconds": 10,
        "actions": [0],
        "controller_reward_agents": [-queue],
        "reward_agents": [-queue],
        "reward_network_mean": -queue,
        "reward_network_sum": -queue,
        "queue_lanes": {"lane": queue},
        "queue_intersections": [queue],
        "queue_network_mean": queue,
        "queue_network_sum": queue,
        "delay_lanes": {"lane": queue / 10.0},
        "lane_vehicle_counts": {"lane": int(queue)},
        "delay_intersections": [queue / 10.0],
        "delay_network_weighted_mean": queue / 10.0,
        "throughput_interval": 1,
        "throughput_cumulative": step,
    }


def episode_summary(controller, agent, training_seed, evaluation_seed):
    return {
        "controller_id": controller,
        "agent": agent,
        "network": "sumohz1x1_config2",
        "training_seed": training_seed,
        "evaluation_seed": evaluation_seed,
        "checkpoint_episode": 10 if agent == "dqn" else None,
        "checkpoint_path": "checkpoints/evaluation/episode_0010.pt" if agent == "dqn" else None,
        "checkpoint_sha256": "a" * 64 if agent == "dqn" else None,
        "simulation_steps": 60,
        "decision_steps": 6,
        "travel_time": 70.0,
        "reward_mean": -1.0,
        "queue": 1.0,
        "delay": 0.1,
        "real_delay": 5.0,
        "throughput": 6,
        "waiting_time": 2.0,
        "unfinished_vehicles": 0,
        "wall_time_seconds": 0.1,
    }


class EvaluationPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_package_hashes_counts_and_timeseries_plots(self):
        collection = {
            "schema_version": 1,
            "package_id": "synthetic",
            "world": "sumo",
            "evaluation_seeds": [10000],
            "sampling_interval_seconds": 10,
            "smoothing_window_seconds": 60,
            "metrics": ["reward", "queue", "delay", "throughput", "travel_time"],
            "expected_episode_count": 2,
            "controllers": [
                {"controller_id": "dqn_seed0", "agent": "dqn",
                 "network": "sumohz1x1_config2", "training_seed": 0},
                {"controller_id": "fixedtime", "agent": "fixedtime",
                 "network": "sumohz1x1_config2", "training_seed": 0},
            ],
        }
        output = self.root / "package"
        writer = EvaluationPackageWriter(output, collection)
        records = []
        for controller, agent, seed in (
            ("dqn_seed0", "dqn", 0), ("fixedtime", "fixedtime", 0),
        ):
            for step in range(1, 7):
                record = decision_record(controller, agent, seed, 10000, step)
                writer.append_record(record)
                records.append(record)
            writer.append_summary(episode_summary(controller, agent, seed, 10000))
        writer.finalize()
        validated = validate_evaluation_package(output)
        self.assertEqual(2, validated["manifest"]["episode_count"])
        self.assertEqual(12, validated["record_count"])

        figures = render_evaluation_timeseries(
            records, self.root / "figures", 60, get_profile("plan1"), dpi=30,
        )
        self.assertEqual(42, len(figures))
        self.assertTrue(all(path.is_file() for path in figures))

        with (output / "records.jsonl").open("a", encoding="utf-8") as handle:
            handle.write("{}\n")
        with self.assertRaisesRegex(IOError, "hash mismatch"):
            validate_evaluation_package(output)

    def test_replay_utilization_uses_counters_without_batch_logging(self):
        agent = DQNAgent.__new__(DQNAgent)
        agent.replay_buffer = deque(maxlen=3)
        agent.replay_total_collected = 0
        agent.replay_sample_total = 0
        agent.replay_sample_counts = {}
        agent.replay_sample_count_histogram = {}
        agent.replay_insert_step = {}
        agent.replay_sample_age_sum = 0.0
        placeholder = [0]
        for key in ("a", "b", "c"):
            agent.remember(
                placeholder, placeholder, placeholder, None, placeholder,
                placeholder, placeholder, False, key,
            )
        samples = [agent.replay_buffer[0], agent.replay_buffer[1]]
        agent._record_replay_samples(samples)
        stats = agent.replay_utilization()
        self.assertEqual(3, stats["collected_transitions"])
        self.assertEqual(2, stats["sampled_transitions"])
        self.assertEqual(2, stats["unique_sampled_transitions"])
        self.assertAlmostEqual(2 / 3, stats["unique_coverage"])
        self.assertEqual(1.0, stats["sample_count_p95"])


if __name__ == "__main__":
    unittest.main()
