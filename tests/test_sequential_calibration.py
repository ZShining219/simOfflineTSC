import json
import os
import tempfile
import unittest

from sequential.calibration import METRIC_FIELDS, compare_evaluator_calibration
from sequential.io import atomic_json


class SequentialCalibrationTests(unittest.TestCase):
    def test_exact_actions_and_all_non_wall_metrics_are_required(self):
        with tempfile.TemporaryDirectory() as directory:
            actions = [[index % 8] for index in range(360)]
            metrics = {
                'travel_time': 1.0, 'reward_mean': -1.0, 'queue': 2.0,
                'delay': 0.1, 'real_delay': 3.0, 'throughput': 4,
                'waiting_time': 5.0, 'unfinished_vehicles': 6,
                'phase_switches': 7, 'phase_switch_frequency': 8.0,
                'action_distribution': {'0': 1.0},
            }
            legacy_path = os.path.join(directory, 'legacy.json')
            atomic_json(legacy_path, {
                'network': 'network', 'decision_count': 360,
                'action_sequence': actions, 'metrics': metrics,
            })
            physical = os.path.join(directory, 'physical')
            os.makedirs(physical)
            summary_path = os.path.join(physical, 'summary.json')
            decisions_path = os.path.join(physical, 'decisions.jsonl')
            atomic_json(summary_path, {**metrics, 'wall_time_seconds': 99.0})
            with open(decisions_path, 'w', encoding='utf-8') as handle:
                for index, action in enumerate(actions, start=1):
                    handle.write(json.dumps({
                        'decision_index': index, 'actions': action,
                    }) + '\n')
            committed_path = os.path.join(physical, 'committed.json')
            atomic_json(committed_path, {
                'summary_path': summary_path, 'decisions_path': decisions_path,
            })
            alias_path = os.path.join(directory, 'alias.json')
            atomic_json(alias_path, {
                'identity': {'checkpoint_digest': 'digest'},
                'physical_committed_path': committed_path,
            })
            report = compare_evaluator_calibration(legacy_path, alias_path)
            self.assertTrue(report['valid'])
            self.assertEqual(set(report['metric_checks']), set(METRIC_FIELDS))
            metrics['queue'] = 3.0
            atomic_json(summary_path, {**metrics, 'wall_time_seconds': 1.0})
            self.assertFalse(compare_evaluator_calibration(
                legacy_path, alias_path
            )['valid'])


if __name__ == '__main__':
    unittest.main()
