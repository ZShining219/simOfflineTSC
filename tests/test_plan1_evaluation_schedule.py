import json
import os
import tempfile
import unittest

import numpy as np

from trainer.tsc_trainer import TSCTrainer


class Plan1EvaluationScheduleTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.trainer = TSCTrainer.__new__(TSCTrainer)
        self.trainer.episodes = 100
        self.trainer.output_path = self.temporary_directory.name
        self.trainer.evaluation_results = {}
        self.trainer.final_evaluation_completed = False
        self.trainer.last_evaluation_record = None
        self.saved = []

        def save_checkpoints(episode):
            self.saved.append(episode)
            return []

        def evaluate(episode, record_type='EVALUATION'):
            travel_times = {0: 10.0, 10: 5.0, 25: 5.0, 100: 6.0}
            self.trainer.last_evaluation_record = {
                'schema_version': 2,
                'record_type': record_type,
                'episode': episode,
                'travel_time': np.float64(travel_times[episode]),
                'action_distribution': {'0': np.float64(1.0)},
            }
            return travel_times[episode]

        self.trainer.save_milestone_checkpoints = save_checkpoints
        self.trainer.train_test = evaluate

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_schedule_requires_sorted_unique_zero_and_final(self):
        resolved = self.trainer._resolve_evaluation_episodes(
            [0, 10, 25, 50, 100]
        )
        self.assertEqual((0, 10, 25, 50, 100), resolved)
        for invalid in ([10, 100], [0, 25, 10, 100], [0, 10, 10, 100], [0, 10, 99]):
            with self.assertRaises(ValueError):
                self.trainer._resolve_evaluation_episodes(invalid)

    def test_scheduled_evaluation_writes_summary_and_selects_earliest_tie(self):
        for episode in (0, 10, 25, 100):
            self.trainer._run_scheduled_evaluation(episode)
        self.assertEqual([0, 10, 25, 100], self.saved)
        self.assertTrue(self.trainer.final_evaluation_completed)
        path = self.trainer._evaluation_summary_path()
        self.assertTrue(os.path.isfile(path))
        with open(path, encoding='utf-8') as handle:
            summary = json.load(handle)
        self.assertEqual(10, summary['best_episode'])
        self.assertEqual(100, summary['final_episode'])
        self.assertEqual('FINAL_EVALUATION', summary['evaluations'][-1]['record_type'])
        self.assertEqual({'0': 1.0}, summary['evaluations'][0]['action_distribution'])


if __name__ == '__main__':
    unittest.main()
