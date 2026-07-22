import json
import os
import tempfile
import unittest

from utils.logger import METRIC_FIELDS, StructuredMetricLogger


class StructuredMetricLoggerTest(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.logger = StructuredMetricLogger(self.temporary_directory.name)
        self.record = {field: None for field in METRIC_FIELDS}
        self.record.update({
            'schema_version': 2,
            'record_type': 'TRAIN',
            'agent': 'dqn',
            'network': 'sumohz1x1',
            'training_seed': 0,
            'episode': 0,
            'simulation_step': 100,
            'decision_step': 10,
            'global_decision_step': 10,
            'gradient_updates': 1,
            'wall_time_seconds': 0.1,
            'action_distribution': {'0': 0.75, '1': 0.25},
        })

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_records_are_append_only_complete_json_lines(self):
        self.logger.append(self.record)
        second = dict(self.record, record_type='FINAL_EVALUATION')
        self.logger.append(second)
        with open(self.logger.path, encoding='utf-8') as handle:
            records = [json.loads(line) for line in handle]
        self.assertEqual(['TRAIN', 'FINAL_EVALUATION'], [r['record_type'] for r in records])
        self.assertTrue(all(tuple(record) == METRIC_FIELDS for record in records))
        self.assertEqual({'0': 0.75, '1': 0.25}, records[0]['action_distribution'])
        self.assertEqual(2, self.logger.validate())

    def test_schema_v1_records_remain_readable(self):
        from utils.logger import METRIC_FIELDS_V1

        record = {field: self.record.get(field) for field in METRIC_FIELDS_V1}
        record['schema_version'] = 1
        self.logger.append(record)
        self.assertEqual(1, self.logger.validate())

    def test_missing_extra_and_invalid_type_are_rejected(self):
        missing = dict(self.record)
        missing.pop('epsilon')
        with self.assertRaisesRegex(ValueError, 'missing'):
            self.logger.append(missing)
        with self.assertRaisesRegex(ValueError, 'extra'):
            self.logger.append(dict(self.record, unexpected=1))
        with self.assertRaisesRegex(ValueError, 'record_type'):
            self.logger.append(dict(self.record, record_type='TEST'))

    def test_validation_rejects_missing_empty_and_corrupt_logs(self):
        with self.assertRaisesRegex(IOError, 'missing'):
            self.logger.validate()
        os.makedirs(os.path.dirname(self.logger.path), exist_ok=True)
        with open(self.logger.path, 'w', encoding='utf-8'):
            pass
        with self.assertRaisesRegex(IOError, 'no records'):
            self.logger.validate()
        with open(self.logger.path, 'w', encoding='utf-8') as handle:
            handle.write('{broken\n')
        with self.assertRaisesRegex(IOError, 'line 1'):
            self.logger.validate()


if __name__ == '__main__':
    unittest.main()
