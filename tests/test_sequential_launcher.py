import os
import shutil
import signal
import tempfile
import unittest

from sequential.io import atomic_json
from sequential.launcher import (
    AttemptLineage, ConcurrencyController, GIB, LogicalRunLock,
    ResourceGate, SequentialLauncher, classify_failure, collect_status,
    is_auto_recoverable,
)


class SequentialLauncherTests(unittest.TestCase):
    def test_logical_run_lock_rejects_duplicate_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, 'run.lock')
            first = LogicalRunLock(path).acquire()
            try:
                with self.assertRaisesRegex(RuntimeError, 'already locked'):
                    LogicalRunLock(path).acquire()
            finally:
                first.release()
            second = LogicalRunLock(path).acquire()
            second.release()

    def test_attempt_lineage_never_overwrites_old_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            lineage = AttemptLineage(directory, 'logical')
            first = lineage.create_attempt()
            marker = os.path.join(first['attempt_dir'], 'evidence.json')
            atomic_json(marker, {'attempt': 1})
            lineage.update_attempt('attempt_1', 'failed', failure_class='interrupted')
            second = lineage.create_attempt(resume_from='/recovery/latest.pt')
            self.assertNotEqual(first['attempt_dir'], second['attempt_dir'])
            self.assertTrue(os.path.isfile(marker))
            self.assertEqual(second['resume_from'], '/recovery/latest.pt')
            lineage.update_attempt('attempt_2', 'completed')
            status = collect_status(directory)
            self.assertEqual(status['counts'], {'completed': 1})
            self.assertEqual(
                status['runs'][0]['effective_attempt'], 'attempt_2'
            )

    def test_resource_gate_enforces_disk_double_estimate_and_memory_per_slot(self):
        disk = shutil._ntuple_diskusage(100 * GIB, 50 * GIB, 50 * GIB)
        gate = ResourceGate(
            disk_provider=lambda path: disk,
            memory_provider=lambda: 31 * GIB,
        )
        result = gate.check('/tmp', pending_output_bytes=20 * GIB, concurrency_slots=8)
        self.assertFalse(result['valid'])
        self.assertEqual(result['disk_required_bytes'], 40 * GIB)
        self.assertEqual(result['memory_required_bytes'], 32 * GIB)
        self.assertEqual(result['failures'], ['memory'])

    def test_resource_failures_downgrade_concurrency_only_after_two(self):
        controller = ConcurrencyController(8)
        self.assertFalse(controller.record_resource_failure())
        self.assertEqual(controller.value, 8)
        self.assertTrue(controller.record_resource_failure())
        self.assertEqual(controller.value, 6)
        self.assertFalse(controller.record_resource_failure())
        self.assertTrue(controller.record_resource_failure())
        self.assertEqual(controller.value, 4)
        for _ in range(4):
            controller.record_resource_failure()
        self.assertEqual(controller.value, 4)

    def test_failure_classification_and_auto_resume_boundary(self):
        self.assertEqual(classify_failure(-signal.SIGTERM), 'interrupted')
        self.assertEqual(classify_failure(1, 'No space left on device'), 'resource')
        self.assertEqual(classify_failure(1, error_type='digest'), 'digest')
        self.assertTrue(is_auto_recoverable('interrupted', '/valid.pt'))
        self.assertFalse(is_auto_recoverable('interrupted', None))
        self.assertFalse(is_auto_recoverable('digest', '/valid.pt'))

    def test_formal_manifest_cannot_launch_without_explicit_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'formal.json')
            atomic_json(manifest, {
                'mode': 'formal', 'launch_authorized': False,
                'children': [{
                    'logical_run_id': 'formal_child',
                    'estimated_output_bytes': 1,
                }],
            })
            launcher = SequentialLauncher(manifest, os.path.join(directory, 'runs'))
            with self.assertRaisesRegex(PermissionError, 'authorize-formal'):
                launcher.launch()


if __name__ == '__main__':
    unittest.main()
