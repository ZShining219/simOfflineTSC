import os
import shutil
import signal
import tempfile
import unittest
from unittest import mock

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

    def test_manifest_status_includes_planned_and_stale_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'manifest.json')
            atomic_json(manifest, {'children': [
                {'logical_run_id': 'running'}, {'logical_run_id': 'planned'},
            ]})
            lineage = AttemptLineage(directory, 'running')
            attempt = lineage.create_attempt()
            lineage.update_attempt('attempt_1', 'running', started_at_unix=1)
            status = collect_status(
                directory, manifest_path=manifest, stale_seconds=1,
            )
            self.assertEqual(status['counts'], {'planned': 1, 'stale': 1})

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

    def test_launcher_runs_real_child_process_and_records_failure_artifacts(self):
        class PassingGate:
            @staticmethod
            def check(output_root, pending_output_bytes, concurrency_slots):
                return {
                    'valid': True, 'disk_free_bytes': 1, 'disk_required_bytes': 1,
                    'memory_available_bytes': 1, 'memory_required_bytes': 1,
                    'concurrency_slots': concurrency_slots,
                }

        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'pilot.json')
            atomic_json(manifest, {
                'mode': 'pilot', 'children': [{
                    'logical_run_id': 'real_failed_child',
                    'networks': ['missing'], 'stage_episodes': [400],
                    'policy': 'clear', 'training_seed': 0,
                    'parent_import_manifest': os.path.join(directory, 'missing.json'),
                    'estimated_output_bytes': 0,
                }],
                'model': {}, 'trainer': {},
            })
            output_root = os.path.join(directory, 'runs')
            result = SequentialLauncher(
                manifest, output_root, resource_gate=PassingGate(),
                initial_concurrency=4,
            ).launch()
            self.assertEqual(result[0]['status'], 'failed')
            lineage = collect_status(output_root)['runs'][0]
            self.assertEqual(lineage['status'], 'failed')
            attempt = lineage['attempts'][0]
            self.assertNotEqual(attempt['returncode'], 0)
            self.assertTrue(os.path.isfile(attempt['stdout_path']))
            self.assertTrue(os.path.isfile(attempt['stderr_path']))
            with open(attempt['stderr_path'], encoding='utf-8') as handle:
                self.assertIn('FileNotFoundError', handle.read())

    def test_external_interrupt_finalizes_running_attempt_and_releases_lock(self):
        class PassingGate:
            @staticmethod
            def check(output_root, pending_output_bytes, concurrency_slots):
                return {
                    'valid': True, 'disk_free_bytes': 1, 'disk_required_bytes': 1,
                    'memory_available_bytes': 1, 'memory_required_bytes': 1,
                    'concurrency_slots': concurrency_slots,
                }

        class DelayedInterruptProcess:
            next_pid = 1000

            def __init__(self, command, stdout=None, stderr=None):
                self.command = command
                self.returncode = None
                self.pid = type(self).next_pid
                type(self).next_pid += 1

            def poll(self):
                return self.returncode

            def send_signal(self, signum):
                # Keep the process alive through the launcher's next poll so
                # the finally branch owns lineage finalization.
                self.pending_signal = signum

            def terminate(self):
                self.returncode = -signal.SIGTERM

            def kill(self):
                self.returncode = -signal.SIGKILL

            def wait(self, timeout=None):
                self.returncode = -getattr(
                    self, 'pending_signal', signal.SIGTERM,
                )
                return self.returncode

        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'pilot.json')
            atomic_json(manifest, {
                'mode': 'pilot', 'children': [{
                    'logical_run_id': 'interrupted_child',
                    'estimated_output_bytes': 0,
                }],
            })
            output_root = os.path.join(directory, 'runs')
            launcher = SequentialLauncher(
                manifest, output_root, resource_gate=PassingGate(),
            )
            interrupted = False

            def interrupt_after_spawn(_):
                nonlocal interrupted
                if launcher.processes and not interrupted:
                    interrupted = True
                    launcher._signal_handler(signal.SIGINT, None)

            with mock.patch(
                'sequential.launcher.subprocess.Popen', DelayedInterruptProcess,
            ), mock.patch(
                'sequential.launcher.time.sleep', interrupt_after_spawn,
            ):
                result = launcher.launch()

            self.assertEqual(result, [{
                'logical_run_id': 'interrupted_child',
                'status': 'interrupted', 'returncode': -signal.SIGINT,
            }])
            lineage = AttemptLineage(output_root, 'interrupted_child')
            attempt = lineage.latest_attempt()
            self.assertEqual(lineage.manifest['status'], 'interrupted')
            self.assertEqual(attempt['failure_class'], 'interrupted')
            self.assertTrue(attempt['finalized_by_launcher'])
            lock = LogicalRunLock(os.path.join(lineage.root, 'run.lock')).acquire()
            lock.release()


if __name__ == '__main__':
    unittest.main()
