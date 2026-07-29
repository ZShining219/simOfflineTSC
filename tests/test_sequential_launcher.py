import os
import shutil
import signal
import subprocess
import tempfile
import unittest
from unittest import mock

from sequential.io import atomic_json
from sequential.cli import main as sequential_main
from sequential.launcher import (
    AttemptLineage, ConcurrencyController, GIB, LogicalRunLock,
    ResourceGate, SequentialLauncher, classify_failure, collect_status,
    is_auto_recoverable, process_exists, reconcile_invalid_ha_runs,
    reconcile_stale_runs,
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

    def test_recovery_context_binds_checkpoint_and_state_to_same_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            lineage = AttemptLineage(directory, 'logical')
            source = lineage.create_attempt()
            source_checkpoint = os.path.join(
                source['attempt_dir'], 'source.pt',
            )
            open(source_checkpoint, 'wb').close()
            source_state = os.path.join(
                source['attempt_dir'], 'current_state.json',
            )
            atomic_json(source_state, {'logical_run_id': 'logical'})
            atomic_json(
                os.path.join(source['attempt_dir'], 'resume_pointer.json'),
                {'checkpoint_path': source_checkpoint},
            )
            lineage.update_attempt(
                source['attempt_id'], 'interrupted',
                failure_class='interrupted',
            )
            environment_failure = lineage.create_attempt(
                resume_from=source_checkpoint,
            )
            lineage.update_attempt(
                environment_failure['attempt_id'], 'failed',
                failure_class='failed',
            )
            invalid = lineage.create_attempt(resume_from=source_checkpoint)
            invalid_checkpoint = os.path.join(
                invalid['attempt_dir'], 'invalid.pt',
            )
            open(invalid_checkpoint, 'wb').close()
            atomic_json(
                os.path.join(invalid['attempt_dir'], 'resume_pointer.json'),
                {'checkpoint_path': invalid_checkpoint},
            )
            lineage.update_attempt(
                invalid['attempt_id'], 'failed', recovery_eligible=False,
            )

            with mock.patch(
                'sequential.launcher.load_full_checkpoint', return_value={},
            ):
                context = AttemptLineage(
                    directory, 'logical',
                ).latest_recovery_context()
            self.assertEqual(context, {
                'checkpoint_path': source_checkpoint,
                'resume_state_path': source_state,
                'source_attempt_id': source['attempt_id'],
            })

    def test_launcher_uses_resume_state_from_checkpoint_source_attempt(self):
        class PassingGate:
            @staticmethod
            def check(output_root, pending_output_bytes, concurrency_slots):
                return {
                    'valid': True, 'disk_free_bytes': 1,
                    'disk_required_bytes': 1, 'memory_available_bytes': 1,
                    'memory_required_bytes': 1,
                    'concurrency_slots': concurrency_slots,
                }

        class ImmediateProcess:
            command = None

            def __init__(self, command, stdout=None, stderr=None):
                type(self).command = command
                self.pid = 1234

            @staticmethod
            def poll():
                return 0

        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'manifest.json')
            atomic_json(manifest, {
                'mode': 'pilot', 'children': [{
                    'logical_run_id': 'logical',
                    'estimated_output_bytes': 0,
                }],
            })
            output_root = os.path.join(directory, 'runs')
            lineage = AttemptLineage(output_root, 'logical')
            source = lineage.create_attempt()
            checkpoint = os.path.join(source['attempt_dir'], 'source.pt')
            open(checkpoint, 'wb').close()
            state = os.path.join(source['attempt_dir'], 'current_state.json')
            atomic_json(state, {'logical_run_id': 'logical'})
            atomic_json(
                os.path.join(source['attempt_dir'], 'resume_pointer.json'),
                {'checkpoint_path': checkpoint},
            )
            lineage.update_attempt(
                source['attempt_id'], 'interrupted',
                failure_class='interrupted',
            )
            failed = lineage.create_attempt(resume_from=checkpoint)
            lineage.update_attempt(
                failed['attempt_id'], 'failed', failure_class='failed',
            )

            with mock.patch(
                'sequential.launcher.load_full_checkpoint', return_value={},
            ), mock.patch(
                'sequential.launcher.subprocess.Popen', ImmediateProcess,
            ):
                SequentialLauncher(
                    manifest, output_root, resource_gate=PassingGate(),
                    initial_concurrency=4,
                ).launch()

            command = ImmediateProcess.command
            self.assertEqual(command[command.index('--resume') + 1], checkpoint)
            self.assertEqual(
                command[command.index('--resume-state') + 1], state,
            )

    def test_invalid_ha_budget_reconciliation_preserves_attempt_and_source(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'manifest.json')
            atomic_json(manifest, {
                'mode': 'formal',
                'protocol_id': 'ha_sodqn_b100_v1',
                'children': [{
                    'logical_run_id': 'invalid',
                    'stage_episodes': [100, 100, 100, 100],
                }],
            })
            lineage = AttemptLineage(directory, 'invalid')
            source = lineage.create_attempt()
            source_checkpoint = os.path.join(
                source['attempt_dir'], 'source.pt',
            )
            open(source_checkpoint, 'wb').close()
            source_state = os.path.join(
                source['attempt_dir'], 'current_state.json',
            )
            atomic_json(source_state, {'logical_run_id': 'invalid'})
            lineage.update_attempt(
                source['attempt_id'], 'interrupted',
                failure_class='interrupted',
                reconciliation={'recovery_checkpoint': source_checkpoint},
            )
            completed = lineage.create_attempt(resume_from=source_checkpoint)
            final_checkpoint = os.path.join(
                completed['attempt_dir'], 'final.pt',
            )
            open(final_checkpoint, 'wb').close()
            atomic_json(
                os.path.join(completed['attempt_dir'], 'current_state.json'),
                {'completed_operations': {
                    'stage_4:checkpoint': {'artifact': final_checkpoint},
                }},
            )
            lineage.update_attempt(
                completed['attempt_id'], 'completed', pid=999,
            )

            def checkpoint(path, expected_type=None):
                if path == final_checkpoint:
                    return {'agent_state': {'counters': {
                        'global_decision_step': 160000,
                        'gradient_updates': 159000,
                        'target_updates': 15900,
                    }}}
                return {}

            with mock.patch(
                'sequential.launcher.load_full_checkpoint', side_effect=checkpoint,
            ):
                report = reconcile_invalid_ha_runs(
                    directory, manifest, ['invalid'],
                    pid_checker=lambda pid: False, now_provider=lambda: 123,
                )

            self.assertEqual(report['reconciled'], ['invalid'])
            updated = AttemptLineage(directory, 'invalid')
            self.assertEqual(len(updated.manifest['attempts']), 2)
            self.assertEqual(updated.manifest['status'], 'failed')
            self.assertIsNone(updated.manifest['effective_attempt'])
            latest = updated.latest_attempt()
            self.assertEqual(latest['failure_class'], 'validation_failed')
            self.assertFalse(latest['recovery_eligible'])
            self.assertTrue(latest['invalidated_by_validation'])
            self.assertEqual(
                latest['validation_failure']['recovery_context'], {
                    'checkpoint_path': source_checkpoint,
                    'resume_state_path': source_state,
                    'source_attempt_id': source['attempt_id'],
                },
            )
            self.assertTrue(os.path.isfile(final_checkpoint))

            second = reconcile_invalid_ha_runs(
                directory, manifest, ['invalid'],
                pid_checker=lambda pid: False,
            )
            self.assertEqual(second['reconciled'], [])
            self.assertEqual(
                second['results'][0]['reason'], 'not_effective_completed',
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

    def test_reconcile_stale_requires_dead_pid_lock_and_valid_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'manifest.json')
            atomic_json(manifest, {'children': [
                {'logical_run_id': 'stale'},
                {'logical_run_id': 'alive'},
                {'logical_run_id': 'no_checkpoint'},
            ]})
            for logical_id, pid in (
                ('stale', 101), ('alive', 202), ('no_checkpoint', 303),
            ):
                lineage = AttemptLineage(directory, logical_id)
                attempt = lineage.create_attempt()
                atomic_json(
                    os.path.join(attempt['attempt_dir'], 'current_state.json'),
                    {'logical_run_id': logical_id},
                )
                lineage.update_attempt(
                    attempt['attempt_id'], 'running', pid=pid,
                    started_at_unix=1,
                )

            def recovery(lineage):
                if lineage.manifest['logical_run_id'] == 'no_checkpoint':
                    return None
                return '/valid/recovery.pt'

            with mock.patch.object(
                AttemptLineage, 'latest_recovery_checkpoint', recovery,
            ), mock.patch(
                'sequential.launcher.os.path.getmtime', return_value=1,
            ):
                report = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: pid == 202,
                    now_provider=lambda: 100,
                )

            self.assertEqual(report['reconciled'], ['stale'])
            stale = AttemptLineage(directory, 'stale')
            self.assertEqual(stale.manifest['status'], 'interrupted')
            self.assertEqual(stale.latest_attempt()['failure_class'], 'interrupted')
            self.assertTrue(
                stale.latest_attempt()['finalized_by_reconciliation'],
            )
            self.assertEqual(
                AttemptLineage(directory, 'alive').manifest['status'], 'running',
            )
            self.assertEqual(
                AttemptLineage(directory, 'no_checkpoint').manifest['status'],
                'running',
            )
            reasons = {
                item['logical_run_id']: item.get('reason')
                for item in report['results']
            }
            self.assertEqual(reasons['alive'], 'pid_alive')
            self.assertEqual(
                reasons['no_checkpoint'], 'no_valid_recovery_checkpoint',
            )

    def _running_lineage(self, directory, logical_id='stale', pid=101):
        manifest = os.path.join(directory, 'manifest.json')
        atomic_json(manifest, {'children': [{'logical_run_id': logical_id}]})
        lineage = AttemptLineage(directory, logical_id)
        attempt = lineage.create_attempt()
        atomic_json(
            os.path.join(attempt['attempt_dir'], 'current_state.json'),
            {'logical_run_id': logical_id},
        )
        lineage.update_attempt(
            attempt['attempt_id'], 'running', pid=pid, started_at_unix=1,
        )
        return manifest, lineage, attempt

    def test_reconcile_stale_skips_held_lock_and_not_stale_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, lineage, _ = self._running_lineage(directory)
            lock = LogicalRunLock(os.path.join(lineage.root, 'run.lock')).acquire()
            try:
                with mock.patch(
                    'sequential.launcher.os.path.getmtime', return_value=1,
                ):
                    report = reconcile_stale_runs(
                        directory, manifest, stale_seconds=10,
                        pid_checker=lambda pid: False, now_provider=lambda: 100,
                    )
            finally:
                lock.release()
            self.assertEqual(report['results'][0]['reason'], 'locked')
            self.assertEqual(lineage.manifest['status'], 'running')

            with mock.patch(
                'sequential.launcher.os.path.getmtime', return_value=95,
            ):
                report = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: False, now_provider=lambda: 100,
                )
            self.assertEqual(report['results'][0]['reason'], 'not_stale')
            self.assertEqual(
                AttemptLineage(directory, 'stale').manifest['status'], 'running',
            )

    def test_reconcile_stale_skips_pid_change_detected_under_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, lineage, _ = self._running_lineage(directory, pid=101)
            original_acquire = LogicalRunLock.acquire

            def change_pid_then_acquire(lock):
                changed = AttemptLineage(directory, 'stale')
                changed.update_attempt('attempt_1', 'running', pid=303)
                return original_acquire(lock)

            with mock.patch.object(
                LogicalRunLock, 'acquire', change_pid_then_acquire,
            ), mock.patch(
                'sequential.launcher.os.path.getmtime', return_value=1,
            ):
                report = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: False, now_provider=lambda: 100,
                )
            self.assertEqual(report['results'][0]['reason'], 'lineage_changed')
            self.assertEqual(
                AttemptLineage(directory, 'stale').manifest['status'], 'running',
            )
            self.assertEqual(
                AttemptLineage(directory, 'stale').latest_attempt()['pid'], 303,
            )

    def test_reconcile_stale_rejects_corrupt_checkpoint_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest, lineage, attempt = self._running_lineage(directory)
            corrupt = os.path.join(attempt['attempt_dir'], 'corrupt.pt')
            with open(corrupt, 'wb') as handle:
                handle.write(b'not a checkpoint')
            atomic_json(
                os.path.join(attempt['attempt_dir'], 'resume_pointer.json'),
                {'checkpoint_path': corrupt},
            )
            with mock.patch(
                'sequential.launcher.os.path.getmtime', return_value=1,
            ):
                rejected = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: False, now_provider=lambda: 100,
                )
            self.assertEqual(
                rejected['results'][0]['reason'],
                'no_valid_recovery_checkpoint',
            )
            self.assertEqual(
                AttemptLineage(directory, 'stale').manifest['status'], 'running',
            )

            with mock.patch.object(
                AttemptLineage, 'latest_recovery_checkpoint',
                return_value='/valid/recovery.pt',
            ), mock.patch(
                'sequential.launcher.os.path.getmtime', return_value=1,
            ):
                first = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: False, now_provider=lambda: 100,
                )
                second = reconcile_stale_runs(
                    directory, manifest, stale_seconds=10,
                    pid_checker=lambda pid: False, now_provider=lambda: 100,
                )
            self.assertEqual(first['reconciled'], ['stale'])
            self.assertTrue(
                first['results'][0]['recovery_checkpoint_validated'],
            )
            self.assertEqual(second['reconciled'], [])
            finalized = AttemptLineage(directory, 'stale')
            self.assertEqual(len(finalized.manifest['attempts']), 1)
            self.assertEqual(finalized.manifest['status'], 'interrupted')

    def test_process_exists_handles_invalid_alive_and_exited_pids(self):
        for invalid in (None, '', 'not-a-pid', 0, -1):
            self.assertFalse(process_exists(invalid))
        self.assertTrue(process_exists(os.getpid()))
        process = subprocess.Popen(['/bin/true'])
        process.wait(timeout=5)
        self.assertFalse(process_exists(process.pid))

    def test_formal_manifest_reconciliation_requires_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'formal.json')
            atomic_json(manifest, {'mode': 'formal', 'children': []})
            with self.assertRaisesRegex(PermissionError, 'authorize-formal'):
                sequential_main([
                    'reconcile-stale', '--manifest', manifest,
                    '--output-root', directory,
                ])

    def test_formal_invalid_reconciliation_requires_authorization(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = os.path.join(directory, 'formal.json')
            atomic_json(manifest, {
                'mode': 'formal', 'protocol_id': 'ha_sodqn_b100_v1',
                'children': [],
            })
            with self.assertRaisesRegex(PermissionError, 'authorize-formal'):
                sequential_main([
                    'reconcile-invalid-ha', '--manifest', manifest,
                    '--output-root', directory,
                    '--logical-run-id', 'invalid',
                ])

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
