import contextlib
import fcntl
import json
import os
import shutil
import signal
import subprocess
import sys
import time

from .checkpoint import RollingRecoveryManager, load_full_checkpoint
from .io import atomic_json, read_json


GIB = 1024 ** 3
RECOVERABLE_FAILURES = {
    'interrupted', 'evaluator_exhausted', 'transient_process', 'resource',
}
NONRECOVERABLE_FAILURES = {
    'config', 'schema', 'digest', 'environment_signature', 'checkpoint_corrupt',
}


class LogicalRunLock:
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.handle = None

    def acquire(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.handle = open(self.path, 'a+', encoding='utf-8')
        try:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self.handle.close()
            self.handle = None
            raise RuntimeError(f'Logical run is already locked: {self.path}') from error
        self.handle.seek(0)
        self.handle.truncate()
        self.handle.write(f'{os.getpid()}\n')
        self.handle.flush()
        os.fsync(self.handle.fileno())
        return self

    def release(self):
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None

    def __enter__(self):
        return self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def available_memory_bytes():
    with open('/proc/meminfo', encoding='utf-8') as handle:
        fields = {}
        for line in handle:
            key, value = line.split(':', 1)
            fields[key] = int(value.strip().split()[0]) * 1024
    return fields.get('MemAvailable', fields.get('MemFree', 0))


class ResourceGate:
    def __init__(self, disk_provider=None, memory_provider=None):
        self.disk_provider = disk_provider or shutil.disk_usage
        self.memory_provider = memory_provider or available_memory_bytes

    def check(self, output_root, pending_output_bytes, concurrency_slots):
        disk = self.disk_provider(output_root)
        memory = int(self.memory_provider())
        required_disk = max(20 * GIB, 2 * int(pending_output_bytes))
        required_memory = 4 * GIB * int(concurrency_slots)
        result = {
            'valid': disk.free >= required_disk and memory >= required_memory,
            'disk_free_bytes': int(disk.free),
            'disk_required_bytes': required_disk,
            'memory_available_bytes': memory,
            'memory_required_bytes': required_memory,
            'concurrency_slots': int(concurrency_slots),
        }
        if not result['valid']:
            failures = []
            if disk.free < required_disk:
                failures.append('disk')
            if memory < required_memory:
                failures.append('memory')
            result['failures'] = failures
        return result


class ConcurrencyController:
    LEVELS = (8, 6, 4)

    def __init__(self, initial=8):
        if initial not in self.LEVELS:
            raise ValueError('Sequential concurrency must start at 8, 6, or 4')
        self.index = self.LEVELS.index(initial)
        self.resource_failures_in_wave = 0

    @property
    def value(self):
        return self.LEVELS[self.index]

    def record_resource_failure(self):
        self.resource_failures_in_wave += 1
        if self.resource_failures_in_wave >= 2 and self.index < len(self.LEVELS) - 1:
            self.index += 1
            self.resource_failures_in_wave = 0
            return True
        return False

    def begin_wave(self):
        self.resource_failures_in_wave = 0


class AttemptLineage:
    def __init__(self, logical_root, logical_run_id):
        self.root = os.path.join(os.path.abspath(logical_root), logical_run_id)
        self.manifest_path = os.path.join(self.root, 'logical_run_manifest.json')
        os.makedirs(self.root, exist_ok=True)
        if os.path.isfile(self.manifest_path):
            self.manifest = read_json(self.manifest_path)
        else:
            self.manifest = {
                'schema_version': 1,
                'logical_run_id': logical_run_id,
                'status': 'planned',
                'attempts': [],
                'effective_attempt': None,
            }
            atomic_json(self.manifest_path, self.manifest)

    def create_attempt(self, resume_from=None):
        number = len(self.manifest['attempts']) + 1
        attempt_id = f'attempt_{number}'
        attempt_dir = os.path.join(self.root, 'attempts', attempt_id)
        os.makedirs(attempt_dir, exist_ok=False)
        record = {
            'attempt_id': attempt_id,
            'attempt_dir': attempt_dir,
            'status': 'planned',
            'resume_from': resume_from,
            'created_at_unix': time.time(),
        }
        self.manifest['attempts'].append(record)
        self.manifest['status'] = 'planned'
        atomic_json(self.manifest_path, self.manifest)
        return record

    def update_attempt(self, attempt_id, status, **fields):
        record = next(
            item for item in self.manifest['attempts']
            if item['attempt_id'] == attempt_id
        )
        record.update({'status': status, **fields})
        self.manifest['status'] = status
        if status == 'completed':
            self.manifest['effective_attempt'] = attempt_id
        atomic_json(self.manifest_path, self.manifest)
        return record

    def latest_attempt(self):
        return self.manifest['attempts'][-1] if self.manifest['attempts'] else None

    def latest_recovery_checkpoint(self):
        for attempt in reversed(self.manifest['attempts']):
            pointer = os.path.join(attempt['attempt_dir'], 'resume_pointer.json')
            if os.path.isfile(pointer):
                path = read_json(pointer).get('checkpoint_path')
                if path and os.path.isfile(path):
                    try:
                        load_full_checkpoint(path)
                        return path
                    except (IOError, ValueError):
                        pass
            committed_dir = os.path.join(
                attempt['attempt_dir'], 'checkpoints', 'committed'
            )
            if os.path.isdir(committed_dir):
                candidates = sorted(
                    (os.path.join(committed_dir, name)
                     for name in os.listdir(committed_dir)
                     if name.endswith('.pt')),
                    reverse=True,
                )
                for path in candidates:
                    try:
                        load_full_checkpoint(path)
                        return path
                    except (IOError, ValueError):
                        continue
            directory = os.path.join(attempt['attempt_dir'], 'checkpoints', 'recovery')
            try:
                return RollingRecoveryManager(directory).latest_valid()[0]
            except (FileNotFoundError, IOError):
                continue
        return None


def classify_failure(returncode, stderr_text='', error_type=None):
    text = (stderr_text or '').lower()
    if returncode in (-signal.SIGINT, -signal.SIGTERM, 128 + signal.SIGINT,
                      128 + signal.SIGTERM):
        return 'interrupted'
    if error_type in NONRECOVERABLE_FAILURES:
        return error_type
    if error_type == 'evaluator_exhausted':
        return 'evaluator_exhausted'
    if any(token in text for token in ('out of memory', 'cannot allocate memory', 'no space left')):
        return 'resource'
    if any(token in text for token in ('connection reset', 'temporarily unavailable', 'broken pipe')):
        return 'transient_process'
    return 'failed'


def is_auto_recoverable(failure_class, recovery_checkpoint):
    return failure_class in RECOVERABLE_FAILURES and recovery_checkpoint is not None


class SequentialLauncher:
    def __init__(self, manifest_path, output_root, python_executable=None,
                 resource_gate=None, initial_concurrency=8):
        self.manifest_path = os.path.abspath(manifest_path)
        self.plan = read_json(self.manifest_path)
        self.output_root = os.path.abspath(output_root)
        os.makedirs(self.output_root, exist_ok=True)
        self.python_executable = python_executable or sys.executable
        self.resource_gate = resource_gate or ResourceGate()
        self.concurrency = ConcurrencyController(initial_concurrency)
        self.processes = {}
        self.stop_requested = False
        self.stop_signal = None

    def selected_children(self, logical_run_ids=None):
        children = self.plan.get('children', [])
        if logical_run_ids is None:
            return children
        requested = set(logical_run_ids)
        selected = [child for child in children if child['logical_run_id'] in requested]
        missing = requested - {child['logical_run_id'] for child in selected}
        if missing:
            raise ValueError(f'Unknown logical run IDs: {sorted(missing)}')
        return selected

    def _resource_gate(self, children):
        pending = sum(int(child.get('estimated_output_bytes', 0)) for child in children)
        result = self.resource_gate.check(
            self.output_root, pending, min(self.concurrency.value, len(children)),
        )
        atomic_json(os.path.join(self.output_root, 'resource_gate.json'), result)
        if not result['valid']:
            self.concurrency.record_resource_failure()
            raise RuntimeError(f'Resource gate failed: {result.get("failures", [])}')
        return result

    def _signal_handler(self, signum, frame):
        self.stop_requested = True
        if self.stop_signal is None:
            self.stop_signal = signum
        for process in self.processes.values():
            if process.poll() is None:
                process.send_signal(signum)

    def launch(self, logical_run_ids=None, authorize_formal=False,
               child_command='run-child'):
        if self.plan.get('mode') == 'formal' and not authorize_formal:
            raise PermissionError(
                'Formal 60-child manifest is frozen but launch requires '
                'explicit --authorize-formal'
            )
        children = self.selected_children(logical_run_ids)
        self._resource_gate(children)
        old_handlers = {
            signum: signal.getsignal(signum)
            for signum in (signal.SIGINT, signal.SIGTERM)
        }
        for signum in old_handlers:
            signal.signal(signum, self._signal_handler)
        try:
            pending = list(children)
            completed = []
            while (pending or self.processes) and not self.stop_requested:
                while pending and len(self.processes) < self.concurrency.value:
                    child = pending.pop(0)
                    logical_id = child['logical_run_id']
                    lineage = AttemptLineage(self.output_root, logical_id)
                    if lineage.manifest['status'] in ('running', 'completed'):
                        raise RuntimeError(
                            f'Logical run cannot be launched from status '
                            f'{lineage.manifest["status"]}: {logical_id}'
                        )
                    lock_path = os.path.join(lineage.root, 'run.lock')
                    lock = LogicalRunLock(lock_path).acquire()
                    previous_attempt = lineage.latest_attempt()
                    recovery = lineage.latest_recovery_checkpoint()
                    attempt = lineage.create_attempt(resume_from=recovery)
                    stdout_path = os.path.join(attempt['attempt_dir'], 'stdout.log')
                    stderr_path = os.path.join(attempt['attempt_dir'], 'stderr.log')
                    stdout = open(stdout_path, 'ab', buffering=0)
                    stderr = open(stderr_path, 'ab', buffering=0)
                    command = [
                        self.python_executable, 'sequential_run.py', child_command,
                        '--manifest', self.manifest_path,
                        '--logical-run-id', logical_id,
                        '--attempt-dir', attempt['attempt_dir'],
                    ]
                    if recovery:
                        command += ['--resume', recovery]
                        if previous_attempt is not None:
                            previous_state = os.path.join(
                                previous_attempt['attempt_dir'], 'current_state.json'
                            )
                            if os.path.isfile(previous_state):
                                command += ['--resume-state', previous_state]
                    lineage.update_attempt(
                        attempt['attempt_id'], 'running', command=command,
                        stdout_path=stdout_path, stderr_path=stderr_path,
                        started_at_unix=time.time(),
                    )
                    process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
                    lineage.update_attempt(
                        attempt['attempt_id'], 'running', pid=process.pid,
                    )
                    self.processes[logical_id] = process
                    process._sequential_context = (lineage, attempt, lock, stdout, stderr)
                time.sleep(0.1)
                for logical_id, process in list(self.processes.items()):
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    lineage, attempt, lock, stdout, stderr = process._sequential_context
                    stdout.close()
                    stderr.close()
                    with open(os.path.join(attempt['attempt_dir'], 'stderr.log'),
                              encoding='utf-8', errors='replace') as handle:
                        stderr_text = handle.read()
                    failure = classify_failure(returncode, stderr_text)
                    status = 'completed' if returncode == 0 else (
                        'interrupted' if failure == 'interrupted' else 'failed'
                    )
                    lineage.update_attempt(
                        attempt['attempt_id'], status, returncode=returncode,
                        failure_class=None if returncode == 0 else failure,
                        finished_at_unix=time.time(),
                    )
                    if failure == 'resource':
                        self.concurrency.record_resource_failure()
                    lock.release()
                    completed.append({
                        'logical_run_id': logical_id, 'status': status,
                        'returncode': returncode,
                    })
                    del self.processes[logical_id]
            return completed
        finally:
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)
            for logical_id, process in list(self.processes.items()):
                lineage, attempt, lock, stdout, stderr = process._sequential_context
                if process.poll() is None:
                    if self.stop_signal is not None:
                        process.send_signal(self.stop_signal)
                    else:
                        process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                stdout.close()
                stderr.close()
                returncode = process.poll()
                with open(os.path.join(attempt['attempt_dir'], 'stderr.log'),
                          encoding='utf-8', errors='replace') as handle:
                    stderr_text = handle.read()
                failure = (
                    'interrupted' if self.stop_signal is not None else
                    classify_failure(returncode, stderr_text)
                )
                status = 'completed' if returncode == 0 else (
                    'interrupted' if failure == 'interrupted' else 'failed'
                )
                lineage.update_attempt(
                    attempt['attempt_id'], status, returncode=returncode,
                    failure_class=None if returncode == 0 else failure,
                    finished_at_unix=time.time(), finalized_by_launcher=True,
                )
                lock.release()
                completed.append({
                    'logical_run_id': logical_id, 'status': status,
                    'returncode': returncode,
                })
                del self.processes[logical_id]


def collect_status(output_root, manifest_path=None, stale_seconds=1800):
    output_root = os.path.abspath(output_root)
    runs = []
    if not os.path.isdir(output_root):
        return {'runs': [], 'counts': {}}
    for name in sorted(os.listdir(output_root)):
        manifest = os.path.join(output_root, name, 'logical_run_manifest.json')
        if os.path.isfile(manifest):
            runs.append(read_json(manifest))
    known = {run['logical_run_id'] for run in runs}
    if manifest_path is not None:
        plan = read_json(manifest_path)
        for child in plan.get('children', []):
            if child['logical_run_id'] not in known:
                runs.append({
                    'schema_version': 1,
                    'logical_run_id': child['logical_run_id'],
                    'status': 'planned', 'attempts': [],
                    'effective_attempt': None,
                })
    now = time.time()
    for run in runs:
        observed = run['status']
        latest = run['attempts'][-1] if run.get('attempts') else None
        if observed == 'running' and latest is not None:
            progress = os.path.join(latest['attempt_dir'], 'current_state.json')
            reference = os.path.getmtime(progress) if os.path.isfile(progress) else float(
                latest.get('started_at_unix', 0)
            )
            if reference and now - reference > int(stale_seconds):
                observed = 'stale'
        run['observed_status'] = observed
    runs.sort(key=lambda item: item['logical_run_id'])
    counts = {}
    for run in runs:
        status = run['observed_status']
        counts[status] = counts.get(status, 0) + 1
    return {'runs': runs, 'counts': counts}
