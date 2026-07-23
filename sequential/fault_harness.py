import json
import os
import signal
import subprocess
import sys
import time

from .io import atomic_json, read_json


def _events(path):
    if not os.path.isfile(path):
        return []
    records = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                break
    return records


def run_fault_recovery(manifest_path, output_root, logical_run_id,
                       timeout_seconds=3600, max_child=4):
    manifest_path = os.path.abspath(manifest_path)
    output_root = os.path.abspath(output_root)
    child = next(
        item for item in read_json(manifest_path)['children']
        if item['logical_run_id'] == logical_run_id
    )
    if child.get('variant') != 'fault' or not child.get('fault_point'):
        raise ValueError('Fault harness requires a manifest fault variant')
    audit_root = os.path.join(output_root, '_fault_harness', logical_run_id)
    os.makedirs(audit_root, exist_ok=True)
    launch_stdout_path = os.path.join(audit_root, 'launch.stdout.log')
    launch_stderr_path = os.path.join(audit_root, 'launch.stderr.log')
    command = [
        sys.executable, 'sequential_run.py', 'launch',
        '--manifest', manifest_path, '--output-root', output_root,
        '--logical-run-id', logical_run_id, '--max-child', str(max_child),
    ]
    started = time.monotonic()
    with open(launch_stdout_path, 'wb') as stdout, open(
        launch_stderr_path, 'wb'
    ) as stderr:
        launcher = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        trigger = None
        attempt = None
        try:
            while time.monotonic() - started < timeout_seconds:
                lineage_path = os.path.join(
                    output_root, logical_run_id, 'logical_run_manifest.json'
                )
                if os.path.isfile(lineage_path):
                    lineage = read_json(lineage_path)
                    if lineage['attempts']:
                        attempt = lineage['attempts'][-1]
                        events_path = os.path.join(
                            attempt['attempt_dir'], 'events.jsonl'
                        )
                        matches = [event for event in _events(events_path)
                                   if event['event_type'] == 'FAULT_TRIGGER_READY']
                        if matches:
                            trigger = matches[-1]
                            pid = attempt.get('pid')
                            if not pid:
                                raise RuntimeError('Running attempt has no persisted PID')
                            os.kill(int(pid), signal.SIGTERM)
                            break
                if launcher.poll() is not None:
                    raise RuntimeError('Launcher exited before the fault trigger')
                time.sleep(0.1)
            if trigger is None:
                raise TimeoutError('Timed out waiting for the persisted fault trigger')
            returncode = launcher.wait(timeout=60)
            if returncode != 0:
                raise RuntimeError(f'Fault launch wrapper exited with {returncode}')
        finally:
            if launcher.poll() is None:
                launcher.terminate()
                launcher.wait(timeout=10)
    lineage = read_json(os.path.join(
        output_root, logical_run_id, 'logical_run_manifest.json'
    ))
    interrupted = lineage['attempts'][-1]
    if interrupted['status'] != 'interrupted' or interrupted['returncode'] != 143:
        raise RuntimeError('Fault attempt did not terminate as SIGTERM interruption')
    resume_stdout_path = os.path.join(audit_root, 'resume.stdout.log')
    resume_stderr_path = os.path.join(audit_root, 'resume.stderr.log')
    resume_command = [
        sys.executable, 'sequential_run.py', 'resume-failed',
        '--manifest', manifest_path, '--output-root', output_root,
        '--max-child', str(max_child),
    ]
    with open(resume_stdout_path, 'wb') as stdout, open(
        resume_stderr_path, 'wb'
    ) as stderr:
        resumed = subprocess.run(
            resume_command, stdout=stdout, stderr=stderr,
            timeout=timeout_seconds, check=False,
        )
    if resumed.returncode != 0:
        raise RuntimeError(f'Resume wrapper exited with {resumed.returncode}')
    lineage = read_json(os.path.join(
        output_root, logical_run_id, 'logical_run_manifest.json'
    ))
    if lineage['status'] != 'completed' or len(lineage['attempts']) < 2:
        raise RuntimeError('Fault recovery did not complete in a new attempt')
    report = {
        'schema_version': 1, 'logical_run_id': logical_run_id,
        'fault_point': child['fault_point'], 'trigger_event': trigger,
        'interrupted_attempt': interrupted['attempt_id'],
        'effective_attempt': lineage['effective_attempt'],
        'attempt_count': len(lineage['attempts']), 'valid': True,
    }
    report_path = os.path.join(audit_root, 'report.json')
    atomic_json(report_path, report)
    return {'report_path': report_path, **report}
