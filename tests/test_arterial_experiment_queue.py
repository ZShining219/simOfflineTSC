import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from tools.run_arterial_experiment_queue import (
    QueueRunner, classify_failure, completion_valid)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def args(tmp_path, manifest, workers=2):
    return SimpleNamespace(
        manifest=str(manifest), state_dir=str(tmp_path / 'state'),
        max_workers=workers, max_retries=0, stagger_seconds=0.0,
        poll_interval=0.01, monitor_interval=0.01,
        max_memory_percent=100.0, max_swap_growth_mb=10 ** 6,
        min_disk_free_gb=0.0, max_load_ratio=10 ** 6,
    )


def test_completion_check_reads_nested_json(tmp_path):
    marker = tmp_path / 'run_status.json'
    write_json(marker, {'result': {'status': 'done'}})
    assert completion_valid({
        'completion': {'path': str(marker), 'json_field': 'result.status',
                       'equals': 'done'}})


def test_queue_runs_independent_tasks_and_persists_csv(tmp_path):
    tasks = []
    for index in range(2):
        marker = tmp_path / f'run{index}' / 'status.json'
        command = [
            sys.executable, '-c',
            ('import json,pathlib; p=pathlib.Path(r"%s"); '
             'p.parent.mkdir(parents=True,exist_ok=True); '
             'p.write_text(json.dumps({"status":"done"}))') % marker,
        ]
        tasks.append({
            'run_id': f'run_{index}', 'command': command,
            'output_path': str(marker.parent),
            'completion': {'path': str(marker), 'json_field': 'status',
                           'equals': 'done'},
        })
    manifest = tmp_path / 'queue.json'
    write_json(manifest, {'schema_version': 1, 'tasks': tasks})
    runner = QueueRunner(args(tmp_path, manifest))
    assert runner.run() == 0
    assert {task['status'] for task in runner.tasks} == {'SUCCEEDED'}
    assert (tmp_path / 'state' / 'run_manifest.csv').is_file()
    assert (tmp_path / 'state' / 'status.json').is_file()


def test_existing_completion_is_skipped(tmp_path):
    marker = tmp_path / 'run' / 'status.json'
    write_json(marker, {'status': 'done'})
    manifest = tmp_path / 'queue.json'
    write_json(manifest, {'tasks': [{
        'run_id': 'complete', 'command': [sys.executable, '-c', 'raise SystemExit(9)'],
        'output_path': str(marker.parent),
        'completion': {'path': str(marker), 'json_field': 'status',
                       'equals': 'done'},
    }]})
    runner = QueueRunner(args(tmp_path, manifest, workers=1))
    assert runner.tasks[0]['status'] == 'SKIPPED_COMPLETED'
    assert runner.run() == 0


def test_failure_classification():
    assert classify_failure(1, 'cannot allocate memory') == 'RESOURCE_FAILURE'
    assert classify_failure(1, 'libsumo connection closed') == 'SIMULATION_FAILURE'
    assert classify_failure(1, 'schema mismatch') == 'CONFIG_FAILURE'


def test_resource_sample_contains_process_and_io_fields(tmp_path):
    manifest = tmp_path / 'queue.json'
    write_json(manifest, {'tasks': [{
        'run_id': 'pending', 'command': [sys.executable, '-c', 'pass'],
        'output_path': str(tmp_path / 'pending'),
        'completion': {'path': str(tmp_path / 'pending' / 'done')},
    }]})
    sample = QueueRunner(args(tmp_path, manifest, workers=1)).resource_sample()
    assert sample['run_process_rss_bytes'] == 0
    assert sample['run_process_count'] == 0
    assert sample['disk_read_bytes_per_second'] >= 0
    assert sample['disk_write_bytes_per_second'] >= 0


def test_latest_checkpoint_and_resume_command(tmp_path):
    output = tmp_path / 'run'
    checkpoints = output / 'checkpoints' / 'resumable'
    checkpoints.mkdir(parents=True)
    (checkpoints / 'episode_0025.pt').write_bytes(b'a')
    (checkpoints / 'episode_0050.pt').write_bytes(b'b')
    task = {
        'run_id': 'resume', 'command': [sys.executable, 'arterial_run.py'],
        'output_path': str(output),
    }
    assert QueueRunner._checkpoint_exists(task)
    assert task['checkpoint_path'].endswith('episode_0050.pt')
    command = QueueRunner._resume_command(task)
    assert command[-4:] == [
        '--resume-output', str(output), '--resume-checkpoint',
        str((checkpoints / 'episode_0050.pt').resolve())]


def test_existing_matching_worker_is_adopted(tmp_path):
    marker = tmp_path / 'run' / 'done.json'
    command = [sys.executable, '-c', 'import time; time.sleep(30)']
    process = subprocess.Popen(command, start_new_session=True)
    try:
        manifest = tmp_path / 'queue.json'
        write_json(manifest, {'tasks': [{
            'run_id': 'orphan', 'command': command,
            'output_path': str(marker.parent),
            'completion': {'path': str(marker), 'json_field': 'status',
                           'equals': 'done'},
        }]})
        state = tmp_path / 'state' / 'run_manifest.json'
        write_json(state, {'tasks': [{
            'run_id': 'orphan', 'status': 'RUNNING', 'pid': process.pid,
        }]})
        runner = QueueRunner(args(tmp_path, manifest, workers=1))
        assert runner.tasks[0]['status'] == 'RUNNING'
        assert set(runner.running) == {'orphan'}
        assert runner.running['orphan']['adopted'] is True
    finally:
        os.killpg(process.pid, 15)
        process.wait(timeout=10)
