"""Reusable CLI planning, process isolation, checkpoint recovery and dashboards."""
import copy
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET

import pytest
import torch
import yaml

from common.paper_experiment import plan, load_manifest, read_json, digest, load_config
from sequential.launcher import LogicalRunLock
from trainer.tsc_trainer import _state_values_equal

ROOT = Path(__file__).resolve().parents[1]
SMOKE = ROOT / 'configs/tsc/paper_experiment_smoke.yml'


def invoke(args, log, wait=True):
    command = [sys.executable, '-m', 'tools.run_paper_baseline', *map(str, args)]
    stream = Path(log).open('w')
    process = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                               start_new_session=True,
                               env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1'))
    if not wait:
        return process, stream
    try:
        code = process.wait(timeout=240)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        stream.close()
    assert code == 0, Path(log).read_text()[-6000:]
    return command


def test_plan_is_deterministic_isolated_and_tamper_evident(tmp_path):
    state = random.getstate()
    a = plan(SMOKE, tmp_path / 'a')
    b = plan(SMOKE, tmp_path / 'b')
    assert random.getstate() == state
    left, right = load_manifest(a), load_manifest(b)
    assert left['protocol_hash'] == right['protocol_hash']
    for source in left['source_sha256']:
        assert (a.parent / 'source_snapshot' / source).read_bytes() == (ROOT / source).read_bytes()
    assert left['budget'] == {'training_episodes': 4, 'validation_episodes': 24, 'test_episodes': 18}
    normal = read_json(a.parent / 'plans/normal_s7.json')
    mixed = read_json(a.parent / 'plans/mixed_s7.json')
    assert [r['sumo_seed'] for r in normal['rows']] == [r['sumo_seed'] for r in mixed['rows']]
    assert all(not r['schedule']['events'] for r in normal['rows'])
    assert {r['kind'] for r in mixed['rows']} == {'normal', 'global_rain'}
    route = next((a.parent / 'assets').glob('*.rou.xml'))
    assert all(v.get('type') == 'pkw' for v in ET.parse(route).getroot().findall('vehicle'))
    assert all(t['command'][1:3] == ['-m', 'tools.run_paper_baseline'] for t in left['tasks'])
    with pytest.raises(FileExistsError):
        plan(SMOKE, a.parent)
    route.write_text(route.read_text() + '\n')
    with pytest.raises(ValueError, match='Frozen experiment input changed'):
        load_manifest(a)


def test_research_plan_budget_and_event_split(tmp_path):
    manifest = load_manifest(plan(ROOT / 'configs/tsc/colight_event150.yml', tmp_path / 'research'))
    assert manifest['budget'] == {'training_episodes': 1500, 'validation_episodes': 1240, 'test_episodes': 1500}
    cases = read_json(Path(manifest['root']) / 'cases.json')
    held_out = {digest(c['schedule']) for group in cases.values() for c in group if c['kind'] != 'normal'}
    from collections import Counter
    for seed in manifest['config']['seeds']:
        rows = read_json(Path(manifest['root']) / 'plans' / f'mixed_s{seed}.json')['rows']
        assert Counter(r['kind'] for r in rows) == {'normal': 75, 'lane_blockage': 25, 'road_closure': 25, 'global_rain': 25}
        assert not any(digest(r['schedule']) in held_out for r in rows if r['kind'] != 'normal')
    # Only planning was performed: no training or queue output is created.
    assert not (Path(manifest['root']) / 'train').exists()


def test_unknown_config_keys_and_seed_overlap_rejected(tmp_path):
    spec = load_config(SMOKE)
    spec['model']['batch_szie'] = 2
    path = tmp_path / 'bad.yml'
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='Unknown'):
        plan(path, tmp_path / 'bad')
    spec['model'].pop('batch_szie')
    spec['events']['test_sumo_seeds'] = spec['events']['validation_sumo_seeds']
    path.write_text(yaml.safe_dump(spec))
    with pytest.raises(ValueError, match='overlap'):
        plan(path, tmp_path / 'bad')


def test_serial_parallel_match_and_dashboard(tmp_path):
    roots, commands = [], []
    for label, cap in [('serial', 1), ('parallel', 4)]:
        manifest_path = plan(SMOKE, tmp_path / label)
        roots.append(manifest_path.parent)
        commands.append(invoke(['launch', '--manifest', manifest_path, '--train-workers', '2',
                                '--eval-workers', '2', '--test-workers', '2', '--max-workers', str(cap)],
                               tmp_path / (label + '.log')))
        manifest = load_manifest(manifest_path)
        state = read_json(manifest_path.parent / 'queue/run_manifest.json')
        assert all(t['status'] == 'SUCCEEDED' for t in state['tasks']), state
        assert len(state['tasks']) == 10
        invoke(['status', '--manifest', manifest_path], tmp_path / (label + '_status.log'))
        invoke(['resume', '--manifest', manifest_path, '--train-workers', '1', '--eval-workers', '1'],
               tmp_path / (label + '_noop_resume.log'))
    for regime in ('normal', 'mixed'):
        a, b = [torch.load(r / f'train/{regime}_s7/checkpoints/resumable/episode_0002.pt', map_location='cpu') for r in roots]
        assert _state_values_equal(a['agents'], b['agents'])
        for name in ('decisions.jsonl', 'updates.jsonl', 'environment.jsonl'):
            assert (roots[0] / f'train/{regime}_s7/monitor' / name).read_bytes() == (
                roots[1] / f'train/{regime}_s7/monitor' / name).read_bytes()
        left, right = [read_json(r / f'test/{regime}_s7/episode_0002/results.json') for r in roots]
        assert all(x['trace_sha256'] == y['trace_sha256'] for x, y in zip(left['cases'], right['cases']))
        assert all(c['vehicle_type'] == 'pkw' for c in left['cases'])
        assert all(c['system_time_per_vehicle'] >= 0 for c in left['cases'])
        assert len(left['isolation_checks']) == 9
        import csv
        for case in left['cases']:
            if case['kind'] == 'global_rain':
                assert case['target_split'] == 'global'
            with Path(case['timeline']).open() as handle:
                for row in csv.DictReader(handle):
                    lanes = json.loads(row['lane_queues_json'])
                    assert sum(lanes.values()) == int(row['queue_vehicles'])
                    assert sum(lanes.get(l, 0) for l in case['local_regions']['closure_area_lanes']) == int(row['closure_area_queue'])
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    events = EventAccumulator(str(roots[0] / 'train/mixed_s7/tensorboard')).Reload()
    assert 'environment/pending_due' in events.Tags()['scalars']
    queue = EventAccumulator(str(roots[0] / 'queue/tensorboard')).Reload()
    assert 'tasks/succeeded' in queue.Tags()['scalars']
    # Simulate a crash after cases finished but before group completion was
    # published. Recovery must reuse complete case evidence, not rerun SUMO.
    evaluation = roots[0] / 'test/mixed_s7/episode_0002'
    timeline = next((evaluation / 'environment/test').glob('*.csv'))
    mtime = timeline.stat().st_mtime_ns
    (evaluation / 'completed.json').unlink()
    invoke(['worker', '--manifest', roots[0] / 'manifest.json', '--task-id', 'test_mixed_s7_e0002'],
           tmp_path / 'evaluation_recovery.log')
    assert timeline.stat().st_mtime_ns == mtime
    assert read_json(evaluation / 'run/run_status.json')['status'] == '已完成'
    # Likewise, final training publication can be finalized without attempting
    # to resume an already-complete legacy Runner.
    training = roots[0] / 'train/normal_s7'
    checkpoint = training / 'checkpoints/resumable/episode_0002.pt'
    checkpoint_mtime = checkpoint.stat().st_mtime_ns
    (training / 'completed.json').unlink()
    invoke(['worker', '--manifest', roots[0] / 'manifest.json', '--task-id', 'normal_s7', '--resume'],
           tmp_path / 'completion_recovery.log')
    assert checkpoint.stat().st_mtime_ns == checkpoint_mtime
    (tmp_path / 'parallel_validation.json').write_text(json.dumps({'status': 'passed',
        'commands': commands, 'simulator': 'SUMO/libsumo', 'network': 'hz4x4', 'training_seed': 7,
        'regimes': ['normal', 'mixed'], 'serial_parallel_agent_states_equal': True,
        'serial_parallel_rewards_losses_environment_equal': True, 'evaluation_traces_equal': True}, indent=2))


def test_interrupted_training_resumes_exactly(tmp_path):
    spec = load_config(SMOKE)
    spec.update(episodes=3, seconds=300, regimes=['mixed'], test_episodes=[3])
    config = tmp_path / 'recovery.yml'
    config.write_text(yaml.safe_dump(spec))
    paths = [plan(config, tmp_path / name) for name in ('continuous', 'interrupted')]
    command = ['worker', '--manifest', paths[0], '--task-id', 'mixed_s7']
    invoke(command, tmp_path / 'continuous.log')
    proc, log = invoke(['worker', '--manifest', paths[1], '--task-id', 'mixed_s7'], tmp_path / 'interrupted.log', wait=False)
    output = paths[1].parent / 'train/mixed_s7'
    try:
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            marker = output / 'published/episode_0001.json'
            decisions = output / 'monitor/decisions.jsonl'
            if marker.exists() and decisions.exists() and len(decisions.read_text().splitlines()) > 30:
                os.killpg(proc.pid, signal.SIGTERM)
                break
            assert proc.poll() is None, (tmp_path / 'interrupted.log').read_text()[-4000:]
            time.sleep(.01)
        else:
            pytest.fail('No published recovery checkpoint')
        assert proc.wait(timeout=30) == 130
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
        log.close()
    resume_command = ['worker', '--manifest', paths[1], '--task-id', 'mixed_s7', '--resume']
    invoke(resume_command, tmp_path / 'resumed.log')
    a, b = [torch.load(p.parent / 'train/mixed_s7/checkpoints/resumable/episode_0003.pt', map_location='cpu') for p in paths]
    assert _state_values_equal(a['agents'], b['agents'])
    for name in ('decisions.jsonl', 'updates.jsonl', 'environment.jsonl'):
        assert (paths[0].parent / 'train/mixed_s7/monitor' / name).read_bytes() == (output / 'monitor' / name).read_bytes()
    assert list((output / 'monitor_orphans').glob('after_*/tensorboard_*'))
    (tmp_path / 'recovery_validation.json').write_text(json.dumps({'status': 'passed',
        'resume_command': [sys.executable, '-m', 'tools.run_paper_baseline', *map(str, resume_command)],
        'training_seed': 7, 'network': 'hz4x4', 'checkpoint_state_equal': True,
        'monitor_records_equal': True}, indent=2))


def test_worker_lock_rejects_duplicate_launch(tmp_path):
    path = plan(SMOKE, tmp_path / 'locked')
    with LogicalRunLock(path.parent / 'locks/mixed_s7.lock'):
        result = subprocess.run([sys.executable, '-m', 'tools.run_paper_baseline', 'worker',
                                 '--manifest', str(path), '--task-id', 'mixed_s7'], cwd=ROOT,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and 'already locked' in result.stderr
        assert not (path.parent / 'train/mixed_s7').exists()
    with LogicalRunLock(path.parent / 'launch.lock'):
        result = subprocess.run([sys.executable, '-m', 'tools.run_paper_baseline', 'launch',
                                 '--manifest', str(path)], cwd=ROOT, capture_output=True, text=True, timeout=30)
        assert result.returncode != 0 and 'already locked' in result.stderr
        assert not (path.parent / 'queue').exists()
