"""Reward isolation and real training/dashboard persistence, including SIGTERM."""
from dataclasses import FrozenInstanceError
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from common.paper_rewards import load_profile
from utils.training_monitor import TrainingMonitor, read_rows, render_run
from world.paper_rewards import SumoRewardSource


ROOT = Path(__file__).resolve().parents[1]


def profile(owner):
    return load_profile(ROOT / f'configs/rewards/{owner}.yml', owner)


def test_author_reward_algebra_and_sign():
    # Selected author-code weight -.25 and TD normalization /20 => -stat/80.
    assert profile('frap').calculate([2, 3]) == {'statistic': 5., 'raw_reward': -1.25, 'reward': -0.0625}
    assert profile('colight').calculate([2, 3])['reward'] == -0.0625
    pressure = profile('presslight')
    assert pressure.calculate([2, 3], [4, 5])['reward'] == -0.05
    assert pressure.calculate([4, 5], [2, 3])['reward'] == -0.05
    assert pressure.calculate([2, 3], [1, 4])['reward'] == 0.
    with pytest.raises(ValueError):
        pressure.calculate([1], [])


@pytest.mark.parametrize('owner', ['frap', 'presslight', 'colight'])
def test_profiles_are_owned_and_immutable(owner, tmp_path):
    source = ROOT / f'configs/rewards/{owner}.yml'
    reward = load_profile(source, owner)
    for other in {'frap', 'presslight', 'colight'} - {owner}:
        with pytest.raises(ValueError, match='ownership'):
            load_profile(source, other)
    with pytest.raises(FrozenInstanceError):
        reward.weight = -1
    data = yaml.safe_load(source.read_text())
    data['weight'] = -1
    changed = tmp_path / 'changed.yml'
    changed.write_text(yaml.safe_dump(data))
    assert load_profile(changed, owner).digest != reward.digest
    assert reward.weight == -.25  # Existing instance cannot follow later file/global changes.
    data['statistic'] = 'some_global_default'
    changed.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match='statistic/scope'):
        load_profile(changed, owner)


def test_sumo_counts_current_speed_full_lane_and_reset():
    source = object.__new__(SumoRewardSource)
    source.incoming = {'J': ('left', 'right')}
    source.outgoing = {'J': ('out',)}
    source.movements = {'J': ('left',)}
    source.unsupported_frap_lanes = {'J': ()}
    ids = {'left': ('stopped', 'moving', '__tsc_event__obstacle'), 'right': ('right_car',), 'out': ('out_car',)}
    speeds = {'stopped': 0., 'moving': 8., '__tsc_event__obstacle': 0., 'right_car': 0., 'out_car': 0.}
    # There is no waiting-history or distance API; reward cannot depend on them.
    world = SimpleNamespace(intersections=[object()], get_current_time=lambda: 0., eng=SimpleNamespace(
        lane=SimpleNamespace(getLastStepVehicleIDs=lambda lane: ids[lane]),
        vehicle=SimpleNamespace(getSpeed=lambda vehicle: speeds[vehicle])))
    source.world, source._key, source._queues = world, None, {}
    assert source.reward(profile('frap'), 'J')['statistic'] == 1
    assert source.reward(profile('colight'), 'J')['statistic'] == 2
    assert source.reward(profile('presslight'), 'J')['statistic'] == 1
    speeds['stopped'] = 5.
    world.intersections = [object()]  # Reset at the same timestamp must invalidate cache.
    assert source.reward(profile('frap'), 'J')['statistic'] == 0


def test_frap_lane_restrictions_do_not_leak_into_other_models():
    source = object.__new__(SumoRewardSource)
    source.incoming = {'J': ('mixed',)}
    source.outgoing = {'J': ('out',)}
    source.movements = {'J': ('mixed',)}
    source.unsupported_frap_lanes = {'J': ('mixed',)}
    with pytest.raises(ValueError, match='FRAP'):
        source.lane_manifest(profile('frap'))
    assert source.lane_manifest(profile('colight'))['J']['incoming'] == ['mixed']
    assert source.lane_manifest(profile('presslight'))['J']['outgoing'] == ['out']


def test_monitor_partial_rows_and_persistence(tmp_path):
    pytest.importorskip('tensorboard')
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    p = profile('frap')
    contract = {'profile': p.to_dict(), 'hash': p.digest}
    (tmp_path / 'reward_contract.json').write_text(json.dumps(contract))
    monitor = TrainingMonitor(tmp_path, contract)
    monitor.decision({'episode': 1, 'decision': 1, 'simulation_time': 10., 'reward_mean': -.1,
                      'reward_min': -.2, 'reward_max': 0., 'stopped_vehicles': 8, 'active_reports': 1})
    monitor.episode({'record_type': 'TRAIN', 'episode': 1, 'reward_mean': np.float32(-.1),
                     'loss_mean': .01, 'travel_time': 120., 'queue': 2., 'throughput': 20,
                     'epsilon': .5})
    monitor.close('interrupted')
    assert json.loads((tmp_path / 'monitor/status.json').read_text())['status'] == 'interrupted'
    assert (tmp_path / 'monitor/episode_curves.png').stat().st_size > 1000
    assert (tmp_path / 'monitor/decision_curves.svg').stat().st_size > 1000
    assert len(read_rows(tmp_path / 'monitor/decisions.jsonl')) == 1
    events = EventAccumulator(str(tmp_path / 'tensorboard')).Reload()
    assert any('reward/frap_' in tag for tag in events.Tags()['scalars'])
    assert 'run/status/text_summary' in events.Tags()['tensors']
    with (tmp_path / 'monitor/decisions.jsonl').open('a') as handle:
        handle.write('{"incomplete":')
    render_run(tmp_path)
    assert len(read_rows(tmp_path / 'monitor/decisions.jsonl')) == 1


def launch(tmp_path, owner, episodes=2, corrected_signals=False, phase_input=False):
    overlay = 'paper_p0.yml' if corrected_signals else 'paper_monitor_smoke.yml'
    config = yaml.safe_load((ROOT / 'configs/tsc' / overlay).read_text())
    if phase_input:
        config['model'].update(phase=True, one_hot=True)
    output = tmp_path / 'run'
    config['command']['output_path'] = str(output)
    config['trainer'].update(episodes=episodes, evaluation_episodes=[0, episodes],
                             resumable_checkpoint_episodes=[0, episodes])
    path = tmp_path / 'run.yml'
    path.write_text(yaml.safe_dump(config))
    cmd = [sys.executable, '-m', 'tools.run_paper_baseline', '-w', 'sumo', '-a', 'paper_' + owner,
           '-n', 'hz4x4', '--seed', '7', '--interface', 'libsumo', '--prefix', 'validation',
           '--experiment-config', str(path)]
    log = (tmp_path / 'process.log').open('w')
    env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', CUDA_VISIBLE_DEVICES='-1')
    process = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    return process, log, output, cmd


@pytest.mark.parametrize('owner,phase_input', [('frap', False), ('presslight', False),
                                            ('colight', False), ('colight', True)],
                         ids=['frap', 'presslight', 'colight', 'colight_phase'])
@pytest.mark.parametrize('corrected_signals', [False, True], ids=['legacy', 'signal_v1'])
def test_real_reward_training_and_dashboard(owner, phase_input, corrected_signals, tmp_path):
    pytest.importorskip('tensorboard')
    if owner == 'colight':
        pytest.importorskip('torch_geometric')
    process, log, output, cmd = launch(tmp_path, owner, corrected_signals=corrected_signals,
                                     phase_input=phase_input)
    try:
        result = process.wait(timeout=90)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()
    assert result == 0, (tmp_path / 'process.log').read_text()[-5000:]
    contract = json.loads((output / 'reward_contract.json').read_text())
    assert contract['profile']['owner'] == owner
    assert contract['profile']['lane_scope'] == profile(owner).lane_scope
    if corrected_signals:
        assert contract['signal_control'] == {'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5}
        binding = json.loads((output / 'experiment_contract.json').read_text())
        assert binding['seeds'] == {'training': 7, 'sumo': 7, 'events': None,
                                    'event_mode': 'fixed_schedule_no_rng'}
        assert len(binding['schedule']['events']) == 3
    assert json.loads((output / 'monitor/status.json').read_text())['status'] == 'completed'
    assert len(read_rows(output / 'monitor/decisions.jsonl')) == 60
    assert len(read_rows(output / 'monitor/episodes.jsonl')) == 4
    assert len(read_rows(output / 'monitor/updates.jsonl')) == 55
    for row in read_rows(output / 'monitor/episodes.jsonl'):
        if row['record_type'] == 'TRAIN':
            decisions = [r['reward_mean'] for r in read_rows(output / 'monitor/decisions.jsonl')
                         if r['episode'] == row['episode']]
            assert row['reward_per_node_mean'] == pytest.approx(np.mean(decisions), rel=1e-5, abs=1e-8)
    assert any(r['active_reports'] for r in read_rows(output / 'monitor/decisions.jsonl'))
    saved = torch.load(output / 'checkpoints/resumable/episode_0002.pt', map_location='cpu')
    assert saved['reward_contract_hash'] == contract['hash']
    if phase_input:
        initial = torch.load(output / 'checkpoints/evaluation/episode_0000.pt', map_location='cpu')
        before = initial['agents'][0]['online_model_state_dict']
        after = saved['agents'][0]['online_model_state_dict']
        assert after['_phase_input_version'].item() == 1
        weight = 'embedding_MLP.embedding_node.node_embedding_0.weight'
        assert after[weight].shape[1] == 20
        assert torch.count_nonzero(after[weight][:, 12:] - before[weight][:, 12:]) > 0
    if corrected_signals:
        assert saved['experiment_contract_hash'] == binding['hash']
    # The actual buffer fed to TD updates contains native-scaled per-interval rewards.
    reward_arrays = [np.asarray(item[1][3]) for agent in saved['agents'] for item in agent['replay_state']['items']]
    assert all(np.all(r <= 0) and np.all(np.isfinite(r)) for r in reward_arrays)
    assert any(np.any(r < 0) for r in reward_arrays)
    assert saved['gradient_updates'] > 0
    for ext in ('csv', 'jsonl'):
        assert (output / f'monitor/decisions.{ext}').stat().st_size > 1000
    (tmp_path / 'validation.json').write_text(json.dumps({'status': 'passed', 'command': cmd,
        'reward_contract_hash': contract['hash'], 'owner': owner, 'simulator': 'SUMO 1.27.1',
        'network': 'hz4x4', 'seed': 7, 'train_episodes': 2, 'seconds': 300,
        'signal_control': contract.get('signal_control', 'legacy'),
        'phase_input': phase_input,
        'scope': 'reward_and_monitor_engineering_not_policy_efficacy'}, indent=2))


def test_graceful_stop_exports_partial_training(tmp_path):
    pytest.importorskip('tensorboard')
    process, log, output, cmd = launch(tmp_path, 'frap', episodes=30)
    try:
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if len(read_rows(output / 'monitor/decisions.jsonl')) >= 3:
                break
            assert process.poll() is None, (tmp_path / 'process.log').read_text()[-3000:]
            time.sleep(.1)
        else:
            pytest.fail('No live decision data within timeout')
        # Stop the trainer, not the independently served TensorBoard process.
        process.terminate()
        assert process.wait(timeout=20) == 130
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        log.close()
    assert json.loads((output / 'monitor/status.json').read_text())['status'] == 'interrupted'
    assert len(read_rows(output / 'monitor/decisions.jsonl')) >= 3
    assert (output / 'monitor/decision_curves.png').stat().st_size > 1000
    assert json.loads((output / 'run_status.json').read_text())['exit_code'] == 130


def test_checkpoint_refuses_wrong_reward_contract(tmp_path):
    from trainer.paper_trainer import PaperTrainer
    trainer = object.__new__(PaperTrainer)
    trainer.reward_contract = {'hash': 'expected'}
    payload = {'schema_version': 1, 'checkpoint_type': 'evaluation', 'episode': 0,
               'global_decision_step': 0, 'gradient_updates': 0, 'config_hash': 'a' * 64,
               'agents': [{'rank': 0, 'online_model_state_dict': {}}], 'reward_contract_hash': 'other'}
    path = tmp_path / 'wrong.pt'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='reward contract mismatch'):
        trainer.load_checkpoint_payload(path, 'evaluation')
    payload['reward_contract_hash'] = 'expected'
    trainer.experiment_contract = {'hash': 'new-events-or-seeds'}
    payload['experiment_contract_hash'] = 'old-events-or-seeds'
    torch.save(payload, path)
    with pytest.raises(ValueError, match='experiment contract mismatch'):
        trainer.load_checkpoint_payload(path, 'evaluation')


@pytest.mark.parametrize('owner,phase_input', [('frap', False), ('presslight', False),
                                            ('colight', False), ('colight', True)],
                         ids=['frap', 'presslight', 'colight', 'colight_phase'])
def test_corrected_training_reproducible(owner, phase_input, tmp_path):
    from trainer.tsc_trainer import _state_values_equal
    payloads, bindings, commands = [], [], []
    for repeat in range(2):
        directory = tmp_path / str(repeat)
        directory.mkdir()
        process, log, output, cmd = launch(directory, owner, corrected_signals=True,
                                         phase_input=phase_input)
        commands.append(cmd)
        try:
            assert process.wait(timeout=90) == 0, (directory / 'process.log').read_text()[-4000:]
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
            log.close()
        payloads.append(torch.load(output / 'checkpoints/resumable/episode_0002.pt', map_location='cpu'))
        bindings.append(json.loads((output / 'experiment_contract.json').read_text()))
    assert _state_values_equal(payloads[0]['agents'], payloads[1]['agents'])
    assert bindings[0]['hash'] == bindings[1]['hash']
    for name in ['decisions.jsonl', 'updates.jsonl']:
        assert read_rows(tmp_path / '0/run/monitor' / name) == read_rows(tmp_path / '1/run/monitor' / name)
    (tmp_path / 'reproducibility.json').write_text(json.dumps({
        'status': 'passed', 'owner': owner, 'training_seed': 7, 'sumo_seed': 7,
        'phase_input': phase_input,
        'event_mode': 'fixed', 'repeats': 2, 'seconds_per_episode': 300, 'episodes': 2,
        'contract_hash': bindings[0]['hash'], 'same_agent_checkpoint_state': True,
        'same_decisions_rewards_losses': True, 'scope': 'p0_engineering_only',
        'commands': commands,
    }, indent=2) + '\n')
