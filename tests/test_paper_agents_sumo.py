"""FRAP/CoLight hz4x4 readiness: actual trainer, events, gradients, checkpoints.

Run from the repository root with the colight interpreter:
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s tests/test_paper_agents_sumo.py

The event test installs events at World creation using a test-only hook. It does
not add an event CLI flag or change frozen agent/trainer sources. Every case
writes readiness.json with simulator, seeds, invocation and source hashes.
"""
import hashlib
import importlib.metadata
import json
import logging
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip('libsumo')
from trainer.tsc_trainer import TSCTrainer
from utils.logger import hash_torch_state_dict
from world.sumo_events import Schedule, install_events, load_schedule
from world.sumo_signal_control import install_signal_control


ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / 'configs/tsc/hz4x4_paper_smoke.yml'


def check_geometry(trainer, method):
    world = trainer.world
    assert len(world.intersections) == 16
    records = []
    if method in {'frap', 'presslight'}:
        assert len(trainer.agents) == 16
        assert len({id(a.model) for a in trainer.agents}) == 16
        assert all(a.sub_agents == 1 for a in trainer.agents)
        for agent, inter in zip(trainer.agents, world.intersections):
            assert agent.id == inter.id
            lanes = [lane for row in agent.ob_generator.lanes for lane in row]
            if method == 'presslight':
                expected = {l for road in inter.in_roads + inter.out_roads
                            for l in inter.road_lane_mapping[road]}
                assert len(lanes) == 24 and set(lanes) == expected
                assert agent.phase and agent.one_hot and agent.ob_length == 32
                records.append({'intersection': inter.id, 'observation_lanes': lanes,
                                'phase_one_hot': 8, 'observation_cutoff_m': world.max_distance})
                continue
            links = world.eng.trafficlight.getControlledLinks(inter.id)
            # Read green-phase signal states and actual controlled links, not
            # the FRAP config, to reconstruct served incoming lanes.
            served = [{lane_link[0] for signal, group in zip(phase.state, links)
                       if signal in ('G', 's') for lane_link in group}
                      for phase in inter.green_phases]
            common = set.intersection(*served)
            assert len(common) == 4  # hz4x4 has four always-open right-turn lanes.
            pairs = [sorted(lanes.index(lane) for lane in row - common) for row in served]
            assert len(pairs) == agent.action_space.n == 8
            assert all(len(pair) == 2 for pair in pairs)
            assert pairs == [sorted(pair) for pair in agent.phase_pairs]
            records.append({'intersection': inter.id, 'observation_lanes': lanes,
                            'always_open_lanes': sorted(common), 'phase_pairs': pairs})
    else:
        assert len(trainer.agents) == 1
        agent = trainer.agents[0]
        assert agent.sub_agents == 16
        assert agent.get_ob().shape == (16, 12)
        expected_width = 12 + (8 if agent.one_hot else 1) if agent.phase else 12
        assert agent.ob_length == expected_width
        graph_ids = [agent.graph['node_idx2id'][i] for i in range(16)]
        world_ids = [i[3:] if i.startswith('GS_') else i for i in world.intersection_ids]
        # Legacy CoLight returns vectors in World order. The graph must match
        # that order for this admitted scenario; do not sort just observations.
        assert graph_ids == world_ids
        assert [item[0] for item in agent.ob_generator] == list(range(16))
        expected_edges = {(i, j) for i, left in enumerate(world.intersections)
                          for j, right in enumerate(world.intersections)
                          if i != j and set(left.out_roads) & set(right.in_roads)}
        actual_edges = set(map(tuple, agent.edge_idx.T.tolist()))
        assert expected_edges == actual_edges
        records.append({'node_ids': graph_ids, 'directed_edges': sorted(actual_edges)})
    return records


@pytest.mark.parametrize('method,phase_mode', [
    ('frap', None), ('colight', None), ('presslight', None),
    ('colight', 'one_hot'), ('colight', 'index')],
    ids=['frap', 'colight', 'presslight', 'colight_phase_one_hot', 'colight_phase_index'])
@pytest.mark.parametrize('events', [False, True], ids=['normal', 'events'])
@pytest.mark.parametrize('corrected_signals', [False, True], ids=['legacy', 'signal_v1'])
def test_hz4x4_training_readiness(method, phase_mode, events, corrected_signals, tmp_path, monkeypatch):
    if method == 'colight':
        pytest.importorskip('torch_scatter')
        pytest.importorskip('torch_geometric')
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    config = yaml.safe_load(OVERLAY.read_text())
    if phase_mode is not None:
        config['model'].update(phase=True, one_hot=phase_mode == 'one_hot')
    config['command']['output_path'] = str(tmp_path / 'run')
    config_path = tmp_path / 'overlay.yml'
    config_path.write_text(yaml.safe_dump(config))
    argv = ['run.py', '-w', 'sumo', '-a', method, '-n', 'hz4x4', '--seed', '7',
            '--interface', 'libsumo', '--prefix', 'readiness', '--experiment-config', str(config_path)]
    monkeypatch.setattr(sys, 'argv', argv)
    namespace = runpy.run_path(str(ROOT / 'run.py'), run_name='paper_agent_readiness')
    logger = logging.Logger(f'readiness_{method}_{events}')
    handler = logging.FileHandler(tmp_path / 'trainer.log')
    logger.addHandler(handler)
    # Pytest owns root logging handlers, which are not the FileHandler expected
    # by this legacy trainer. Give Runner a private file logger for this test.
    monkeypatch.setitem(namespace['Runner'].run.__globals__, 'setup_logging', lambda level: logger)
    runner = namespace['Runner'](namespace['args'])
    (Path(runner.output_path) / config.get('logger', {}).get('log_dir', 'logger')).mkdir(exist_ok=True)
    schedule = load_schedule(ROOT / 'configs/events/hz4x4.yml') if events else Schedule(())
    original_create = TSCTrainer.create_world
    original_reset = TSCTrainer.create_agents
    runtime_holder, geometry, boundaries, initial_hashes, gradient_visits = [], [], [], [], set()
    departed, arrived = set(), set()
    phase_gradient_visits = set()

    def create_world(trainer):
        original_create(trainer)
        if corrected_signals:
            install_signal_control(trainer.world, {
                'version': 'sumo-green-yellow-v1', 'yellow_seconds': 5})
        runtime = install_events(trainer.world, schedule)
        runtime_holder.append(runtime)
        original_step = trainer.world.step

        def step(actions=None):
            value = original_step(actions)
            now = trainer.world.get_current_time()
            if corrected_signals:
                raw = runtime._engine
                assert raw.simulation.getCollidingVehiclesNumber() == 0
                assert raw.simulation.getStartingTeleportNumber() == 0
                departed.update(trainer.world.last_entered_vehicle_ids)
                arrived.update(trainer.world.last_exited_vehicle_ids)
                assert departed - arrived == set(trainer.world.eng.vehicle.getIDList())
                if now % trainer.action_interval == 0:
                    for inter in trainer.world.intersections:
                        assert raw.trafficlight.getRedYellowGreenState(inter.id) == inter.green_phases[inter.current_phase].state
            if events and now in {e.begin for e in schedule.events} | {e.end for e in schedule.events}:
                boundaries.append({'time': now, 'reports': [r.__dict__ for r in runtime.reports()]})
            return value

        trainer.world.step = step

    def create_agents(trainer):
        original_reset(trainer)
        initial_hashes.extend(hash_torch_state_dict(a.model.state_dict()) for a in trainer.agents)
        # World's constructor closes its initial connection; geometry checks run
        # on the first real reset, without changing any random state.
        original_world_reset = trainer.world.reset

        def reset():
            value = original_world_reset()
            departed.clear()
            arrived.clear()
            if not geometry:
                geometry.extend(check_geometry(trainer, method))
            return value

        trainer.world.reset = reset
        for rank, agent in enumerate(trainer.agents):
            # Register a hook on a real model weight, proving training gradients
            # reached every local/shared policy instead of merely filling replay.
            parameter = next(agent.model.parameters())
            def hook(grad, rank=rank):
                assert torch.isfinite(grad).all()
                if torch.count_nonzero(grad):
                    gradient_visits.add(rank)
                if phase_mode and torch.count_nonzero(grad[:, 12:]):
                    phase_gradient_visits.add(rank)
            parameter.register_hook(hook)

    monkeypatch.setattr(TSCTrainer, 'create_world', create_world)
    monkeypatch.setattr(TSCTrainer, 'create_agents', create_agents)
    try:
        runner.run()
        trainer = runner.trainer
        assert trainer.global_decision_step == 60
        expected_agents = 1 if method == 'colight' else 16
        assert trainer.gradient_updates == 55 * expected_agents
        assert gradient_visits == set(range(expected_agents))
        if phase_mode:
            assert phase_gradient_visits == {0}
            ag = trainer.agents[0]
            samples = list(ag.replay_buffer)[-2:]
            current, following, _, _ = ag._batchwise(samples)
            for idx, (_, row) in enumerate(samples):
                for features, phase in [(current.x[idx * 16:(idx + 1) * 16, 12:], row[1]),
                                        (following.x[idx * 16:(idx + 1) * 16, 12:], row[5])]:
                    expected = np.eye(8)[phase] if ag.one_hot else np.asarray(phase)[:, None]
                    np.testing.assert_array_equal(features.numpy(), expected)
        final_hashes = [hash_torch_state_dict(a.model.state_dict()) for a in trainer.agents]
        assert all(a != b for a, b in zip(initial_hashes, final_hashes))
        assert all(len(a.replay_buffer) == 60 for a in trainer.agents)
        assert len(trainer.evaluation_isolation_checks) == 2
        assert all(row['online_model_unchanged'] and row['replay_unchanged'] and row['rng_unchanged']
                   for row in trainer.evaluation_isolation_checks)
        if events:
            assert len(boundaries) == 6 * 4  # initial/final eval plus two train episodes
            for row in boundaries:
                for public in row['reports']:
                    event = next(e for e in schedule.events if e.event_id == public['event_id'])
                    assert public['status'] == ('active' if row['time'] < event.end else 'cleared')
        else:
            assert not boundaries and not runtime_holder[0].reports()
        assert all(not hasattr(a, 'runtime') and not hasattr(a, 'text_encoder') for a in trainer.agents)
        evaluation = Path(trainer._checkpoint_path('evaluation', 2))
        resumable = Path(trainer._checkpoint_path('resumable', 2))
        payload = trainer.load_checkpoint_payload(str(resumable), 'resumable')
        assert len(payload['agents']) == expected_agents
        saved_epsilon = [a.epsilon for a in trainer.agents]
        for a in trainer.agents:
            with torch.no_grad():
                next(a.model.parameters()).add_(1)
        trainer.load_online_checkpoint(str(evaluation), expected_type='evaluation')
        assert [hash_torch_state_dict(a.model.state_dict()) for a in trainer.agents] == final_hashes
        for a in trainer.agents:
            a.replay_buffer.clear()
            a.epsilon = 0.
        trainer.load_resumable_checkpoint(str(resumable))
        assert [a.epsilon for a in trainer.agents] == saved_epsilon
        assert all(len(a.replay_buffer) == 60 for a in trainer.agents)
        files = ['agent/' + method + '.py', 'trainer/tsc_trainer.py', 'requirements-colight.txt',
                 'configs/tsc/hz4x4_paper_smoke.yml', 'configs/tsc/' + method + '.yml',
                 'tests/test_paper_agents_sumo.py', 'world/sumo_events/runtime.py',
                 'configs/events/hz4x4.yml']
        evidence = {'status': 'passed', 'scope': 'engineering_readiness_not_efficacy',
                    'method': method, 'events': events, 'training_seed': 7, 'sumo_seed': 7,
                    'phase_mode': phase_mode or 'disabled',
                    'phase_columns_received_gradient': bool(phase_gradient_visits),
                    'signal_control': getattr(trainer.world, 'signal_control_config', 'legacy'),
                    'decision_phase_matches_physical_green': corrected_signals,
                    'checked_no_collisions_teleports_or_vehicle_loss': corrected_signals,
                    'network': 'hz4x4', 'sumo_version': trainer.world.eng.getVersion(),
                    'python': sys.executable, 'runner_argv': argv, 'sumo_command': trainer.world.sumo_cmd,
                    'event_installation': 'test-only World creation hook calling public install_events',
                    'test_command': 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q -s tests/test_paper_agents_sumo.py',
                    'agents': expected_agents, 'train_episodes': 2, 'seconds_per_episode': 300,
                    'decision_steps': trainer.global_decision_step, 'gradient_updates': trainer.gradient_updates,
                    'geometry': geometry, 'boundaries': boundaries,
                    'evaluation_isolation': trainer.evaluation_isolation_checks,
                    'evaluation_checkpoint_roundtrip': True, 'resumable_checkpoint_roundtrip': True,
                    'versions': {n: importlib.metadata.version(n) for n in ('torch', 'torch-scatter', 'torch-geometric')
                                 if importlib.util.find_spec(n.replace('-', '_'))},
                    'sha256': {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in files}}
        evidence['sha256']['world/sumo_signal_control.py'] = hashlib.sha256(
            (ROOT / 'world/sumo_signal_control.py').read_bytes()).hexdigest()
        assert trainer.world.sumo_cmd[trainer.world.sumo_cmd.index('--seed') + 1] == '7'
        (tmp_path / 'readiness.json').write_text(json.dumps(evidence, indent=2) + '\n')
        print('Readiness evidence:', tmp_path / 'readiness.json')
    finally:
        if hasattr(runner, 'trainer'):
            runner.trainer.world.close()
        handler.close()
        torch.set_num_threads(threads)


def test_colight_graph_batches_do_not_mix():
    pytest.importorskip('torch_scatter')
    pytest.importorskip('torch_geometric')
    from agent.colight import MultiHeadAttModel
    from torch_geometric.data import Batch, Data
    torch.manual_seed(7)
    model = MultiHeadAttModel(8, 4, 8, 2, 0)
    edge = torch.tensor([[0, 1, 2], [1, 2, 0]])
    left, right = torch.randn(3, 8), torch.randn(3, 8) + 4
    graph = Batch.from_data_list([Data(x=left, edge_index=edge), Data(x=right, edge_index=edge)])
    actual = model(graph.x, graph.edge_index)
    expected = torch.cat([model(left, edge), model(right, edge)])
    torch.testing.assert_close(actual, expected)
    actual.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
