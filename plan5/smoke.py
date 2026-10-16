"""Real-SUMO Phase0 smoke and boundary-checkpoint equivalence runners."""
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from sequential.core import canonical_digest
from sequential.core import capture_rng_state
from sequential.io import atomic_json, sha256_file
from sequential.trainer import SequentialStageTrainer
from agent import utils as agent_utils

from .agent import Plan5CTXDDQNAgent, Plan5DDQNAgent, nested_initialize
from .checkpoint import restore_checkpoint, save_checkpoint
from .config import load_config
from .ppo_adapter import Plan5PPOAdapter
from .sumo_runtime import build_world


def seed_all(seed):
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))


def restore_training_rng(state):
    random.setstate(state['python_random_state'])
    np.random.set_state(state['numpy_random_state'])
    torch.set_rng_state(state['torch_cpu_rng_state'])
    if torch.cuda.is_available() and state['torch_cuda_rng_states']:
        torch.cuda.set_rng_state_all(state['torch_cuda_rng_states'])


def _configs(config):
    ddqn = config.resolved['ddqn']
    model = {
        'phase': True, 'one_hot': True, 'gamma': ddqn['gamma'],
        'grad_clip': ddqn['gradient_clip'],
        'epsilon_decay': ddqn['epsilon_decay'],
        'epsilon_min': ddqn['epsilon_min'],
        'epsilon': ddqn['epsilon_initial'],
        'learning_rate': ddqn['learning_rate'],
    }
    trainer = {
        'batch_size': ddqn['batch_size'],
        'learning_start': ddqn['learning_start'],
        'buffer_size': ddqn['replay_capacity'],
        'target_update_interval': ddqn['target_sync_successful_updates'],
        'steps': config.simulation_duration,
        'action_interval': config.decision_interval,
        'update_model_rate': 1,
    }
    return model, trainer


def _build_agent(algorithm_id, world, seed, config, ppo_parameters=None):
    seed_all(seed)
    if algorithm_id == 'DDQN':
        model, trainer = _configs(config)
        return Plan5DDQNAgent(world, 0, model, trainer), None
    if algorithm_id == 'CTXDDQN':
        model, trainer = _configs(config)
        source = Plan5DDQNAgent(world, 0, model, trainer)
        source_rng = capture_rng_state()
        target = Plan5CTXDDQNAgent(world, 0, model, trainer)
        raw_batch = np.arange(16 * 8, dtype=np.float32).reshape(8, 16)
        paired = nested_initialize(source, target, raw_batch)
        # Match the exact stream position of the paired DDQN construction.
        restore_training_rng(source_rng)
        paired['training_rng_digest'] = canonical_digest(source_rng)
        return target, paired
    if algorithm_id == 'PPO':
        if ppo_parameters is None:
            ppo_parameters = {
                'learning_rate': 2.5e-4,
                'entropy_coefficient': 0.001,
            }
        return Plan5PPOAdapter(
            world, 0, ppo_parameters['learning_rate'],
            ppo_parameters['entropy_coefficient'], training_seed=seed,
        ), None
    raise ValueError(f'Unsupported Plan5 algorithm: {algorithm_id}')


def _train_ppo_episode(agent, world, episode):
    world.reset(); agent.rebind_environment(world)
    observation = agent.get_ob()
    records = []
    for decision in range(1, 361):
        phase = agent.get_phase()
        feature = agent._feature(observation, phase)[0]
        action = agent.get_action(observation, phase, test=False)
        rewards = []
        for _ in range(10):
            world.step(action)
            rewards.append(float(np.asarray(agent.get_reward()).reshape(-1)[0]))
        next_observation = agent.get_ob()
        next_phase = agent.get_phase()
        reward = float(np.mean(rewards))
        truncated = decision == 360
        update = agent.observe(
            next_observation, next_phase, reward,
            terminated=False, truncated=truncated,
        )
        records.append({
            'episode': int(episode), 'decision': decision,
            'action': int(np.asarray(action).reshape(-1)[0]),
            'reward': reward,
            'model_input': feature.tolist(),
            'minibatch_updates': update['minibatch_updates'],
        })
        observation = next_observation
    audit = _episode_audit(records, input_key='model_input')
    return {
        'episode': int(episode), 'decision_steps': 360,
        'minibatch_updates': sum(r['minibatch_updates'] for r in records),
        'mean_reward': float(np.mean([r['reward'] for r in records])),
        'reward_std': audit['reward_std'],
        'action_counts': audit['action_counts'],
        'action_frequencies': audit['action_frequencies'],
        'model_input_summary': audit['model_input_summary'],
        'counters_end': {
            'global_decision_step': int(agent.global_decision_step),
            'rollout_updates': int(agent.rollout_updates),
            'minibatch_updates': int(agent.core.n_updates),
        },
        'trajectory_digest': canonical_digest(records),
    }


def _episode_audit(records, *, input_key):
    actions = [int(np.asarray(item['action']).reshape(-1)[0]) for item in records]
    if len(actions) != 360 or any(not 0 <= action < 8 for action in actions):
        raise ValueError('Plan5 episode action audit requires 360 actions in 0..7')
    counts = [actions.count(action) for action in range(8)]
    features = np.asarray([item[input_key] for item in records], dtype=np.float32)
    if features.ndim != 2 or features.shape[0] != 360 \
            or features.shape[1] not in {16, 20}:
        raise ValueError('Plan5 episode model-input audit has invalid shape')
    rewards = np.asarray([
        float(np.asarray(item['reward']).reshape(-1)[0]) for item in records
    ], dtype=np.float64)
    return {
        'action_counts': counts,
        'action_frequencies': [count / 360.0 for count in counts],
        'reward_std': float(np.std(rewards)),
        'model_input_summary': {
            'feature_dim': int(features.shape[1]),
            'mean': features.mean(axis=0).astype(float).tolist(),
            'std': features.std(axis=0).astype(float).tolist(),
            'min': features.min(axis=0).astype(float).tolist(),
            'max': features.max(axis=0).astype(float).tolist(),
        },
    }


def _dqn_model_input(agent, algorithm_id, transition):
    if algorithm_id == 'CTXDDQN':
        return agent._ctx_feature(
            transition['state'], transition['phase'],
        )[0].tolist()
    phase = agent_utils.idx2onehot(
        transition['phase'], agent.action_space.n,
    ).astype(np.float32)
    return np.concatenate([
        np.asarray(transition['state'], dtype=np.float32), phase,
    ], axis=1)[0].tolist()


def _train_episodes(agent, world, algorithm_id, start_episode, end_episode,
                    config):
    summaries = []
    if algorithm_id in {'DDQN', 'CTXDDQN'}:
        _, trainer_config = _configs(config)
        trainer = SequentialStageTrainer(agent, world, trainer_config)
        for episode in range(start_episode, end_episode + 1):
            result = trainer.train_episode(episode, episode)
            audited = [
                {
                    **row,
                    'model_input': _dqn_model_input(agent, algorithm_id, row),
                }
                for row in result['transitions']
            ]
            audit = _episode_audit(audited, input_key='model_input')
            summaries.append({
                'episode': episode, 'decision_steps': result['decision_steps'],
                'gradient_updates': result['gradient_updates'],
                'loss_mean': result['loss_mean'],
                'mean_reward': float(np.mean([
                    float(np.asarray(row['reward']).reshape(-1)[0])
                    for row in result['transitions']
                ])),
                'reward_std': audit['reward_std'],
                'action_counts': audit['action_counts'],
                'action_frequencies': audit['action_frequencies'],
                'model_input_summary': audit['model_input_summary'],
                'epsilon_end': float(agent.epsilon),
                'replay_size_end': len(agent.replay.records),
                'counters_end': {
                    'global_decision_step': int(
                        agent.counters.global_decision_step
                    ),
                    'gradient_updates': int(agent.counters.gradient_updates),
                    'target_updates': int(agent.counters.target_updates),
                },
                'trajectory_digest': canonical_digest(result['transitions']),
            })
    else:
        for episode in range(start_episode, end_episode + 1):
            summaries.append(_train_ppo_episode(agent, world, episode))
    return summaries


def run_smoke(algorithm_id, output_dir, *, seed=999, episodes=None,
              network='sumohz1x1', interface='traci', resume_from=None,
              verify_evaluation_isolation=True):
    config = load_config()
    episodes = int(episodes or (1 if algorithm_id == 'PPO' else 3))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    world = None
    started = time.perf_counter()
    try:
        world, command = build_world(
            network, output_dir / 'sumo', interface=interface, context=True,
        )
        agent, paired = _build_agent(algorithm_id, world, seed, config)
        if algorithm_id in {'DDQN', 'CTXDDQN'}:
            agent.begin_stage(1, 'S2', 'clear')
        else:
            agent.begin_stage(1, 'S2')
        start_episode = 1
        if resume_from is not None:
            payload = restore_checkpoint(
                agent, resume_from,
                {'algorithm_id': algorithm_id, 'training_seed': int(seed)},
            )
            start_episode = int(payload['episode']) + 1
        summaries = _train_episodes(
            agent, world, algorithm_id, start_episode, episodes, config,
        )
        checkpoint_path = output_dir / f'episode_{episodes:04d}.pt'
        checkpoint = save_checkpoint(
            agent, checkpoint_path,
            {'algorithm_id': algorithm_id, 'training_seed': int(seed),
             'network': network, 'run_type': 'smoke'},
            episode=episodes, stage=1,
            extra_state={'summaries': summaries},
        )
        evaluation_isolation = None
        if verify_evaluation_isolation:
            from .frozen_evaluator import (
                Plan5FrozenEvaluator, save_evaluation_snapshot,
            )
            snapshot_path = output_dir / 'evaluation_snapshot.pt'
            save_evaluation_snapshot(
                agent, snapshot_path,
                {'algorithm_id': algorithm_id, 'training_seed': int(seed),
                 'network': network, 'run_type': 'smoke'},
            )
            evaluation_isolation = Plan5FrozenEvaluator(
                python_executable=os.sys.executable,
            ).evaluate(
                agent, snapshot_path, network,
                output_dir / 'frozen_evaluation', interface=interface,
                parent_checkpoint=checkpoint_path,
            )
        state = agent.full_state_dict()
        if algorithm_id in {'DDQN', 'CTXDDQN'}:
            counters = state['counters']
            if counters['gradient_updates'] < 1 or counters['target_updates'] < 1:
                raise RuntimeError('DDQN-family smoke did not update/sync target')
        else:
            agent.assert_rollout_empty()
            counters = state['counters']
            if counters['rollout_updates'] != episodes:
                raise RuntimeError('PPO smoke rollout update count mismatch')
        report = {
            'schema_version': 1, 'algorithm_id': algorithm_id,
            'config_sha256': config.sha256(),
            'network': network, 'training_seed': int(seed),
            'episodes': episodes, 'steps_per_episode': 3600,
            'decisions_per_episode': 360,
            'resolved_sumo_command': command,
            'invocation_command': [
                os.sys.executable, '-m', 'plan5', 'smoke',
                '--algorithm', algorithm_id, '--output',
                str(output_dir.resolve()), '--seed', str(int(seed)),
                '--episodes', str(episodes), '--interface', interface,
            ] + ([] if verify_evaluation_isolation else [
                '--skip-evaluation-isolation'
            ]),
            'exit_code': 0,
            'summaries': summaries, 'counters': counters,
            'paired_initialization': paired,
            'parameter_ownership': (
                agent.parameter_ownership()
                if algorithm_id == 'PPO' else None
            ),
            'checkpoint_path': str(checkpoint_path.resolve()),
            'checkpoint_sha256': sha256_file(checkpoint_path),
            'canonical_state_digest': checkpoint['canonical_state_digest'],
            'evaluation_isolation': evaluation_isolation,
            'wall_time_seconds': time.perf_counter() - started,
            'valid': True,
        }
        atomic_json(output_dir / 'smoke_report.json', report)
        return report
    finally:
        if world is not None: world.close()


def resume_equivalence(algorithm_id, output_dir, *, seed=998,
                       network='sumohz1x1', interface='traci'):
    config = load_config()
    final_episode = 2 if algorithm_id == 'PPO' else 4
    split_episode = 1 if algorithm_id == 'PPO' else 2
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=False)

    def execute(path, stop, resume=None):
        world = None
        try:
            world, command = build_world(
                network, path / 'sumo', interface=interface, context=True,
            )
            agent, _ = _build_agent(algorithm_id, world, seed, config)
            if algorithm_id in {'DDQN', 'CTXDDQN'}:
                agent.begin_stage(1, 'S2', 'clear')
            else:
                agent.begin_stage(1, 'S2')
            start = 1
            if resume:
                payload = restore_checkpoint(
                    agent, resume,
                    {'algorithm_id': algorithm_id, 'training_seed': seed},
                )
                start = int(payload['episode']) + 1
            summaries = _train_episodes(
                agent, world, algorithm_id, start, stop, config,
            )
            checkpoint_path = path / f'episode_{stop:04d}.pt'
            payload = save_checkpoint(
                agent, checkpoint_path,
                {'algorithm_id': algorithm_id, 'training_seed': seed,
                 'network': network, 'run_type': 'resume_equivalence'},
                episode=stop, stage=1,
            )
            return {
                'state': agent.full_state_dict(), 'summaries': summaries,
                'checkpoint_path': checkpoint_path,
                'checkpoint_sha256': sha256_file(checkpoint_path),
                'checkpoint': payload, 'resolved_sumo_command': command,
                'start_episode': start, 'stop_episode': stop,
                'resumed_from': None if resume is None else str(
                    Path(resume).resolve()
                ),
            }
        finally:
            if world is not None: world.close()

    continuous = execute(
        output_dir / 'continuous', final_episode,
    )
    prefix = execute(
        output_dir / 'split_prefix', split_episode,
    )
    resumed = execute(
        output_dir / 'resumed', final_episode,
        resume=prefix['checkpoint_path'],
    )
    state_equal = canonical_digest(continuous['state']) == canonical_digest(
        resumed['state']
    )
    metrics_equal = continuous['summaries'] == (
        prefix['summaries'] + resumed['summaries']
    )
    counters_equal = (
        continuous['state'].get('counters')
        == resumed['state'].get('counters')
    )
    def execution_record(item):
        return {
            'start_episode': item['start_episode'],
            'stop_episode': item['stop_episode'],
            'resumed_from': item['resumed_from'],
            'resolved_sumo_command': item['resolved_sumo_command'],
            'summaries': item['summaries'],
            'checkpoint_path': str(item['checkpoint_path'].resolve()),
            'checkpoint_sha256': item['checkpoint_sha256'],
            'canonical_state_digest': item['checkpoint'][
                'canonical_state_digest'
            ],
            'counters': item['state'].get('counters'),
        }
    report = {
        'schema_version': 1, 'algorithm_id': algorithm_id,
        'training_seed': seed, 'network': network,
        'final_episode': final_episode, 'split_episode': split_episode,
        'steps_per_episode': 3600, 'decisions_per_episode': 360,
        'invocation_command': [
            os.sys.executable, '-m', 'plan5', 'resume-equivalence',
            '--algorithm', algorithm_id, '--output',
            str(output_dir.resolve()), '--seed', str(int(seed)),
            '--interface', interface,
        ],
        'continuous_state_digest': canonical_digest(continuous['state']),
        'resumed_state_digest': canonical_digest(resumed['state']),
        'continuous_checkpoint_digest': continuous['checkpoint'][
            'canonical_state_digest'
        ],
        'resumed_checkpoint_digest': resumed['checkpoint'][
            'canonical_state_digest'
        ],
        'state_equal': state_equal, 'metrics_equal': metrics_equal,
        'counters_equal': counters_equal,
        'executions': {
            'continuous': execution_record(continuous),
            'split_prefix': execution_record(prefix),
            'resumed': execution_record(resumed),
        },
        'exit_code': 0,
        'valid': state_equal and metrics_equal and counters_equal,
    }
    if not report['valid']:
        raise RuntimeError(f'{algorithm_id} resume equivalence failed')
    atomic_json(output_dir / 'resume_equivalence_report.json', report)
    return report
