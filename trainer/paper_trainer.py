"""Opt-in paper reward and monitoring integration over the existing trainer.

Registered only by tools.run_paper_baseline, leaving frozen standard entrypoints
unchanged. New experiments opt into versioned signal control via configuration.
"""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np
import torch

from common.registry import Registry
from trainer.tsc_trainer import TSCTrainer
from utils.training_monitor import TrainingMonitor
from world.paper_rewards import SumoRewardSource
from world.sumo_events import Schedule, install_events, load_schedule
from world.sumo_signal_control import install_signal_control
from common.paper_experiment import checked_json, digest, file_digest, write_json


class MetricMirror:
    def __init__(self, original, monitor):
        self.original, self.monitor = original, monitor

    def append(self, record):
        self.original.append(record)
        self.monitor.episode(record)

    def validate(self, *args, **kwargs):
        return self.original.validate(*args, **kwargs)

    @property
    def path(self):
        return self.original.path


class PaperTrainer(TSCTrainer):
    def train(self):
        settings = Registry.mapping['trainer_mapping']['setting'].param
        extension = settings.get('training_extension')
        if extension and not self.resume_episode:
            from common.paper_experiment import extension_checkpoint
            # This explicitly audited stage transition is separate from crash
            # resume. Never mutate the source checkpoint or weaken its checks.
            payload = extension_checkpoint(self, extension)
            self._extension_payload = payload
            try:
                self.load_resumable_checkpoint(extension['checkpoint'])
            finally:
                del self._extension_payload
            self.resume_episode = payload['episode']
            self.episode_events.cursor = payload['episode']
            self.evaluation_results[payload['episode']] = {'status': 'published'}
            self.save_checkpoint('evaluation', payload['episode'])
            self.save_checkpoint('resumable', payload['episode'])
            write_json(Path(self.output_path) / 'extension_initialized.json', {
                'parent_checkpoint': extension['checkpoint'], 'parent_sha256': extension['checkpoint_sha256'],
                'parent_episode': self.resume_episode, 'global_decision_step': self.global_decision_step,
                'gradient_updates': self.gradient_updates,
                'replay_lengths': [len(a.replay_buffer) for a in self.agents],
                'epsilon': [a.epsilon for a in self.agents],
                'state_transfer': 'online,target,optimizer,replay,counters,epsilon,python/numpy/torch RNG'})
        return super().train()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        profiles = {a.reward_profile.digest for a in self.agents}
        if len(profiles) != 1:
            raise ValueError('Every node in a run must use the same owned reward contract')
        profile = self.agents[0].reward_profile
        contract = {'schema_version': 'paper-reward-binding-v1', 'profile': profile.to_dict(),
                    'profile_hash': profile.digest, 'network_sha256': self.world.paper_reward_source.net_sha256,
                    'lanes': self.world.paper_reward_source.lane_manifest(profile),
                    'temporal_aggregation': 'per-second rewards averaged once over action_interval',
                    'action_interval': self.action_interval,
                    'scale_application': 'weight/normal_factor applied before replay; no second TD normalization',
                    'qualification': 'reward-adapter engineering validation; full paper fidelity not established',
                    'known_limitations': ['Legacy signal transition requires correction before formal efficacy claims.',
                                          'Legacy policy observations/architectures/updates remain unchanged.']}
        root = Path(__file__).resolve().parents[1]
        sources = ['common/paper_rewards.py', 'world/paper_rewards.py', 'agent/paper_baselines.py',
                   f'agent/{profile.owner}.py']
        signal_config = getattr(self.world, 'signal_control_config', None)
        if signal_config is not None:
            contract['signal_control'] = signal_config
            contract['known_limitations'] = [
                'Author-reward SUMO adaptation; full paper fidelity not established.',
                'Legacy policy observations/architectures/updates remain unchanged.']
            sources.append('world/sumo_signal_control.py')
        contract['implementation_sha256'] = {
            p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in sources}
        contract['hash'] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        self.reward_contract = contract
        path = Path(self.output_path) / 'reward_contract.json'
        if path.exists() and json.loads(path.read_text())['hash'] != contract['hash']:
            raise ValueError('Run already belongs to a different reward contract')
        path.write_text(json.dumps(contract, indent=2) + '\n')
        self.experiment_contract = None
        if signal_config is not None:
            command = Registry.mapping['command_mapping']['setting'].param
            settings = Registry.mapping['trainer_mapping']['setting'].param
            # A path is not a schedule identity. Archive the parsed immutable
            # schedule plus actual assets and implementation for corrected runs.
            files = set(sources + ['world/world_sumo.py', 'trainer/tsc_trainer.py',
                                  'trainer/paper_trainer.py', 'environment.py',
                                  'generator/lane_vehicle.py', 'generator/intersection_phase.py',
                                  'tools/run_paper_baseline.py', 'trainer/base_trainer.py',
                                  'agent/base.py', 'agent/rl_agent.py', 'agent/utils.py',
                                  'common/utils.py', 'common/registry.py',
                                  'common/paper_experiment.py', 'common/experiment_queue.py',
                                  'run.py', 'utils/logger.py', 'utils/training_monitor.py'])
            files.update(str(p.relative_to(root)) for p in (root / 'world/sumo_events').glob('*.py'))
            binding = {
                'schema_version': 'paper-experiment-binding-v1',
                'reward_contract_hash': contract['hash'], 'signal_control': signal_config,
                'seeds': {'training': self.seed, 'sumo': command['sumo_seed'],
                          'events': None, 'event_mode': 'fixed_schedule_no_rng'},
                'schedule': self.event_runtime.schedule.to_dict(),
                'schedule_sha256': self.event_runtime.schedule_sha256,
                'network_sha256': contract['network_sha256'],
                'route_sha256': hashlib.sha256(Path(self.world.route).read_bytes()).hexdigest(),
                'simulator_config_sha256': hashlib.sha256(Path(self.path).read_bytes()).hexdigest(),
                'trainer': settings,
                'model': Registry.mapping['model_mapping']['setting'].param,
                'td_boundary': 'continuing_task_bootstrap_at_time_limit_no_reset_transition',
                'implementation_sha256': {p: hashlib.sha256((root / p).read_bytes()).hexdigest()
                                          for p in sorted(files)},
                'python_version': sys.version,
                'torch_num_threads': torch.get_num_threads(),
                'packages': {p: importlib.metadata.version(p) for p in
                             ('torch', 'numpy', 'libsumo', 'sumolib', 'gym', 'PyYAML') +
                             (('torch-scatter', 'torch-geometric') if profile.owner == 'colight' else ())},
            }
            if '-c' in self.world.sumo_cmd:
                sumocfg = Path(self.world.sumo_cmd[self.world.sumo_cmd.index('-c') + 1])
                binding['sumocfg_sha256'] = hashlib.sha256(sumocfg.read_bytes()).hexdigest()
            if getattr(self, 'episode_events', None) is not None:
                binding['episode_plan_hash'] = self.episode_events.plan_hash
                binding['protocol_hash'] = self.episode_events.plan['protocol_hash']
                binding['seeds']['event_mode'] = 'frozen_episode_plan'
                binding['seeds']['events'] = self.episode_events.plan['event_seed']
            binding = json.loads(json.dumps(binding))
            binding['hash'] = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
            self.experiment_contract = binding
            (Path(self.output_path) / 'experiment_contract.json').write_text(json.dumps(binding, indent=2) + '\n')
        self.monitor = TrainingMonitor(self.output_path, contract,
            render_every=Registry.mapping['trainer_mapping']['setting'].param.get('monitor_render_every', 1))
        if Registry.mapping['command_mapping']['setting'].param.get('resume_checkpoint'):
            self.monitor.restore_tensorboard()
        if getattr(self, 'episode_events', None) is not None:
            self.episode_events.callback = lambda row: self.monitor.environment(row,
                (self.episode_events.cursor - 1) * (self.steps // self.action_interval)
                + int(row['simulation_time']) // self.action_interval)
        self.monitor.writer.add_text('protocol/run', json.dumps({
            'agent': Registry.mapping['command_mapping']['setting'].param['agent'],
            'network': Registry.mapping['command_mapping']['setting'].param['network'],
            'training_seed': self.seed, 'episodes': self.episodes, 'seconds_per_episode': self.steps,
            'action_interval': self.action_interval, 'reward_contract_hash': contract['hash'],
        }, indent=2), 0)
        self.structured_metrics = MetricMirror(self.structured_metrics, self.monitor)

    def create_world(self):
        command = Registry.mapping['command_mapping']['setting'].param
        if command['world'] != 'sumo':
            raise ValueError('paper reward v1 is validated for SUMO only')
        settings = Registry.mapping['trainer_mapping']['setting'].param
        self.episode_events = None
        plan = None
        if settings.get('episode_plan'):
            plan = checked_json(settings['episode_plan'], settings['episode_plan_hash'])
            if (len(plan['rows']) != settings['episodes'] or plan['seconds'] != settings['steps']
                    or settings['steps'] != settings['test_steps']):
                raise ValueError('Frozen episode plan does not match training budget')
        if 'signal_control' in settings:
            for field in ('seed', 'sumo_seed'):
                if type(command.get(field)) is not int or command[field] < 0:
                    raise ValueError('Corrected experiments require explicit nonnegative seed and sumo_seed')
            interval = settings['action_interval']
            if type(interval) is not int or interval <= 0:
                raise ValueError('action_interval must be a positive integer')
            if any(type(settings[k]) is not int or settings[k] <= 0 or settings[k] % interval
                   for k in ('steps', 'test_steps')):
                raise ValueError('Training and evaluation horizons must be positive multiples of action_interval')
        super().create_world()
        if 'signal_control' in settings:
            if settings['action_interval'] <= settings['signal_control']['yellow_seconds']:
                raise ValueError('action_interval must exceed yellow_seconds')
            install_signal_control(self.world, settings['signal_control'])
        if plan is not None:
            from world.sumo_events.episodes import EpisodeEvents
            self.episode_events = EpisodeEvents(self.world, settings['episode_plan'],
                settings['episode_plan_hash'], Registry.mapping['logger_mapping']['path'].path)
            # Unbound snapshot is sufficient to build the immutable contract.
            from world.sumo_events import SumoEventRuntime
            self.event_runtime = SumoEventRuntime(self.world.net, Schedule(()))
        else:
            path = settings.get('event_schedule')
            self.event_runtime = install_events(self.world, load_schedule(path) if path else Schedule(()))
            if 'signal_control' in settings and any(
                    event.end >= min(settings['steps'], settings['test_steps'])
                    for event in self.event_runtime.schedule.events):
                raise ValueError('Fixed engineering schedules must clear before training/evaluation horizons')
        self.world.paper_reward_source = SumoRewardSource(self.world)

    def _run_scheduled_evaluation(self, completed_episodes):
        if self.episode_events is None:
            return super()._run_scheduled_evaluation(completed_episodes)
        ready = Path(self.output_path) / 'published' / f'episode_{completed_episodes:04d}.json'
        if not ready.exists():
            self.save_checkpoint('evaluation', completed_episodes)
            self.save_checkpoint('resumable', completed_episodes)
        self.evaluation_results[completed_episodes] = {'episode': completed_episodes, 'status': 'external_pending'}
        # Independent workers evaluate immutable checkpoint files. No in-process
        # validation resets consume the training plan or policy RNG.
        if completed_episodes == self.episodes:
            self.final_evaluation_completed = True

    def writeStructuredLog(self, record_type, episode, *args, **kwargs):
        record = super().writeStructuredLog(record_type, episode, *args, **kwargs)
        if self.episode_events is not None and record_type == 'TRAIN':
            result = self.episode_events.finish()
            self.monitor.traffic_episode(result, episode)
        return record

    def _write_shared_intersection_metrics(self, episode, decision_step, rewards, actions):
        super()._write_shared_intersection_metrics(episode, decision_step, rewards, actions)
        profile = self.agents[0].reward_profile
        values = np.asarray(rewards, dtype=float).reshape(-1)
        self.monitor.decision({'episode': int(episode), 'decision': self.global_decision_step + 1,
                               'simulation_time': float(self.world.get_current_time()),
                               'reward_mean': float(values.mean()), 'reward_min': float(values.min()),
                               'reward_max': float(values.max()),
                               'stopped_vehicles': int(sum(self.world.paper_reward_source.queues(profile.stopped_speed_mps).values())),
                               'active_reports': sum(r.status == 'active' for r in
                                   (self.episode_events.current if self.episode_events else self.event_runtime).reports())})

    def save_checkpoint(self, checkpoint_type, episode):
        path = super().save_checkpoint(checkpoint_type, episode)
        if path:
            payload = torch.load(path, map_location='cpu')
            payload['reward_contract_hash'] = self.reward_contract['hash']
            payload['reward_profile'] = self.reward_contract['profile']
            if self.experiment_contract is not None:
                payload['experiment_contract_hash'] = self.experiment_contract['hash']
            self._atomic_torch_save(payload, path)
            if self.episode_events is not None and checkpoint_type == 'resumable':
                evaluation = Path(self._checkpoint_path('evaluation', episode))
                publication = {'protocol_hash': self.episode_events.plan['protocol_hash'], 'episode': episode,
                               'evaluation': str(evaluation), 'evaluation_sha256': file_digest(evaluation),
                               'resumable': str(path), 'resumable_sha256': file_digest(path),
                               'experiment_contract_hash': self.experiment_contract['hash']}
                ready = Path(self.output_path) / 'published' / f'episode_{episode:04d}.json'
                if ready.exists() and json.loads(ready.read_text()) != publication:
                    raise RuntimeError('Refusing to replace an already published checkpoint identity')
                write_json(ready, publication)
        return path

    def optimizer_update_from_replay(self):
        losses = super().optimizer_update_from_replay()
        self.monitor.optimization(self.global_decision_step, self.gradient_updates, losses)
        return losses

    def load_checkpoint_payload(self, path, expected_type=None):
        if hasattr(self, '_extension_payload'):
            if expected_type != 'resumable':
                raise ValueError('Extension state is only valid for full training restore')
            return self._extension_payload
        payload = super().load_checkpoint_payload(path, expected_type)
        if payload.get('reward_contract_hash') != self.reward_contract['hash']:
            raise ValueError('Checkpoint reward contract mismatch (including legacy/other-model rewards)')
        binding = getattr(self, 'experiment_contract', None)
        if binding is not None and payload.get('experiment_contract_hash') != binding['hash']:
            raise ValueError('Checkpoint experiment contract mismatch (seeds, events, signal control or assets)')
        return payload

    def load_resumable_checkpoint(self, path):
        payload = super().load_resumable_checkpoint(path)
        if self.episode_events is not None:
            self.episode_events.cursor = payload['episode']
            self.evaluation_results[payload['episode']] = {'status': 'published'}
            if payload['episode'] == self.episodes:
                self.final_evaluation_completed = True
        return payload
