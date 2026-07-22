import os
import time
import copy
import json
import random
import tempfile
from collections import deque
import numpy as np
import torch
from common.metrics import Metrics
from environment import TSCEnv
from common.registry import Registry
from trainer.base_trainer import BaseTrainer
from utils.logger import StructuredMetricLogger, hash_torch_state_dict


def _state_values_equal(left, right):
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(
            _state_values_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, type(left)):
        return len(left) == len(right) and all(
            _state_values_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class EvaluationIsolationGuard:
    """Fail evaluation if it mutates any Milestone 0 protected training state."""

    def __init__(self, trainer, record_type):
        self.trainer = trainer
        self.record_type = record_type
        self.remember_calls = 0
        self.remember_bindings = []

    def _snapshot(self):
        agents = []
        for agent in self.trainer.agents:
            model = getattr(agent, 'model', None)
            target = getattr(agent, 'target_model', None)
            optimizer = getattr(agent, 'optimizer', None)
            agents.append({
                'online_hash': None if model is None else hash_torch_state_dict(model.state_dict()),
                'target_hash': None if target is None else hash_torch_state_dict(target.state_dict()),
                'optimizer': None if optimizer is None else copy.deepcopy(optimizer.state_dict()),
                'epsilon': getattr(agent, 'epsilon', None),
                'replay_length': None if not hasattr(agent, 'replay_buffer') else len(agent.replay_buffer),
            })
        return {
            'agents': agents,
            'dataset_writes': len(self.trainer.dataset),
            'gradient_updates': self.trainer.gradient_updates,
            'global_decision_step': self.trainer.global_decision_step,
        }

    def __enter__(self):
        self.before = self._snapshot()
        for agent in self.trainer.agents:
            if not hasattr(agent, 'remember'):
                continue
            had_instance_value = 'remember' in agent.__dict__
            instance_value = agent.__dict__.get('remember')

            def blocked_remember(*args, **kwargs):
                self.remember_calls += 1
                raise RuntimeError('Evaluation attempted to call agent.remember')

            self.remember_bindings.append((agent, had_instance_value, instance_value))
            agent.remember = blocked_remember
        return self

    def __exit__(self, exception_type, exception, traceback):
        for agent, had_instance_value, instance_value in self.remember_bindings:
            if had_instance_value:
                agent.remember = instance_value
            else:
                agent.__dict__.pop('remember', None)
        if exception_type is not None:
            return False
        after = self._snapshot()
        if self.remember_calls:
            raise RuntimeError('Evaluation called agent.remember')
        if not _state_values_equal(self.before, after):
            raise RuntimeError(
                f'Evaluation mutated protected training state: {self.record_type}'
            )
        self.trainer.evaluation_isolation_checks.append({
            'record_type': self.record_type,
            'remember_calls': self.remember_calls,
            'replay_lengths': [agent['replay_length'] for agent in after['agents']],
            'dataset_writes': after['dataset_writes'],
            'gradient_updates': after['gradient_updates'],
            'global_decision_step': after['global_decision_step'],
        })
        return False


@Registry.register_trainer("tsc")
class TSCTrainer(BaseTrainer):
    '''
    Register TSCTrainer for traffic signal control tasks.
    '''
    def __init__(
        self,
        logger,
        gpu=0,
        cpu=False,
        name="tsc"
    ):
        super().__init__(
            logger=logger,
            gpu=gpu,
            cpu=cpu,
            name=name
        )
        self.episodes = Registry.mapping['trainer_mapping']['setting'].param['episodes']
        self.steps = Registry.mapping['trainer_mapping']['setting'].param['steps']
        self.test_steps = Registry.mapping['trainer_mapping']['setting'].param['test_steps']
        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.action_interval = Registry.mapping['trainer_mapping']['setting'].param['action_interval']
        self.save_rate = Registry.mapping['logger_mapping']['setting'].param['save_rate']
        self.learning_start = Registry.mapping['trainer_mapping']['setting'].param['learning_start']
        self.update_model_rate = Registry.mapping['trainer_mapping']['setting'].param['update_model_rate']
        self.update_target_rate = Registry.mapping['trainer_mapping']['setting'].param['update_target_rate']
        self.test_when_train = Registry.mapping['trainer_mapping']['setting'].param['test_when_train']
        self.global_decision_step = 0
        self.gradient_updates = 0
        self.target_updates = 0
        self.evaluation_isolation_checks = []
        self.structured_metrics = StructuredMetricLogger(
            Registry.mapping['logger_mapping']['path'].path
        )
        self.output_path = Registry.mapping['logger_mapping']['path'].path
        with open(os.path.join(self.output_path, 'run_manifest.json'), encoding='utf-8') as handle:
            self.config_hash = json.load(handle)['config_hash']
        # replay file is only valid in cityflow now. 
        # TODO: support SUMO and Openengine later
        
        # TODO: support other dataset in the future
        self.dataset = Registry.mapping['dataset_mapping'][Registry.mapping['command_mapping']['setting'].param['dataset']](
            os.path.join(Registry.mapping['logger_mapping']['path'].path,
                         Registry.mapping['logger_mapping']['setting'].param['data_dir'])
        )
        self.dataset.initiate(ep=self.episodes, step=self.steps, interval=self.action_interval)
        self.yellow_time = Registry.mapping['trainer_mapping']['setting'].param['yellow_length']
        # consists of path of output dir + log_dir + file handlers name
        self.log_file = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                     Registry.mapping['logger_mapping']['setting'].param['log_dir'],
                                     os.path.basename(self.logger.handlers[-1].baseFilename).rstrip('_BRF.log') + '_DTL.log'
                                     )

    def create_world(self):
        '''
        create_world
        Create world, currently support CityFlow World, SUMO World and Citypb World.

        :param: None
        :return: None
        '''
        # traffic setting is in the world mapping
        self.world = Registry.mapping['world_mapping'][Registry.mapping['command_mapping']['setting'].param['world']](
            self.path, Registry.mapping['command_mapping']['setting'].param['thread_num'],interface=Registry.mapping['command_mapping']['setting'].param['interface'])

    def create_metrics(self):
        '''
        create_metrics
        Create metrics to evaluate model performance, currently support reward, queue length, delay(approximate or real) and throughput.

        :param: None
        :return: None
        '''
        if Registry.mapping['command_mapping']['setting'].param['delay_type'] == 'apx':
            lane_metrics = ['rewards', 'queue', 'delay']
            world_metrics = ['real avg travel time', 'throughput']
        else:
            lane_metrics = ['rewards', 'queue']
            world_metrics = ['delay', 'real avg travel time', 'throughput']
        self.metric = Metrics(lane_metrics, world_metrics, self.world, self.agents)

    def create_agents(self):
        '''
        create_agents
        Create agents for traffic signal control tasks.

        :param: None
        :return: None
        '''
        self.agents = []
        agent = Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, 0)
        print(agent)
        num_agent = int(len(self.world.intersections) / agent.sub_agents)
        self.agents.append(agent)  # initialized N agents for traffic light control
        for i in range(1, num_agent):
            self.agents.append(Registry.mapping['model_mapping'][Registry.mapping['command_mapping']['setting'].param['agent']](self.world, i))

        # for magd agents should share information 
        if Registry.mapping['model_mapping']['setting'].param['name'] == 'magd':
            for ag in self.agents:
                ag.link_agents(self.agents)

    def create_env(self):
        '''
        create_env
        Create simulation environment for communication with agents.

        :param: None
        :return: None
        '''
        # TODO: finalized list or non list
        self.env = TSCEnv(self.world, self.agents, self.metric)

    def train(self):
        '''
        train
        Train the agent(s).

        :param: None
        :return: None
        '''
        total_decision_num = self.global_decision_step
        flush = 0
        for e in range(self.episodes):
            phase_started_at = time.perf_counter()
            # TODO: check this reset agent
            self.metric.clear()
            last_obs = self.env.reset()  # agent * [sub_agent, feature]

            for a in self.agents:
                a.reset()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay and e % self.save_rate == 0:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"episode_{e}.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            episode_loss = []
            i = 0
            while i < self.steps:
                if i % self.action_interval == 0:
                    last_phase = np.stack([ag.get_phase() for ag in self.agents])  # [agent, intersections]

                    if total_decision_num > self.learning_start:
                        actions = []
                        for idx, ag in enumerate(self.agents):
                            actions.append(ag.get_action(last_obs[idx], last_phase[idx], test=False))                            
                        actions = np.stack(actions)  # [agent, intersections]
                    else:
                        actions = np.stack([ag.sample() for ag in self.agents])

                    actions_prob = []
                    for idx, ag in enumerate(self.agents):
                        actions_prob.append(ag.get_action_prob(last_obs[idx], last_phase[idx]))

                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)  # [agent, intersection]
                    self.metric.update(rewards)

                    cur_phase = np.stack([ag.get_phase() for ag in self.agents])
                    for idx, ag in enumerate(self.agents):
                        ag.remember(last_obs[idx], last_phase[idx], actions[idx], actions_prob[idx], rewards[idx],
                            obs[idx], cur_phase[idx], dones[idx], f'{e}_{i//self.action_interval}_{ag.id}')
                    flush += 1
                    if flush == self.buffer_size - 1:
                        flush = 0
                        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
                    total_decision_num += 1
                    self.global_decision_step = total_decision_num
                    last_obs = obs
                if total_decision_num > self.learning_start and\
                        total_decision_num % self.update_model_rate == self.update_model_rate - 1:

                    cur_loss_q = np.stack(self.optimizer_update_from_replay())
                    episode_loss.append(cur_loss_q)
                if total_decision_num > self.learning_start and \
                        total_decision_num % self.update_target_rate == self.update_target_rate - 1:
                    [ag.update_target_network() for ag in self.agents]
                    self.target_updates += 1

                if all(dones):
                    break
            mean_loss = np.mean(np.array(episode_loss)) if episode_loss else None
            
            self.writeLog("TRAIN", e, self.metric.real_average_travel_time(),\
                0 if mean_loss is None else mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput())
            self.writeStructuredLog(
                'TRAIN', e, i, self.metric.decision_num, mean_loss,
                time.perf_counter() - phase_started_at,
            )
            self.logger.info("step:{}/{}, q_loss:{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(i, self.steps,\
                0 if mean_loss is None else mean_loss, self.metric.rewards(), self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            if e % self.save_rate == 0:
                [ag.save_model(e=e) for ag in self.agents]
                self.save_milestone_checkpoints(e)
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]
        self.save_milestone_checkpoints(self.episodes)

    def optimizer_update_from_replay(self):
        losses = []
        for agent in self.agents:
            losses.append(agent.train())
            self.gradient_updates += 1
        return losses

    def _supports_dqn_checkpoint(self):
        return bool(self.agents) and all(
            all(hasattr(agent, name) for name in (
                'model', 'target_model', 'optimizer', 'epsilon', 'replay_buffer',
            )) and agent.model is not None and agent.target_model is not None
            for agent in self.agents
        )

    def _checkpoint_path(self, checkpoint_type, episode):
        return os.path.join(
            self.output_path, 'checkpoints', checkpoint_type,
            f'episode_{episode:04d}.pt',
        )

    @staticmethod
    def _atomic_torch_save(payload, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        descriptor, temporary_path = tempfile.mkstemp(
            prefix='tmp-checkpoint-', suffix='.pt', dir=os.path.dirname(path)
        )
        os.close(descriptor)
        try:
            torch.save(payload, temporary_path)
            os.replace(temporary_path, path)
        except Exception:
            if os.path.exists(temporary_path):
                os.unlink(temporary_path)
            raise

    def save_checkpoint(self, checkpoint_type, episode):
        if checkpoint_type not in {'evaluation', 'resumable'}:
            raise ValueError(f'Invalid checkpoint_type: {checkpoint_type}')
        if not self._supports_dqn_checkpoint():
            return None
        payload = {
            'schema_version': 1,
            'checkpoint_type': checkpoint_type,
            'episode': episode,
            'global_decision_step': self.global_decision_step,
            'gradient_updates': self.gradient_updates,
            'config_hash': self.config_hash,
            'agents': [],
        }
        for rank, agent in enumerate(self.agents):
            agent_payload = {
                'rank': getattr(agent, 'rank', rank),
                'online_model_state_dict': copy.deepcopy(agent.model.state_dict()),
            }
            if checkpoint_type == 'resumable':
                agent_payload.update({
                    'target_model_state_dict': copy.deepcopy(agent.target_model.state_dict()),
                    'optimizer_state_dict': copy.deepcopy(agent.optimizer.state_dict()),
                    'epsilon': agent.epsilon,
                    'replay_state': {
                        'capacity': agent.replay_buffer.maxlen,
                        'items': list(agent.replay_buffer),
                    },
                })
            payload['agents'].append(agent_payload)
        if checkpoint_type == 'resumable':
            payload.update({
                'training_counters': {
                    'global_decision_step': self.global_decision_step,
                    'gradient_updates': self.gradient_updates,
                    'target_updates': self.target_updates,
                    'epoch': self.epoch,
                    'step': self.step,
                },
                'python_random_state': random.getstate(),
                'numpy_random_state': np.random.get_state(),
                'torch_cpu_rng_state': torch.get_rng_state(),
                'torch_cuda_rng_states': torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else [],
            })
        self.validate_checkpoint_payload(payload)
        path = self._checkpoint_path(checkpoint_type, episode)
        self._atomic_torch_save(payload, path)
        return path

    def save_milestone_checkpoints(self, episode):
        if not self._supports_dqn_checkpoint():
            return []
        return [
            self.save_checkpoint('evaluation', episode),
            self.save_checkpoint('resumable', episode),
        ]

    @staticmethod
    def validate_checkpoint_payload(payload, expected_type=None):
        required = {
            'schema_version', 'checkpoint_type', 'episode',
            'global_decision_step', 'gradient_updates', 'config_hash', 'agents',
        }
        missing = sorted(required - set(payload)) if isinstance(payload, dict) else sorted(required)
        if missing:
            raise ValueError(f'Checkpoint missing required fields: {missing}')
        checkpoint_type = payload['checkpoint_type']
        if checkpoint_type not in {'evaluation', 'resumable'}:
            raise ValueError(f'Invalid checkpoint_type: {checkpoint_type}')
        if expected_type is not None and checkpoint_type != expected_type:
            raise ValueError(
                f'Checkpoint type mismatch: expected {expected_type}, got {checkpoint_type}'
            )
        if payload['schema_version'] != 1 or not isinstance(payload['agents'], list):
            raise ValueError('Invalid checkpoint schema_version or agents')
        if not isinstance(payload['config_hash'], str) or len(payload['config_hash']) != 64:
            raise ValueError('Invalid checkpoint config_hash')
        agent_required = {'rank', 'online_model_state_dict'}
        if checkpoint_type == 'resumable':
            agent_required |= {
                'target_model_state_dict', 'optimizer_state_dict', 'epsilon', 'replay_state',
            }
            top_required = {
                'training_counters', 'python_random_state', 'numpy_random_state',
                'torch_cpu_rng_state', 'torch_cuda_rng_states',
            }
            missing_top = sorted(top_required - set(payload))
            if missing_top:
                raise ValueError(f'Resumable checkpoint missing fields: {missing_top}')
            counter_required = {
                'global_decision_step', 'gradient_updates', 'target_updates', 'epoch', 'step',
            }
            missing_counters = sorted(counter_required - set(payload['training_counters']))
            if missing_counters:
                raise ValueError(
                    f'Resumable checkpoint missing training counters: {missing_counters}'
                )
        for agent_payload in payload['agents']:
            missing_agent = sorted(agent_required - set(agent_payload))
            if missing_agent:
                raise ValueError(f'Checkpoint agent missing fields: {missing_agent}')
            if checkpoint_type == 'resumable':
                replay = agent_payload['replay_state']
                if not isinstance(replay, dict) or not {'capacity', 'items'} <= set(replay):
                    raise ValueError('Invalid resumable replay_state')
        return payload

    @classmethod
    def load_checkpoint_payload(cls, path, expected_type=None):
        try:
            payload = torch.load(path, map_location='cpu')
        except Exception as error:
            raise IOError(f'Cannot load checkpoint: {path}') from error
        return cls.validate_checkpoint_payload(payload, expected_type=expected_type)

    def load_evaluation_checkpoint(self, path):
        payload = self.load_checkpoint_payload(path, expected_type='evaluation')
        if len(payload['agents']) != len(self.agents):
            raise ValueError('Checkpoint agent count does not match trainer')
        for rank, (agent, agent_payload) in enumerate(zip(self.agents, payload['agents'])):
            if agent_payload['rank'] != getattr(agent, 'rank', rank):
                raise ValueError('Checkpoint agent rank does not match trainer')
            agent.model.load_state_dict(agent_payload['online_model_state_dict'])
        return payload

    def load_resumable_checkpoint(self, path):
        payload = self.load_checkpoint_payload(path, expected_type='resumable')
        if payload['config_hash'] != self.config_hash:
            raise ValueError('Checkpoint config_hash does not match current run')
        if len(payload['agents']) != len(self.agents):
            raise ValueError('Checkpoint agent count does not match trainer')
        for rank, (agent, agent_payload) in enumerate(zip(self.agents, payload['agents'])):
            if agent_payload['rank'] != getattr(agent, 'rank', rank):
                raise ValueError('Checkpoint agent rank does not match trainer')
            agent.model.load_state_dict(agent_payload['online_model_state_dict'])
            agent.target_model.load_state_dict(agent_payload['target_model_state_dict'])
            agent.optimizer.load_state_dict(agent_payload['optimizer_state_dict'])
            agent.epsilon = agent_payload['epsilon']
            replay = agent_payload['replay_state']
            agent.replay_buffer = deque(replay['items'], maxlen=replay['capacity'])
        counters = payload['training_counters']
        self.global_decision_step = counters['global_decision_step']
        self.gradient_updates = counters['gradient_updates']
        self.target_updates = counters['target_updates']
        self.epoch = counters['epoch']
        self.step = counters['step']
        random.setstate(payload['python_random_state'])
        np.random.set_state(payload['numpy_random_state'])
        torch.set_rng_state(payload['torch_cpu_rng_state'])
        if torch.cuda.is_available() and payload['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(payload['torch_cuda_rng_states'])
        return payload

    def train_test(self, e):
        '''
        train_test
        Evaluate model performance after each episode training process.

        :param e: number of episode
        :return self.metric.real_average_travel_time: travel time of vehicles
        '''
        with EvaluationIsolationGuard(self, 'EVALUATION'):
            phase_started_at = time.perf_counter()
            obs = self.env.reset()
            self.metric.clear()
            for a in self.agents:
                a.reset()
            for i in range(self.test_steps):
                if i % self.action_interval == 0:
                    phases = np.stack([ag.get_phase() for ag in self.agents])
                    actions = []
                    for idx, ag in enumerate(self.agents):
                        actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                    actions = np.stack(actions)
                    rewards_list = []
                    for _ in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric.update(rewards)
                if all(dones):
                    break
            self.logger.info("Test step:{}/{}, travel time :{}, rewards:{}, queue:{}, delay:{}, throughput:{}".format(\
                e, self.episodes, self.metric.real_average_travel_time(), self.metric.rewards(),\
                self.metric.queue(), self.metric.delay(), int(self.metric.throughput())))
            self.writeLog("TEST", e, self.metric.real_average_travel_time(),\
                100, self.metric.rewards(),self.metric.queue(),self.metric.delay(), self.metric.throughput())
            self.writeStructuredLog(
                'EVALUATION', e, min(i + 1, self.test_steps), self.metric.decision_num,
                None, time.perf_counter() - phase_started_at,
            )
        return self.metric.real_average_travel_time()

    def test(self, drop_load=True):
        '''
        test
        Test process. Evaluate model performance.

        :param drop_load: decide whether to load pretrained model's parameters
        :return self.metric: including queue length, throughput, delay and travel time
        '''
        with EvaluationIsolationGuard(self, 'FINAL_EVALUATION'):
            phase_started_at = time.perf_counter()
            if Registry.mapping['command_mapping']['setting'].param['world'] == 'cityflow':
                if self.save_replay:
                    self.env.eng.set_save_replay(True)
                    self.env.eng.set_replay_file(os.path.join(self.replay_file_dir, f"final.txt"))
                else:
                    self.env.eng.set_save_replay(False)
            self.metric.clear()
            if not drop_load:
                [ag.load_model(self.episodes) for ag in self.agents]
            attention_mat_list = []
            obs = self.env.reset()
            for a in self.agents:
                a.reset()
            for i in range(self.test_steps):
                if i % self.action_interval == 0:
                    phases = np.stack([ag.get_phase() for ag in self.agents])
                    actions = []
                    for idx, ag in enumerate(self.agents):
                        actions.append(ag.get_action(obs[idx], phases[idx], test=True))
                    actions = np.stack(actions)
                    rewards_list = []
                    for j in range(self.action_interval):
                        obs, rewards, dones, _ = self.env.step(actions.flatten())
                        i += 1
                        rewards_list.append(np.stack(rewards))
                    rewards = np.mean(rewards_list, axis=0)
                    self.metric.update(rewards)
                if all(dones):
                    break
            self.logger.info("Final Travel Time is %.4f, mean rewards: %.4f, queue: %.4f, delay: %.4f, throughput: %d" % (self.metric.real_average_travel_time(), \
                self.metric.rewards(), self.metric.queue(), self.metric.delay(), self.metric.throughput()))
            self.writeStructuredLog(
                'FINAL_EVALUATION', self.episodes, min(i + 1, self.test_steps),
                self.metric.decision_num, None, time.perf_counter() - phase_started_at,
            )
        return self.metric

    def writeStructuredLog(
        self, record_type, episode, simulation_step, decision_step, loss_mean,
        wall_time_seconds,
    ):
        command = Registry.mapping['command_mapping']['setting'].param
        reward_values = self.metric.lane_metrics.get('rewards')
        reward_sum = None if reward_values is None else float(np.sum(reward_values))
        epsilon_values = [
            float(agent.epsilon) for agent in self.agents if hasattr(agent, 'epsilon')
        ]
        self.structured_metrics.append({
            'schema_version': 1,
            'record_type': record_type,
            'agent': command['agent'],
            'network': command['network'],
            'training_seed': command['seed'],
            'episode': episode,
            'simulation_step': simulation_step,
            'decision_step': decision_step,
            'global_decision_step': self.global_decision_step,
            'gradient_updates': self.gradient_updates,
            'travel_time': self.metric.real_average_travel_time(),
            'reward_mean': self.metric.rewards(),
            'reward_sum': reward_sum,
            'queue': self.metric.queue(),
            'delay': self.metric.delay(),
            'throughput': self.metric.throughput(),
            'loss_mean': loss_mean,
            'epsilon': None if not epsilon_values else float(np.mean(epsilon_values)),
            'wall_time_seconds': wall_time_seconds,
        })

    def writeLog(self, mode, step, travel_time, loss, cur_rwd, cur_queue, cur_delay, cur_throughput):
        '''
        writeLog
        Write log for record and debug.

        :param mode: "TRAIN" or "TEST"
        :param step: current step in simulation
        :param travel_time: current travel time
        :param loss: current loss
        :param cur_rwd: current reward
        :param cur_queue: current queue length
        :param cur_delay: current delay
        :param cur_throughput: current throughput
        :return: None
        '''
        res = Registry.mapping['model_mapping']['setting'].param['name'] + '\t' + mode + '\t' + str(
            step) + '\t' + "%.1f" % travel_time + '\t' + "%.1f" % loss + "\t" +\
            "%.2f" % cur_rwd + "\t" + "%.2f" % cur_queue + "\t" + "%.2f" % cur_delay + "\t" + "%d" % cur_throughput
        log_handle = open(self.log_file, "a")
        log_handle.write(res + "\n")
        log_handle.close()
