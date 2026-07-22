import os
import time
import copy
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
        total_decision_num = 0
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

                    current_losses = []
                    for agent in self.agents:
                        current_losses.append(agent.train())
                        self.gradient_updates += 1
                    cur_loss_q = np.stack(current_losses)  # TODO: training
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
            self.logger.info("episode:{}/{}, real avg travel time:{}".format(e, self.episodes, self.metric.real_average_travel_time()))
            for j in range(len(self.world.intersections)):
                self.logger.debug("intersection:{}, mean_episode_reward:{}, mean_queue:{}".format(j, self.metric.lane_rewards()[j],\
                     self.metric.lane_queue()[j]))
            if self.test_when_train:
                self.train_test(e)
        # self.dataset.flush([ag.replay_buffer for ag in self.agents])
        [ag.save_model(e=self.episodes) for ag in self.agents]

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
