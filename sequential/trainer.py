import numpy as np


class SequentialTrainingEnvironment:
    """Small training-only environment; evaluation uses a separate process."""

    def __init__(self, world, agent):
        self.world = world
        self.agent = agent

    def reset(self):
        self.world.reset()
        # SUMO reset replaces Intersection objects; refresh every generator
        # through the single audited agent API before reading observations.
        self.agent.rebind_environment(self.world)
        return self.agent.get_ob()

    def step(self, action):
        self.world.step(np.asarray(action).reshape(-1))
        return self.agent.get_ob(), self.agent.get_reward(), False, {}


class SequentialStageTrainer:
    def __init__(self, agent, world, trainer_config, trajectory_sink=None,
                 decision_hook=None, replay_diagnostics=None):
        self.agent = agent
        self.world = world
        self.steps = int(trainer_config['steps'])
        self.action_interval = int(trainer_config['action_interval'])
        self.update_model_rate = int(trainer_config.get('update_model_rate', 1))
        if self.steps <= 0 or self.action_interval <= 0:
            raise ValueError('Sequential training steps and action interval must be positive')
        if self.update_model_rate != 1:
            raise ValueError('Frozen Sequential protocol requires update_model_rate=1')
        self.trajectory_sink = trajectory_sink
        self.decision_hook = decision_hook
        self.replay_diagnostics = replay_diagnostics
        self.environment = SequentialTrainingEnvironment(world, agent)

    def rebind_environment(self, world, expected_signature=None):
        audit = self.agent.rebind_environment(
            world, expected_signature=expected_signature,
        )
        self.world = world
        self.environment = SequentialTrainingEnvironment(world, self.agent)
        return audit

    def apply_replay_policy(self, stage_index, network, policy):
        return self.agent.begin_stage(stage_index, network, policy)

    def train_episode(self, local_episode, global_episode):
        observation = self.environment.reset()
        losses = []
        transitions = []
        simulation_step = 0
        decision_index = 0
        while simulation_step < self.steps:
            decision_index += 1
            phase = self.agent.get_phase()
            action = self.agent.get_action(observation, phase, test=False)
            rewards = []
            terminated = False
            next_observation = observation
            for _ in range(self.action_interval):
                if simulation_step >= self.steps:
                    break
                next_observation, reward, terminated, _ = self.environment.step(action)
                rewards.append(np.asarray(reward))
                simulation_step += 1
                if terminated:
                    break
            mean_reward = np.mean(np.stack(rewards), axis=0)
            next_phase = self.agent.get_phase()
            truncated = simulation_step >= self.steps and not terminated
            metadata = self.agent.remember(
                observation, phase, action, mean_reward,
                next_observation, next_phase,
                local_episode=local_episode, decision_index=decision_index,
                terminated=terminated, truncated=truncated,
            )
            if self.replay_diagnostics is not None:
                self.replay_diagnostics.record_transition(metadata)
            transition = {
                'stage_index': self.agent.current_stage_index,
                'local_episode': int(local_episode),
                'global_episode': int(global_episode),
                'decision_index': decision_index,
                'global_decision_step': metadata.written_global_step,
                'state': np.array(observation, copy=True),
                'phase': np.array(phase, copy=True),
                'action': np.array(action, copy=True),
                'reward': np.array(mean_reward, copy=True),
                'next_state': np.array(next_observation, copy=True),
                'next_phase': np.array(next_phase, copy=True),
                'terminated': bool(terminated),
                'truncated': bool(truncated),
                'transition_id': metadata.transition_id,
            }
            transitions.append(transition)
            if self.trajectory_sink is not None:
                self.trajectory_sink(transition)
            update = self.agent.successful_gradient_update()
            if update is not None:
                losses.append(update)
                if self.replay_diagnostics is not None:
                    self.replay_diagnostics.record_update(
                        update, self.agent.counters.gradient_updates,
                        replay_composition=self.agent.replay.composition(),
                    )
            if self.decision_hook is not None:
                self.decision_hook({
                    'stage_index': self.agent.current_stage_index,
                    'local_episode': int(local_episode),
                    'global_episode': int(global_episode),
                    'decision_index': decision_index,
                    'simulation_step': simulation_step,
                    'transition_id': metadata.transition_id,
                })
            observation = next_observation
            if terminated:
                break
        return {
            'stage_index': self.agent.current_stage_index,
            'local_episode': int(local_episode),
            'global_episode': int(global_episode),
            'simulation_steps': simulation_step,
            'decision_steps': decision_index,
            'gradient_updates': len(losses),
            'loss_mean': (
                None if not losses else
                float(np.mean([item['loss'] for item in losses]))
            ),
            'transitions': transitions,
            'updates': losses,
        }
