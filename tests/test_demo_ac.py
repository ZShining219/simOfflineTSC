import copy
import random
import unittest

import numpy as np
import torch

from sequential.core import (
    ReplayMetadata, ReplayRecord, SequentialReplay, TargetUpdateScheduler,
    TrainingPayload,
)
from sequential.demo_ac import DemoACAgent
from agent.dqn import DQNNet


class Space:
    n = 8


def _demo_agent(n_demos=80, n_lanes=8):
    agent = DemoACAgent.__new__(DemoACAgent)
    agent.model = DQNNet(16, 8)          # actor
    agent.critic = DQNNet(16, 8)
    agent.critic.load_state_dict(agent.model.state_dict())
    agent.target_model = DQNNet(16, 8)
    agent.target_model.load_state_dict(agent.critic.state_dict())
    agent.model_config = {'learning_rate': 0.001}
    agent.trainer_config = {}
    agent.optimizer = torch.optim.Adam(agent.model.parameters(), lr=1e-3)
    agent.critic_optimizer = torch.optim.Adam(
        agent.critic.parameters(), lr=1e-3)
    agent.gamma_a = 0.8
    agent.margin = 0.8
    agent.gumbel_tau = 1.0
    agent.critic_l2 = 1e-5
    agent.pretrain_batch = 32
    agent.n_step = 10
    agent.demo_target_interval = 50
    agent.phase = True
    agent.one_hot = True
    agent.action_space = Space()
    agent.ob_length = 16
    agent.batch_size = 32
    agent.learning_start = 5
    agent.policy = 'demo_ac'
    agent.current_stage_index = 1
    agent.current_network = 'net'
    agent.counters = type('Counters', (), {
        'global_decision_step': 0, 'gradient_updates': 0,
        'target_updates': 0})()
    agent.target_scheduler = TargetUpdateScheduler(50, 49)
    agent._act_rng = np.random.RandomState(0)
    agent.replay = SequentialReplay(5000)
    agent._critic_updates = 0
    agent._pretrain_done = {'actor': False, 'critic': False}
    agent._demo_data = None
    agent._reward_shaper = None
    agent._reward_shaper_pending = False
    agent._shaper_episode = None
    rng = np.random.RandomState(3)
    demos = {
        'obs': rng.randn(n_demos, 16).astype(np.float32),
        'action': rng.randint(0, 8, n_demos),
        'reward': -rng.rand(n_demos).astype(np.float32) * 3,
        'next_obs': rng.randn(n_demos, 16).astype(np.float32),
        'done': np.zeros(n_demos, dtype=bool),
        'episode_id': np.repeat(np.arange(4), n_demos // 4),
    }
    demos['done'][n_demos // 4 - 1] = True
    agent.load_demos(demos)
    return agent


def _append_transition(agent, reward=-1.0):
    idx = agent.counters.global_decision_step
    payload = TrainingPayload(
        np.random.rand(1, 8).astype(np.float32), np.asarray([idx % 8]),
        np.asarray([idx % 8]), np.asarray([reward]),
        np.random.rand(1, 8).astype(np.float32),
        np.asarray([(idx + 1) % 8]),
    )
    metadata = ReplayMetadata(
        f'net:stage1:global{idx:09d}', 'net', 1, 0, idx, idx + 1)
    agent.replay.append(ReplayRecord(payload, metadata))
    agent.counters.global_decision_step = idx + 1


class DemoACPretrainTest(unittest.TestCase):
    def test_actor_pretrain_moves_prob_toward_expert(self):
        agent = _demo_agent()
        demos = agent._demo_data
        obs = torch.as_tensor(demos['obs'][:32])
        acts = torch.as_tensor(demos['action'][:32])
        with torch.no_grad():
            before = torch.softmax(agent.model(obs), 1).gather(
                1, acts[:, None]).mean()
        agent.pretrain(60, pretrain_actor=True, pretrain_critic=False,
                       seed=0)
        with torch.no_grad():
            after = torch.softmax(agent.model(obs), 1).gather(
                1, acts[:, None]).mean()
        self.assertGreater(float(after), float(before))
        self.assertTrue(agent._pretrain_done['actor'])
        self.assertFalse(agent._pretrain_done['critic'])

    def test_critic_pretrain_uses_per_and_updates_priorities(self):
        agent = _demo_agent()
        agent.pretrain(20, pretrain_actor=False, pretrain_critic=True,
                       seed=0)
        self.assertTrue(agent._pretrain_done['critic'])
        self.assertTrue(
            np.any(agent._demo_priorities != 1.0),
            'PER priorities should be updated after critic steps')
        self.assertEqual(20, agent._critic_updates)

    def test_n_step_targets_stay_within_episode(self):
        agent = _demo_agent(n_demos=40)
        targets = agent._n_step_targets(np.arange(10))
        # rewards are negative so n-step sums are finite and negative
        self.assertEqual(10, len(targets))
        self.assertTrue(np.all(np.isfinite(targets)))

    def test_begin_stage_resets_optimizers_and_target(self):
        agent = _demo_agent()
        agent.pretrain(10, seed=0)
        # perturb critic away from target
        for p in agent.critic.parameters():
            p.data.add_(0.01)
        agent.begin_stage(1, 'net', 'demo_ac')
        for pc, pt in zip(agent.critic.parameters(),
                          agent.target_model.parameters()):
            torch.testing.assert_close(pc, pt)
        # fresh Adam => no momentum buffers
        self.assertEqual(0, len(agent.optimizer.state_dict()['state']))
        self.assertEqual(
            0, len(agent.critic_optimizer.state_dict()['state']))

    def test_online_update_uses_latest_transition(self):
        agent = _demo_agent()
        agent.replay = SequentialReplay(5000)
        for _ in range(7):
            _append_transition(agent)
        update = agent.successful_gradient_update()
        self.assertIsNotNone(update)
        self.assertIn('critic_loss', update)
        self.assertIn('advantage', update)
        self.assertEqual(1, agent.counters.gradient_updates)

    def test_warmup_gate(self):
        agent = _demo_agent()
        agent.replay = SequentialReplay(5000)
        for _ in range(3):
            _append_transition(agent)
        self.assertIsNone(agent.successful_gradient_update())


if __name__ == '__main__':
    unittest.main()
