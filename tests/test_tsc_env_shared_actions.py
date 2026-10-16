import sys
import types

import numpy as np

sys.modules.setdefault('libsumo', types.ModuleType('libsumo'))

from environment import TSCEnv


class Space:
    n = 8


class Agent:
    sub_agents = 6
    action_space = Space()

    def get_ob(self):
        return np.zeros((6, 4))

    def get_reward(self):
        return np.zeros(6)


class World:
    intersection_ids = [f'i{x}' for x in range(6)]
    eng = object()

    def step(self, actions):
        self.last_actions = np.asarray(actions)

    def reset(self):
        pass


def test_shared_env_declares_six_local_actions_and_jointly_executes_vector():
    world = World()
    env = TSCEnv(world, [Agent()], metric=None)
    assert env.action_space.nvec.tolist() == [8] * 6
    observations, rewards, dones, _ = env.step(np.arange(6))
    assert world.last_actions.tolist() == list(range(6))
    assert np.asarray(observations).shape == (1, 6, 4)
    assert np.asarray(rewards).shape == (1, 6)
    assert dones == [False] * 6


def test_id_keyed_interface_preserves_intersection_mapping_and_five_values():
    world = World()
    env = TSCEnv(world, [Agent()], metric=None)
    observations = env.reset_by_intersection()
    assert list(observations) == world.intersection_ids
    result = env.step_by_intersection(
        {intersection_id: index
         for index, intersection_id in enumerate(world.intersection_ids)},
        truncated=True)
    next_observations, rewards, terminated, truncated, info = result
    assert list(next_observations) == world.intersection_ids
    assert rewards == {intersection_id: 0.0
                       for intersection_id in world.intersection_ids}
    assert terminated is False and truncated is True
    assert info['network_reward_aggregation'] == 'mean'
