import gym
import numpy as np


class TSCEnv(gym.Env):
    """
    Environment for Traffic Signal Control task.
    Parameters
    ----------
    world: World object
    agents: list of agents, corresponding to each intersection in world.intersections
    metric: Metric object, used to calculate evaluation metric
    """

    def __init__(self, world, agents, metric):
        """
        :param world: one world object to interact with agents. Support multi world
        objects in different TSCEnvs.
        :param agents: single agents, each control all intersections. Or multi agents,
        each control one intersection.
        actions is a list of actions, agents is a list of agents.
        :param metric: metrics to evaluate policy.
        """
        self.world = world
        self.eng = self.world.eng
        self.n_agents = len(agents) * agents[0].sub_agents
        # test agents number == intersection number
        assert len(world.intersection_ids) == self.n_agents
        self.agents = agents
        action_dims = [
            agent.action_space.n
            for agent in agents for _ in range(agent.sub_agents)
        ]
        # One discrete dimension per controlled intersection.
        self.action_space = gym.spaces.MultiDiscrete(action_dims)
        self.metric = metric

    def step(self, actions):
        """
        :param actions: keep action as N_agents * 1
        """
        if not actions.shape:
            assert(self.n_agents == 1)
            actions = actions[np.newaxis]
        else:
            assert len(actions) == self.n_agents
        self.world.step(actions)

        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]
            # obs = np.expand_dims(np.array(obs),axis=1)
            rewards = [agent.get_reward() for agent in self.agents]
            # rewards = np.expand_dims(np.array(rewards),axis=1)
        else:
            obs = [self.agents[0].get_ob()]
            rewards = [self.agents[0].get_reward()]
        dones = [False] * self.n_agents
        # infos = {"metric": self.metric.update()}
        infos = {}

        return obs, rewards, dones, infos

    def reset(self):
        self.world.reset()
        if not len(self.agents) == 1:
            obs = [agent.get_ob() for agent in self.agents]  # [agent, sub_agent==1, feature]
            # obs = np.expand_dims(np.array(obs),axis=1)
        else:
            obs = [self.agents[0].get_ob()]  # [agent==1, sub_agent, feature]
        return obs

    def reset_by_intersection(self):
        """Reset and expose ID-keyed local observations for shared control."""
        observations = self.reset()
        if len(self.agents) != 1 or self.agents[0].sub_agents != self.n_agents:
            raise ValueError(
                'ID-keyed interface requires one shared multi-intersection agent')
        local = np.asarray(observations[0])
        return {intersection_id: np.array(local[index], copy=True)
                for index, intersection_id in enumerate(self.world.intersection_ids)}

    def step_by_intersection(self, actions, truncated=False):
        """Execute ID-keyed local actions and return a five-value interface."""
        expected = tuple(self.world.intersection_ids)
        if set(actions) != set(expected):
            missing = sorted(set(expected) - set(actions))
            extra = sorted(set(actions) - set(expected))
            raise ValueError(
                f'Action IDs do not match controlled intersections; '
                f'missing={missing}, extra={extra}')
        vector = np.asarray([actions[key] for key in expected], dtype=np.int64)
        observations, rewards, dones, legacy_info = self.step(vector)
        local_observations = np.asarray(observations[0])
        local_rewards = np.asarray(rewards[0]).reshape(-1)
        observation_map = {key: np.array(local_observations[index], copy=True)
                           for index, key in enumerate(expected)}
        reward_map = {key: float(local_rewards[index])
                      for index, key in enumerate(expected)}
        info = dict(legacy_info)
        info.update({
            'intersection_ids': list(expected),
            'local_rewards': reward_map,
            'network_reward': float(np.mean(local_rewards)),
            'network_reward_aggregation': 'mean',
        })
        if hasattr(self.world, 'get_lane_vehicle_count'):
            info['lane_vehicle_count'] = self.world.get_lane_vehicle_count()
        if hasattr(self.world, 'get_lane_waiting_vehicle_count'):
            info['lane_waiting_vehicle_count'] = (
                self.world.get_lane_waiting_vehicle_count())
        if hasattr(self.world, 'get_pressure'):
            info['pressure'] = self.world.get_pressure()
        return (observation_map, reward_map, bool(all(dones)),
                bool(truncated), info)
