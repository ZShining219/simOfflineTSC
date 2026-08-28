"""Shared deterministic SUMO evaluation used by FixedTime and fixed probe."""
import copy
import os
from pathlib import Path

import numpy as np
import torch

from sequential.config import simulator_config_path
from sequential.core import canonical_digest
from sequential.io import atomic_json, sha256_file


def build_world(network, output_dir, *, interface='traci', context=True):
    from common import interface as registry_interface
    from common.registry import Registry
    import world.world_sumo as world_sumo

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    registry_interface.Command_Setting_Interface({
        'command': {'sumo_seed': None},
    })
    Registry.mapping['logger_mapping']['path'].path = output_dir
    world = world_sumo.World(
        simulator_config_path(network), 1, interface=interface,
        plan5_context=bool(context),
    )
    world.configure_evaluation_output(output_dir)
    world.reset()
    command = list(world.sumo_cmd)
    if '--seed' in command or '--random' in command:
        world.close()
        raise ValueError('Plan5 SUMO command contains forbidden randomness flags')
    return world, command


def _raw16(agent):
    from agent import utils as agent_utils
    observation = np.asarray(agent.get_ob(), dtype=np.float32)
    phase = np.asarray(agent.get_phase())
    one_hot = agent_utils.idx2onehot(
        phase, agent.action_space.n
    ).astype(np.float32)
    raw = np.concatenate([observation, one_hot], axis=1).reshape(-1)
    if raw.shape != (16,):
        raise ValueError(f'Plan5 evaluator expected raw16, got {raw.shape}')
    return observation, phase, raw


def _incoming_waiting(world, intersection):
    waiting = world.get_lane_waiting_vehicle_count()
    lanes = [
        lane for road in intersection.in_roads
        for lane in intersection.road_lane_mapping[road]
    ]
    values = [float(waiting[lane]) for lane in lanes]
    if len(values) != 8:
        raise ValueError(f'Plan5 expected 8 incoming lanes, got {len(values)}')
    return values


def evaluate_policy(network, output_dir, policy, *, interface='traci',
                    collect_probe=False, source_identity=None):
    """Evaluate one deterministic 360-decision controller trajectory."""
    world = None
    try:
        world, command = build_world(
            network, output_dir, interface=interface, context=True,
        )
        policy.bind(world)
        # Generators subscribe after build_world's reset; populate those new
        # subscriptions before the t=0 observation is read.
        world._update_infos()
        decisions = []
        reward_sum = 0.0
        queue_sum = 0.0
        delay_sum = 0.0
        for decision_index in range(1, 361):
            observation, phase, raw = _raw16(policy)
            context = world.get_plan5_arrival_context()
            action = np.asarray(
                policy.action(observation, phase), dtype=np.int64
            ).reshape(-1)
            if action.shape != (1,) or not 0 <= int(action[0]) < 8:
                raise ValueError('Plan5 evaluator action is not one of 8 actions')
            per_step_rewards = []
            for _ in range(10):
                world.step(action)
                waiting = _incoming_waiting(world, policy.intersection)
                per_step_rewards.append(-float(np.mean(waiting)) * 12.0)
            mean_reward = float(np.mean(per_step_rewards))
            reward_sum += mean_reward
            waiting = _incoming_waiting(world, policy.intersection)
            queue_sum += float(np.sum(waiting))
            lane_counts = world.get_lane_vehicle_count()
            lane_delay = world.get_lane_delay()
            vehicle_total = sum(float(v) for v in lane_counts.values())
            weighted_delay = sum(
                float(lane_delay.get(lane, 0.0)) * float(count)
                for lane, count in lane_counts.items()
            )
            delay_sum += 0.0 if not vehicle_total else weighted_delay / vehicle_total
            row = {
                'decision_index': decision_index,
                'simulation_time_s': float(context['time_s']),
                'raw16': raw.tolist(),
                'ctx4': [float(value) for value in context['rates']],
                'arrival_counts': [int(value) for value in context['counts']],
                'source_action': int(action[0]),
                'reward': mean_reward,
            }
            if collect_probe:
                if source_identity is None:
                    raise ValueError('Probe rows require source identity')
                row.update(copy.deepcopy(source_identity))
            decisions.append(row)
        trajectory_digest = canonical_digest(decisions)
        for row in decisions:
            if collect_probe:
                row['source_trajectory_sha256'] = trajectory_digest
        summary = {
            'network': network, 'decision_steps': 360,
            'simulation_duration_s': 3600,
            'travel_time': float(world.get_average_travel_time()),
            'throughput': int(world.get_cur_throughput()),
            'mean_reward': reward_sum / 360.0,
            'mean_queue': queue_sum / 360.0,
            'mean_delay': delay_sum / 360.0,
            'trajectory_digest': trajectory_digest,
            'resolved_sumo_command': command,
            'resolved_sumo_command_digest': canonical_digest(command),
            'boundary_entry_mapping': copy.deepcopy(
                world.plan5_boundary_entry_mapping
            ),
            'entry_events': copy.deepcopy(world.plan5_all_entry_events),
        }
        return {'summary': summary, 'decisions': decisions}
    finally:
        if world is not None:
            world.close()


class FixedTimePolicy:
    def __init__(self, t_fixed=30):
        self.t_fixed = int(t_fixed)

    def bind(self, world):
        self.world = world
        self.intersection = world.intersections[0]
        from sequential.evaluator import InferenceOnlyDQN
        from generator import IntersectionPhaseGenerator, LaneVehicleGenerator
        import gym
        self.ob_generator = LaneVehicleGenerator(
            world, self.intersection, ['lane_count'], in_only=True, average=None,
        )
        self.phase_generator = IntersectionPhaseGenerator(
            world, self.intersection, ['phase'], targets=['cur_phase'],
            negative=False,
        )
        self.action_space = gym.spaces.Discrete(len(self.intersection.phases))

    def get_ob(self):
        return np.asarray([self.ob_generator.generate()], dtype=np.float32)

    def get_phase(self):
        return np.concatenate([self.phase_generator.generate()]).astype(np.int8)

    def action(self, observation, phase):
        if self.intersection.current_phase_time < self.t_fixed:
            return np.asarray([self.intersection.current_phase], dtype=np.int64)
        return np.asarray([
            (self.intersection.current_phase + 1) % len(self.intersection.phases)
        ], dtype=np.int64)


class Plan1DQNPolicy:
    def __init__(self, checkpoint_path):
        self.checkpoint_path = os.path.abspath(checkpoint_path)

    def bind(self, world):
        import gym
        from sequential.evaluator import InferenceOnlyDQN
        checkpoint = torch.load(self.checkpoint_path, map_location='cpu')
        state = checkpoint['agents'][0]['online_model_state_dict']
        snapshot = {
            'model': {
                'input_dim': 16, 'output_dim': 8,
                'phase': True, 'one_hot': True,
            },
            'online_model_state_dict': state,
        }
        self.inference = InferenceOnlyDQN(world, snapshot)
        self.intersection = self.inference.inter
        self.action_space = gym.spaces.Discrete(
            self.inference.model.dense_3.out_features
        )

    def get_ob(self):
        return self.inference.get_ob()

    def get_phase(self):
        return self.inference.get_phase()

    def action(self, observation, phase):
        return self.inference.get_action(observation, phase)


def causal_gate_report(evaluation):
    events = evaluation['summary']['entry_events']
    decisions = evaluation['decisions']
    directions = ('North', 'South', 'East', 'West')
    offline = []
    for row in decisions:
        now = float(row['simulation_time_s'])
        counts = [sum(
            1 for event in events
            if now - 60.0 < float(event['time_s']) <= now
            and event['direction'] == direction
        ) for direction in directions]
        offline.append(counts)
    exact = all(
        list(row['arrival_counts']) == counts
        for row, counts in zip(decisions, offline)
    )
    first_zero = decisions[0]['arrival_counts'] == [0, 0, 0, 0]
    # Adding future-only events cannot alter an already computed prefix.
    probe_time = float(decisions[min(5, len(decisions) - 1)]['simulation_time_s'])
    prefix = [event for event in events if float(event['time_s']) <= probe_time]
    future = prefix + [{
        'time_s': probe_time + 1.0, 'direction': 'North',
        'vehicle_id': 'future-mutant', 'edge_id': 'future',
    }]
    def counts_at(items):
        return [sum(
            1 for event in items
            if probe_time - 60.0 < float(event['time_s']) <= probe_time
            and event['direction'] == direction
        ) for direction in directions]
    future_invariant = counts_at(prefix) == counts_at(future)
    report = {
        'past_only': exact, 'future_mutation_invariant': future_invariant,
        'reset_zero_padding': first_zero, 'offline_recompute_exact': exact,
    }
    report['valid'] = all(report.values())
    return report
