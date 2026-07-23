import copy
import json
import multiprocessing
import os
import tempfile
import time
import traceback

import numpy as np
import torch

from agent import utils as agent_utils
from agent.dqn import DQNNet
from common.metrics import Metrics
from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

from .core import canonical_digest, online_parameter_digest
from .io import atomic_json, read_json


EVALUATION_SCHEMA_VERSION = 1


def _atomic_torch_save(payload, path):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix='.tmp-snapshot-', suffix='.pt', dir=directory,
    )
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def save_online_state_snapshot(state_dict, model, path, identity):
    if os.path.exists(path):
        raise FileExistsError(f'Evaluation snapshot is immutable: {path}')
    state = copy.deepcopy(state_dict)
    payload = {
        'schema_version': EVALUATION_SCHEMA_VERSION,
        'checkpoint_type': 'online_only',
        'identity': copy.deepcopy(identity),
        'online_model_state_dict': state,
        'online_parameter_digest': online_parameter_digest(state),
        'model': copy.deepcopy(model),
    }
    _atomic_torch_save(payload, path)
    return payload


def save_online_snapshot(agent, path, identity):
    return save_online_state_snapshot(
        agent.model.state_dict(), {
            'input_dim': int(agent.ob_length),
            'output_dim': int(agent.action_space.n),
            'phase': bool(agent.phase),
            'one_hot': bool(agent.one_hot),
        }, path, identity,
    )


class InferenceOnlyDQN:
    def __init__(self, world, snapshot):
        self.world = world
        self.rank = 0
        self.sub_agents = 1
        self.phase = bool(snapshot['model']['phase'])
        self.one_hot = bool(snapshot['model']['one_hot'])
        self.model = DQNNet(
            int(snapshot['model']['input_dim']),
            int(snapshot['model']['output_dim']),
        )
        self.model.load_state_dict(snapshot['online_model_state_dict'])
        self.model.eval()
        self.epsilon = 0.0
        self._bind(world)

    def _bind(self, world):
        self.world = world
        inter_id = world.intersection_ids[self.rank]
        self.inter = world.id2intersection[inter_id]
        self.ob_generator = LaneVehicleGenerator(
            world, self.inter, ['lane_count'], in_only=True, average=None,
        )
        self.phase_generator = IntersectionPhaseGenerator(
            world, self.inter, ['phase'], targets=['cur_phase'], negative=False,
        )
        self.reward_generator = LaneVehicleGenerator(
            world, self.inter, ['lane_waiting_count'], in_only=True,
            average='all', negative=True,
        )
        self.queue = LaneVehicleGenerator(
            world, self.inter, ['lane_waiting_count'], in_only=True,
            average=None, negative=False,
        )
        self.delay = LaneVehicleGenerator(
            world, self.inter, ['lane_delay'], in_only=True,
            average='all', negative=False,
        )
        state_dim = int(self.ob_generator.ob_length)
        action_dim = len(self.inter.phases)
        expected_input = state_dim + action_dim if self.phase and self.one_hot else (
            state_dim + 1 if self.phase else state_dim
        )
        if expected_input != self.model.dense_1.in_features:
            raise ValueError('Evaluation state dimension does not match snapshot')
        if action_dim != self.model.dense_3.out_features:
            raise ValueError('Evaluation action dimension does not match snapshot')

    def get_ob(self):
        return np.asarray([self.ob_generator.generate()], dtype=np.float32)

    def get_phase(self):
        return np.concatenate([self.phase_generator.generate()]).astype(np.int8)

    def get_reward(self):
        return np.squeeze(np.asarray([self.reward_generator.generate()])) * 12

    def get_queue(self):
        return float(np.sum(self.queue.generate()))

    def get_delay(self):
        return float(np.mean(self.delay.generate()))

    def get_action(self, observation, phase):
        if self.phase:
            phase_feature = (
                agent_utils.idx2onehot(phase, self.model.dense_3.out_features)
                if self.one_hot else phase
            )
            feature = np.concatenate([observation, phase_feature], axis=1)
        else:
            feature = observation
        with torch.no_grad():
            values = self.model(torch.as_tensor(feature, dtype=torch.float32))
        return np.argmax(values.cpu().numpy(), axis=1)


def _write_jsonl_atomic(path, records):
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix='.tmp-decisions-', dir=directory)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _evaluation_worker(request_path):
    request = read_json(request_path)
    attempt_dir = request['attempt_dir']
    world = None
    started = time.perf_counter()
    try:
        snapshot = torch.load(request['snapshot_path'], map_location='cpu')
        if snapshot.get('checkpoint_type') != 'online_only':
            raise ValueError('Evaluator requires an online-only snapshot')
        if online_parameter_digest(snapshot['online_model_state_dict']) != request[
            'checkpoint_digest'
        ]:
            raise ValueError('Evaluation snapshot digest mismatch')
        from common import interface as registry_interface
        from common.registry import Registry
        import world.world_sumo as world_sumo

        registry_interface.Command_Setting_Interface({
            'command': {'sumo_seed': None}
        })
        Registry.mapping['logger_mapping']['path'].path = attempt_dir
        os.makedirs(attempt_dir, exist_ok=True)
        world = world_sumo.World(
            request['simulator_config'], 1, interface=request['interface'],
        )
        world.configure_evaluation_output(attempt_dir)
        # Subscribe before reset so World._update_infos populates every field;
        # then bind again because reset replaces Intersection instances.
        agent = InferenceOnlyDQN(world, snapshot)
        world.reset()
        agent._bind(world)
        metric = Metrics(
            ['rewards', 'queue', 'delay'],
            ['delay', 'real avg travel time', 'throughput'], world, [agent],
        )
        metric.clear()
        observation = agent.get_ob()
        action_counts = {}
        phase_switches = 0
        previous_actions = None
        decisions = []
        simulation_step = 0
        for decision_index in range(1, request['decision_count'] + 1):
            phase = agent.get_phase()
            action = agent.get_action(observation, phase)
            flattened = np.asarray(action).reshape(-1)
            for value in flattened:
                key = str(int(value))
                action_counts[key] = action_counts.get(key, 0) + 1
            if previous_actions is not None:
                phase_switches += int(np.sum(flattened != previous_actions))
            previous_actions = flattened.copy()
            rewards = []
            for _ in range(request['action_interval']):
                world.step(flattened)
                simulation_step += 1
                observation = agent.get_ob()
                rewards.append(np.asarray(agent.get_reward()))
            mean_reward = np.mean(np.stack(rewards), axis=0)
            metric.update(np.asarray([mean_reward]))
            decisions.append({
                'schema_version': EVALUATION_SCHEMA_VERSION,
                'decision_index': decision_index,
                'simulation_step': simulation_step,
                'actions': flattened.astype(int).tolist(),
                'reward': np.asarray(mean_reward).reshape(-1).astype(float).tolist(),
                'queue': float(agent.get_queue()),
                'approximate_delay': float(agent.get_delay()),
                'throughput': int(world.get_cur_throughput()),
            })
        summary = {
            'schema_version': EVALUATION_SCHEMA_VERSION,
            'checkpoint_digest': request['checkpoint_digest'],
            'evaluation_network': request['evaluation_network'],
            'evaluation_protocol_digest': request['evaluation_protocol_digest'],
            'simulation_steps': simulation_step,
            'decision_steps': metric.decision_num,
            'travel_time': float(metric.real_average_travel_time()),
            'reward_mean': float(metric.rewards()),
            'queue': float(metric.queue()),
            'delay': float(metric.delay()),
            'real_delay': float(metric.real_delay()),
            'throughput': int(metric.throughput()),
            'waiting_time': float(metric.waiting_time()),
            'unfinished_vehicles': int(metric.unfinished_vehicles()),
            'phase_switches': phase_switches,
            'phase_switch_frequency': float(
                phase_switches / max(1, metric.decision_num - 1)
            ),
            'action_distribution': {
                key: value / max(1, sum(action_counts.values()))
                for key, value in sorted(action_counts.items())
            },
            'wall_time_seconds': time.perf_counter() - started,
        }
        _write_jsonl_atomic(os.path.join(attempt_dir, 'decisions.jsonl'), decisions)
        atomic_json(os.path.join(attempt_dir, 'summary.json'), summary)
        atomic_json(os.path.join(attempt_dir, 'success.json'), {
            'valid': True,
            'summary_path': os.path.join(attempt_dir, 'summary.json'),
            'decisions_path': os.path.join(attempt_dir, 'decisions.jsonl'),
        })
    except BaseException as error:
        atomic_json(os.path.join(attempt_dir, 'error.json'), {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc(),
        })
        raise
    finally:
        if world is not None:
            close_report = world.close()
            atomic_json(os.path.join(attempt_dir, 'world_close.json'), close_report)


class IndependentEvaluator:
    def __init__(self, output_root, retries=3, timeout_seconds=300,
                 worker_target=_evaluation_worker):
        if retries != 3:
            raise ValueError('Sequential evaluator protocol requires exactly 3 attempts')
        self.output_root = os.path.abspath(output_root)
        self.retries = retries
        self.timeout_seconds = int(timeout_seconds)
        self.worker_target = worker_target

    @staticmethod
    def protocol_digest(protocol):
        return canonical_digest(protocol)

    def evaluate(self, snapshot_path, evaluation_network, protocol, identity):
        snapshot = torch.load(snapshot_path, map_location='cpu')
        checkpoint_digest = snapshot['online_parameter_digest']
        protocol_digest = self.protocol_digest(protocol)
        physical_key = canonical_digest({
            'checkpoint_digest': checkpoint_digest,
            'evaluation_network': evaluation_network,
            'evaluation_protocol_digest': protocol_digest,
        })
        physical_dir = os.path.join(self.output_root, 'physical', physical_key)
        committed_path = os.path.join(physical_dir, 'committed.json')
        reused = os.path.isfile(committed_path)
        if not reused:
            os.makedirs(physical_dir, exist_ok=True)
            attempted = {
                int(name.split('_', 1)[1])
                for name in os.listdir(physical_dir)
                if name.startswith('attempt_') and name.split('_', 1)[1].isdigit()
            }
            for attempt in range(1, self.retries + 1):
                if attempt in attempted:
                    continue
                attempt_dir = os.path.join(physical_dir, f'attempt_{attempt}')
                os.makedirs(attempt_dir, exist_ok=False)
                request = {
                    'snapshot_path': os.path.abspath(snapshot_path),
                    'checkpoint_digest': checkpoint_digest,
                    'evaluation_network': evaluation_network,
                    'evaluation_protocol_digest': protocol_digest,
                    'simulator_config': os.path.abspath(protocol['simulator_config']),
                    'interface': protocol.get('interface', 'libsumo'),
                    'action_interval': int(protocol['action_interval']),
                    'decision_count': int(protocol['steps']) // int(
                        protocol['action_interval']
                    ),
                    'attempt_dir': attempt_dir,
                }
                request_path = os.path.join(attempt_dir, 'request.json')
                atomic_json(request_path, request)
                context = multiprocessing.get_context('spawn')
                process = context.Process(target=self.worker_target, args=(request_path,))
                try:
                    process.start()
                    process.join(self.timeout_seconds)
                except BaseException:
                    if process.pid is not None and process.is_alive():
                        process.terminate()
                        process.join(10)
                        if process.is_alive():
                            process.kill()
                            process.join(10)
                    raise
                if process.is_alive():
                    process.terminate()
                    process.join(10)
                    if process.is_alive():
                        process.kill()
                        process.join(10)
                    atomic_json(os.path.join(attempt_dir, 'timeout.json'), {
                        'timeout_seconds': self.timeout_seconds,
                    })
                success_path = os.path.join(attempt_dir, 'success.json')
                if process.exitcode == 0 and os.path.isfile(success_path):
                    success = read_json(success_path)
                    committed = {
                        'schema_version': EVALUATION_SCHEMA_VERSION,
                        'physical_key': physical_key,
                        'checkpoint_digest': checkpoint_digest,
                        'evaluation_network': evaluation_network,
                        'evaluation_protocol_digest': protocol_digest,
                        'successful_attempt': attempt,
                        'summary_path': success['summary_path'],
                        'decisions_path': success['decisions_path'],
                    }
                    atomic_json(committed_path, committed)
                    break
            if not os.path.isfile(committed_path):
                raise RuntimeError(
                    f'Evaluation failed after three attempts: {physical_key}'
                )
        committed = read_json(committed_path)
        logical_identity = dict(identity)
        required_identity = {
            'stage_index', 'training_network', 'evaluation_network',
            'local_episode', 'global_episode',
        }
        missing = sorted(required_identity - set(logical_identity))
        if missing:
            raise ValueError(f'Evaluation identity missing fields: {missing}')
        logical_identity.update({
            'checkpoint_digest': checkpoint_digest,
            'evaluation_protocol_digest': protocol_digest,
        })
        alias_key = canonical_digest(logical_identity)
        alias_path = os.path.join(self.output_root, 'aliases', f'{alias_key}.json')
        if not os.path.exists(alias_path):
            atomic_json(alias_path, {
                'schema_version': EVALUATION_SCHEMA_VERSION,
                'identity': logical_identity,
                'physical_committed_path': committed_path,
                'physical_key': physical_key,
            })
        return {
            'physical': committed,
            'alias_path': alias_path,
            'reused': reused,
        }
