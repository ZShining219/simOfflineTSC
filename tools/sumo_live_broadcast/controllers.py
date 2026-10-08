"""Controller adapters for the live SUMO broadcast.

Every controller exposes the same surface so the broadcast engine can treat
traditional agents, DQN-family snapshots, and a human operator uniformly:

    bind(world, ranks)  attach to a live ``world_sumo.World`` for the given
                        controlled intersection ranks (default: rank 0 only)
    decide(rank) -> int pick the next held green-phase action for one junction
    reset()             drop per-run state (optional)

Adapters never start SUMO themselves and never mutate experiment data; they
only read the bound world and return action indices.  Heavy imports (torch,
the agent package) stay lazy so this module is importable for --help and unit
tests without simulator or ML dependencies.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Q-network weight loading
# ---------------------------------------------------------------------------

def _arch_from_state_dict(state_dict):
    if any(str(key).startswith('advantage.') for key in state_dict):
        return 'dueling_mlp', 'dueling_double_dqn'
    return 'dqn_mlp', 'independent_dqn'


def _state_dict_dims(state_dict, architecture):
    try:
        if architecture == 'dueling_mlp':
            input_dim = int(state_dict['feature.0.weight'].shape[1])
            output_dim = int(state_dict['advantage.weight'].shape[0])
        else:
            input_dim = int(state_dict['dense_1.weight'].shape[1])
            output_dim = int(state_dict['dense_3.weight'].shape[0])
    except (KeyError, IndexError, AttributeError) as exc:
        raise ValueError(
            f'Cannot infer Q-network dimensions from state dict keys: '
            f'{sorted(state_dict)[:8]}'
        ) from exc
    return input_dim, output_dim


def _looks_like_state_dict(payload):
    if not isinstance(payload, dict) or not payload:
        return False
    sample = next(iter(payload.values()))
    return hasattr(sample, 'shape') and hasattr(sample, 'dtype')


def load_qnet_payload(path):
    """Normalize a supported weight file into a uniform snapshot payload.

    Accepted layouts:
      * sequential "snapshot" files (``online_model_state_dict`` + ``model``
        dimension metadata) produced by ``save_online_state_snapshot``;
      * sequential resumable checkpoints (``agents[0]`` payload);
      * bare ``state_dict`` files such as legacy ``model/NNN_0.pt``.

    Returns a dict with ``state_dict``, ``input_dim``, ``output_dim``,
    ``phase``, ``one_hot``, ``architecture``, ``algorithm_id``, ``kind`` and
    ``source_path``.
    """
    import torch

    payload = torch.load(str(path), map_location='cpu')
    if not isinstance(payload, dict):
        raise ValueError(f'Not a torch checkpoint file: {path}')

    if 'online_model_state_dict' in payload:
        state_dict = payload['online_model_state_dict']
        model_meta = payload.get('model') or {}
        architecture, algorithm_id = _arch_from_state_dict(state_dict)
        architecture = str(model_meta.get('architecture_name') or architecture)
        algorithm_id = (
            'dueling_double_dqn' if architecture == 'dueling_mlp'
            else normalize_known_algorithm(payload)
        )
        input_dim = model_meta.get('input_dim')
        output_dim = model_meta.get('output_dim')
        if input_dim is None or output_dim is None:
            input_dim, output_dim = _state_dict_dims(state_dict, architecture)
        return {
            'kind': 'snapshot',
            'state_dict': state_dict,
            'input_dim': int(input_dim),
            'output_dim': int(output_dim),
            'phase': bool(model_meta.get('phase', True)),
            'one_hot': bool(model_meta.get('one_hot', True)),
            'architecture': architecture,
            'algorithm_id': algorithm_id,
            'source_path': str(path),
            'identity': payload.get('identity'),
        }

    agents = payload.get('agents')
    if isinstance(agents, (list, tuple)) and agents and isinstance(agents[0], dict) \
            and 'online_model_state_dict' in agents[0]:
        return _payload_from_state_dict(
            agents[0]['online_model_state_dict'], 'sequential_checkpoint', path,
        )

    if _looks_like_state_dict(payload):
        return _payload_from_state_dict(payload, 'raw_state_dict', path)

    raise ValueError(
        f'Unrecognized model file format: {path} '
        f'(keys: {sorted(payload)[:10]})'
    )


def normalize_known_algorithm(payload):
    algorithm_id = str(payload.get('algorithm_id') or 'independent_dqn')
    if 'dueling' in algorithm_id:
        return 'dueling_double_dqn'
    return 'double_dqn' if 'double' in algorithm_id else 'independent_dqn'


def _payload_from_state_dict(state_dict, kind, path):
    architecture, algorithm_id = _arch_from_state_dict(state_dict)
    input_dim, output_dim = _state_dict_dims(state_dict, architecture)
    return {
        'kind': kind,
        'state_dict': state_dict,
        'input_dim': input_dim,
        'output_dim': output_dim,
        # Legacy files carry no feature metadata; the binding check below
        # validates the assumption against the live observation size.
        'phase': True,
        'one_hot': True,
        'architecture': architecture,
        'algorithm_id': algorithm_id,
        'source_path': str(path),
        'identity': None,
    }


# ---------------------------------------------------------------------------
# Controller adapters
# ---------------------------------------------------------------------------

class LegacyAgentController:
    """Adapter over ``agent/`` classes with the (world, rank) constructor.

    One adapter serves every controlled rank: each rank gets its own agent
    instance because the legacy classes store per-intersection generators.
    """

    def __init__(self, controller_id, label, registry_name):
        self.id = controller_id
        self.label = label
        self.kind = 'builtin'
        self.registry_name = registry_name
        self._agents = {}

    def bind(self, world, ranks=None):
        import agent  # noqa: F401 - registers model classes
        from common.registry import Registry

        agent_cls = Registry.mapping['model_mapping'][self.registry_name]
        self._agents = {
            int(rank): agent_cls(world, int(rank))
            for rank in (ranks or [0])
        }
        return self

    def decide(self, rank=0):
        agent = self._agents[int(rank)]
        phase = agent.get_phase()
        action = agent.get_action(None, phase, test=True)
        return int(action)

    def reset(self):
        for agent in self._agents.values():
            if hasattr(agent, 'reset'):
                agent.reset()


class SnapshotController:
    """DQN-family inference adapter over a normalized Q-network payload.

    Works for online-DQN model dumps, sequential snapshots, and resumable
    checkpoints regardless of how the weights were trained (DHOA/CONT share
    the same deployed network).
    """

    def __init__(self, controller_id, label, weights_path, detail=None):
        self.id = controller_id
        self.label = label
        self.kind = 'snapshot'
        self.weights_path = str(weights_path)
        self.detail = detail or {}
        self._payload = None
        self._model = None
        self._agents = {}

    @property
    def payload(self):
        if self._payload is None:
            self._payload = load_qnet_payload(self.weights_path)
        return self._payload

    def bind(self, world, ranks=None):
        from sequential.agent import build_q_network

        payload = self.payload
        if self._model is None:
            self._model = build_q_network(
                payload['algorithm_id'], payload['input_dim'],
                payload['output_dim'],
            )
            self._model.load_state_dict(payload['state_dict'])
            self._model.eval()
        # Independent-DQN deployment shares one weight set across junctions;
        # each rank only needs its own generators/intersection binding.
        self._agents = {
            int(rank): _build_inference_agent(
                world, payload, int(rank), model=self._model,
            )
            for rank in (ranks or [0])
        }
        return self

    def decide(self, rank=0):
        return _inference_action(self._agents[int(rank)])

    def reset(self):
        pass


class ManualController:
    """Human operator: per-junction hold of the last requested green phase.

    With no explicit request on a junction the controller simply keeps
    whichever green phase is currently targeted there, so traffic never
    waits on the operator.
    """

    def __init__(self, controller_id='manual', label='人工接管'):
        self.id = controller_id
        self.label = label
        self.kind = 'manual'
        self._world = None
        self._requested = {}

    def bind(self, world, ranks=None):
        self._world = world
        self._requested = {}
        return self

    def request_phase(self, rank, phase_index):
        self._requested[int(rank)] = int(phase_index)

    def release(self, rank=None):
        if rank is None:
            self._requested.clear()
        else:
            self._requested.pop(int(rank), None)

    def pending(self):
        return dict(self._requested)

    def decide(self, rank=0):
        rank = int(rank)
        if rank in self._requested:
            return self._requested[rank]
        inter = self._world.id2intersection[
            self._world.intersection_ids[rank]
        ]
        return int(inter.virtual_phase)

    def reset(self):
        self._requested = {}


def _build_inference_agent(world, payload, rank=0, model=None):
    """Assemble an InferenceOnlyDQN-equivalent bound to one junction rank."""
    from sequential.agent import build_q_network
    from sequential.evaluator import InferenceOnlyDQN

    agent = InferenceOnlyDQN.__new__(InferenceOnlyDQN)
    agent.world = world
    agent.rank = rank
    agent.sub_agents = 1
    agent.phase = bool(payload['phase'])
    agent.one_hot = bool(payload['one_hot'])
    agent.model = model
    if agent.model is None:
        agent.model = build_q_network(
            payload['algorithm_id'], payload['input_dim'],
            payload['output_dim'],
        )
        agent.model.load_state_dict(payload['state_dict'])
        agent.model.eval()
    agent.epsilon = 0.0
    _bind_generators(agent, world)
    return agent


def _inference_action(agent):
    """``InferenceOnlyDQN.get_action`` without assuming ``dense_*`` layers.

    The stock implementation reads ``model.dense_3.out_features`` for the
    phase one-hot width, which breaks dueling heads; the shared model's real
    output width is recovered via ``_output_dim`` instead.
    """
    import torch
    from agent import utils as agent_utils

    observation = agent.get_ob()
    phase = agent.get_phase()
    if agent.phase:
        out_dim = _output_dim(agent.model)
        phase_feature = (
            agent_utils.idx2onehot(phase, out_dim)
            if agent.one_hot else phase
        )
        feature = np.concatenate([observation, phase_feature], axis=1)
    else:
        feature = observation
    with torch.no_grad():
        values = agent.model(torch.as_tensor(feature, dtype=torch.float32))
    return int(np.argmax(values.cpu().numpy(), axis=1)[0])


def _bind_generators(agent, world):
    """Mirror ``InferenceOnlyDQN._bind`` without assuming dense_* layers."""
    from generator import IntersectionPhaseGenerator, LaneVehicleGenerator

    inter_id = world.intersection_ids[agent.rank]
    agent.inter = world.id2intersection[inter_id]
    agent.ob_generator = LaneVehicleGenerator(
        world, agent.inter, ['lane_count'], in_only=True, average=None,
    )
    agent.phase_generator = IntersectionPhaseGenerator(
        world, agent.inter, ['phase'], targets=['cur_phase'], negative=False,
    )
    agent.reward_generator = LaneVehicleGenerator(
        world, agent.inter, ['lane_waiting_count'], in_only=True,
        average='all', negative=True,
    )
    agent.queue = LaneVehicleGenerator(
        world, agent.inter, ['lane_waiting_count'], in_only=True,
        average=None, negative=False,
    )
    agent.delay = LaneVehicleGenerator(
        world, agent.inter, ['lane_delay'], in_only=True,
        average='all', negative=False,
    )
    state_dim = int(agent.ob_generator.ob_length)
    action_dim = len(agent.inter.phases)
    expected_input = state_dim + action_dim if agent.phase and agent.one_hot else (
        state_dim + 1 if agent.phase else state_dim
    )
    first = _first_linear(agent.model)
    if expected_input != int(first.in_features):
        raise ValueError(
            f'Model input dim {first.in_features} does not match live '
            f'observation dim {expected_input}'
        )
    last_out = _output_dim(agent.model)
    if action_dim != last_out:
        raise ValueError(
            f'Model action dim {last_out} does not match intersection phase '
            f'count {action_dim}'
        )


def _first_linear(model):
    for module in model.modules():
        if hasattr(module, 'in_features'):
            return module
    raise ValueError('Q-network has no linear layer')


def _output_dim(model):
    for module in reversed(list(model.modules())):
        if hasattr(module, 'out_features'):
            return int(module.out_features)
    raise ValueError('Q-network has no linear layer')


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

BUILTIN_CONTROLLERS = (
    ('fixedtime', 'FixedTime', 'fixedtime'),
    ('maxpressure', 'MaxPressure', 'maxpressure'),
    ('sotl', 'SOTL', 'sotl'),
)


def build_builtin_controllers():
    return [
        LegacyAgentController(controller_id, label, registry_name)
        for controller_id, label, registry_name in BUILTIN_CONTROLLERS
    ] + [ManualController()]


def build_snapshot_controller(controller_id, label, weights_path,
                              detail=None):
    return SnapshotController(controller_id, label, weights_path,
                              detail=detail)
