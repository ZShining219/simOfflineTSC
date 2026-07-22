"""Offline DQN agents that preserve the repository's existing DQN semantics.

The public d3rlpy package is used as the upstream Offline-RL dependency and
versioned implementation reference.  Its stock DQN/CQL losses are deliberately
not called directly because d3rlpy 2.0.4 uses chosen-action Huber loss and
Double-DQN CQL, while this project requires full-Q-vector MSE and plain DQN.
"""

import importlib

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent.dqn import DQNAgent
from common.registry import Registry


SUPPORTED_D3RLPY_VERSION = '2.0.4'


def resolve_offline_backend(requested):
    """Resolve and describe the selected implementation without changing deps."""
    if requested not in {'d3rlpy', 'native', 'auto'}:
        raise ValueError(f'Unsupported offline backend: {requested}')
    if requested == 'native':
        return {
            'requested': requested,
            'name': 'native',
            'implementation': 'project_semantics_adapter',
            'd3rlpy_version': None,
            'fallback_reason': 'explicit native backend request',
        }
    try:
        d3rlpy = importlib.import_module('d3rlpy')
    except ImportError as error:
        if requested == 'd3rlpy':
            raise RuntimeError(
                'Offline backend d3rlpy requires d3rlpy==2.0.4. '
                'Install requirements-offline.txt without replacing the project '
                'Torch/Gym/NumPy environment.'
            ) from error
        return {
            'requested': requested,
            'name': 'native',
            'implementation': 'project_semantics_adapter',
            'd3rlpy_version': None,
            'fallback_reason': 'd3rlpy is not installed',
        }
    version = getattr(d3rlpy, '__version__', None)
    if version != SUPPORTED_D3RLPY_VERSION:
        if requested == 'd3rlpy':
            raise RuntimeError(
                f'Unsupported d3rlpy version {version!r}; '
                f'expected {SUPPORTED_D3RLPY_VERSION}'
            )
        return {
            'requested': requested,
            'name': 'native',
            'implementation': 'project_semantics_adapter',
            'd3rlpy_version': version,
            'fallback_reason': 'installed d3rlpy version is incompatible',
        }
    return {
        'requested': requested,
        'name': 'd3rlpy',
        'implementation': 'project_semantics_adapter',
        'd3rlpy_version': version,
        'fallback_reason': None,
    }


class OfflineDQNMixin:
    """Fixed-data optimizer shared by Batch-DQN and CQL-DQN."""

    offline_algorithm = None

    def __init__(self, world, rank):
        model = Registry.mapping['model_mapping']['setting'].param
        self.backend = resolve_offline_backend(model.get('offline_backend', 'd3rlpy'))
        if self.backend['name'] == 'd3rlpy':
            d3rlpy = importlib.import_module('d3rlpy')
            d3rlpy.seed(
                int(Registry.mapping['command_mapping']['setting'].param['seed'])
            )
        super().__init__(world, rank)
        if self.backend['name'] == 'd3rlpy':
            factory_module = importlib.import_module('d3rlpy.models.optimizers')
            factory = factory_module.RMSpropFactory(
                alpha=0.9, eps=1e-7, centered=False
            )
            self.optimizer = factory.create(
                self.model.parameters(), lr=self.learning_rate
            )
            self.backend['upstream_components'] = [
                'd3rlpy.seed', 'd3rlpy.models.optimizers.RMSpropFactory'
            ]
        else:
            self.backend['upstream_components'] = []
        self.cql_alpha = float(model.get('cql_alpha', 0.0))
        self.epsilon = 0.0

    def train_offline_batch(self, batch):
        observations = torch.as_tensor(batch.observations, dtype=torch.float32)
        next_observations = torch.as_tensor(
            batch.next_observations, dtype=torch.float32
        )
        actions = torch.as_tensor(batch.actions, dtype=torch.long).reshape(-1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32).reshape(-1)

        with torch.no_grad():
            next_q = self.target_model(next_observations, train=True)
            td_target = rewards + self.gamma * torch.max(next_q, dim=1)[0]
            full_target = self.model(observations, train=True).detach().clone()
            full_target[torch.arange(len(actions)), actions] = td_target

        predicted_q = self.model(observations, train=True)
        td_loss = self.criterion(predicted_q, full_target)
        selected_q = predicted_q.gather(1, actions.reshape(-1, 1)).reshape(-1)
        conservative_loss = (
            torch.logsumexp(predicted_q, dim=1) - selected_q
        ).mean()
        loss = td_loss + self.cql_alpha * conservative_loss

        self.optimizer.zero_grad()
        loss.backward()
        gradient_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            'loss': float(loss.detach().cpu()),
            'td_loss': float(td_loss.detach().cpu()),
            'conservative_loss': float(conservative_loss.detach().cpu()),
            'gradient_norm': float(np.asarray(gradient_norm.detach().cpu())),
        }


@Registry.register_model('batch_dqn')
class BatchDQNAgent(OfflineDQNMixin, DQNAgent):
    offline_algorithm = 'batch_dqn'


@Registry.register_model('cql_dqn')
class CQLDQNAgent(OfflineDQNMixin, DQNAgent):
    offline_algorithm = 'cql_dqn'
