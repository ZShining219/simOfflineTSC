"""Offline DQN agents that preserve the repository's existing DQN semantics.

The public d3rlpy package is used as the upstream Offline-RL dependency and
versioned implementation reference.  Its stock DQN/CQL losses are deliberately
not called directly because d3rlpy 2.0.4 uses chosen-action Huber loss and
Double-DQN CQL, while this project requires full-Q-vector MSE and plain DQN.
"""

import importlib
from dataclasses import dataclass

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_

from agent.dqn import DQNAgent
from common.registry import Registry


SUPPORTED_D3RLPY_VERSION = '2.0.4'


@dataclass
class OfflineLossTensors:
    q_values: torch.Tensor
    full_target: torch.Tensor
    td_target: torch.Tensor
    actions: torch.Tensor
    td_loss: torch.Tensor
    cql_loss: torch.Tensor
    total_loss: torch.Tensor


def normalize_discrete_actions(actions, batch_size, action_dim, device=None):
    raw = torch.as_tensor(actions, device=device)
    if raw.dtype == torch.bool or raw.dtype.is_floating_point:
        raise TypeError('Offline discrete actions must have an integer dtype')
    normalized = raw.to(dtype=torch.long).reshape(-1)
    if normalized.shape != (batch_size,):
        raise ValueError(
            f'Offline actions must contain exactly {batch_size} values; '
            f'got shape {tuple(raw.shape)}'
        )
    if torch.any(normalized < 0) or torch.any(normalized >= action_dim):
        raise ValueError(f'Offline action is outside [0, {action_dim})')
    return normalized


def build_full_q_mse_target(q_values, actions, td_target):
    """Replace only the data-action entries in a detached Q-vector copy."""
    if q_values.ndim != 2:
        raise ValueError('q_values must have shape [batch_size, action_dim]')
    batch_size, action_dim = q_values.shape
    actions = normalize_discrete_actions(
        actions, batch_size, action_dim, device=q_values.device
    )
    td_target = torch.as_tensor(
        td_target, dtype=q_values.dtype, device=q_values.device
    ).reshape(-1)
    if td_target.shape != (batch_size,):
        raise ValueError('td_target must have shape [batch_size]')
    full_target = q_values.detach().clone()
    full_target[torch.arange(batch_size, device=q_values.device), actions] = (
        td_target.detach()
    )
    if full_target.requires_grad or full_target.grad_fn is not None:
        raise RuntimeError('Offline full-Q target must be detached')
    return full_target, actions


def compute_offline_loss_tensors(
    model, target_model, observations, next_observations, actions, rewards,
    gamma, cql_alpha, criterion, retain_q_grad=False,
):
    observations = torch.as_tensor(observations, dtype=torch.float32)
    next_observations = torch.as_tensor(next_observations, dtype=torch.float32)
    rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    if observations.ndim != 2 or next_observations.shape != observations.shape:
        raise ValueError('Offline observations must be matching rank-2 tensors')
    if rewards.shape != (len(observations),):
        raise ValueError('Offline rewards must contain one value per observation')
    alpha = float(cql_alpha)
    with torch.no_grad():
        next_q = target_model(next_observations, train=True)
        td_target = rewards + float(gamma) * torch.max(next_q, dim=1)[0]
    q_values = model(observations, train=True)
    if retain_q_grad:
        q_values.retain_grad()
    full_target, actions = build_full_q_mse_target(
        q_values, actions, td_target
    )
    td_loss = criterion(q_values, full_target)
    selected_q = q_values.gather(1, actions[:, None]).squeeze(1)
    cql_loss = (torch.logsumexp(q_values, dim=1) - selected_q).mean()
    total_loss = td_loss + alpha * cql_loss
    return OfflineLossTensors(
        q_values=q_values, full_target=full_target, td_target=td_target.detach(),
        actions=actions, td_loss=td_loss, cql_loss=cql_loss,
        total_loss=total_loss,
    )


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
                int(Registry.mapping['command_mapping']['setting'].param.get(
                    'model_seed',
                    Registry.mapping['command_mapping']['setting'].param['seed'],
                ))
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
        loss_tensors = compute_offline_loss_tensors(
            self.model, self.target_model, batch.observations,
            batch.next_observations, batch.actions, batch.rewards, self.gamma,
            self.cql_alpha, self.criterion,
        )
        self.optimizer.zero_grad()
        loss_tensors.total_loss.backward()
        gradient_norm = clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return {
            'total_loss': float(loss_tensors.total_loss.detach().cpu()),
            'td_loss': float(loss_tensors.td_loss.detach().cpu()),
            'cql_loss': float(loss_tensors.cql_loss.detach().cpu()),
            'gradient_norm': float(np.asarray(gradient_norm.detach().cpu())),
        }


@Registry.register_model('batch_dqn')
class BatchDQNAgent(OfflineDQNMixin, DQNAgent):
    offline_algorithm = 'batch_dqn'


@Registry.register_model('cql_dqn')
class CQLDQNAgent(OfflineDQNMixin, DQNAgent):
    offline_algorithm = 'cql_dqn'
