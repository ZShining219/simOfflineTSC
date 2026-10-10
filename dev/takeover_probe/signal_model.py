"""Scene-guided signal agent: encoder -> z -> per-phase Q head.

Single-layer pivot of the dispatch-router design: the scene-guided
attention module IS the controller's representation backbone, deciding
signal phases directly every decision tick — not a meta-policy over
child controllers.

Arms (``variant``):

  Scene-channel arms (lanes + text + global -> z):
    'sga'         composite module: lane self-attn + text cross-attn +
                  identity binding + scene content into z
    'attn'        text-as-query cross-attention over lanes
    'selfattn'    lane self-attn, text concatenated at fusion
    'concat'      pooled-state + text MLP

  State-only baselines (lanes + global -> z, no scene channel —
  the conventional observation pipeline the task book compares
  against; pair them with phys='queue' lane features):
    'selfattn_st' self-attention over lane tokens
    'concat_st'   lane-mean + global MLP
    'mlp_st'      flat MLP over all lane tokens (richest state arm)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from dev.takeover_probe.dispatch_model import (
    CompositeSceneEncoder,
    ConcatEncoder,
    SceneAttention,
    SelfAttentionEncoder,
)


class StateSelfAttentionEncoder(nn.Module):
    """Self-attention over lane tokens + global vector; no scene text."""

    def __init__(self, lane_dim, global_dim, d_model=64, heads=4):
        super().__init__()
        self.lane_proj = nn.Linear(lane_dim, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Linear(d_model + global_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )

    def forward(self, lanes, text, glob):
        x = self.lane_proj(lanes)
        ctx, w = self.self_attn(x, x, x, need_weights=True)
        z = self.fuse(torch.cat([ctx.mean(dim=1), glob], dim=-1))
        return z, w.mean(dim=1)


class StateConcatEncoder(nn.Module):
    """Lane-mean pooled state + global -> MLP; no attention, no text."""

    def __init__(self, lane_dim, global_dim, d_model=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(lane_dim + global_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )

    def forward(self, lanes, text, glob):
        z = self.mlp(torch.cat([lanes.mean(dim=1), glob], dim=-1))
        w = torch.zeros(lanes.shape[0], lanes.shape[1],
                        device=lanes.device)
        return z, w


class StateFlatMLPEncoder(nn.Module):
    """Flat MLP over every lane token + global; no attention, no text.

    Strongest state-only arm: no pooling bottleneck — the full lane
    vector reaches the MLP directly.
    """

    def __init__(self, n_lanes, lane_dim, global_dim, d_model=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(n_lanes * lane_dim + global_dim, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model * 2), nn.ReLU(),
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
        )

    def forward(self, lanes, text, glob):
        z = self.mlp(torch.cat(
            [lanes.flatten(start_dim=1), glob], dim=-1))
        w = torch.zeros(lanes.shape[0], lanes.shape[1],
                        device=lanes.device)
        return z, w


STATE_ONLY = ('selfattn_st', 'concat_st', 'mlp_st')


class SceneResidualPolicy(nn.Module):
    """Frozen-base Q + bounded scene-guided residual — the "guided"
    form: a converged controller keeps its learned driving; the scene
    module only learns a small, gated correction for the cases the
    base answers badly (events).

        Q(s, scene) = q_base
                      + dose_frac * spread(q_base)      # tilt budget
                        * gate(bind_mass)               # engage?
                        * tanh(delta(z))                # learned dir

      q_base : per-phase values of the frozen base DQN, supplied per
               call and stored inside transitions — the deployed
               controller stays an opaque function, its feature
               pipeline is never reimplemented on the batch path.
      gate   : piecewise-linear ramp on the max-edge cross-attention
               bind mass — 0 below ``bind_tau`` so an unbound/no text
               report leaves the policy EXACTLY equal to the base.
      delta  : zero-init linear head on the condensed z — at birth the
               residual is identically zero whatever the gate says.

    Variants with no binding product ('selfattn_st' etc.) are not
    supported — the gate IS the scene-conditional switch; without it
    there is nothing to distinguish "scene says engage" from noise.
    """

    def __init__(self, lane_dim, text_dim, global_dim, n_phases,
                 d_model=64, variant='sga', residual_pool='mean',
                 geom=None, n_edges=1, edge_idx=None,
                 dose_frac=0.35, bind_tau=0.4):
        super().__init__()
        if variant == 'sga':
            self.encoder = CompositeSceneEncoder(
                lane_dim, text_dim, global_dim, d_model,
                residual_pool=residual_pool, geom=geom)
        elif variant == 'attn':
            self.encoder = SceneAttention(
                lane_dim, text_dim, global_dim, d_model,
                residual_pool=residual_pool, geom=geom)
        else:
            raise ValueError(
                f'residual policy needs a binding variant, got {variant!r}')
        self.variant = variant
        self.n_phases = int(n_phases)
        self.dose_frac = float(dose_frac)
        self.bind_tau = float(bind_tau)
        self.delta = nn.Linear(d_model, self.n_phases)
        nn.init.zeros_(self.delta.weight)
        nn.init.zeros_(self.delta.bias)
        self.n_edges = int(n_edges)
        if edge_idx is None:
            edge_idx = np.zeros(0, dtype=np.int64)
        self.register_buffer(
            'edge_idx', torch.as_tensor(edge_idx, dtype=torch.long))

    def bind_gate(self, w):
        """(B,L) cross-attn weights -> (B,1) engagement gate in [0,1]."""
        mass = w.new_zeros(w.shape[0], self.n_edges)
        mass.scatter_add_(
            1, self.edge_idx.view(1, -1).expand(w.shape[0], -1), w)
        return ((mass.max(dim=-1, keepdim=True).values - self.bind_tau)
                / (1.0 - self.bind_tau)).clamp(0.0, 1.0)

    def forward(self, batch, q_base):
        z, w = self.encoder(batch['lanes'], batch['text'],
                            batch['global'])
        d = torch.tanh(self.delta(z))
        spread = (q_base.max(dim=-1).values
                  - q_base.min(dim=-1).values).clamp_min(1e-6)
        q = q_base + self.dose_frac * spread.unsqueeze(-1) \
            * self.bind_gate(w) * d
        return q, w

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
        if q_base is None:
            raise ValueError('SceneResidualPolicy.act requires q_base — '
                             'run with q_base_fn (frozen base values)')
        qb = torch.as_tensor(np.asarray(q_base, dtype=np.float32))
        q, w = self.forward(batch, qb.view(1, -1))
        if epsilon > 0 and rng is not None and rng.random() < epsilon:
            idx = int(rng.integers(self.n_phases))
        else:
            idx = int(q.argmax(dim=-1).item())
        return idx, q.detach().cpu().numpy()[0].tolist(), w

    def save(self, path, meta=None):
        torch.save({
            'kind': 'residual',
            'variant': self.variant,
            'n_phases': self.n_phases,
            'dose_frac': self.dose_frac,
            'bind_tau': self.bind_tau,
            'n_edges': self.n_edges,
            'edge_idx': self.edge_idx.cpu().numpy().tolist(),
            'state_dict': self.state_dict(),
            'meta': meta or {},
        }, str(path))

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(str(path), map_location=map_location,
                             weights_only=False)
        meta = payload.get('meta') or {}
        agent = cls(
            meta['lane_dim'], meta['text_dim'], meta['global_dim'],
            payload['n_phases'], d_model=meta.get('d_model', 64),
            variant=payload['variant'],
            residual_pool=meta.get('residual_pool', 'mean'),
            geom=meta.get('geom'),
            n_edges=payload.get('n_edges', 1),
            edge_idx=payload.get('edge_idx'),
            dose_frac=payload.get('dose_frac', 0.35),
            bind_tau=payload.get('bind_tau', 0.4),
        )
        agent.load_state_dict(payload['state_dict'])
        agent.eval()
        return agent, payload


class SceneSignalAgent(nn.Module):
    """Scene encoder + linear per-phase Q head (DQN action = phase id)."""

    def __init__(self, lane_dim, text_dim, global_dim, n_phases,
                 d_model=64, variant='sga', residual_pool='mean',
                 geom=None, n_lanes=None):
        super().__init__()
        self.variant = variant
        self.n_phases = int(n_phases)
        if variant == 'attn':
            self.encoder = SceneAttention(
                lane_dim, text_dim, global_dim, d_model,
                residual_pool=residual_pool, geom=geom)
        elif variant == 'sga':
            self.encoder = CompositeSceneEncoder(
                lane_dim, text_dim, global_dim, d_model,
                residual_pool=residual_pool, geom=geom)
        elif variant == 'selfattn':
            self.encoder = SelfAttentionEncoder(
                lane_dim, text_dim, global_dim, d_model)
        elif variant == 'concat':
            self.encoder = ConcatEncoder(
                lane_dim, text_dim, global_dim, d_model)
        elif variant == 'selfattn_st':
            self.encoder = StateSelfAttentionEncoder(lane_dim, global_dim,
                                                     d_model)
        elif variant == 'concat_st':
            self.encoder = StateConcatEncoder(lane_dim, global_dim,
                                              d_model)
        elif variant == 'mlp_st':
            if n_lanes is None:
                raise ValueError('mlp_st needs n_lanes')
            self.encoder = StateFlatMLPEncoder(n_lanes, lane_dim,
                                               global_dim, d_model)
        else:
            raise ValueError(f'unknown variant {variant!r}')
        self.head = nn.Linear(d_model, self.n_phases)

    def encode(self, batch):
        return self.encoder(batch['lanes'], batch['text'], batch['global'])

    def q_values(self, batch):
        z, w = self.encode(batch)
        return self.head(z), w

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
        q, w = self.q_values(batch)
        if epsilon > 0 and rng is not None and rng.random() < epsilon:
            idx = int(rng.integers(self.n_phases))
        else:
            idx = int(q.argmax(dim=-1).item())
        return idx, q.detach().cpu().numpy()[0].tolist(), w

    def save(self, path, meta=None):
        torch.save({
            'variant': self.variant,
            'n_phases': self.n_phases,
            'state_dict': self.state_dict(),
            'meta': meta or {},
        }, str(path))

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(str(path), map_location=map_location,
                             weights_only=False)
        meta = payload.get('meta') or {}
        agent = cls(
            meta['lane_dim'], meta['text_dim'], meta['global_dim'],
            payload['n_phases'], d_model=meta.get('d_model', 64),
            variant=payload['variant'],
            residual_pool=meta.get('residual_pool', 'mean'),
            geom=meta.get('geom'),
            n_lanes=meta.get('n_lanes'),
        )
        agent.load_state_dict(payload['state_dict'])
        agent.eval()
        return agent, payload
