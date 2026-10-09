"""Scene-guided attention + SMDP router for controller dispatch.

Two encoder variants sharing one router head:

  variant='attn'   scene text embedding is the query over lane tokens
                   (cross-attention); residual path keeps raw traffic
                   state so attention cannot drop necessary information.
  variant='concat' text and mean-pooled lanes are concatenated into an
                   MLP of matched capacity — the no-attention baseline C.

Router scores each registered controller with a learned per-controller
embedding:  Q(z, k) = MLP([z, e_k, z ⊙ e_k]) -> scalar.
The controller registry stays decoupled from the scene encoder: adding a
controller = appending an embedding row + calibration data.

Text fields carry only report-time information (type / location /
severity / estimated duration); lane tokens carry live traffic state.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SceneAttention(nn.Module):
    """scene text (query) attends over lane tokens (key/value)."""

    def __init__(self, lane_dim, text_dim, global_dim, d_model=64, heads=4):
        super().__init__()
        self.lane_proj = nn.Linear(lane_dim, d_model)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True)
        # residual path: raw traffic summary bypasses attention
        self.residual = nn.Linear(lane_dim * 0 + global_dim + lane_dim,
                                  d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )
        self.d_model = d_model

    def forward(self, lanes, text, glob):
        """lanes (B,L,F), text (B,T), glob (B,G) -> z (B,d), attn_w (B,L)."""
        kv = self.lane_proj(lanes)                    # (B,L,d)
        q = self.text_proj(text).unsqueeze(1)         # (B,1,d)
        ctx, w = self.attn(q, kv, kv, need_weights=True)
        ctx = ctx.squeeze(1)                          # (B,d)
        pooled = lanes.mean(dim=1)                    # (B,F)
        raw = self.residual(torch.cat([glob, pooled], dim=-1))
        z = self.fuse(torch.cat([ctx, raw], dim=-1))
        return z, w.squeeze(1)


class SelfAttentionEncoder(nn.Module):
    """Self-attention Transformer arm: lanes attend to lanes.

    The scene text is projected and concatenated at fusion time, but it is
    never a query — attention here is intra-modal only.  This is the task-book
    comparison arm: identical attention machinery and identical inputs, but no
    scene-guidance of the attention itself.
    """

    def __init__(self, lane_dim, text_dim, global_dim, d_model=64, heads=4):
        super().__init__()
        self.lane_proj = nn.Linear(lane_dim, d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, heads, batch_first=True)
        self.text_proj = nn.Linear(text_dim, d_model)
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 2 + global_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )

    def forward(self, lanes, text, glob):
        x = self.lane_proj(lanes)                     # (B,L,d)
        ctx, w = self.self_attn(x, x, x, need_weights=True)  # w: (B,L,L)
        pooled = ctx.mean(dim=1)                      # (B,d)
        z = self.fuse(torch.cat(
            [pooled, self.text_proj(text), glob], dim=-1))
        # per-lane salience proxy = received attention mass (column mean)
        return z, w.mean(dim=1)


class ConcatEncoder(nn.Module):
    """No-attention baseline: text + pooled state -> MLP."""

    def __init__(self, lane_dim, text_dim, global_dim, d_model=64):
        super().__init__()
        in_dim = text_dim + global_dim + lane_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, d_model), nn.ReLU(),
        )

    def forward(self, lanes, text, glob):
        pooled = lanes.mean(dim=1)
        z = self.mlp(torch.cat([text, glob, pooled], dim=-1))
        w = torch.zeros(lanes.shape[0], lanes.shape[1],
                        device=lanes.device)
        return z, w


class DispatchRouter(nn.Module):
    """Scene encoder + per-controller Q heads (SMDP action = controller)."""

    def __init__(self, lane_dim, text_dim, global_dim, controller_ids,
                 d_model=64, variant='attn'):
        super().__init__()
        self.variant = variant
        self.controller_ids = list(controller_ids)
        if variant == 'attn':
            self.encoder = SceneAttention(
                lane_dim, text_dim, global_dim, d_model)
        elif variant == 'selfattn':
            self.encoder = SelfAttentionEncoder(
                lane_dim, text_dim, global_dim, d_model)
        else:
            self.encoder = ConcatEncoder(
                lane_dim, text_dim, global_dim, d_model)
        self.ctrl_emb = nn.Embedding(len(controller_ids), d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 3, d_model), nn.ReLU(),
            nn.Linear(d_model, 1),
        )

    def encode(self, batch):
        z, w = self.encoder(batch['lanes'], batch['text'], batch['global'])
        return z, w

    def q_values(self, batch):
        z, w = self.encode(batch)
        emb = self.ctrl_emb.weight                    # (K,d)
        k, d = emb.shape
        zz = z.unsqueeze(1).expand(-1, k, -1)         # (B,K,d)
        ee = emb.unsqueeze(0).expand(z.shape[0], -1, -1)
        q = self.head(torch.cat([zz, ee, zz * ee], dim=-1)).squeeze(-1)
        return q, w

    def act(self, batch, epsilon=0.0, rng=None):
        q, w = self.q_values(batch)
        if epsilon > 0 and rng is not None and rng.random() < epsilon:
            idx = int(rng.integers(len(self.controller_ids)))
        else:
            idx = int(q.argmax(dim=-1).item())
        return idx, q.detach().cpu().numpy()[0].tolist(), w

    def add_controller(self, cid, init='clone'):
        """Extend the registry by one arm without touching the encoder.

        The shared head consumes ``[z, e_k, z*e_k]`` and is
        controller-count agnostic; only ``ctrl_emb`` gains a row.
        init: 'clone' copies the most-similar existing row (starts as a
        duplicate -> score parity, learns to differentiate); 'zero'
        starts neutral.
        """
        if cid in self.controller_ids:
            raise ValueError(f'{cid} already registered')
        k, d = self.ctrl_emb.weight.shape
        emb = nn.Embedding(k + 1, d)
        with torch.no_grad():
            emb.weight[:k] = self.ctrl_emb.weight
            if init == 'clone':
                emb.weight[k] = self.ctrl_emb.weight.mean(dim=0)
            else:
                emb.weight[k].zero_()
        self.ctrl_emb = emb
        self.controller_ids.append(cid)
        return self

    def freeze_rows_except_last(self):
        """Mask embedding grads so only the newest row trains."""
        k = self.ctrl_emb.weight.shape[0] - 1
        mask = torch.zeros_like(self.ctrl_emb.weight)
        mask[k] = 1.0
        self.ctrl_emb.weight.register_hook(lambda g: g * mask)

    def save(self, path, meta=None):
        payload = {
            'variant': self.variant,
            'controller_ids': self.controller_ids,
            'state_dict': self.state_dict(),
            'meta': meta or {},
        }
        torch.save(payload, str(path))

    @classmethod
    def load(cls, path, map_location='cpu'):
        payload = torch.load(str(path), map_location=map_location,
                             weights_only=False)
        meta = payload.get('meta') or {}
        router = cls(
            meta['lane_dim'], meta['text_dim'], meta['global_dim'],
            payload['controller_ids'], d_model=meta.get('d_model', 64),
            variant=payload['variant'],
        )
        router.load_state_dict(payload['state_dict'])
        router.eval()
        return router, payload


def collate(samples):
    return {
        'lanes': torch.as_tensor(
            np.stack([s['lanes'] for s in samples]), dtype=torch.float32),
        'text': torch.as_tensor(
            np.stack([s['text'] for s in samples]), dtype=torch.float32),
        'global': torch.as_tensor(
            np.stack([s['global'] for s in samples]), dtype=torch.float32),
        'event_lane_idx': [s.get('event_lane_idx', []) for s in samples],
    }
