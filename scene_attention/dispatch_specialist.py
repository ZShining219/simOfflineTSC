"""Attention-key dispatch: scene attention binds the live report to an
edge, the bound (kind, edge) key looks up a specialist controller in a
registry; no match -> the deployed base controller keeps deciding.

Single decision layer, the dispatch itself is a RULE (not a learned
meta-policy — the archived SMDP ``DispatchRouter`` failed on exactly
that second learned layer)::

    batch -> binder.encode -> w per-lane cross-attn
    key   = kind if max_edge edge_mass(w) >= bind_tau else None
    ctl   = registry.get(key, base)
    a     = ctl(batch)

Arm semantics used by ``dispatch_eval.py``:

* ``attn``   key from the trained encoder's binding (the demo arm);
* ``oracle`` key from ``fb.event_fields`` — a perfect report channel,
             separating 'binding accuracy' from 'dispatch mechanism';
* blind/wrong reports are produced by ``text_override`` at eval level:
   no binding -> key None -> base, so the failure mode is 'stay on the
   deployed controller', not 'run a blinded specialist'.

``registry`` values are any object with ``act(batch, ...)`` compatible
with ``run_signal_episode``'s agent path — a trained specialist
(``SceneJointPolicy`` ckpt) or a text-free rule specialist
(``SceneGuidedMP`` with bind_gate=False, for which the dispatch is the
only scene channel).  ``audit`` accumulates per-tick dispatch rows for
the attention-response / takeover trace.
"""

from __future__ import annotations

import numpy as np
import torch


class KeyDispatchAgent:
    """key -> specialist registry; falls back to base act() on miss."""

    def __init__(self, binder, fb, world, base_act, registry,
                 bind_tau=0.4, oracle=False, kinds=('congest',)):
        self.binder = binder        # obj with .encode(batch) -> (z, w)
        self.fb = fb
        self.world = world
        self.base_act = base_act    # callable(batch,...) -> (a,q,w)
        self.registry = dict(registry)
        self.bind_tau = float(bind_tau)
        self.oracle = bool(oracle)
        self.kinds = tuple(kinds)
        self.audit = []
        self._tick = 0

    def _kind(self, batch):
        """Report kind from the text channel onehot; 'none' if empty."""
        text = batch['text']
        vec = text[0].detach().cpu().numpy().ravel() \
            if text.ndim == 2 else text.detach().cpu().numpy().ravel()
        kinds = ['congest', 'lane_close', 'none']
        if vec[:2].max() <= 0:
            return 'none'
        return kinds[int(np.argmax(vec[:2]))]

    def _edge_mass(self, w):
        masses = {}
        for edge, idx in self.fb.edge_to_laneidx.items():
            idx = np.asarray(idx, dtype=int)
            if idx.size:
                masses[edge] = float(w[idx].sum())
        return masses

    def _key(self, batch, w):
        """(kind, bound_edge, mass) or None when dispatch must not fire."""
        if self.oracle:
            fields, _ = self.fb.event_fields(
                self.world.get_current_time())
            kind = fields.get('kind', 'none')
            edge = fields.get('edge', '')
            mass = 1.0
        else:
            kind = self._kind(batch)
            masses = self._edge_mass(w)
            edge = max(masses, key=masses.get) if masses else ''
            mass = masses.get(edge, 0.0)
        if kind in self.kinds and mass >= self.bind_tau and edge:
            return kind, edge, mass
        return None

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
        _, w = self.binder.encode(batch)
        w = w[0].detach().cpu().numpy()
        key = self._key(batch, w)
        if key is not None:
            kind, edge, mass = key
            spec = self.registry.get(kind)
        else:
            kind, edge, mass = 'none', '', 0.0
            spec = None
        if spec is not None:
            a, q, _ = spec.act(batch, epsilon=epsilon, rng=rng,
                               q_base=q_base)
            choice = 'specialist'
        else:
            a, q, _ = self.base_act(batch, epsilon=epsilon, rng=rng,
                                    q_base=q_base)
            choice = 'base'
        sa = getattr(self.binder, 'last_sa', None)
        self.audit.append({
            'tick': self._tick, 'kind': kind, 'edge': edge,
            'mass': round(mass, 4), 'choice': choice,
            'sa_sal': ([round(float(v), 4) for v in sa]
                       if sa is not None else None),
        })
        self._tick += 1
        return a, q, torch.as_tensor(w).unsqueeze(0)


def wrap_controller(controller, rank=0):
    """Adapt a ``decide(rank)`` builtin/snapshot controller to the
    agent-path act() contract so it can serve as base or specialist."""
    def _act(batch, epsilon=0.0, rng=None, q_base=None):
        return int(controller.decide(rank)), None, None
    return _act
