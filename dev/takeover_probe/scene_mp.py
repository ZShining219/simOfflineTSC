"""Scene-guided MaxPressure: scene attention weights reweight the
pressure computation of the classic maxpressure rule.

Single-layer intervention structure, no RL loop::

    lane/text/global -> encoder -> (z, w per-lane attention)
    gate   = 1 if max_edge bind_mass(w_cross) >= tau else 0
    fade_l = min(queue_l / gate_q, 1)             # incipient-queue shape
    pressure_p = sum_{(up,down) in phase_p.lanelinks}
                     (count_up - count_down)
                 + beta * gate * sum_{(up,down) in phase_p.lanelinks}
                       w_dir[up] * fade_up
    phase = argmax_p pressure_p            (hold if phase_time < t_min)

Two products of the scene module do different jobs, matching what the
mechanism is empirically good at:

* ``w_cross`` (text->lane cross-attention, identity bound) acts as the
  ENGAGEMENT GATE: when the module has bound the live report to a
  concrete edge (edge-mass >= tau), the adaptive bonus engages; when no
  report is bound — no text, blinded text, or a report about nothing
  physical — the controller is exactly vanilla maxpressure.  The scene
  channel thus decides *when* the event-response mode applies.
* ``w_dir`` (traffic-side lane salience — the composite encoder's own
  lane self-attention for 'sga', the arm's own w otherwise) shapes the
  bonus direction: phases get credited for serving lanes whose queues
  are *incipient* (q < gate_q), i.e. anticipatory maxpressure.

Empirics behind the design (single-intersection S2, surge events):

* diffuse/queue-following direction + fade-in bonus: queue_int -22%,
  event-queue -22%, clear 27.5s -> 5s vs vanilla MP.
* concentrated named-edge direction (raw cross-attn w): queue_int 3x
  WORSE — a persistent preferential boost over-serves the reported
  approach regardless of where demand actually is.  The binding is for
  gating/audit, not for steering the green.
* multiplicative boost (boost*count): also bad — locks the controller
  onto the packed named lane.

``SceneGuidedMP`` mirrors ``SceneSignalAgent.act`` so it drops into the
existing ``run_signal_episode`` agent path (records carry the attention
weights for bind-mass audit).
"""

from __future__ import annotations

import numpy as np
import torch


class SceneGuidedMP:
    """Adapter: encoder + FeatureBuilder + world -> boosted MP action."""

    def __init__(self, agent, fb, world, beta=40.0, t_min=10, gate=8.0,
                 fade='in', bind_gate=None, bind_tau=0.4,
                 direction='auto', dose_frac=0.35):
        self.agent = agent          # SceneSignalAgent (encoder used; head ignored)
        self.fb = fb
        self.world = world
        self.beta = float(beta)
        self.t_min = float(t_min)
        self.gate = float(gate)
        # 'in'  : bonus ramps with the lane's queue, capped at gate
        #         (preferential/anticipatory service at queue onset)
        # 'out' : bonus fades as the lane's queue materialises
        #         (pre-visibility gating, use beta<0 for metering)
        # 'band': triangular — bonus lives only in the incipient window
        #         0 < q < gate (first-response to the reported approach)
        self.fade = fade
        # scene arms ('sga', 'attn') gate the bonus on binding confidence;
        # state-only arms cannot bind -> bonus always on
        if bind_gate is None:
            bind_gate = getattr(agent, 'variant', None) in ('sga', 'attn')
        self.bind_gate = bool(bind_gate)
        self.bind_tau = float(bind_tau)
        # 'auto'    : every arm's own traffic salience doubles as the
        #             bonus direction — sga/attn use their lane
        #             self-attention (their cross-attn w is the binding
        #             product reserved for the gate); others use own w
        # 'queue'   : bonus direction = incipient-queue shape itself
        #             (fixed rule; the encoder's w only gates)
        # 'count'   : bonus direction = in-transit lane count shape —
        #             demand-following AND pre-emptive (the surge is on
        #             the lane before it queues); the encoder's w gates
        # 'qmix'    : direction = uniform floor + queue share — diffuse
        #             demand-following (the shape the best state-only
        #             salience empirically takes); the encoder's w gates
        # 'attn'    : bonus direction = encoder's traffic salience
        #             (sga -> its lane self-attn, others -> own w)
        # 'binding' : bonus direction = raw cross-attn w (named-edge
        #             concentration — empirically harmful, kept as
        #             an ablation)
        if direction == 'auto':
            direction = 'attn'
        self.direction = direction
        # dose cap: the attention bonus may tilt the argmax by at most
        # dose_frac of the live pressure landscape — the same beta is
        # therefore safe on quiet episodes (small pressures -> small
        # absolute bonus) and meaningful under load
        self.dose_frac = float(dose_frac)

    def _direction(self, batch, w_cross):
        """Lane direction weights used to shape the bonus."""
        if self.direction == 'attn':
            enc = self.agent.encoder
            if getattr(self.agent, 'variant', None) == 'sga' and \
                    hasattr(enc, 'self_attn'):
                x = enc.lane_proj(batch['lanes'])
                _, sa_w = enc.self_attn(x, x, x, need_weights=True)
                return sa_w.mean(dim=1)[0].detach().cpu().numpy()
            return w_cross
        return None          # 'queue' / 'binding' handled in act()

    def act(self, batch, epsilon=0.0, rng=None):
        inter = self.world.intersections[0]
        z, w = self.agent.encode(batch)         # (1,d), (1,L)
        w = w[0].detach().cpu().numpy()
        if inter.current_phase_time < self.t_min:
            return int(inter.virtual_phase), None, \
                torch.as_tensor(w).unsqueeze(0)
        beta_eff = self.beta
        if self.bind_gate:
            edge_mass = 0.0
            for idx in self.fb.edge_to_laneidx.values():
                idx = np.asarray(idx, dtype=int)
                if idx.size:
                    edge_mass = max(edge_mass, float(w[idx].sum()))
            if edge_mass < self.bind_tau:
                beta_eff = 0.0
        w_dir = self._direction(batch, w)
        counts = self.world.get_info('lane_count') or {}
        queues = self.world.get_info('lane_waiting_count') or {}
        q_mean = (sum(queues.values()) / max(1, len(queues))) or 1.0
        lidx = {name: i for i, name in enumerate(self.fb.lanes)}
        pressures = []
        bonuses = []
        for p in range(len(inter.phases)):
            pr = 0.0
            bonus = 0.0
            n_b = 0
            for s, e in inter.phase_available_lanelinks[p]:
                pr += counts.get(s, 0.0) - counts.get(e, 0.0)
                if s in lidx:
                    q_s = (counts.get(s, 0.0) if self.direction == 'count'
                           else queues.get(s, 0.0))
                    if self.fade == 'in':
                        fade = min(1.0, q_s / self.gate)
                    elif self.fade == 'band':
                        fade = (min(1.0, q_s / self.gate)
                                * max(0.0, 1.0 - q_s / self.gate))
                    else:
                        fade = max(0.0, 1.0 - q_s / self.gate)
                    if self.direction in ('queue', 'count'):
                        d = 1.0
                    elif self.direction == 'binding':
                        d = w[lidx[s]]
                    elif self.direction == 'qmix':
                        d = 0.5 + 0.5 * queues.get(s, 0.0) / q_mean
                    else:
                        d = w_dir[lidx[s]]
                    bonus += d * fade
                    n_b += 1
            # 'queue'/'count' direction: per-link mean — phases must not
            # win the bonus merely by having more lanelinks
            if self.direction in ('queue', 'count') and n_b:
                bonus /= n_b
            pressures.append(pr)
            bonuses.append(beta_eff * bonus)
        max_pr = max(abs(v) for v in pressures) or 1.0
        max_bo = max(abs(v) for v in bonuses) or 0.0
        if max_bo > self.dose_frac * max_pr:
            scale = self.dose_frac * max_pr / max_bo
            bonuses = [v * scale for v in bonuses]
        pressures = [float(p_ + b_) for p_, b_ in zip(pressures, bonuses)]
        a = int(np.argmax(pressures))
        return a, pressures, torch.as_tensor(w).unsqueeze(0)
