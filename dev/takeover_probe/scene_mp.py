"""Scene-guided base-controller intervention: scene attention weights
add a bounded, gated bias to a frozen base controller's per-phase
values — the classic maxpressure rule (``SceneGuidedMP``) or a
deployed DQN's Q values (``SceneGuidedDQN``).

Single-layer intervention structure, no RL loop::

    lane/text/global -> encoder -> (z, w per-lane attention)
    gate   = 1 if max_edge bind_mass(w_cross) >= tau else 0
    fade_l = min(queue_l / gate_q, 1)             # incipient-queue shape
    value_p  = base_p                            # MP pressure / DQN Q
                 + beta * gate * sum_{(up,down) in phase_p.lanelinks}
                       w_dir[up] * fade_up
    phase = argmax_p value_p              (hold if phase_time < t_min)

The bonus is capped at ``dose_frac`` of the base controller's live
value scale (max |pressure| for MP, max Q - min Q for DQN), so the same
beta never hijacks the base — it can only tilt near-ties.

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

    def _base_and_dose(self, inter, counts):
        """Per-phase base values of the underlying controller plus the
        reference scale the attention bonus is allowed to tilt (dose
        cap).  MaxPressure: vehicle-count pressure, scale = max
        |pressure|."""
        base = [
            sum(counts.get(s, 0.0) - counts.get(e, 0.0)
                for s, e in inter.phase_available_lanelinks[p])
            for p in range(len(inter.phases))]
        return base, max(abs(v) for v in base) or 1.0

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
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
        base, dose_scale = self._base_and_dose(inter, counts)
        bonuses = []
        for p in range(len(inter.phases)):
            bonus = 0.0
            n_b = 0
            for s, e in inter.phase_available_lanelinks[p]:
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
            bonuses.append(beta_eff * bonus)
        max_bo = max(abs(v) for v in bonuses) or 0.0
        if max_bo > self.dose_frac * dose_scale:
            scale = self.dose_frac * dose_scale / max_bo
            bonuses = [v * scale for v in bonuses]
        values = [float(b_ + bo_) for b_, bo_ in zip(base, bonuses)]
        a = int(np.argmax(values))
        return a, values, torch.as_tensor(w).unsqueeze(0)


class SceneGuidedDQN(SceneGuidedMP):
    """Same attention bias applied to a frozen DQN's per-phase Q values
    instead of the maxpressure rule.

    Q values live on a learned, checkpoint-specific scale, so the dose
    cap is measured against the LIVE Q spread (max Q - min Q): the bias
    tilts the argmax hardest exactly where the base policy is
    indifferent between phases, and barely registers where it is
    confident — a fixed beta means the same fractional tilt regardless
    of the checkpoint's value scale.
    """

    def __init__(self, agent, fb, world, dqn_agent, **kw):
        super().__init__(agent, fb, world, **kw)
        self.dqn = dqn_agent        # bound inference agent (rank 0)

    def _base_and_dose(self, inter, counts):
        from tools.sumo_live_broadcast.controllers import \
            inference_q_values
        q = inference_q_values(self.dqn)
        if len(q) != len(inter.phases):
            raise ValueError(
                f'DQN action dim {len(q)} != live phase count '
                f'{len(inter.phases)}')
        spread = float(q.max() - q.min()) or 1.0
        return list(map(float, q)), spread


def bound_dqn_agent(world, weights_path, rank=0):
    """Load a deployed Q-network and bind an inference-only agent to
    ``world`` at ``rank`` — the frozen base for ``SceneGuidedDQN``."""
    from tools.sumo_live_broadcast.controllers import (
        _build_inference_agent, load_qnet_payload,
    )
    payload = load_qnet_payload(weights_path)
    return _build_inference_agent(world, payload, rank=rank)
