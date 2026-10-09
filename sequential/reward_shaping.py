"""Reward shaping for the HITL B channel (spec §6.2/§6.3/§9).

r_train = r0 + lambda * z_f,  z_f = clip((f - mu_f) / max(sigma_f,1e-6), -3, 3)

  * B1: f = rule teacher score h = -0.5*mean(q/c) - 0.5*max(min(u/T_u,2))
  * B2: f = frozen PreferenceRewardNet forward on spec features
        x = [obs | one_hot(accepted_action) | next_obs | q/c lanes | u/T_u]
  * B0 / 'none': pass-through (r_train = r0)

The agent computes per-transition aux features live at remember() time:
per-lane waiting counts from the binding's queue generator, service gaps
tracked from the executed action's served-lane map and lane occupancy.
Raw r0 is preserved in payload.aux['r_base']; the TD loss consumes
payload.reward == r_train.  Offline legs keep their recorded reward:
shaping applies to the online stream only (B and C arms are separate
experimental axes; mixing is rejected in validation).
"""

from __future__ import annotations

import json
import os

import numpy as np

T_U_SECONDS = 60.0
JAM_SPACING_M = 7.5


class ServiceGapTracker:
    """Per-lane service gap u_l: grows by dt while the lane carries
    vehicles and none of its movements is permitted; resets when served
    or empty.  Reset per episode (track resets when local_episode rolls).
    """

    def __init__(self, phase_served_lanes, n_lanes, dt):
        self.phase_served = phase_served_lanes
        self.n_lanes = int(n_lanes)
        self.dt = float(dt)
        self.gaps = np.zeros(self.n_lanes, dtype=np.float64)

    def reset(self):
        self.gaps = np.zeros(self.n_lanes, dtype=np.float64)

    def advance(self, action, occupancy):
        """Advance gaps across one decision interval driven by `action`
        (green phase index); returns the u_l vector at interval close."""
        served = np.zeros(self.n_lanes, dtype=bool)
        if 0 <= int(action) < len(self.phase_served):
            for lane in self.phase_served[int(action)]:
                if lane < self.n_lanes:
                    served[lane] = True
        demand = np.asarray(occupancy, dtype=np.float64).reshape(-1) > 0
        self.gaps = np.where((~served) & demand[:self.n_lanes],
                             self.gaps + self.dt, 0.0)
        return self.gaps.copy()


class RewardShaper:
    """mode: 'none' | 'b1' | 'b2'; lambda is the spec amplitude λ."""

    def __init__(self, mode, lane_capacities, phase_served_lanes,
                 action_interval, action_dim, lam=0.1,
                 calibration=None, model_path=None, T_u=T_U_SECONDS):
        self.mode = str(mode).lower()
        if self.mode not in ('none', 'b0', 'b1', 'b2'):
            raise ValueError(f'unknown reward_mode: {mode}')
        self.lam = float(lam)
        self.action_dim = int(action_dim)
        self.T_u = float(T_u)
        self.capacities = np.asarray(lane_capacities, dtype=np.float64)
        self.tracker = ServiceGapTracker(
            phase_served_lanes, len(lane_capacities), action_interval)
        self.calibration = calibration or {}
        self._net = None
        if self.mode == 'b2':
            if not model_path or not os.path.isfile(model_path):
                raise ValueError('B2 requires a frozen reward model path')
            import torch
            from .reward_model import PreferenceRewardNet
            blob = torch.load(model_path, map_location='cpu')
            self._net = PreferenceRewardNet(int(blob['feature_dim']))
            self._net.load_state_dict(blob['state_dict'])
            self._net.eval()
            self._torch = torch

    def _norm(self, value, kind):
        stats = self.calibration.get(kind)
        if not stats:
            raise ValueError(f'missing {kind} normalization stats')
        sigma = max(float(stats['sigma']), 1e-6)
        return float(np.clip(
            (value - float(stats['mu'])) / sigma, -3.0, 3.0))

    def teacher_score(self, q_l, gaps):
        q_ratio = np.asarray(q_l, dtype=np.float64) / self.capacities
        Q = float(q_ratio.mean())
        U = float(np.minimum(gaps / self.T_u, 2.0).max())
        return -0.5 * Q - 0.5 * U, Q, U, q_ratio

    def _model_score(self, obs, action, next_obs, q_ratio, gap_norm):
        action_dim = self.action_dim
        one_hot = np.zeros(action_dim, dtype=np.float32)
        if 0 <= int(action) < action_dim:
            one_hot[int(action)] = 1.0
        x = np.concatenate([
            np.asarray(obs, dtype=np.float32).reshape(-1), one_hot,
            np.asarray(next_obs, dtype=np.float32).reshape(-1),
            q_ratio.astype(np.float32).reshape(-1),
            gap_norm.astype(np.float32).reshape(-1),
        ])
        with self._torch.no_grad():
            return float(self._net(self._torch.as_tensor(x)[None])[0])

    def shape(self, r_base, observation, action, next_observation,
              queue_per_lane, occupancy, phase_after_action):
        """Return (r_train, aux) for one executed transition.

        `phase_after_action` = the green action index actually committed
        for this interval; occupancy = per-lane vehicle counts at the
        interval close (next_state lane block).
        """
        gaps = self.tracker.advance(phase_after_action, occupancy)
        h, Q, U, q_ratio = self.teacher_score(queue_per_lane, gaps)
        gap_norm = np.minimum(gaps / self.T_u, 2.0)
        aux = {
            'r_base': float(r_base),
            'h': float(h), 'Q': Q, 'U': U,
            'q_ratio': q_ratio.tolist(),
            'gap_norm': gap_norm.tolist(),
            'mode': self.mode,
        }
        if self.mode in ('none', 'b0'):
            aux['z'] = 0.0
            return float(r_base), aux
        # spec fallback: a collapsed reward signal degenerates to r0.
        kind = 'h' if self.mode == 'b1' else 'f_hat'
        if (self.calibration.get(kind) or {}).get('collapsed'):
            aux['z'] = 0.0
            aux['fallback'] = 'collapsed'
            return float(r_base), aux
        if self.mode == 'b1':
            z = self._norm(h, 'h')
        else:
            f_hat = self._model_score(
                observation, action, next_observation, q_ratio, gap_norm)
            aux['f_hat'] = float(f_hat)
            z = self._norm(f_hat, 'f_hat')
        aux['z'] = z
        return float(r_base) + self.lam * z, aux


def lane_capacities_from_lengths(lane_lengths):
    import math
    return [max(1.0, math.floor(float(v) / JAM_SPACING_M))
            for v in lane_lengths]


def load_calibration(path):
    with open(path) as fh:
        data = json.load(fh)
    return data
