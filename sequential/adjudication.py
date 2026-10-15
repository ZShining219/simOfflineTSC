"""Adjudication layer — unified sample-time reward scoring (HITL B').

Replaces remember()-time shaping (legacy B1/B2 path in reward_shaping.py)
with consumption-time scoring: every replay/archive transition stores the
RAW environment reward plus adjudication feature material
``{'q_ratio', 'gap_norm'}``; the effective reward consumed by the TD loss
is computed per sampled batch, identically for the online leg and any
historical/expert legs::

    r_eff = r_base + lambda * z,   z = clip(sum_j w_j * z_j / sum_j|w_j|,
                                            -clip, clip)
    z_j   = clip((score_j - mu_j) / max(sigma_j, eps), -clip, clip)

Judges (pluggable panel):
  * ``terms``        — parameterized governance vocabulary:
                       raw = sum_k w_k * phi_k(features)
  * ``reward_model`` — frozen PreferenceRewardNet (B2-style scorer)
  * ``qgap``         — frozen CONVERGED controller checkpoint;
                       raw = Q(s,a_taken) - max_a Q(s,a)

Per-judge calibration (mu/sigma) normalizes score scales before
weighting, so heterogeneous judges share a common z unit.  A failed
judge contributes z=0 and its weight is renormalized out — the panel
degrades gracefully instead of collapsing to r0.

Leg adjudication features are reconstructed once per visible archive
view by replaying each episode's (action, next_state occupancy) through
``ServiceGapTracker`` — the same machinery the online leg uses live.
Episodes are keyed by (run_id, episode_id) so same-numbered episodes
from different behavior runs never merge.
"""
from __future__ import annotations

import os

import numpy as np

from .reward_shaping import ServiceGapTracker

ADJ_CLIP = 3.0
ADJ_LAMBDA = 0.1
DEFAULT_QGAP_MIN_EPISODE = 20


# ---------------------------------------------------------------- terms
# Each extractor maps one transition's feature dict -> float.
def _t_queue_mean(f):
    return float(np.mean(f['q_ratio']))


def _t_queue_max(f):
    return float(np.max(f['q_ratio']))


def _t_unserved_max(f):
    return float(np.max(f['gap_norm']))


def _t_unserved_mean(f):
    return float(np.mean(f['gap_norm']))


TERM_EXTRACTORS = {
    'queue_per_capacity': _t_queue_mean,
    'queue_max': _t_queue_max,
    'unserved_max': _t_unserved_max,
    'unserved_mean': _t_unserved_mean,
}


class TermsJudge:
    """Parameterized governance score: raw = sum_k w_k * phi_k(feat)."""

    type = 'terms'

    def __init__(self, terms):
        unknown = set(terms) - set(TERM_EXTRACTORS)
        if unknown:
            raise ValueError(f'Unknown adjudication terms: {sorted(unknown)}')
        self.terms = {k: float(v) for k, v in terms.items() if float(v) != 0.0}
        if not self.terms:
            raise ValueError('TermsJudge requires at least one nonzero weight')
        self.artifact_digest = {'terms': dict(sorted(self.terms.items()))}

    def score(self, feats):
        total = 0.0
        for name, weight in self.terms.items():
            total += weight * TERM_EXTRACTORS[name](feats)
        return total


class RewardModelJudge:
    """Frozen preference reward network (B2 scorer)."""

    type = 'reward_model'

    def __init__(self, model_path):
        import torch
        from .reward_model import PreferenceRewardNet
        if not model_path or not os.path.isfile(model_path):
            raise ValueError('reward_model judge requires a frozen model path')
        blob = torch.load(model_path, map_location='cpu')
        # fork_rng: nn.Linear init consumes torch RNG; keep the agent's
        # rng_state_digest stable across environment rebind audits.  CUDA
        # devices must be forked too: torch.load/deserialization can consume
        # cuda RNG, and initial-state/rebind digest audits capture it.
        _devs = list(range(torch.cuda.device_count())
                   ) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=_devs):
            self.net = PreferenceRewardNet(int(blob['feature_dim']))
            self.net.load_state_dict(blob['state_dict'])
        self.net.eval()
        self._torch = torch
        from .io import sha256_file
        self.artifact_digest = {
            'model_path_sha256': sha256_file(model_path),
        }

    def score(self, feats):
        obs = np.asarray(feats['obs'], dtype=np.float32).reshape(-1)
        next_obs = np.asarray(feats['next_obs'], dtype=np.float32).reshape(-1)
        one_hot = np.zeros(int(feats['action_dim']), dtype=np.float32)
        action = int(feats['action'])
        if 0 <= action < one_hot.size:
            one_hot[action] = 1.0
        x = np.concatenate([
            obs, one_hot, next_obs,
            np.asarray(feats['q_ratio'], dtype=np.float32).reshape(-1),
            np.asarray(feats['gap_norm'], dtype=np.float32).reshape(-1),
        ])
        if x.size != self.net.net[0].in_features:
            raise ValueError(
                f'reward_model judge feature dim {x.size} != '
                f'{self.net.net[0].in_features}')
        with self._torch.no_grad():
            return float(self.net(self._torch.as_tensor(x)[None])[0])


class QGapJudge:
    """Frozen CONVERGED controller checkpoint: raw = Q(s,a) - max_a Q(s,a).

    Negative when the executed action differs from what this controller
    would take; ~0 when the controller endorses it.  Judges must be
    converged DQN-series snapshots — an episode-0/random network's
    Q-gap is noise, not expertise.  Eligibility is enforced at load:
    ``episode >= min_episode`` AND ``gradient_updates > 0`` unless the
    spec explicitly sets ``allow_unconverged: true`` (test hook; the
    flag is recorded in the manifest so the override stays auditable).
    """

    type = 'qgap'

    def __init__(self, checkpoint_path, rank=0, min_episode=None,
                 allow_unconverged=False):
        import torch
        from agent.dqn import DQNNet
        if not checkpoint_path or not os.path.isfile(checkpoint_path):
            raise ValueError('qgap judge requires a controller checkpoint')
        blob = torch.load(checkpoint_path, map_location='cpu')
        agents = blob.get('agents')
        if not agents:
            raise ValueError('qgap checkpoint has no agents payload')
        episode = int(blob.get('episode', -1))
        updates = int(blob.get('gradient_updates', 0))
        min_episode = (DEFAULT_QGAP_MIN_EPISODE if min_episode is None
                       else int(min_episode))
        if not allow_unconverged:
            if episode < min_episode or updates <= 0:
                raise ValueError(
                    'qgap judge requires a converged DQN-series checkpoint: '
                    f'episode={episode} (need >= {min_episode}), '
                    f'gradient_updates={updates} (need > 0); pass '
                    'allow_unconverged=true only for tests')
        if isinstance(agents, dict):
            key = rank if rank in agents else sorted(agents)[0]
        else:
            key = int(rank) if int(rank) < len(agents) else 0
        state = agents[key]['online_model_state_dict']
        if 'dense_1.weight' not in state or 'dense_3.weight' not in state:
            raise ValueError(
                'qgap judge supports dqn_mlp checkpoints only; '
                'non-MLP architectures need their own judge type')
        in_dim = int(state['dense_1.weight'].shape[1])
        out_dim = int(state['dense_3.weight'].shape[0])
        # fork_rng: nn.Linear init consumes torch RNG; keep the agent's
        # rng_state_digest stable across environment rebind audits.  CUDA
        # devices must be forked too (see RewardModelJudge).
        _devs = list(range(torch.cuda.device_count())
                   ) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=_devs):
            self.net = DQNNet(in_dim, out_dim)
            self.net.load_state_dict(state)
        self.net.eval()
        self._in_dim = in_dim
        self._torch = torch
        from .io import sha256_file
        self.artifact_digest = {
            'checkpoint_sha256': sha256_file(checkpoint_path),
            'checkpoint_episode': episode,
            'checkpoint_gradient_updates': updates,
            'allow_unconverged': bool(allow_unconverged),
        }

    def score(self, feats):
        obs = np.asarray(feats['obs'], dtype=np.float32).reshape(1, -1)
        if obs.shape[1] != self._in_dim:
            raise ValueError(
                f'qgap judge obs dim {obs.shape[1]} != {self._in_dim}')
        with self._torch.no_grad():
            q = self.net(self._torch.as_tensor(obs))[0]
        action = int(feats['action'])
        if 0 <= action < int(q.shape[0]):
            return float(q[action] - q.max())
        return 0.0


JUDGE_TYPES = {
    'terms': TermsJudge,
    'reward_model': RewardModelJudge,
    'qgap': QGapJudge,
}


class Adjudicator:
    """Weighted judge panel producing z for a transition feature dict.

    The panel spec is the versioned judge manifest: one immutable
    ``{'lambda', 'clip', 'judges': [...]}`` mapping per run/stage.  Its
    digest (configuration + judge artifact digests) binds a run to
    exactly one reward interpretation — changing the panel is a stage
    boundary, never a mid-run drift.
    """

    def __init__(self, spec, action_dim):
        self.lam = float(spec.get('lambda', ADJ_LAMBDA))
        self.clip = float(spec.get('clip', ADJ_CLIP))
        self.action_dim = int(action_dim)
        self.judges = []
        for entry in spec.get('judges', ()):
            jtype = str(entry.get('type'))
            cls = JUDGE_TYPES.get(jtype)
            if cls is None:
                raise ValueError(f'Unknown judge type: {jtype}')
            import inspect
            accepted = set(inspect.signature(cls.__init__).parameters)
            kwargs = {k: v for k, v in entry.items()
                      if k in accepted and k != 'self'}
            judge = cls(**kwargs)
            self.judges.append({
                'judge': judge, 'type': jtype,
                'weight': float(entry.get('weight', 1.0)),
                'mu': float(entry.get('mu', 0.0)),
                'sigma': float(entry.get('sigma', 1.0)),
            })
        if not self.judges:
            raise ValueError('adjudication requires at least one judge')
        from .historical_archive import stable_digest
        self.digest = stable_digest({
            'lambda': self.lam, 'clip': self.clip,
            'action_dim': self.action_dim,
            'judges': [{
                'type': e['type'], 'weight': e['weight'],
                'mu': e['mu'], 'sigma': e['sigma'],
                'artifact': getattr(e['judge'], 'artifact_digest', None),
            } for e in self.judges],
        })

    def _z_judge(self, entry, feats):
        sigma = max(entry['sigma'], 1e-6)
        raw = entry['judge'].score(feats)
        return float(np.clip(
            (raw - entry['mu']) / sigma, -self.clip, self.clip)), raw

    def z(self, obs, action, next_obs, q_ratio, gap_norm):
        """Return (z, detail) for one transition."""
        feats = {
            'obs': obs, 'action': int(action), 'next_obs': next_obs,
            'action_dim': self.action_dim,
            'q_ratio': np.asarray(q_ratio, dtype=np.float64),
            'gap_norm': np.asarray(gap_norm, dtype=np.float64),
        }
        z_sum = 0.0
        w_sum = 0.0
        detail = {'judges': {}, 'fallback': []}
        for i, entry in enumerate(self.judges):
            try:
                z_j, raw = self._z_judge(entry, feats)
            except Exception:
                z_j, raw = 0.0, None
                detail['fallback'].append(entry['type'])
            detail['judges'][f'{entry["type"]}_{i}'] = {
                'z': z_j, 'raw': raw, 'weight': entry['weight'],
            }
            z_sum += entry['weight'] * z_j
            w_sum += abs(entry['weight'])
        z = z_sum / w_sum if w_sum > 0 else 0.0
        z = float(np.clip(z, -self.clip, self.clip))
        detail['z'] = z
        return z, detail

    def effective_reward(self, r_base, z):
        return float(r_base) + self.lam * float(z)


def build_adjudicator(trainer_config, action_dim):
    """None or Adjudicator from trainer_config['adjudication']."""
    spec = trainer_config.get('adjudication')
    if spec is None:
        return None
    if not isinstance(spec, dict):
        raise ValueError('adjudication spec must be a mapping')
    if spec.get('enabled', True) is False:
        return None
    return Adjudicator(spec, action_dim)


# ------------------------------------------------------- leg features
def leg_features_from_view(view, ctx):
    """{transition_id: {'q_ratio','gap_norm'}} for every visible transition.

    Replays each episode's executed actions through ServiceGapTracker in
    decision_step order, using next_observations[:, :n_lanes] as the
    occupancy at interval close — mirrors the online computation.
    Episodes are grouped by (run_id, episode_id): same-numbered episodes
    from different behavior runs are distinct trajectories.
    """
    capacities = np.asarray(ctx['lane_capacities'], dtype=np.float64)
    n_lanes = capacities.size
    out = {}
    for network in view.networks:
        arrays = view.archive.datasets[network]._arrays[network]
        indices = np.asarray(view.indices_by_network[network])
        run_ids = np.asarray(arrays['run_id'], dtype=object)[indices]
        episodes = np.asarray(arrays['episode_id'])[indices]
        steps = np.asarray(arrays['decision_step'])[indices]
        actions = np.asarray(arrays['actions'])[indices]
        next_obs = np.asarray(arrays['next_observations'])[indices]
        tids = np.asarray(arrays['transition_id'], dtype=object)[indices]
        tracker = ServiceGapTracker(
            ctx['phase_served_lanes'], n_lanes, ctx['action_interval'])
        keys = np.asarray(
            [f'{r}#{e}' for r, e in zip(run_ids, episodes)], dtype=object)
        for key in np.unique(keys):
            positions = np.flatnonzero(keys == key)
            positions = positions[np.argsort(steps[positions])]
            tracker.reset()
            for pos in positions:
                occupancy = np.asarray(
                    next_obs[pos, :n_lanes], dtype=np.float64)
                gaps = tracker.advance(int(actions[pos]), occupancy)
                out[str(tids[pos])] = {
                    'q_ratio': occupancy / capacities,
                    'gap_norm': np.minimum(gaps / ctx['T_u'], 2.0),
                }
    return out


def build_ctx_from_world(world, inter, signature, action_space_n):
    """Derive {lane_capacities, phase_served_lanes} from the live world —
    same derivation as RewardShaper (lazy: needs SUMO lane lengths)."""
    from .reward_shaping import lane_capacities_from_lengths
    lane_ids = list(signature['incoming_lane_mapping'])
    lane_lengths = [float(world.eng.lane.getLength(l)) for l in lane_ids]
    capacities = lane_capacities_from_lengths(lane_lengths)
    lane_index = {l: i for i, l in enumerate(lane_ids)}
    served = [
        [lane_index[l] for l in inter.phase_available_startlanes[a]
         if l in lane_index]
        for a in range(action_space_n)
    ]
    return {'lane_capacities': capacities, 'phase_served_lanes': served}


def load_topology_ctx(path, action_interval, T_u=60.0):
    """ctx from a dump_lane_topology.py JSON (explicit override).

    Reads the produced schema: ``in_lanes`` (ordering),
    ``lane_capacity_veh`` (per-lane map), ``phase_served_lanes``.
    Also accepts a flat ``lane_capacities`` list for hand-built files.
    """
    import json
    data = json.load(open(path))
    if 'lane_capacities' in data:
        capacities = list(data['lane_capacities'])
    else:
        capacities = [
            float(data['lane_capacity_veh'][lane])
            for lane in data['in_lanes']
        ]
    return {
        'lane_capacities': capacities,
        'phase_served_lanes': [list(v) for v in data['phase_served_lanes']],
        'action_interval': float(data.get('action_interval', action_interval)),
        'T_u': float(data.get('T_u', T_u)),
    }
