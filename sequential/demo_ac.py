"""DemoACAgent — spec §8 DemoLight-style actor/critic demonstration
pretraining, implemented on the shared Sequential runtime contract.

Reuses the DQNNet trunk (d→20→20→P) for BOTH actor logits and critic
Q-vector, so the same episode-0 checkpoint initialises A arms bit-for-bit
and evaluator snapshots (argmax over actor logits == argmax pi) need no
branching.

Spec parameters (pilot defaults, all trainer_config-overridable):
  gamma_A=0.8, margin=0.8, gumbel tau=1.0, critic L2=1e-5,
  pretrain batch=32, n-step=10, critic target sync every 50 critic
  updates, PER alpha=0.6, beta 0.4->1 linear over pretrain steps,
  actor stream = fixed uniform shuffle (own RNG, independent of critic
  PER stream).

Online phase (spec §8 "per-decision update"):
  critic: 0.5*(Q(s,a) - y)^2,  y = r + gamma*(1-done)*Q_t(s', a'),
          a' ~ pi(.|s') sampled on the actor's own RNG
  actor:  -log pi(a|s) * A,    A = stopgrad(Q(s,a) - sum_b pi(s,b)Q(s,b))
  One update each per decision; the current transition is used directly
  (demos are NOT replayed online; online replay still records for
  auditability/checkpoint parity).

Deviation ledger (documented, spec §8 forbids silent equivalence):
  * demo source = our packaged MP-correction archive (spec demo harvest
    uses 5 SOTL configs x 360 transitions; pool size deviation recorded
    in the experiment record)
  * mask = all phases legal (S2); legal-action masks are identity here
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from agent import utils as agent_utils
from agent.dqn import DQNNet

from .agent import SequentialDQNAgent
from .core import TargetUpdateScheduler

GAMMA_A = 0.8
MARGIN = 0.8
GUMBEL_TAU = 1.0
CRITIC_L2 = 1e-5
PRETRAIN_BATCH = 32
N_STEP = 10
PRETRAIN_TARGET_SYNC = 50
PER_ALPHA = 0.6


class DemoACAgent(SequentialDQNAgent):
    """Actor-critic with demonstration pretraining on the Sequential
    contract.  `self.model` is the actor (drives autonomous eval via
    argmax == argmax pi); `self.critic`/`self.target_model` hold the
    online/target Q nets."""

    def __init__(self, world, rank, model_config, trainer_config,
                 binding_factory=None, algorithm_id='independent_dqn'):
        super().__init__(
            world, rank, model_config, trainer_config,
            binding_factory=binding_factory, algorithm_id=algorithm_id,
        )
        # model = actor; critic + critic target are separate nets.
        self.critic = DQNNet(self.ob_length, self.action_space.n)
        self.critic.load_state_dict(self.model.state_dict())
        self.target_model.load_state_dict(self.critic.state_dict())
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=float(model_config['learning_rate']))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(model_config['learning_rate']))
        self.gamma_a = float(trainer_config.get('demo_gamma', GAMMA_A))
        self.margin = float(trainer_config.get('demo_margin', MARGIN))
        self.gumbel_tau = float(
            trainer_config.get('demo_gumbel_tau', GUMBEL_TAU))
        self.critic_l2 = float(
            trainer_config.get('demo_critic_l2', CRITIC_L2))
        self.pretrain_batch = int(
            trainer_config.get('demo_pretrain_batch', PRETRAIN_BATCH))
        self.n_step = int(trainer_config.get('demo_n_step', N_STEP))
        self.demo_target_interval = int(
            trainer_config.get('demo_target_interval', PRETRAIN_TARGET_SYNC))
        self._act_rng = np.random.RandomState(
            int(trainer_config.get('training_seed', 0)) + 7919)
        self._critic_updates = 0
        self._pretrain_done = {'actor': False, 'critic': False}
        self._demo_data = None

    @classmethod
    def from_initial_state(cls, world, rank, model_config, trainer_config,
                           initial_state, **kwargs):
        """Load the same ep0 resumable checkpoint as HA arms.

        Deviations from HASODQNAgent.from_initial_state, all spec-mandated:
          * actor/model  <- online weights; critic <- online weights;
            critic target <- target weights (same arch, bit-identical)
          * optimizers are fresh Adam (RMSprop state is not portable and
            spec requires momentum reset at the online boundary anyway)
          * digest verification covers online/target parameter + RNG
            digests; optimizer digest is skipped (fresh Adam).
        """
        import random as _random
        from .agent import SequentialCounters
        from .core import SequentialReplay
        agent = cls(
            world, rank, model_config, trainer_config, **kwargs)
        checkpoint = torch.load(
            initial_state['checkpoint_path'], map_location='cpu')
        if (checkpoint.get('checkpoint_type') != 'resumable'
                or checkpoint.get('episode') != 0):
            raise ValueError('DemoAC must start from an episode-0 checkpoint')
        payload = checkpoint['agents'][rank]
        agent.model.load_state_dict(payload['online_model_state_dict'])
        agent.critic.load_state_dict(payload['online_model_state_dict'])
        agent.target_model.load_state_dict(payload['target_model_state_dict'])
        agent.epsilon = float(payload['epsilon'])
        counters = checkpoint['training_counters']
        agent.counters = SequentialCounters(
            int(counters['global_decision_step']),
            int(counters['gradient_updates']), int(counters['target_updates']))
        if any((agent.counters.global_decision_step,
                agent.counters.gradient_updates,
                agent.counters.target_updates)):
            raise ValueError('DemoAC initial-state counters must be zero')
        _random.setstate(checkpoint['python_random_state'])
        np.random.set_state(checkpoint['numpy_random_state'])
        torch.set_rng_state(checkpoint['torch_cpu_rng_state'])
        if torch.cuda.is_available() and checkpoint['torch_cuda_rng_states']:
            torch.cuda.set_rng_state_all(checkpoint['torch_cuda_rng_states'])
        current = agent.training_state_digests()
        for key in ('online_parameter_digest', 'target_parameter_digest',
                    'rng_state_digest'):
            expected = initial_state.get('digests', {}).get(key)
            if expected and current[key] != expected:
                raise ValueError(f'Initial-state digest mismatch: {key}')
        agent.replay = SequentialReplay(int(trainer_config['buffer_size']))
        return agent

    # ------------------------------------------------------------- obs util
    def _feature(self, state, phase):
        obs = np.asarray(state, dtype=np.float32).reshape(-1)
        if self.phase and self.one_hot:
            obs = np.concatenate([
                obs,
                agent_utils.idx2onehot(
                    np.asarray(phase).reshape(-1).astype(int),
                    self.action_space.n)[0],
            ])
        return obs

    def _features_batch(self, states, phases):
        return torch.as_tensor(
            np.stack([self._feature(s, p) for s, p in zip(states, phases)]),
            dtype=torch.float32)

    # ------------------------------------------------------------ acting
    def get_action(self, observation, phase, test=False):
        feature = torch.as_tensor(
            np.concatenate([
                np.asarray(observation, dtype=np.float32),
                agent_utils.idx2onehot(
                    np.asarray(phase).reshape(-1).astype(int),
                    self.action_space.n)
                if self.phase and self.one_hot
                else np.zeros((np.asarray(observation).shape[0], 0),
                              dtype=np.float32),
            ], axis=1).astype(np.float32))
        with torch.no_grad():
            logits = self.model(feature)
            if test:
                return np.argmax(logits.cpu().numpy(), axis=1)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return np.asarray(
            [self._act_rng.choice(self.action_space.n, p=row)
             for row in probs])

    # ------------------------------------------------------ pretraining
    def load_demos(self, demo_arrays):
        """demo_arrays: dict of np arrays — obs (N,d) flat, next_obs, action,
        reward, done, episode_id.  n-step suffixes are derived from
        episode_id ordering (rows must stay in episode order)."""
        self._demo_data = {k: np.asarray(v) for k, v in demo_arrays.items()}
        self._demo_priorities = np.ones(len(self._demo_data['action']))
        self._demo_beta = 0.4

    def _n_step_targets(self, idxs):
        """yn = sum_k gamma^k r_{t+k} within same episode + gamma^m *
        max_a' Q_t(s_{t+m}) bootstrap when the episode continues."""
        d = self._demo_data
        out = np.zeros(len(idxs), dtype=np.float32)
        bootstrap_obs = []
        bootstrap_pos = []
        for i, t in enumerate(idxs):
            ep = d['episode_id'][t]
            total = 0.0
            m = 0
            for k in range(self.n_step):
                j = t + k
                if j >= len(d['reward']) or d['episode_id'][j] != ep:
                    break
                total += (self.gamma_a ** k) * float(d['reward'][j])
                m = k + 1
                if d['done'][j]:
                    break
            # bootstrap at t+m when the episode chain continues
            j = t + m
            if (j < len(d['reward']) and d['episode_id'][j] == ep
                    and not d['done'][j - 1]):
                bootstrap_obs.append(d['next_obs'][j - 1])
                bootstrap_pos.append((i, m))
            out[i] = total
        if bootstrap_pos:
            obs = torch.as_tensor(
                np.stack(bootstrap_obs), dtype=torch.float32)
            with torch.no_grad():
                q_next = self.target_model(obs).max(dim=1).values.numpy()
            for (i, m), qn in zip(bootstrap_pos, q_next):
                out[i] += (self.gamma_a ** m) * float(qn)
        return out

    def pretrain(self, steps, pretrain_actor=True, pretrain_critic=True,
                 seed=0):
        """Spec §8: actor fixed-uniform stream + critic PER stream, fully
        independent RNGs; critic target sync every 50 critic updates."""
        if self._demo_data is None:
            raise ValueError('pretrain requires load_demos() first')
        d = self._demo_data
        n = len(d['action'])
        actor_rng = np.random.RandomState(int(seed) + 13)
        critic_rng = np.random.RandomState(int(seed) + 29)
        log = {'actor_loss': [], 'critic_loss': []}
        obs_t = torch.as_tensor(d['obs'], dtype=torch.float32)
        act_t = torch.as_tensor(d['action'], dtype=torch.long)
        rew_t = torch.as_tensor(d['reward'], dtype=torch.float32)
        nobs_t = torch.as_tensor(d['next_obs'], dtype=torch.float32)
        done_t = torch.as_tensor(d['done'], dtype=torch.float32)
        order = actor_rng.permutation(n)
        cursor = 0
        for step in range(int(steps)):
            if pretrain_actor:
                if cursor + self.pretrain_batch > n:
                    order = actor_rng.permutation(n)
                    cursor = 0
                idx = order[cursor:cursor + self.pretrain_batch]
                cursor += self.pretrain_batch
                logits = self.model(obs_t[idx])
                logp = torch.log_softmax(logits, dim=1)
                # spec-selected reading: gumbel-softmax(log pi) probs,
                # NLL on the expert action.
                g = nn.functional.gumbel_softmax(
                    logp, tau=self.gumbel_tau, hard=False)
                a_onehot = nn.functional.one_hot(
                    act_t[idx], self.action_space.n).float()
                loss_a = -(torch.log(g + 1e-9) * a_onehot).sum(1).mean()
                self.optimizer.zero_grad()
                loss_a.backward()
                self.optimizer.step()
                log['actor_loss'].append(float(loss_a))
            if pretrain_critic:
                beta = 0.4 + 0.6 * (step / max(1, int(steps) - 1))
                p = self._demo_priorities ** PER_ALPHA
                p = p / p.sum()
                idx = critic_rng.choice(
                    n, size=min(self.pretrain_batch, n),
                    replace=False, p=p)
                w = (n * p[idx]) ** (-beta)
                w = w / w.max()
                w = torch.as_tensor(w, dtype=torch.float32)
                x = obs_t[idx]
                q_all = self.critic(x)
                q_a = q_all.gather(1, act_t[idx][:, None]).squeeze(1)
                with torch.no_grad():
                    a_star = self.critic(nobs_t[idx]).argmax(dim=1)
                    y1 = rew_t[idx] + self.gamma_a * (1 - done_t[idx]) * (
                        self.target_model(nobs_t[idx])
                        .gather(1, a_star[:, None]).squeeze(1))
                yn = torch.as_tensor(
                    self._n_step_targets(idx), dtype=torch.float32)
                margin_vec = (q_all + self.margin * (
                    1.0 - nn.functional.one_hot(
                        act_t[idx], self.action_space.n).float())
                ).max(dim=1).values - q_a
                per = w * (
                    (q_a - y1) ** 2 + (q_a - yn) ** 2 + margin_vec)
                l2 = self.critic_l2 * sum(
                    (pr ** 2).sum() for pr in self.critic.parameters())
                loss_c = per.mean() + l2
                self.critic_optimizer.zero_grad()
                loss_c.backward()
                self.critic_optimizer.step()
                self._demo_priorities[idx] = (
                    (q_a - y1).abs().detach().numpy() + 1e-6)
                self._critic_updates += 1
                if self._critic_updates % self.demo_target_interval == 0:
                    self.target_model.load_state_dict(self.critic.state_dict())
                log['critic_loss'].append(float(loss_c))
        self._pretrain_done = {
            'actor': bool(pretrain_actor), 'critic': bool(pretrain_critic)}
        return log

    # ------------------------------------------------------- online phase
    def begin_stage(self, stage_index, network, policy='demo_ac'):
        if policy != 'demo_ac':
            raise ValueError('DemoACAgent requires policy=demo_ac')
        self.policy = policy
        self.current_stage_index = int(stage_index)
        self.current_network = network
        self.replay.stage_insertions = 0
        # spec boundary: fresh optimizers + target copies critic at the
        # online start (pretrain momentum must not leak into online phase).
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(self.model_config['learning_rate']))
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=float(self.model_config['learning_rate']))
        self.target_model.load_state_dict(self.critic.state_dict())
        return {'policy': policy, 'replay_size': len(self.replay.records)}

    def is_update_ready(self):
        if self.policy is None:
            raise RuntimeError('Replay policy has not been applied')
        # per-decision update: one fresh transition suffices after warmup
        return (len(self.replay.records) > self.learning_start
                and len(self.replay.records) >= 1)

    def successful_gradient_update(self):
        """Per-decision update on the most recent transition (spec §8)."""
        if not self.is_update_ready() or not self.replay.records:
            return None
        record = self.replay.records[-1]
        payload = record.payload
        state_t = torch.as_tensor(
            np.concatenate([
                np.asarray(payload.state, dtype=np.float32),
                agent_utils.idx2onehot(
                    np.asarray(payload.phase).reshape(-1).astype(int),
                    self.action_space.n)
                if self.phase and self.one_hot else
                np.zeros((np.asarray(payload.state).shape[0], 0),
                         dtype=np.float32),
            ], axis=1).astype(np.float32))
        next_t = torch.as_tensor(
            np.concatenate([
                np.asarray(payload.next_state, dtype=np.float32),
                agent_utils.idx2onehot(
                    np.asarray(payload.next_phase).reshape(-1).astype(int),
                    self.action_space.n)
                if self.phase and self.one_hot else
                np.zeros((np.asarray(payload.next_state).shape[0], 0),
                         dtype=np.float32),
            ], axis=1).astype(np.float32))
        action = int(np.asarray(payload.action).reshape(-1)[0])
        reward = float(np.asarray(payload.reward).reshape(-1)[0])
        done = float(bool(payload.terminated) or bool(payload.truncated))
        # --- critic: 0.5*(Q(s,a) - y)^2, a' ~ pi(s')
        with torch.no_grad():
            next_logits = self.model(next_t)
            next_probs = torch.softmax(next_logits, dim=1).numpy()[0]
            a_prime = int(self._act_rng.choice(
                self.action_space.n, p=next_probs))
            y = reward + self.gamma_a * (1 - done) * float(
                self.target_model(next_t)[0, a_prime])
        q_now = self.critic(state_t)[0, action]
        critic_loss = 0.5 * (q_now - y) ** 2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        self._critic_updates += 1
        target_synced = False
        if self._critic_updates % self.demo_target_interval == 0:
            self.target_model.load_state_dict(self.critic.state_dict())
            target_synced = True
            self.counters.target_updates += 1
        # --- actor: -log pi(a|s) * A, A = sg(Q - E_pi Q)
        with torch.no_grad():
            q_s = self.critic(state_t)[0]
            pi_s = torch.softmax(self.model(state_t), dim=1)[0]
            advantage = float(q_now.detach() - (pi_s * q_s).sum())
        logp = torch.log_softmax(self.model(state_t), dim=1)[0, action]
        actor_loss = -logp * advantage
        self.optimizer.zero_grad()
        actor_loss.backward()
        self.optimizer.step()
        self.counters.gradient_updates += 1
        return {
            'loss': float(actor_loss.detach().cpu().item()),
            'critic_loss': float(critic_loss.detach().cpu().item()),
            'advantage': advantage,
            'sample_transition_ids': [record.metadata.transition_id],
            'sample_sources': [record.metadata.source_network],
            'sample_ages': [0],
            'target_synced': target_synced,
        }

    # ------------------------------------------------------ checkpoints
    def full_state_dict(self):
        state = super().full_state_dict()
        state['architecture_name'] = 'demo_ac'
        state['demo_ac'] = {
            'critic_state_dict': copy.deepcopy(self.critic.state_dict()),
            'critic_optimizer_state_dict': copy.deepcopy(
                self.critic_optimizer.state_dict()),
            'critic_updates': self._critic_updates,
            'pretrain_done': dict(self._pretrain_done),
            'demo_target_interval': self.demo_target_interval,
        }
        return state

    def load_full_state_dict(self, state):
        super().load_full_state_dict(state)
        block = state.get('demo_ac')
        if block:
            self.critic.load_state_dict(block['critic_state_dict'])
            self.critic_optimizer.load_state_dict(
                block['critic_optimizer_state_dict'])
            self._critic_updates = int(block.get('critic_updates', 0))
            self._pretrain_done = dict(block.get('pretrain_done', {}))
            # optimizer (actor) state restored by super(); target_model is
            # the critic target in this class.
