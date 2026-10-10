import dataclasses
import os
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
import torch

from sequential.adjudication import (
    Adjudicator, QGapJudge, RewardModelJudge, TermsJudge,
    build_adjudicator, leg_features_from_view)
from sequential.core import ReplayRecord, TrainingPayload, ReplayMetadata
from sequential.historical_archive import HistoricalBatch


def _feats(action=0, obs=None, next_obs=None):
    return {
        'obs': np.zeros(8, dtype=np.float64) if obs is None else obs,
        'action': int(action), 'action_dim': 4,
        'next_obs': np.zeros(8) if next_obs is None else next_obs,
        'q_ratio': np.asarray([0.2, 0.0, 0.5, 0.1]),
        'gap_norm': np.asarray([0.0, 0.0, 1.0, 0.5]),
    }


class TermsJudgeTest(unittest.TestCase):
    def test_weighted_terms(self):
        judge = TermsJudge({'queue_per_capacity': -0.5,
                            'unserved_max': -0.5})
        # q mean = 0.2; gap max = 1.0 -> -0.1 - 0.5 = -0.6
        self.assertAlmostEqual(-0.6, judge.score(_feats()), places=9)

    def test_unknown_term_rejected(self):
        with self.assertRaises(ValueError):
            TermsJudge({'safety_margin': 1.0})

    def test_all_zero_weights_rejected(self):
        with self.assertRaises(ValueError):
            TermsJudge({'queue_max': 0.0})


class AdjudicatorTest(unittest.TestCase):
    def _panel(self, judges, **kw):
        return Adjudicator(
            {'lambda': kw.pop('lam', 0.1), 'clip': 3.0, 'judges': judges}, 4)

    def test_single_judge_identity_scale(self):
        # mu=sigma calibrated so raw -0.6 -> z = (-0.6 - (-0.6))/1 = 0
        adj = self._panel([{
            'type': 'terms', 'terms': {'unserved_max': -1.0},
            'weight': 1.0, 'mu': -1.0, 'sigma': 1.0,
        }])
        z, detail = adj.z(None, 0, None,
                          _feats()['q_ratio'], _feats()['gap_norm'])
        self.assertAlmostEqual(0.0, z, places=9)
        self.assertAlmostEqual(
            -1.0 + 0.1 * 0.0,
            adj.effective_reward(-1.0, z), places=9)

    def test_weighted_aggregation_deterministic(self):
        judges = [
            {'type': 'terms', 'terms': {'queue_per_capacity': -1.0},
             'weight': 2.0, 'mu': 0.0, 'sigma': 1.0},
            {'type': 'terms', 'terms': {'unserved_max': -1.0},
             'weight': 1.0, 'mu': 0.0, 'sigma': 1.0},
        ]
        adj = self._panel(judges)
        f = _feats()
        z1, d1 = adj.z(f['obs'], 0, f['next_obs'],
                       f['q_ratio'], f['gap_norm'])
        z2, _ = adj.z(f['obs'], 0, f['next_obs'],
                      f['q_ratio'], f['gap_norm'])
        # z_j1=-0.2, z_j2=-1.0 -> (2*-0.2 + 1*-1.0)/3 = -0.4666..
        self.assertAlmostEqual(-1.4 / 3.0, z1, places=9)
        self.assertEqual(z1, z2)
        self.assertEqual(2, len(d1['judges']))

    def test_failed_judge_drops_weight_and_scores_zero(self):
        class Boom:
            artifact_digest = None

            def score(self, feats):
                raise RuntimeError('collapsed')

        adj = Adjudicator({'judges': [{
            'type': 'terms', 'terms': {'queue_max': -1.0}}]}, 4)
        adj.judges = [
            {'judge': Boom(), 'type': 'boom', 'weight': 3.0,
             'mu': 0.0, 'sigma': 1.0},
            {'judge': TermsJudge({'unserved_max': -1.0}), 'type': 'terms',
             'weight': 1.0, 'mu': 0.0, 'sigma': 1.0},
        ]
        f = _feats()
        z, detail = adj.z(f['obs'], 0, f['next_obs'],
                          f['q_ratio'], f['gap_norm'])
        self.assertIn('boom', detail['fallback'])
        # boom contributes 0 with weight 3, terms -1.0 with weight 1
        self.assertAlmostEqual(-1.0 / 4.0, z, places=9)

    def test_manifest_digest_stable_and_sensitive(self):
        spec = {'judges': [{'type': 'terms',
                            'terms': {'queue_max': -1.0},
                            'weight': 1.0, 'mu': 0.0, 'sigma': 1.0}]}
        d1 = Adjudicator(spec, 4).digest
        d2 = Adjudicator(dict(spec), 4).digest
        self.assertEqual(d1, d2)
        changed = {'judges': [{'type': 'terms',
                               'terms': {'queue_max': -1.0},
                               'weight': 2.0, 'mu': 0.0, 'sigma': 1.0}]}
        self.assertNotEqual(d1, Adjudicator(changed, 4).digest)

    def test_build_gating(self):
        self.assertIsNone(build_adjudicator({}, 4))
        self.assertIsNone(
            build_adjudicator({'adjudication': {'enabled': False}}, 4))
        with self.assertRaises(ValueError):
            build_adjudicator({'adjudication': {'judges': []}}, 4)


class QGapJudgeTest(unittest.TestCase):
    def _ckpt(self, episode, updates):
        from agent.dqn import DQNNet
        net = DQNNet(8, 4)
        path = tempfile.mktemp(suffix='.pt')
        torch.save({
            'checkpoint_type': 'resumable', 'episode': episode,
            'gradient_updates': updates,
            'agents': [{'online_model_state_dict': net.state_dict()}],
        }, path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        return path

    def test_episode0_checkpoint_rejected(self):
        with self.assertRaises(ValueError):
            QGapJudge(self._ckpt(episode=0, updates=0))

    def test_unconverged_below_min_episode_rejected(self):
        with self.assertRaises(ValueError):
            QGapJudge(self._ckpt(episode=5, updates=5000))

    def test_converged_checkpoint_accepted(self):
        judge = QGapJudge(self._ckpt(episode=20, updates=5000))
        self.assertEqual(
            20, judge.artifact_digest['checkpoint_episode'])

    def test_explicit_override_loads_but_flags(self):
        judge = QGapJudge(
            self._ckpt(episode=0, updates=0), allow_unconverged=True)
        self.assertTrue(judge.artifact_digest['allow_unconverged'])

    def test_score_is_q_gap(self):
        judge = QGapJudge(self._ckpt(episode=20, updates=1))
        raw = judge.score(_feats(obs=np.zeros(8)))
        self.assertLessEqual(raw, 0.0)


class RewardModelJudgeTest(unittest.TestCase):
    def test_scores_with_frozen_net(self):
        from sequential.reward_model import PreferenceRewardNet
        feat_dim = 8 + 4 + 8 + 4 + 4
        net = PreferenceRewardNet(feat_dim)
        path = tempfile.mktemp(suffix='.pt')
        torch.save({'feature_dim': feat_dim, 'state_dict': net.state_dict()},
                   path)
        self.addCleanup(lambda: os.path.exists(path) and os.unlink(path))
        judge = RewardModelJudge(path)
        value = judge.score(_feats(obs=np.zeros(8), next_obs=np.zeros(8)))
        self.assertIsInstance(value, float)


class LegFeaturesTest(unittest.TestCase):
    def _view(self):
        """2 networks x 2 'episodes'; episode numbering collides across
        run_ids to prove (run_id, episode_id) grouping."""
        n_lanes, n_obs = 4, 8
        arrays = {
            'run_id': np.asarray(['r1'] * 6 + ['r2'] * 6, dtype=object),
            'episode_id': np.asarray([1] * 3 + [2] * 3 + [1] * 3 + [2] * 3),
            'decision_step': np.asarray([1, 2, 3] * 4),
            'actions': np.asarray([0, 0, 1] * 4, dtype=np.int64),
            'observations': np.zeros((12, n_obs), dtype=np.float32),
            'next_observations': np.tile(
                np.arange(n_obs, dtype=np.float32), (12, 1)),
            'transition_id': np.asarray(
                [f't{i}' for i in range(12)], dtype=object),
        }
        dataset = SimpleNamespace(_arrays={'net': arrays})
        archive = SimpleNamespace(datasets={'net': dataset})
        view = SimpleNamespace(
            archive=archive, networks=('net',),
            indices_by_network={'net': np.arange(12)},
        )
        ctx = {
            'lane_capacities': [10.0] * n_lanes,
            'phase_served_lanes': [[0, 1], [2, 3]],
            'action_interval': 10.0, 'T_u': 60.0,
        }
        return view, ctx

    def test_reconstruction_deterministic_and_keyed(self):
        view, ctx = self._view()
        f1 = leg_features_from_view(view, ctx)
        f2 = leg_features_from_view(view, ctx)
        self.assertEqual(12, len(f1))
        for tid in f1:
            np.testing.assert_array_equal(f1[tid]['q_ratio'],
                                          f2[tid]['q_ratio'])
            np.testing.assert_array_equal(f1[tid]['gap_norm'],
                                          f2[tid]['gap_norm'])
        # occupancy = next_obs[:4] = [0,1,2,3] -> q_ratio [0,.1,.2,.3]
        np.testing.assert_allclose(
            f1['t0']['q_ratio'], [0.0, 0.1, 0.2, 0.3])
        # action 0 serves lanes 0,1: unserved lanes 2,3 accumulate 10s
        np.testing.assert_allclose(
            f1['t0']['gap_norm'], [0, 0, 10 / 60.0, 10 / 60.0])

    def test_run_id_collision_does_not_merge_episodes(self):
        # r1#ep1 (t0..t2) and r2#ep1 (t6..t8) both advance the same
        # action sequence; tracker state resets per (run, episode), so
        # step-1 gaps are identical in both runs — proving no merge.
        view, ctx = self._view()
        f = leg_features_from_view(view, ctx)
        np.testing.assert_array_equal(
            f['t0']['gap_norm'], f['t6']['gap_norm'])
        np.testing.assert_array_equal(
            f['t1']['gap_norm'], f['t7']['gap_norm'])


class _Space:
    n = 4


class UnifiedScoringTest(unittest.TestCase):
    """The core invariant: online and leg transitions carrying the same
    content receive the same r_eff under the same judge manifest."""

    def _agent(self):
        from sequential.ha_agent import HASODQNAgent
        a = HASODQNAgent.__new__(HASODQNAgent)
        a.phase = True
        a.one_hot = True
        a.action_space = _Space()
        a.adjudicator = Adjudicator({'judges': [{
            'type': 'terms',
            'terms': {'unserved_max': -1.0},
            'weight': 1.0, 'mu': 0.0, 'sigma': 1.0,
        }]}, 4)
        a._adj_ctx = {'lane_capacities': [10.0] * 4,
                      'phase_served_lanes': [[0, 1], [2, 3]],
                      'action_interval': 10.0, 'T_u': 60.0}
        a._adj_tracker = None
        a._adj_episode = None
        a._leg_feature_cache = {}
        return a

    def _record(self, tid, action, next_state, reward, feats):
        return ReplayRecord(
            TrainingPayload(
                np.zeros((1, 8), dtype=np.float32),
                np.asarray([action], dtype=np.int8),
                np.asarray([action]), np.asarray([reward]),
                np.asarray([next_state], dtype=np.float32),
                np.asarray([action], dtype=np.int8),
            ),
            ReplayMetadata(tid, 'current', 1, 1, 0, 0,
                           reward_aux={'adjudication': feats,
                                       'r_base': reward}),
        )

    def _leg_batch(self, tids, actions, next_states, rewards):
        n = len(tids)
        return HistoricalBatch(
            observations=np.zeros((n, 12), dtype=np.float32),
            actions=np.asarray(actions, dtype=np.int64),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(next_states, dtype=np.float32),
            terminated=np.zeros(n, dtype=np.bool_),
            truncated=np.zeros(n, dtype=np.bool_),
            source_networks=np.asarray(['past'] * n, dtype=object),
            local_indices=np.arange(n),
            transition_ids=np.asarray(tids, dtype=object),
            behavior_training_seeds=np.zeros(n, dtype=np.int64),
            episode_ids=np.ones(n, dtype=np.int64),
            decision_steps=np.arange(n) + 1,
            run_ids=np.asarray(['r'] * n, dtype=object),
            shard_sha256=np.asarray(['x'] * n, dtype=object),
        )

    def test_same_transition_same_effective_reward(self):
        agent = self._agent()
        next_state = np.asarray([1, 0, 4, 2, 0, 0, 0, 0], dtype=np.float32)
        feats = {'q_ratio': [0.1, 0.0, 0.4, 0.2],
                 'gap_norm': [0.0, 0.0, 0.5, 0.0]}
        rec = self._record('t1', action=1, next_state=next_state,
                           reward=-2.0, feats=feats)
        online_r, st = agent._apply_adjudication(
            [rec], torch.as_tensor([-2.0]))
        self.assertEqual(0, st['missing_features'])

        agent._leg_feature_cache['offline'] = {'t1': feats}
        batch = self._leg_batch(['t1'], [1], [next_state], [-2.0])
        leg_r, lst = agent._apply_adjudication_leg(
            torch.as_tensor([-2.0]), batch, 'offline')
        self.assertEqual(0, lst['missing_features'])
        # identical content -> identical effective reward
        self.assertAlmostEqual(float(online_r[0]), float(leg_r[0]),
                               places=6)
        # z = (0 - (-0.5))/1 * w -> clip; r_eff = -2.0 + 0.1*(-0.5)
        self.assertAlmostEqual(-2.05, float(leg_r[0]), places=6)

    def test_missing_features_score_zero_not_crash(self):
        agent = self._agent()
        rec = self._record('t1', action=0,
                           next_state=np.zeros(8), reward=-2.0,
                           feats=None)
        rec = dataclasses.replace(
            rec, metadata=dataclasses.replace(
                rec.metadata, reward_aux={'r_base': -2.0}))
        r, st = agent._apply_adjudication([rec], torch.as_tensor([-2.0]))
        self.assertEqual(1, st['missing_features'])
        self.assertAlmostEqual(-2.0, float(r[0]), places=6)  # z=0 -> r_base

    def test_leg_missing_tid_scores_zero(self):
        agent = self._agent()
        agent._leg_feature_cache['expert'] = {}
        batch = self._leg_batch(['zz'], [0],
                                [np.zeros(8, dtype=np.float32)], [-2.0])
        r, st = agent._apply_adjudication_leg(
            torch.as_tensor([-2.0]), batch, 'expert')
        self.assertEqual(1, st['missing_features'])
        self.assertAlmostEqual(-2.0, float(r[0]), places=6)


class ManifestBindingTest(unittest.TestCase):
    """Judge manifest is bound into the checkpoint: silent panel swaps
    on resume must raise, matching digests must pass."""

    def _agent(self):
        from test_ha_agent import agent
        return agent(0.0)

    def _adjudicator(self, weight=1.0):
        return Adjudicator({'judges': [{
            'type': 'terms', 'terms': {'unserved_max': -1.0},
            'weight': weight, 'mu': 0.0, 'sigma': 1.0}]}, 8)

    def test_resume_with_same_manifest_passes(self):
        a = self._agent()
        a.adjudicator = self._adjudicator()
        state = a.full_state_dict()
        b = self._agent()
        b.adjudicator = self._adjudicator()
        b.load_full_state_dict(state)  # no raise

    def test_resume_with_changed_manifest_raises(self):
        a = self._agent()
        a.adjudicator = self._adjudicator(weight=1.0)
        state = a.full_state_dict()
        b = self._agent()
        b.adjudicator = self._adjudicator(weight=2.0)
        with self.assertRaises(ValueError):
            b.load_full_state_dict(state)

    def test_adding_panel_to_plain_checkpoint_raises(self):
        a = self._agent()          # no adjudicator -> saved as None
        state = a.full_state_dict()
        b = self._agent()
        b.adjudicator = self._adjudicator()
        with self.assertRaises(ValueError):
            b.load_full_state_dict(state)

    def test_removing_panel_raises(self):
        a = self._agent()
        a.adjudicator = self._adjudicator()
        state = a.full_state_dict()
        b = self._agent()
        with self.assertRaises(ValueError):
            b.load_full_state_dict(state)


if __name__ == '__main__':
    unittest.main()
