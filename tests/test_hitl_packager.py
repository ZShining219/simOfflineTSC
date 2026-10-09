"""HITL session packager tests: synthetic recorder sessions -> archives.

The packager (dev/hitl_live/package.py) is exercised entirely on hand-built
``transitions.jsonl``/``manifest.json`` fixtures — no SUMO required.  A tiny
roadnet XML fixture feeds ``_scene_semantics`` so the produced dataset
manifest carries real semantic hashes.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dev.hitl_live.package import (
    PackagingError, _AUTHORSHIP_ORDER, package_session,
)
from dataset.offline_trajectory_dataset import (
    OfflineTrajectoryDataset, validate_offline_dataset,
)

_STATE_DIM = 4
_ACTION_DIM = 2

_NET_XML = """<net>
  <edge id="e_in" from="n_in" to="J0"><lane id="e_in_0" index="0"/></edge>
  <edge id="e_out" from="J0" to="n_out"><lane id="e_out_0" index="0"/></edge>
  <connection from="e_in" to="e_out" fromLane="0" toLane="0"
              tl="J0" linkIndex="0"/>
  <tlLogic id="J0" type="static" programID="0">
    <phase duration="30" state="GG"/>
    <phase duration="30" state="rG"/>
  </tlLogic>
</net>
"""


def _write_fixture(root):
    """Minimal sim cfg + roadnet so _scene_semantics has real inputs."""
    root = Path(root)
    data = root / 'data'
    data.mkdir(parents=True)
    (data / 'net.xml').write_text(_NET_XML, encoding='utf-8')
    cfg = {
        'network': 'testnet', 'dir': str(data), 'roadnetFile': 'net.xml',
        'interval': 1.0, 'seed': 0,
    }
    cfg_path = root / 'sim.cfg'
    cfg_path.write_text(json.dumps(cfg), encoding='utf-8')
    return cfg_path


def _row(ep, ds, gs, state, next_state, *, rank=0, junction='J0',
         phase=0, next_phase=0, action=0, reward=-2.0, reward_steps=10,
         truncated=False, hitl=None, metrics=None, **extra):
    row = {
        'schema_version': 1,
        'episode_id': ep,
        'decision_step': ds,
        'global_step': gs,
        'rank': rank,
        'junction': junction,
        't0': 10.0 + ds * 10,
        't1': 20.0 + ds * 10,
        'state': list(state),
        'current_phase': phase,
        'action': action,
        'reward': reward,
        'reward_steps': reward_steps,
        'reward_error': False,
        'next_state': list(next_state),
        'next_phase': next_phase,
        'terminated': False,
        'truncated': truncated,
        'mid_interval_override': False,
        'hitl': hitl or {
            'student_id': 'dqn_s2', 'student_action': action,
            'queried': False, 'query_reason': None, 'expert_id': None,
            'expert_action': None, 'accepted_action': action,
            'intervened': False,
        },
        'metrics': metrics if metrics is not None else {
            'junctions': {'J0': {'queue': 5.0, 'approximate_delay': 2.0}},
            'real_delay': 10.0, 'throughput': 42.0, 'waiting_time': 3.5,
        },
    }
    row.update(extra)
    return row


def _episode_rows(ep, gs_base, count, *, hitl_at=None):
    """Contiguous chained rows: ds 0..count-1, state carries a counter."""
    rows = []
    for i in range(count):
        state = [float(gs_base + i), 0.0, 0.0, 0.0]
        next_state = [float(gs_base + i + 1), 0.0, 0.0, 0.0]
        hitl = None
        if hitl_at is not None and i == hitl_at:
            hitl = {
                'student_id': 'dqn_s2', 'student_action': 1,
                'queried': True, 'query_reason': 'periodic',
                'expert_id': 'maxpressure', 'expert_action': 0,
                'accepted_action': 0, 'intervened': True,
            }
        rows.append(_row(
            ep, i, gs_base + i, state, next_state,
            phase=i % _ACTION_DIM, next_phase=(i + 1) % _ACTION_DIM,
            action=0 if hitl else 1,
            truncated=(i == count - 1), hitl=hitl))
    return rows


def _write_session(root, rows, **meta_over):
    root = Path(root)
    root.mkdir(parents=True)
    manifest = {
        'schema_version': 1,
        'session_id': root.name,
        'scene': 'S2',
        'source_config': meta_over.pop('source_config'),
        'action_interval': 10,
        'expert_id': 'maxpressure',
        'gate': {'mode': 'periodic', 'period': 3, 'budget': 36},
        'junctions': {'0': 'J0'},
        'action_dims': {'0': _ACTION_DIM},
        'episodes': [],
        'counts': {'transitions': len(rows)},
    }
    manifest.update(meta_over)
    (root / 'manifest.json').write_text(
        json.dumps(manifest), encoding='utf-8')
    with open(root / 'transitions.jsonl', 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')
    return root


class HitlPackagerTest(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.work = Path(self._tmp.name)
        self.cfg = _write_fixture(self.work / 'fixture')

    def _package(self, rows, session_name='sess_01', **kwargs):
        session = _write_session(
            self.work / 'sessions' / session_name, rows,
            source_config=str(self.cfg))
        out = self.work / 'archives' / session_name
        return package_session(
            str(session), str(out),
            config_path=str(self.cfg), **kwargs), out

    def test_valid_session_produces_loadable_dataset(self):
        rows = _episode_rows(1, 0, 3, hitl_at=1) + _episode_rows(2, 3, 2)
        report, out = self._package(rows, network='testnet',
                                    behavior_seed=7)
        self.assertTrue(report['valid'], report)
        self.assertEqual(report['transition_count'], 5)
        self.assertEqual(report['episodes_packaged'], 2)
        self.assertEqual(report['episodes_quarantined'], [])
        load = report['dataset_load']
        self.assertEqual(load['transition_count'], 5)
        self.assertEqual(load['observation_dim'], _STATE_DIM + _ACTION_DIM)
        # on-disk structure
        self.assertTrue((out / 'trajectory' / 'manifest.json').is_file())
        self.assertTrue((out / 'trajectory' / 'validation.json').is_file())
        self.assertTrue((out / 'source_shards.jsonl').is_file())
        self.assertTrue(
            (out / 'datasets' / 'testnet' / 'hitl' / 'manifest.json')
            .is_file())
        self.assertTrue((out / 'hitl' / 'labels.jsonl').is_file())
        # shard contents are Plan-1 shaped
        with np.load(out / 'trajectory' / 'episodes' / 'episode_0001.npz',
                     allow_pickle=False) as data:
            self.assertEqual(data['state'].shape, (3, 1, _STATE_DIM))
            self.assertEqual(data['current_phase'].shape, (3, 1))
            self.assertEqual(data['action'].shape, (3, 1))
            self.assertTrue(bool(data['truncated'][-1]))
            self.assertEqual(list(data['decision_step']), [1, 2, 3])
            self.assertEqual(list(data['global_step']), [1, 2, 3])
            self.assertEqual(
                data['behavior_mode'].tolist(),
                ['hitl_autonomous', 'hitl_expert_took_over',
                 'hitl_autonomous'])
            self.assertAlmostEqual(float(data['queue'][0]), 5.0)
            self.assertAlmostEqual(float(data['throughput'][0]), 42.0)

    def test_dataset_manifest_passes_loader_validation(self):
        rows = _episode_rows(1, 0, 4)
        report, out = self._package(rows, network='testnet')
        manifest = out / 'datasets' / 'testnet' / 'hitl' / 'manifest.json'
        result = validate_offline_dataset(str(manifest))
        self.assertTrue(result['valid'])
        dataset = OfflineTrajectoryDataset(str(manifest), seed=0)
        self.assertEqual(dataset.transition_count, 4)
        batch = dataset.sample_batch(2)
        self.assertEqual(
            batch.observations.shape, (2, _STATE_DIM + _ACTION_DIM))
        stats = dataset.statistics()
        self.assertEqual(stats['testnet']['transition_count'], 4)

    def test_broken_state_chain_quarantines_episode(self):
        rows = _episode_rows(1, 0, 3)
        ep2 = _episode_rows(2, 3, 3)
        ep2[1]['state'] = [999.0, 0.0, 0.0, 0.0]  # chain break
        report, _ = self._package(rows + ep2, network='testnet')
        self.assertEqual(report['episodes_packaged'], 1)
        self.assertEqual(report['episodes_quarantined'], ['2'])
        self.assertIn('state chain', str(report['quarantine_reasons']))

    def test_invalid_action_quarantines_episode(self):
        rows = _episode_rows(1, 0, 3)
        bad = _episode_rows(2, 3, 2)
        bad[0]['action'] = _ACTION_DIM + 3
        report, _ = self._package(rows + bad, network='testnet')
        self.assertEqual(report['episodes_packaged'], 1)
        self.assertIn('out of range', str(report['quarantine_reasons']))

    def test_missing_field_quarantines_and_labels(self):
        rows = _episode_rows(1, 0, 2)
        bad = _episode_rows(2, 2, 2)
        del bad[0]['state']
        report, out = self._package(rows + bad, network='testnet')
        self.assertEqual(report['episodes_packaged'], 1)
        labels = [
            json.loads(line) for line in
            (out / 'hitl' / 'labels.jsonl').read_text().splitlines()
            if line.strip()
        ]
        dropped = [r for r in labels if not r['packaged']]
        self.assertEqual(len(dropped), 2)  # both rows of ep2 unpackaged
        self.assertTrue(any(
            'missing fields' in str(r['quarantine_reasons'])
            for r in dropped))

    def test_malformed_json_raises(self):
        session = _write_session(
            self.work / 'sessions' / 'sess_bad', _episode_rows(1, 0, 2),
            source_config=str(self.cfg))
        with open(session / 'transitions.jsonl', 'a',
                  encoding='utf-8') as handle:
            handle.write('{"episode_id": 1, broken\n')
        with self.assertRaises(PackagingError):
            package_session(
                str(session), str(self.work / 'archives' / 'sess_bad'),
                config_path=str(self.cfg), network='testnet')

    def test_expert_intervention_metadata_preserved(self):
        hitl = {
            'student_id': 'dqn_s2', 'student_action': 1,
            'queried': True, 'query_reason': 'queue_growth',
            'expert_id': 'maxpressure', 'expert_action': 0,
            'accepted_action': 0, 'intervened': True,
        }
        rows = _episode_rows(1, 0, 2, hitl_at=0)
        rows[0]['hitl'] = hitl
        report, out = self._package(rows, network='testnet')
        labels = [
            json.loads(line) for line in
            (out / 'hitl' / 'labels.jsonl').read_text().splitlines()
            if line.strip()
        ]
        self.assertEqual(labels[0]['hitl']['expert_id'], 'maxpressure')
        self.assertTrue(labels[0]['hitl']['intervened'])
        self.assertEqual(labels[0]['behavior_mode'],
                         'hitl_expert_took_over')
        self.assertTrue(all(tag in _AUTHORSHIP_ORDER for tag in
                            ('hitl_operator', 'hitl_expert_took_over',
                             'hitl_expert_agreed', 'hitl_autonomous')))
        index = json.loads(
            (out / 'hitl' / 'index.json').read_text())
        self.assertEqual(
            index['authorship_counts']['hitl_expert_took_over'], 1)
        self.assertTrue(report['valid'])

    def test_strict_mode_raises_on_quarantine(self):
        bad = _episode_rows(1, 0, 2)
        bad[0]['action'] = 99
        with self.assertRaises(PackagingError):
            self._package(bad, network='testnet', strict=True)

    def test_yellow_phase_index_quarantines(self):
        rows = _episode_rows(1, 0, 2)
        rows[0]['next_phase'] = 5  # generated yellow index > action_dim
        rows += _episode_rows(2, 2, 2)
        report, _ = self._package(rows, network='testnet')
        self.assertEqual(report['episodes_quarantined'], ['1'])
        self.assertEqual(report['episodes_packaged'], 1)
        self.assertIn('out of range', str(report['quarantine_reasons']))

    def test_semantic_hashes_present(self):
        rows = _episode_rows(1, 0, 2)
        _, out = self._package(rows, network='testnet')
        manifest = json.loads(
            (out / 'datasets' / 'testnet' / 'hitl' / 'manifest.json')
            .read_text())
        semantics = manifest['scene_semantics']['testnet']
        for field in ('roadnet_sha256', 'roadnet_control_topology_sha256',
                      'tl_program_sha256', 'action_mapping_sha256',
                      'feature_schema_sha256', 'reward_definition_sha256'):
            self.assertEqual(len(semantics[field]), 64)
        self.assertEqual(semantics['green_action_count'], _ACTION_DIM)
        self.assertEqual(semantics['controlled_intersection_count'], 1)


if __name__ == '__main__':
    unittest.main()
