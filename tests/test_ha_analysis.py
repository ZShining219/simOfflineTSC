import json
import os
import tempfile
import unittest

from sequential.ha_analysis import _resolve_analysis_whitelist, _selection_table


def run(logical_id, archive_mode, method, ratio, adaptation,
        average_forgetting, worst_forgetting, retention):
    return {
        'logical_run_id': logical_id, 'archive_mode': archive_mode,
        'method': method, 'offline_ratio': ratio,
        'current_adaptation_aulc_mean': adaptation,
        'average_forgetting': average_forgetting,
        'worst_forgetting': worst_forgetting,
        'average_retention': retention,
    }


class HAAnalysisTest(unittest.TestCase):
    def test_preregistered_selection_enforces_adaptation_and_retention_gate(self):
        runs = [
            run('CONT', 'NONE', 'CONT', 0, 1.0, 10, 20, .8),
            run('D25', 'P1C', 'DHOA', .25, 1.04, 8, 18, .82),
            run('D75', 'P1C', 'DHOA', .75, 1.08, 5, 12, .9),
            run('CQ50', 'P1C', 'CQ', .5, 1.03, 7, 16, .84),
            run('CQA50', 'P1C', 'CQA', .5, 1.02, 6, 15, .86),
        ]
        table, selected = _selection_table(runs)
        selected_ids = {row['logical_run_id'] for row in selected}
        self.assertEqual({'CONT', 'D25', 'CQ50', 'CQA50'}, selected_ids)
        d75 = next(row for row in table if row['logical_run_id'] == 'D75')
        self.assertFalse(d75['passes_five_percent_adaptation_gate'])
        self.assertFalse(d75['eligible'])

    def test_audit_whitelist_resolves_frozen_plan1_reference(self):
        with tempfile.TemporaryDirectory() as root:
            reference = os.path.join(root, 'plan1.csv')
            with open(reference, 'w', encoding='utf-8') as handle:
                handle.write('run_path,network\n')
            catalog = os.path.join(root, 'catalog.json')
            with open(catalog, 'w', encoding='utf-8') as handle:
                json.dump({'whitelist_path': reference}, handle)
            audit_path = os.path.join(root, 'audit.json')
            with open(audit_path, 'w', encoding='utf-8') as handle:
                json.dump({
                    'valid': True, 'run_count': 1,
                    'runs': [{
                        'logical_run_id': 'run-1',
                        'attempt_dir': os.path.join(root, 'attempt_1'),
                        'ha_audit': {'valid': True},
                    }],
                }, handle)
            resolved = _resolve_analysis_whitelist({
                'initial_state_catalog': catalog,
                'children': [{'logical_run_id': 'run-1'}],
            }, audit_path)
            self.assertEqual('ha_audit_json', resolved['kind'])
            self.assertEqual(reference, resolved['reference_whitelist_path'])
            self.assertEqual(1, resolved['run_count'])

    def test_audit_whitelist_rejects_identity_mismatch(self):
        with tempfile.TemporaryDirectory() as root:
            audit_path = os.path.join(root, 'audit.json')
            with open(audit_path, 'w', encoding='utf-8') as handle:
                json.dump({'valid': True, 'run_count': 0, 'runs': []}, handle)
            with self.assertRaisesRegex(ValueError, 'identity set mismatch'):
                _resolve_analysis_whitelist({
                    'children': [{'logical_run_id': 'required'}],
                }, audit_path)


if __name__ == '__main__':
    unittest.main()
