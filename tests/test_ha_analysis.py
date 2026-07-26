import unittest

from sequential.ha_analysis import _selection_table


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


if __name__ == '__main__':
    unittest.main()
