import unittest

from sequential.cross_algorithm import (
    ALGORITHMS,
    CONDITION_IDS,
    load_cross_algorithm_config,
)
from sequential.cross_analysis import _algorithm_effects, _algorithm_pair_effects


class CrossAlgorithmManifestTest(unittest.TestCase):
    def test_frozen_pilot_matrix_config(self):
        loaded = load_cross_algorithm_config(
            'configs/sequential/ha_cross_algorithm_v1.yml'
        )
        self.assertEqual(('independent_dqn', 'double_dqn', 'dueling_double_dqn'), ALGORITHMS)
        self.assertEqual(
            ('CONT_FIFO', 'P1C_DHOA_R25', 'P1C_CQ_R75', 'P1C_CQA_R75'),
            CONDITION_IDS,
        )
        self.assertEqual(loaded['config']['pilot']['stage_episodes'], [12, 12, 12, 12])
        self.assertEqual(loaded['config']['formal']['stage_episodes'], [100, 100, 100, 100])
        self.assertEqual(loaded['config']['formal']['orders'], ['O1', 'O2', 'O3', 'O4'])
        self.assertEqual(loaded['base']['protocol_id'], 'ha_sodqn_b100_v1')

    def test_effects_pair_each_algorithm_with_its_own_control(self):
        runs = []
        for algorithm, baseline, adapted in (
                ('independent_dqn', 1.0, 0.9),
                ('double_dqn', 1.1, 0.95)):
            for order, seed in (('O1', 0), ('O2', 0)):
                runs.append({
                    'algorithm_id': algorithm, 'condition_id': 'CONT_FIFO',
                    'order_id': order, 'training_seed': seed,
                    'current_adaptation_aulc_mean': baseline,
                    'average_forgetting': 10.0, 'average_retention': .8,
                })
                runs.append({
                    'algorithm_id': algorithm, 'condition_id': 'P1C_DHOA_R25',
                    'order_id': order, 'training_seed': seed,
                    'current_adaptation_aulc_mean': adapted,
                    'average_forgetting': 8.0, 'average_retention': .85,
                })
        effects, groups = _algorithm_effects(
            runs, {'seed': 1, 'resamples': 100},
        )
        self.assertEqual(2, len(effects))
        self.assertEqual({2}, {row['paired_unit_count'] for row in effects})
        pair = _algorithm_pair_effects(groups, {'seed': 1, 'resamples': 100})
        self.assertEqual(1, len(pair))
        self.assertEqual(2, pair[0]['paired_unit_count'])


if __name__ == '__main__':
    unittest.main()
