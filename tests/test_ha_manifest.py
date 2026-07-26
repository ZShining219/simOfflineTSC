import unittest

from sequential.ha_manifest import load_ha_config, stage_specifications


class HAManifestTest(unittest.TestCase):
    def test_frozen_config_and_preregistered_stage_matrices(self):
        config = load_ha_config('configs/sequential/ha_sodqn_b100.yml')
        smoke, smoke_budget = stage_specifications(config, 'smoke')
        self.assertEqual(4, len(smoke))
        self.assertEqual([12, 12, 12, 12], smoke_budget)
        self.assertIn(('O2', 0, 'NONE', 'CONT', 0.0), smoke)
        self.assertIn(('O2', 0, 'P1C', 'CQA', 0.5), smoke)
        e0, full_budget = stage_specifications(config, 'E0')
        self.assertEqual([('O2', 0, 'P1C', 'DHOA', 0.5)], e0)
        self.assertEqual([100, 100, 100, 100], full_budget)
        e1, _ = stage_specifications(config, 'E1')
        self.assertEqual(16, len(e1))
        self.assertEqual(16, len(set(e1)))

    def test_dependent_stages_require_explicit_selection(self):
        config = load_ha_config('configs/sequential/ha_sodqn_b100.yml')
        for stage in ('E2', 'E3', 'E4'):
            with self.assertRaisesRegex(ValueError, 'explicit'):
                stage_specifications(config, stage)


if __name__ == '__main__':
    unittest.main()
