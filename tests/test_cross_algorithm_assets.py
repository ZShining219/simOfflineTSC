import json
import os
import tempfile
import unittest

from sequential.historical_assets import (
    build_historical_asset_plan,
    validate_historical_asset_plan,
)


class CrossAlgorithmAssetPlanTests(unittest.TestCase):
    def test_asset_chain_has_frozen_twenty_run_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'assets.json')
            payload = build_historical_asset_plan(
                output,
                'configs/sequential/ha_cross_algorithm_assets_v1.yml',
            )
            result = validate_historical_asset_plan(output)
            self.assertEqual(payload['protocol_id'], 'ha_cross_algorithm_assets_v1')
            self.assertEqual(result['source_count'], 20)
            self.assertEqual(result['missing_path_count'], 7)

    def test_asset_plan_digest_detects_edit(self):
        with tempfile.TemporaryDirectory() as directory:
            output = os.path.join(directory, 'assets.json')
            build_historical_asset_plan(
                output,
                'configs/sequential/ha_cross_algorithm_assets_v1.yml',
            )
            with open(output, encoding='utf-8') as handle:
                payload = json.load(handle)
            payload['status'] = 'ready'
            with open(output, 'w', encoding='utf-8') as handle:
                json.dump(payload, handle)
            with self.assertRaisesRegex(ValueError, 'digest mismatch'):
                validate_historical_asset_plan(output)


if __name__ == '__main__':
    unittest.main()
