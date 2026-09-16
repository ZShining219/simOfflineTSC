import unittest

from tools.citysky_sumo import pipeline


class CitySkySumoPipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = pipeline.read_source(pipeline.DEFAULT_INPUT)
        cls.complete, cls.shift, cls.simulation_end = pipeline.source_design(cls.source)

    def test_source_counts_and_all_od_pairs(self):
        self.assertEqual(len(self.source), 5166)
        self.assertEqual(len(self.complete), 4319)
        self.assertEqual(
            self.complete.groupby(["entry_edge", "exit_edge"]).ngroups,
            16,
        )
        self.assertAlmostEqual(self.shift, 41.91)

    def test_movement_design_includes_u_turns(self):
        counts = self.complete["movement"].value_counts().to_dict()
        self.assertEqual(counts["through"], 2844)
        self.assertEqual(counts["right"], 872)
        self.assertEqual(counts["left"], 593)
        self.assertEqual(counts["u_turn"], 10)

    def test_lane_design_has_one_connection_for_each_od(self):
        for entry_arm in pipeline.ARMS:
            for exit_arm in pipeline.ARMS:
                movement = pipeline.movement_for_pair(
                    f"{entry_arm}2J", f"J2{exit_arm}"
                )
                lanes = pipeline._lane_pairs(movement)
                self.assertGreaterEqual(len(lanes), 1)
        self.assertEqual(len(pipeline._lane_pairs("through")), 2)

    def test_signal_reconstruction_uses_source_labels(self):
        _, samples, _, report = pipeline.reconstruct_signal(
            self.source, self.shift, self.simulation_end
        )
        self.assertGreater(report["accepted_samples"], 8000)
        self.assertGreaterEqual(report["zero_offset_match_rate"], 0.95)
        self.assertTrue(
            samples.loc[samples["label_status"] == "accepted", "label_match"]
            .astype(bool)
            .any()
        )


if __name__ == "__main__":
    unittest.main()
