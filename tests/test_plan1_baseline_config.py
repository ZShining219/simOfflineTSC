import unittest
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Plan1BaselineConfigTest(unittest.TestCase):
    def test_traditional_baselines_use_one_explicit_episode(self):
        for agent in ("fixedtime", "maxpressure"):
            path = PROJECT_ROOT / "configs" / "tsc" / f"{agent}.yml"
            with path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(
                1, config["trainer"].get("episodes"),
                f"{agent} must label its single final evaluation as episode 1",
            )

    def test_dqn_ab_sparse_schedule_is_explicit(self):
        path = PROJECT_ROOT / "configs" / "tsc" / "dqn.yml"
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(5, config["trainer"].get("episodes"))
        self.assertEqual(
            [0, 5],
            config["trainer"].get("evaluation_episodes"),
        )
        self.assertEqual(
            [0, 5],
            config["trainer"].get("resumable_checkpoint_episodes"),
        )


if __name__ == "__main__":
    unittest.main()
