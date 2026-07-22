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

    def test_dqn_formal_per_episode_schedule_is_frozen(self):
        path = PROJECT_ROOT / "configs" / "tsc" / "dqn.yml"
        with path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        self.assertEqual(400, config["trainer"].get("episodes"))
        self.assertEqual(
            list(range(401)),
            config["trainer"].get("evaluation_episodes"),
        )
        self.assertEqual(
            [0, 10, 25, 50, 100, 150, 200, 250, 300, 350, 400],
            config["trainer"].get("resumable_checkpoint_episodes"),
        )


if __name__ == "__main__":
    unittest.main()
