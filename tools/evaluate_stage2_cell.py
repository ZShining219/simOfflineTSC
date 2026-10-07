#!/usr/bin/env python3
"""Evaluate one immutable online snapshot with the exact formal protocol."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sequential.config import simulator_config_path
from sequential.evaluator import IndependentEvaluator

def main():
    p = argparse.ArgumentParser()
    for name in ("snapshot", "network", "output_root", "training_network", "controller_id"):
        p.add_argument(f"--{name.replace('_','-')}", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    result = IndependentEvaluator(a.output_root, retries=3, timeout_seconds=180).evaluate(
        a.snapshot, a.network,
        {"simulator_config": simulator_config_path(a.network), "interface": "libsumo", "steps": 3600, "action_interval": 10, "sumo_seed_mode": "fixed_default"},
        {"stage_index": 2, "training_network": a.training_network, "evaluation_network": a.network, "local_episode": a.episode, "global_episode": 100 + a.episode, "controller_id": a.controller_id, "training_seed": a.seed},
    )
    print(json.dumps(result, sort_keys=True))

if __name__ == "__main__": main()
