#!/usr/bin/env python3
"""Evaluate one trained checkpoint under a single deterministic episode.

This is the per-agent evaluation route for controllers whose checkpoint
format is not the DQN-style ``checkpoints/evaluation`` schema (currently
MPLight).  It loads the checkpoint into the registered agent, then invokes
``run.py`` with an evaluation-only overlay; the training loop, trainer and
world are unchanged.

The evaluation conditions (network/demand pack, SUMO seed, horizon, action
interval and signal-control semantics) are defined by ``--config``.  Use the
same conditions for every controller in a comparison.

Example (legacy signal semantics):
    python tools/eval_baseline_checkpoint.py \
        --agent mplight --network xiasha1_dense_high \
        --config configs/tsc/eval_baseline_checkpoint.yml \
        --checkpoint data/output_data/tsc/sumo_mplight/xiasha1_dense_high/<run>/model/400_0.pt \
        --seed 0 --sumo_seed 10000 --prefix mplight_eval_xiasha1_dense_high_ss10000

For corrected green/yellow semantics use
``configs/tsc/eval_baseline_checkpoint_gy5.yml`` instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_payload(path):
    payload = torch.load(path, map_location='cpu')
    if isinstance(payload, dict) and 'agents' in payload:
        agents = payload['agents']
        if not isinstance(agents, list) or len(agents) != 1:
            raise ValueError('Checkpoint must contain exactly one agent')
        return agents[0]['online_model_state_dict'], payload
    if isinstance(payload, dict) and 'model_state_dict' in payload:
        return payload['model_state_dict'], payload
    return payload, {}


def _agent_class(name):
    import agent  # noqa: F401  (registers all model classes)
    from common.registry import Registry
    try:
        return Registry.mapping['model_mapping'][name]
    except KeyError as error:
        raise SystemExit(f'Unknown agent {name!r}; registered: '
                         f'{sorted(Registry.mapping["model_mapping"])}') from error


def run(args):
    checkpoint = Path(args.checkpoint).resolve()
    if not checkpoint.is_file():
        raise SystemExit(f'Checkpoint not found: {checkpoint}')
    state, payload = _checkpoint_payload(checkpoint)
    agent_class = _agent_class(args.agent)
    original_init = agent_class.__init__

    def patched_init(self, world, rank):
        original_init(self, world, rank)
        self.model.load_state_dict(state)
        if 'optimizer_state_dict' in payload and hasattr(self, 'optimizer'):
            self.optimizer.load_state_dict(payload['optimizer_state_dict'])
        # DQN-style agents keep a separate target model.  MPLight keeps its
        # target inside the pfrl inner agent.  SAC has a distinct target bundle.
        if hasattr(self, 'agents_iner') and hasattr(self.agents_iner, 'target_model'):
            self.agents_iner.target_model.load_state_dict(self.model.state_dict())
        elif hasattr(self, 'target_model') and self.target_model is not None:
            try:
                self.target_model.load_state_dict(self.model.state_dict())
            except RuntimeError:
                if 'target_model_state_dict' in payload:
                    self.target_model.load_state_dict(
                        payload['target_model_state_dict'])
        if hasattr(self, 'model'):
            self.model.eval()
        if hasattr(self, 'target_model') and self.target_model is not None:
            self.target_model.eval()

    agent_class.__init__ = patched_init

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    sys.argv = [
        str(ROOT / 'run.py'),
        '-w', 'sumo',
        '-a', args.agent,
        '-n', args.network,
        '--seed', str(args.seed),
        '--ngpu', args.ngpu,
        '--interface', 'libsumo',
        '--prefix', args.prefix,
        '--episodes', '1',
        '--experiment-config', str(config_path),
    ]
    if args.sumo_seed is not None:
        sys.argv += ['--sumo_seed', str(args.sumo_seed)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')

    run_dir = (ROOT / 'data' / 'output_data' / 'tsc' /
               f'sumo_{args.agent}' / args.network / args.prefix)
    records_path = run_dir / 'metrics' / 'records.jsonl'
    final = None
    if records_path.is_file():
        for line in records_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get('record_type') == 'FINAL_EVALUATION':
                final = record
    if final is None:
        raise SystemExit(
            f'No FINAL_EVALUATION record in {records_path}; '
            'check the run log for the failure.')

    summary = {
        'schema_version': 1,
        'agent': args.agent,
        'network': args.network,
        'prefix': args.prefix,
        'training_seed': args.seed,
        'sumo_seed': args.sumo_seed,
        'sumo_seed_mode': (
            'fixed_default' if args.sumo_seed is None else 'explicit'
        ),
        'config': str(config_path.resolve()),
        'checkpoint': str(checkpoint),
        'checkpoint_sha256': _sha256(checkpoint),
        'run_dir': str(run_dir.resolve()),
        'travel_time': final.get('travel_time'),
        'throughput': final.get('throughput'),
        'unfinished_vehicles': final.get('unfinished_vehicles'),
        'queue': final.get('queue'),
        'real_delay': final.get('real_delay'),
        'delay': final.get('delay'),
        'reward_mean': final.get('reward_mean'),
        'phase_switches': final.get('phase_switches'),
        'action_distribution': final.get('action_distribution'),
    }
    output = Path(args.output) if args.output else run_dir / 'baseline_eval_summary.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--network', required=True)
    parser.add_argument('--config', required=True,
                        help='evaluation-only overlay (see configs/tsc/eval_baseline_checkpoint*.yml)')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--sumo_seed', type=int, default=None,
                        help='explicit SUMO seed; omit to keep the SUMO '
                             'fixed_default seed used by training runs')
    parser.add_argument('--ngpu', default='-1')
    parser.add_argument('--output', default=None)
    run(parser.parse_args())


if __name__ == '__main__':
    main()
