#!/usr/bin/env python3
"""ATT-ENTITY-004 baseline eval: run non-TARL agents under the same event
protocol (hz4x4_random_v1 plan eval schedule: lane_blockage + road_closure +
global_rain, all within 900-1200s) and record per-decision network queue.

Patches Metrics.update to append {simulation_time, queue_now} per decision
step (queue_now = sum over agents of get_queue(), i.e. network waiting count,
same quantity as run_tarl_struct_cf.py's queue_now).

For RL baselines (colight/mplight/dqn) --checkpoint loads
checkpoint['agents'][0]['online_model_state_dict'] into agent.model.
fixedtime/maxpressure need no checkpoint.

Usage:
    python3.10 tools/att004_baseline_eval.py --agent colight \
        --config configs/tsc/att_entity_004/generated/baseval_colight.yml \
        --checkpoint <run>/checkpoints/resumable/episode_0200.pt \
        --seed 7 --sumo_seed 7 --prefix bcmp_colight_s7_e200_ss7 \
        --output data/output_data/analysis/att_entity_004/baseline_cmp_probe/<tag>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _flush_dump(dump_dir, buf):
    """One episode's per-agent rows -> npz (t/rank/action/queue_i/ob)."""
    rows = buf['rows']
    if not rows:
        return
    buf['ep'] += 1
    ep = buf['ep']
    ranks = sorted({r['rank'] for r in rows})
    ts = sorted({r['t'] for r in rows})
    t_idx = {t: i for i, t in enumerate(ts)}
    r_idx = {r: i for i, r in enumerate(ranks)}
    T, R = len(ts), len(ranks)
    act_dim = max(len(r['action']) for r in rows)
    action = np.full((R, T, act_dim), -1, dtype=np.int64)
    q_dim = max(len(r['queue_i']) for r in rows)
    queue_i = np.full((R, T, q_dim), np.nan, dtype=np.float32)
    ob_dim = max(len(r['ob']) for r in rows)
    ob = np.full((R, T, ob_dim), np.nan, dtype=np.float32)
    for r in rows:
        i, j = r_idx[r['rank']], t_idx[r['t']]
        action[i, j, :len(r['action'])] = r['action']
        queue_i[i, j, :len(r['queue_i'])] = r['queue_i']
        ob[i, j, :len(r['ob'])] = r['ob']
    np.savez_compressed(dump_dir / f'ep{ep:03d}.npz', t=np.asarray(ts),
                        rank=np.asarray(ranks), action=action,
                        queue_per_agent=queue_i, ob=ob)
    buf['rows'] = []
    buf['last_t'] = {}


def run(args):
    import agent  # noqa: F401  (registers all model classes)
    import common.metrics as metrics_mod
    from common.registry import Registry

    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / 'baseline_probe_records.jsonl'

    if args.checkpoint:
        agent_class = Registry.mapping['model_mapping'][args.agent]
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'agents' in checkpoint:
            saved_state = checkpoint['agents'][0]['online_model_state_dict']
        else:
            saved_state = checkpoint['model_state_dict']
        original_init = agent_class.__init__

        def patched_init(self, world, rank):
            original_init(self, world, rank)
            self.model.load_state_dict(saved_state)
            self.model.eval()

        agent_class.__init__ = patched_init

    original_update = metrics_mod.Metrics.update

    def patched_update(self, rewards=None):
        original_update(self, rewards)
        queue_now = float(
            np.sum([np.sum(ag.get_queue()) for ag in self.agents]))
        with records_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps({
                'simulation_time': float(self.world.get_current_time()),
                'queue_now': queue_now,
            }) + '\n')

    metrics_mod.Metrics.update = patched_update

    if args.dump_dir:
        dump_dir = Path(args.dump_dir).resolve()
        dump_dir.mkdir(parents=True, exist_ok=True)
        agent_cls2 = Registry.mapping['model_mapping'][args.agent]
        orig_ga = agent_cls2.get_action
        buf = {'ep': -1, 'last_t': {}, 'rows': []}

        def dumping_ga(self, ob, phase, test=True):
            # force deterministic policy even on train-role episodes
            action = orig_ga(self, ob, phase, test=True)
            t = float(self.world.get_current_time())
            rk = self.rank
            if t <= buf['last_t'].get(rk, -1.0) and rk == 0 and buf['rows']:
                _flush_dump(dump_dir, buf)
            buf['last_t'][rk] = t
            try:
                q_i = np.asarray(self.get_queue(), dtype=np.float32).reshape(-1).tolist()
            except Exception:
                q_i = [float('nan')]
            buf['rows'].append(dict(
                ep=buf['ep'] + 1, t=t, rank=rk,
                action=np.asarray(action).reshape(-1).tolist(),
                queue_i=q_i,
                ob=np.asarray(ob, dtype=np.float32).reshape(-1).tolist()))
            return action

        agent_cls2.get_action = dumping_ga
        # Train-role episodes take sample() while learning_start is huge:
        # redirect to the deterministic policy so dumps are policy rollouts.
        agent_cls2.sample = lambda self: self.get_action(
            self.get_ob(), self.get_phase(), test=True)
        for _n, _fn in {
            'remember': lambda self, *a, **k: None,
            'train': lambda self: np.array(0., dtype=np.float32),
            'is_training_ready': lambda self: False,
            'update_target_network': lambda self: None,
            'save_model': lambda self, **k: None,
        }.items():
            if not hasattr(agent_cls2, _n):
                setattr(agent_cls2, _n, _fn)
        if not hasattr(agent_cls2, 'id'):
            agent_cls2.id = 0
        import atexit
        atexit.register(lambda: _flush_dump(dump_dir, buf))

    sys.argv = [str(ROOT / 'run.py'), '-w', 'sumo', '-a', args.agent,
                '-n', args.network, '--seed', str(args.seed), '--ngpu', '-1',
                '--sumo_seed',
                str(args.sumo_seed if args.sumo_seed is not None else args.seed),
                '--interface', 'libsumo', '--prefix', args.prefix,
                '--experiment-config', str(ROOT / args.config)]
    runpy.run_path(str(ROOT / 'run.py'), run_name='__main__')

    if args.dump_dir:
        dump_dir = Path(args.dump_dir).resolve()
        plan_log = (ROOT / 'data/output_data/tsc' / f'sumo_{args.agent}'
                    / args.network / args.prefix / 'event_plan_log.jsonl')
        if plan_log.exists():
            (dump_dir / 'event_plan_log.jsonl').write_text(plan_log.read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--network', default='hz4x4')
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--sumo_seed', type=int, default=None)
    parser.add_argument('--dump_dir', default=None,
                        help='if set, dump per-agent per-decision arrays '
                             '(t/rank/action/queue_i/ob) as per-episode npz')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
