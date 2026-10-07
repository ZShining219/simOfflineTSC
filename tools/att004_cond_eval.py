#!/usr/bin/env python3
"""ATT-ENTITY-004 formal eval protocol: paired 5-condition x N-rep probes.

For each run + checkpoint episode, launches run_tarl_struct_cf.py under each
TARL_TEXT_CONDITION x each sumo seed, then aggregates episode metrics into
mean±sd tables plus paired canonical-vs-{empty,corrupt} differences.

Conditions: canonical / empty / wrong_location / wrong_type / both_wrong.
Metrics per eval episode: travel_time, throughput, queue (all),
queue in event window (900-1200s), queue post-window, unfinished.

Usage:
    python3.10 tools/att004_cond_eval.py --run-dir <out_dir> --episode 100 \
        --arm-config configs/.../a3s_condeval_e0.yml --reps 3 --sumo-base 7
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PYX = (ROOT / 'data/output_data/cross_algorithm/plan5_b100/environment/'
       'micromamba/envs/colight/bin/python3.10')
CONDS = ('canonical', 'empty', 'wrong_location', 'wrong_type', 'both_wrong')
WINDOW = (900.0, 1200.0)


def run_probe(run_dir, ckpt, cond, sumo_seed, prefix, out_dir, cfg, seed,
              network='hz4x4', agent='tarl_attention'):
    env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'CUDA_VISIBLE_DEVICES': '-1', 'TARL_EVENT_CONDITION': 'normal',
           'TARL_TEXT_CONDITION': cond}
    cmd = [str(PYX), 'tools/run_tarl_struct_cf.py',
           '--checkpoint', str(ckpt), '--config', str(cfg),
           '--agent', agent, '--network', network, '--prefix', prefix,
           '--output', str(out_dir), '--seed', str(seed),
           '--sumo_seed', str(sumo_seed)]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print(f'  FAILED {prefix}\n{r.stderr[-2000:]}', file=sys.stderr)
        return None
    return out_dir / 'tarl_struct_cf_records.jsonl'


def harvest(records_path, run_out):
    recs = [json.loads(l) for l in open(records_path)]
    ev = recs[:360]                     # eval@0 episode
    s = json.load(open(Path(run_out) / 'evaluation/summary.json'))
    e = [x for x in s['evaluations'] if x['episode'] == 0][0]
    qw = [r['queue_now'] for r in ev if WINDOW[0] <= r['simulation_time'] <= WINDOW[1]]
    ql = [r['queue_now'] for r in ev if r['simulation_time'] > WINDOW[1]]
    pres = [r for r in ev if r['n_present_nodes'] > 0]
    n_nodes = len(ev[0]['variants']['canonical']['action']) if ev else 16
    ach = {v: float(np.mean([r['variants'][v]['action_changed_count']
                             for r in pres]) / n_nodes) if pres else float('nan')
           for v in ('empty', 'wrong-location', 'wrong-type', 'both-wrong')}
    return {'travel_time': e['travel_time'], 'queue': e['queue'],
            'delay': e['delay'], 'throughput': e.get('throughput'),
            'unfinished': e.get('unfinished_vehicles'),
            'waiting': e.get('waiting_time'),
            'window_queue': float(np.mean(qw)),
            'post_queue': float(np.mean(ql)),
            'n_event_steps': len(pres), 'action_chg': ach}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run-dir', required=True,
                    help='pilot output dir containing checkpoints/')
    ap.add_argument('--episode', default='100',
                    help='checkpoint episode int, or "auto" = select best of '
                         '{20,50,100} by canonical rep0 TT first')
    ap.add_argument('--arm-config', required=True)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--sumo-base', type=int, required=True,
                    help='base sumo seed; rep r uses base+100r (r0 = existing')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--tag', required=True, help='cell tag e.g. a3s_s17')
    ap.add_argument('--out-root',
                    default='data/output_data/analysis/att_entity_004')
    ap.add_argument('--network', default='hz4x4')
    ap.add_argument('--agent', default='tarl_attention',
                    help='registered agent name (tarl_concat/tarl_selfattn/...)')
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_root = Path(args.out_root)
    if str(args.episode) == 'auto':
        # Validation-based checkpoint selection: canonical x rep0 on each
        # available candidate, then run the full protocol on the winner.
        scored = []
        for e in (20, 50, 100):
            ck = (run_dir / 'checkpoints/resumable' / f'episode_{e:04d}.pt')
            if not ck.exists():
                continue
            tag = f'{args.tag}_sel_e{e}_canonical_v0'
            rp = run_probe(run_dir, ck, 'canonical', args.sumo_base,
                           f'fe_{tag}', out_root / f'fe_{tag}',
                           args.arm_config, args.seed,
                           network=args.network, agent=args.agent)
            run_out = (ROOT / f'data/output_data/tsc/sumo_{args.agent}/'
                       f'{args.network}/fe_{tag}')
            row = harvest(rp, run_out)
            scored.append((row['travel_time'], e))
            print(f'select ckpt e{e}: canonical TT={row["travel_time"]:.1f}')
        assert scored, 'no candidate checkpoints found'
        scored.sort()
        episode = scored[0][1]
        print(f'auto-selected episode_{episode:04d} (TT={scored[0][0]:.1f})')
    else:
        episode = int(args.episode)
    ckpt = run_dir / 'checkpoints/resumable' / f'episode_{episode:04d}.pt'
    assert ckpt.exists(), ckpt
    table = {}
    for cond in CONDS:
        rows = []
        for rep in range(args.reps):
            tag = f'{args.tag}_ep{episode}_{cond}_v{rep}'
            odir = out_root / f'fe_{tag}'
            prefix = f'fe_{tag}'
            rp = run_probe(run_dir, ckpt, cond, args.sumo_base + 100 * rep,
                           prefix, odir, args.arm_config, args.seed,
                           network=args.network, agent=args.agent)
            run_out = (ROOT / f'data/output_data/tsc/sumo_{args.agent}/'
                       f'{args.network}/{prefix}')
            if rp and rp.exists():
                rows.append(harvest(rp, run_out))
        table[cond] = rows
        tts = [r['travel_time'] for r in rows]
        wq = [r['window_queue'] for r in rows]
        print(f'{cond:15s} n={len(rows)} TT={np.mean(tts):.1f}±'
              f'{np.std(tts):.1f} winQ={np.mean(wq):.1f}')
    can = [r['travel_time'] for r in table['canonical']]
    for cond in CONDS[1:]:
        other = [r['travel_time'] for r in table[cond]]
        diffs = [c - o for c, o in zip(can, other)]
        print(f'canonical−{cond}: diffs={[round(d,1) for d in diffs]} '
              f'mean={np.mean(diffs):+.2f}±{np.std(diffs):.2f}')
    out = out_root / f'fe_summary_{args.tag}_ep{episode}.json'
    out.write_text(json.dumps(table, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
