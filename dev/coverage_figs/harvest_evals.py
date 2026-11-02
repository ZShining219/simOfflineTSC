#!/usr/bin/env python3
"""跨语料 stage×scene 冻结评估收割 -> cache/evals.json

- sequential/ 系（seq8_demand, ha_sodqn, plan34_b100/b400）：
    <run>/attempts/attempt_*/events.jsonl 的 evaluation:stage_k:local_E:<net>
    每 (stage,net) 取 local 下标最大者（=stage 末冻结评估格），
    → artifact → <bucket>/_shared_evaluation/physical/<key>/attempt_1/summary.json
- seq8_anchor（soddqnp_r25，s8f_ 批）：
    tsc/sumo_dqn/sumohz1x1_*/s8f_soddqnp_<ORD>_sd<S>_ev<K>_<scene>/
    evaluation/summary.json 末条。

输出行：{corpus, order, arm, seed, eval_stage, eval_scene, ep, tt, queue,
thr, unfin, wait}；eval_scene 用场景短名（sumohz1x1→'base'）。

用法：python dev/coverage_figs/harvest_evals.py
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data/output_data')
CACHE = os.path.join(ROOT, 'dev/coverage_figs/cache')

SEQ8_BATCHES = {
    'N1': 'sequential/seq8_norep_formal_20261015',
    'R1': 'sequential/seq8_rep_formal_20261015',
    'D1': 'sequential/seq8_drift_formal_20261015',
}
SEQ8_ORDERS = {
    'N1': ['sparse_ultra', 'fluct_burst', 'lt_extreme', 'asym_wonly',
           'dense_extreme', 'st_dom', 'asym_we', 'fluct_ramp'],
    'R1': ['sparse_low', 'lt_high', 'fluct_burst', 'dense_high',
           'sparse_low', 'lt_high', 'fluct_burst', 'dense_high'],
    'D1': ['st_dom', 'sumohz1x1', 'lt_high', 'lt_extreme',
           'lt_wsing', 'asym_wonly', 'asym_we', 'asym_ns'],
}
SEQ8_RUNS = [(order, batch, arm_dir, arm, sd)
             for order, batch in SEQ8_BATCHES.items()
             for arm_dir, arm in
             {'CONT': 'CONT', 'SELF-DHOA-R50': 'DHOA'}.items()
             for sd in ('SD0', 'SD1')]

HA_BUCKET = 'ha_sodqn/formal_e7705f7_20260726'
P34_BUCKETS = {'plan34_b100': 'sequential/plan34_b100_formal_60_20260723',
               'plan34_b400': 'sequential/plan34_b400_formal_60_20260723'}


def scene_tag(net):
    return 'base' if net == 'sumohz1x1' else net.replace('sumohz1x1_', '')


def best_attempt(run_dir):
    best, best_n = None, -1
    for att in glob.glob(os.path.join(run_dir, 'attempt*', '*')) + [run_dir]:
        if not os.path.isdir(att):
            continue
        n = len(glob.glob(os.path.join(att, 'trajectory/episodes/*.npz')))
        if n > best_n:
            best, best_n = att, n
    return best


def read_networks(att_dir):
    for name in ('child_run_manifest.json', 'logical_run_manifest.json'):
        p = os.path.join(att_dir, name)
        if os.path.isfile(p):
            try:
                nets = json.load(open(p)).get('networks')
                if nets:
                    return nets
            except Exception:
                pass
    return []


def _summary_metrics(path):
    s = json.load(open(path))
    return dict(tt=s['travel_time'], queue=s['queue'], thr=s['throughput'],
                unfin=s['unfinished_vehicles'], wait=s['waiting_time'])


def harvest_events(run_dir, shared):
    """events.jsonl -> {(stage,net): metrics}，每 (stage,net) 取 local 最大。"""
    att = best_attempt(run_dir)
    if not att:
        return {}, []
    ev = os.path.join(att, 'events.jsonl')
    if not os.path.exists(ev):
        return {}, []
    cells = {}
    nets = read_networks(att)
    for line in open(ev):
        e = json.loads(line)
        key = e.get('payload', {}).get('operation_key', '')
        if not key.startswith('evaluation:'):
            continue
        _, st, loc, net = key.split(':')
        try:
            k_stage = int(st.split('_')[1])
            k_loc = int(loc.split('_')[1])
        except (IndexError, ValueError):
            continue
        cur = cells.get((k_stage, net))
        if cur is not None and cur[0] >= k_loc:
            continue
        try:
            a = json.load(open(e['payload']['artifact']))
            s = _summary_metrics(os.path.join(
                shared, 'physical', a['physical_key'],
                'attempt_1', 'summary.json'))
            cells[(k_stage, net)] = (k_loc, s)
        except Exception:
            pass
    return {k: v[1] for k, v in cells.items()}, nets


def harvest_seq8():
    recs = []
    for order, batch, arm_dir, arm, sd in SEQ8_RUNS:
        run = f'{arm_dir}-{order}-{sd}'
        rd = os.path.join(OUT, batch, run)
        cells, _ = harvest_events(rd, os.path.join(OUT, batch,
                                                   '_shared_evaluation'))
        for (st, net), m in cells.items():
            recs.append(dict(corpus='seq8', order=order, arm=arm, seed=sd,
                             eval_stage=st, eval_scene=scene_tag(net),
                             ep=100, **m))
        print(f'{run}: cells={len(cells)}')
    return recs


def harvest_ha():
    recs = []
    broot = os.path.join(OUT, HA_BUCKET)
    shared = os.path.join(broot, '_shared_evaluation')
    for run in sorted(os.listdir(broot)):
        m = re.match(r'(.+)-O(\d)-SD(\d)$', run)
        rd = os.path.join(broot, run)
        if not m or not os.path.isdir(os.path.join(rd, 'attempts')):
            continue
        cells, nets = harvest_events(rd, shared)
        for (st, net), mtr in cells.items():
            recs.append(dict(corpus='ha', order=f'O{m.group(2)}',
                             arm=m.group(1), seed=f'SD{m.group(3)}',
                             eval_stage=st, eval_scene=scene_tag(net),
                             ep=100, **mtr))
        if not cells:
            print(f'  {run}: no cells')
    print(f'ha: {len(recs)} cells')
    return recs


def harvest_p34():
    recs = []
    for corpus, batch in P34_BUCKETS.items():
        broot = os.path.join(OUT, batch)
        shared = os.path.join(broot, '_shared_evaluation')
        for run in sorted(os.listdir(broot)):
            m = re.match(r'plan34_b\d+_O(\d)_seed(\d)_(.+)$', run)
            if not m or not os.path.isdir(os.path.join(broot, run)):
                continue
            cells, _ = harvest_events(os.path.join(broot, run), shared)
            for (st, net), mtr in cells.items():
                recs.append(dict(corpus=corpus, order=f'O{m.group(1)}',
                                 arm=m.group(3), seed=f'SD{m.group(2)}',
                                 eval_stage=st, eval_scene=scene_tag(net),
                                 ep=100, **mtr))
        print(f'{corpus}: {len(recs)} cells')
    return recs


def harvest_anchor():
    recs = []
    pat = os.path.join(
        OUT, 'tsc/sumo_dqn', 'sumohz1x1*',
        's8f_soddqnp_*_sd*_ev*_*', 'evaluation', 'summary.json')
    for sf in sorted(glob.glob(pat)):
        name = os.path.basename(os.path.dirname(os.path.dirname(sf)))
        r = name.split('_')
        order, sdl = r[2], r[3][2:]
        k = int(r[4][2:])
        tag = '_'.join(r[5:])
        ev = json.load(open(sf)).get('evaluations', [])
        if not ev:
            continue
        e = ev[-1]
        recs.append(dict(corpus='seq8', order=order, arm='ANCHOR',
                         seed=f'SD{sdl}', eval_stage=k,
                         eval_scene=('base' if tag == 'base' else tag),
                         ep=int(e['episode']), tt=e['travel_time'],
                         queue=e['queue'], thr=e['throughput'],
                         unfin=e['unfinished_vehicles'],
                         wait=e['waiting_time']))
    print(f'ANCHOR: cells={len(recs)}')
    return recs


def main():
    os.makedirs(CACHE, exist_ok=True)
    recs = (harvest_seq8() + harvest_ha() + harvest_p34()
            + harvest_anchor())
    with open(os.path.join(CACHE, 'evals.json'), 'w') as fh:
        json.dump(recs, fh, indent=1, ensure_ascii=False)
    print(f'wrote {len(recs)} eval cells -> {CACHE}/evals.json')


if __name__ == '__main__':
    main()
