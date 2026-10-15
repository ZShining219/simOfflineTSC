"""Harvest cl_hz1x1 anchor-arm eval matrices + degradation comparison.

Same cell convention as matrix_report_so.json: per chain a lower-triangular
map ``stage -> scene_tag -> metrics`` (mean over eval sumo seeds).  Train
runs are ``{prefix}_{arm}_{order}_sd{seed}_st{k}`` and eval cells
``..._ev{k}_{tag}_es{es}`` under ``data/output_data/tsc/sumo_{agent}/``.
Baselines are the shared cl1 single-task runs in matrix_report.json.

Additionally aggregates, per chain and per arm:
  * worst_old_degradation  max over cells (st, scene) with scene trained
                           before stage st of  ATT_cell - ATT_base(scene)
  * final_stage_worst      same max restricted to the last stage row
  * diagonal_delta         ATT_cell(st, just-trained scene) - ATT_base(scene)

Usage:  python dev/cl_hz1x1/harvest_soa.py <prefix> [out.json]
        prefix is the run-id prefix, e.g. soas (screen) or soa1 (main).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'

NETS = {'s1': 'sumohz1x1', 'c4': 'sumohz1x1_config4',
        'c2': 'sumohz1x1_config2', 'c3': 'sumohz1x1_config3'}
NET_TAG = {v: k for k, v in NETS.items()}
ORDERS = {
    'o2': [NETS['s1'], NETS['c4'], NETS['c2'], NETS['c3']],
    'o3': [NETS['c2'], NETS['c3'], NETS['s1'], NETS['c4']],
}
EVAL_SEEDS = [101, 102, 103]
SEEDS = [0, 1, 2]
METRICS = ['travel_time', 'throughput', 'unfinished_vehicles', 'queue']
# anchor arm -> (backbone for baseline lookup, agent dir suffix)
BACKBONE = {'soddqn': ('ddqn', 'dqn'), 'soduel': ('duel', 'dqn'),
            'sofrap': ('frap', 'frap'), 'sodqn': ('dqn', 'dqn')}


def arm_backbone(arm):
    for stem, pair in BACKBONE.items():
        if arm.startswith(stem):
            return pair
    raise KeyError(f'unknown arm stem: {arm}')


def load_final_eval(run_dir):
    path = Path(run_dir) / 'evaluation' / 'summary.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    evals = [e for e in data.get('evaluations', [])
             if e.get('record_type') in ('FINAL_EVALUATION', 'EVALUATION')]
    if not evals:
        return None
    return {m: evals[-1].get(m) for m in METRICS}


def cell_metrics(agent_dir, prefix, arm, order, seed, stage, eval_net):
    vals = defaultdict(list)
    for es in EVAL_SEEDS:
        rid = (f'{prefix}_{arm}_{order}_sd{seed}_ev{stage}_'
               f'{NET_TAG[eval_net]}_es{es}')
        got = load_final_eval(OUT / f'sumo_{agent_dir}' / eval_net / rid)
        if got is None:
            return None
        for m, v in got.items():
            if v is not None:
                vals[m].append(v)
    return {m: sum(v) / len(v) for m, v in vals.items()}


def load_baselines():
    base = json.loads((ART / 'matrix_report.json').read_text())['baselines']
    return base  # key: '{backbone}_{tag}_sd{seed}' -> metrics dict


def main():
    prefix = sys.argv[1] if len(sys.argv) > 1 else 'soas'
    dest = ART / (sys.argv[2] if len(sys.argv) > 2
                  else f'matrix_report_soa_{prefix}.json')
    base = load_baselines()
    report = {}
    for arm_stem, (backbone, agent_dir) in BACKBONE.items():
        # discover arms present on disk for this stem
        arm_dirs = set()
        for net in NETS.values():
            for p in (OUT / f'sumo_{agent_dir}' / net).glob(
                    f'{prefix}_{arm_stem}*'):
                name = p.name.split('_o')[0]
                arm_dirs.add(name[len(prefix) + 1:])
        for arm in sorted(arm_dirs):
            for order, nets in ORDERS.items():
                for seed in SEEDS:
                    key = f'{arm}_{order}_sd{seed}'
                    chain = {}
                    complete = True
                    worst, final_worst = None, None
                    diag_deltas = []
                    for st in range(1, 5):
                        row = {}
                        for j in range(st):
                            cm = cell_metrics(agent_dir, prefix, arm, order,
                                              seed, st, nets[j])
                            if cm is None:
                                complete = False
                                continue
                            row[NET_TAG[nets[j]]] = cm
                            b = base.get(
                                f'{backbone}_{NET_TAG[nets[j]]}_sd{seed}')
                            if b and b.get('travel_time') is not None:
                                delta = cm['travel_time'] - b['travel_time']
                                if j == st - 1:
                                    diag_deltas.append(delta)
                                else:
                                    worst = delta if worst is None else \
                                        max(worst, delta)
                                    if st == 4:
                                        final_worst = delta \
                                            if final_worst is None else \
                                            max(final_worst, delta)
                        chain[str(st)] = row
                    chain['complete'] = complete
                    chain['worst_old_degradation'] = worst
                    chain['final_stage_worst'] = final_worst
                    chain['diagonal_deltas'] = diag_deltas
                    report[key] = chain
    dest.write_text(json.dumps(report, indent=1), encoding='utf-8')

    # console summary per arm
    arms = sorted({k.rsplit('_o', 1)[0] for k in report})
    print(f'{"arm":24s} {"n":>3s} {"worst_mean":>11s} {"worst_max":>10s} '
          f'{"diag_mean":>10s} {"diag_max":>9s}')
    for arm in arms:
        rows = [v for k, v in report.items()
                if k.rsplit('_o', 1)[0] == arm and v['complete']]
        w = [r['worst_old_degradation'] for r in rows
             if r['worst_old_degradation'] is not None]
        dg = [d for r in rows for d in r['diagonal_deltas']]
        if not w:
            continue
        print(f'{arm:24s} {len(rows):3d} {sum(w)/len(w):11.2f} '
              f'{max(w):10.2f} {sum(dg)/len(dg):10.2f} {max(dg):9.2f}')
    print(f'\nreport -> {dest}')


if __name__ == '__main__':
    main()
