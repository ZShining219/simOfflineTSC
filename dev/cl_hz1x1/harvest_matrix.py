"""Harvest the cl_hz1x1 continual-learning eval matrix from finished runs.

Reads each eval cell's ``evaluation/summary.json`` (written by the eval-only
``episodes: 0`` runs) plus the single-task baseline runs' final evaluation,
and prints/writes per-condition 4x4 lower-triangular matrices of travel_time /
throughput / unfinished_vehicles (mean over eval sumo seeds 101-103).

Usage:
    python dev/cl_hz1x1/harvest_matrix.py
"""
import json
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
ARMS = {'ddqn': 'dqn', 'duel': 'dqn', 'frap': 'frap', 'ppov2': 'ppo_pfrl'}
CONDS = {'clear': ['ddqn', 'duel', 'frap', 'ppov2'],
         'fifo': ['ddqn', 'duel', 'frap']}
SEEDS = [0, 1, 2]
EVAL_SEEDS = [101, 102, 103]
METRICS = ['travel_time', 'throughput', 'unfinished_vehicles', 'queue']


def load_final_eval(run_dir):
    path = Path(run_dir) / 'evaluation' / 'summary.json'
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    evals = [e for e in data.get('evaluations', [])
             if e.get('record_type') in ('FINAL_EVALUATION', 'EVALUATION')]
    if not evals:
        return None
    last = evals[-1]
    return {m: last.get(m) for m in METRICS}


def cell_metrics(arm, cond, order, seed, stage, eval_net):
    """Mean over eval seeds for one matrix cell; None if incomplete."""
    vals = defaultdict(list)
    for es in EVAL_SEEDS:
        rid = (f'cl1_{arm}_{cond}_{order}_sd{seed}_ev{stage}_'
               f'{NET_TAG[eval_net]}_es{es}')
        got = load_final_eval(OUT / f'sumo_{ARMS[arm]}' / eval_net / rid)
        if got is None:
            return None
        for m, v in got.items():
            if v is not None:
                vals[m].append(v)
    return {m: sum(v) / len(v) for m, v in vals.items()}


def baseline_final(arm, net, seed):
    rid = f'cl1_{arm}_base_{NET_TAG[net]}_sd{seed}'
    return load_final_eval(OUT / f'sumo_{ARMS[arm]}' / net / rid)


def main():
    report = {'conditions': {}, 'baselines': {}}
    for cond, arms in CONDS.items():
        for arm in arms:
            key = f'{arm}_{cond}'
            report['conditions'][key] = {}
            for order, nets in ORDERS.items():
                for seed in SEEDS:
                    matrix = {m: [[None] * 4 for _ in range(4)]
                              for m in METRICS}
                    complete = True
                    for st in range(4):
                        for j in range(st + 1):
                            cm = cell_metrics(arm, cond, order, seed,
                                              st + 1, nets[j])
                            if cm is None:
                                complete = False
                                continue
                            for m in METRICS:
                                matrix[m][st][j] = cm[m]
                    report['conditions'][key][f'{order}_sd{seed}'] = {
                        'complete': complete, 'matrix': matrix}
    for arm in ARMS:
        for tag, net in NETS.items():
            for seed in SEEDS:
                got = baseline_final(arm, net, seed)
                report['baselines'][f'{arm}_{tag}_sd{seed}'] = got

    dest = ART / 'matrix_report.json'
    dest.write_text(json.dumps(report, indent=1), encoding='utf-8')

    # compact travel_time console view
    for key, per in report['conditions'].items():
        print(f'\n== {key} (travel_time, rows=stage k, cols=net index) ==')
        for oseed, blk in per.items():
            flag = '' if blk['complete'] else '  [incomplete]'
            print(f'  {oseed}:{flag}')
            nets = ORDERS[oseed[:2]]
            hdr = ' '.join(f'{NET_TAG[n]:>8}' for n in nets)
            print(f'    st\\net {hdr}')
            for st in range(4):
                row = blk['matrix']['travel_time'][st]
                cells = ' '.join(
                    f'{v:8.1f}' if v is not None else '       -'
                    for v in row[:st + 1])
                print(f'    st{st + 1}   {cells}')
    print(f'\nreport -> {dest}')


if __name__ == '__main__':
    main()
