#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATT-ENTITY-004 probe3 supplement: SUPERVISED low-rank bottleneck on h_it.

probe3's PCA curve is unsupervised (variance-preserving).  This probe asks the
task-relevant question: does a k-dim bottleneck TRAINED FOR THE TASK carry
more congestion info than the best k-dim PCA subspace?  Model:
h_it -> Linear(k) -> ReLU -> softmax head (sklearn MLPClassifier with one
hidden layer of size k == bottleneck).  Same 4-class congestion labels and
episode-level train/test split as probe1.  Output: probes/probe3_sup.json +
probe3_sup.md table (pca vs sup per dim, per cell).

Usage: python3 tools/att004_probe3_sup.py [--cell a3s_s7_ep100_ss7]
"""
import json, argparse, glob, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from att004_probes import node_labels, LIB

DIMS = [4, 8, 16, 32, 64]

def sup_curve(cell_dir):
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score
    Xs, Ys, E = [], [], []
    for f in sorted(glob.glob(os.path.join(cell_dir, 'ep*.npz'))):
        d = np.load(f); i = int(os.path.basename(f)[2:5])
        Xs.append(d['h_it']); Ys.append(node_labels(d['queue_per_node']))
        E += [i] * len(d['h_it'])
    E = np.array(E)
    out = {}
    for k in DIMS:
        tr = E % 4 != 3; te = ~tr
        m = MLPClassifier(hidden_layer_sizes=(k,), max_iter=300, random_state=0)
        m.fit(np.concatenate(Xs)[tr].reshape(-1, 128), np.concatenate(Ys)[tr].ravel())
        pr = m.predict(np.concatenate(Xs)[te].reshape(-1, 128))
        out[k] = round(float(f1_score(np.concatenate(Ys)[te].ravel(), pr,
                                      average='macro')), 4)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell')
    ap.add_argument('--out-dir', default='data/output_data/analysis/att_entity_004/probes')
    a = ap.parse_args()
    rep = json.load(open(os.path.join(a.out_dir, 'probe_report.json')))
    cells = [a.cell] if a.cell else sorted(c for c in rep['cells'] if c.startswith('a3s'))
    res = {}
    for c in cells:
        res[c] = {'sup': sup_curve(os.path.join(LIB, c)),
                  'pca': {int(k): v.get('p1_macroF1') for k, v in
                          (rep['cells'][c].get('probe3_pca') or {}).items()}}
        print('done', c, flush=True)
    fn = f'probe3_sup_{cells[0]}.json' if a.cell else 'probe3_sup.json'
    with open(os.path.join(a.out_dir, fn), 'w') as f:
        json.dump({'dims': DIMS, 'note': 'sup = 128->k->4 MLP bottleneck, '
                  'same labels/split as probe1', 'cells': res}, f, indent=1)
    print('wrote', fn)

if __name__ == '__main__':
    main()
