"""Fit linear probes on TARL representation dumps (Test 1 diagnostic).

Input: one or more probe dirs produced by tools/run_tarl_repr_probe.py
(repr_records.npz + repr_labels.jsonl). For each representation level
(z_text / z_fuse / h_policy) and each label task, trains a multinomial
logistic regression and reports balanced accuracy / F1 vs majority baseline.

Tasks (labels come from the privileged schedule -- analysis only):
  event_kind       per-decision kind of THE active event, restricted to
                   decisions with exactly one active event (+ 'none')
  event_junction   per-decision target junction of the active event
                   (16 junctions + 'global' + 'none'), n_active==1 subset
  node_is_target   per-NODE binary: is this junction the event target?
                   uses target_mask, all decisions
  movement         per-decision movement class for single lane_blockage
  speed_factor     per-decision rain severity bucket for single global_rain

Split: group-aware by event_id (events never shared across train/test),
falling back to contiguous time blocks. Leakage note: z_fuse/h_policy may
decode location from traffic dynamics, not text -- run the probe under
TARL_TEXT_CONDITION=empty to bound that channel.

Usage:
  python tools/analyze_repr_probe.py --dirs dir1 dir2 [--names a b] --out out.csv
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def load_dir(d):
    d = Path(d)
    arr = np.load(d / 'repr_records.npz')
    labels = [json.loads(l) for l in (d / 'repr_labels.jsonl').open()]
    n = arr['z_text'].shape[0]
    assert n == len(labels), (n, len(labels))
    return {k: arr[k] for k in arr.files}, labels


def decision_labels(labels):
    """Per-decision scalar targets; returns dict task -> (y, group, keep_mask)."""
    out = {}
    kind, junc, move, sf, grp = [], [], [], [], []
    for i, l in enumerate(labels):
        evs = l['events']
        single = evs if len(evs) == 1 else []
        eid = evs[0]['event_id'] if single else 'none'
        grp.append(eid if single else f'multi_{i}')  # multi rows excluded anyway
        kind.append(single[0]['kind'] if single else ('none' if not evs else None))
        junc.append(single[0]['junction'] if single else ('none' if not evs else None))
        mv = single[0].get('movements') if single else None
        move.append('+'.join(sorted(mv)) if mv else ('none' if not evs else None))
        s = single[0].get('speed_factor') if single else None
        sf.append(round(s, 2) if s is not None else ('none' if not evs else None))
    out['event_kind'] = (np.array(kind, dtype=object), np.array(grp))
    out['event_junction'] = (np.array(junc, dtype=object), np.array(grp))
    out['movement'] = (np.array(move, dtype=object), np.array(grp))
    out['speed_factor'] = (np.array(sf, dtype=object), np.array(grp))
    return out


def group_split(y, grp, test_frac=0.4, seed=0):
    """Split indices so no group (event_id / 'none'-block) spans both sides."""
    rng = np.random.RandomState(seed)
    n = len(y)
    groups = np.array([g if g != 'none' else f'none_{i // 40}' for i, g in enumerate(grp)])
    uniq = np.unique(groups)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_groups = set(uniq[:n_test])
    test_idx = np.array([i for i in range(n) if groups[i] in test_groups])
    train_idx = np.array([i for i in range(n) if groups[i] not in test_groups])
    return train_idx, test_idx


def fit_probe(X, y, grp, min_support=8):
    """Logistic probe. X: (n, d). Returns dict of metrics or None."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    keep = np.array([v is not None for v in y])
    X, y, grp = X[keep], np.asarray([str(v) for v in y[keep]]), grp[keep]
    if len(np.unique(y)) < 2:
        return None
    tr, te = group_split(y, grp)
    if len(te) < 10 or len(tr) < 40:
        return None
    counts = Counter(y[te])
    maj = max(counts.values()) / len(te)
    classes = sorted(set(y))
    try:
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1500, C=1.0, class_weight='balanced')
        clf.fit(sc.transform(X[tr]), y[tr])
        pred = clf.predict(sc.transform(X[te]))
    except Exception as exc:  # ill-conditioned; report as failure
        return {'error': str(exc)}
    acc = float(np.mean(pred == y[te]))
    # balanced acc + macro-F1
    per_c, f1s = [], []
    for c in classes:
        m = y[te] == c
        if m.sum() == 0:
            continue
        per_c.append(float(np.mean(pred[m] == c)))
        tp = float(np.sum((pred == c) & (y[te] == c)))
        fp = float(np.sum((pred == c) & (y[te] != c)))
        fn = float(np.sum((pred != c) & (y[te] == c)))
        f1s.append(2 * tp / (2 * tp + fp + fn + 1e-9))
    return {'bal_acc': float(np.mean(per_c)), 'acc': acc,
            'f1_macro': float(np.mean(f1s)), 'majority': float(maj),
            'n_train': int(len(tr)), 'n_test': int(len(te)),
            'n_classes': len(classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dirs', nargs='+', required=True)
    ap.add_argument('--names', nargs='+', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    names = args.names or [Path(d).name for d in args.dirs]
    rows = []
    LEVELS = ['z_text', 'z_fuse', 'h_policy']
    for name, d in zip(names, args.dirs):
        arrs, labels = load_dir(d)
        tasks = decision_labels(labels)
        node_target = np.array([l['target_mask'] for l in labels])  # (T,N)
        for lvl in LEVELS:
            X = arrs[lvl]  # (T,N,D)
            # per-decision tasks: mean-pool over nodes AND keep target-node rows
            Xd = X.mean(axis=1)
            for task, (y, grp) in tasks.items():
                if task == 'movement':
                    pass
                r = fit_probe(Xd, y, grp)
                if r:
                    r.update(dir=name, level=lvl, task=task, view='meanpool')
                    rows.append(r)
            # node_is_target: per-node rows, event decisions only
            yb = node_target.reshape(-1)
            Xn = X.reshape(-1, X.shape[-1])
            grp_n = np.repeat([f'{i // 40}' for i in range(len(labels))], X.shape[1])
            # balance: subsample negatives to 3x positives
            pos = np.where(yb == 1)[0]
            neg = np.where(yb == 0)[0]
            rng = np.random.RandomState(0)
            if len(pos) >= 20:
                neg = rng.choice(neg, size=min(len(neg), 3 * len(pos)), replace=False)
                sel = np.sort(np.concatenate([pos, neg]))
                r = fit_probe(Xn[sel], yb[sel].astype(str), grp_n[sel])
                if r:
                    r.update(dir=name, level=lvl, task='node_is_target', view='pernode')
                    rows.append(r)
    import csv
    keys = ['dir', 'level', 'task', 'view', 'bal_acc', 'acc', 'f1_macro',
            'majority', 'n_train', 'n_test', 'n_classes']
    lines = []
    for r in rows:
        lines.append(' '.join(f'{k}={r.get(k, ""):.4f}' if isinstance(r.get(k), float)
                              else f'{k}={r.get(k, "")}' for k in keys))
    text = '\n'.join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + '\n')


if __name__ == '__main__':
    main()
