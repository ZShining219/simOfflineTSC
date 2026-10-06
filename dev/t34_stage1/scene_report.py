#!/usr/bin/env python3
"""Stage-1 scene-classification report card from run_tarl_struct_cf.py dumps.

Reads dump cells produced by tools/run_tarl_struct_cf.py --dump_dir
(ep*.npz + event_plan_log.jsonl + probe/tarl_struct_cf_records.jsonl) and
fits linear probes answering the bench/att_hz4x4 stage-1 gate questions:

  * scene-level event-kind presence from fused representation (h_it / z_post),
    per kind P/R/F1 — including HELD-OUT plans (t3/t4) the arm never trained on;
  * non-text bound: same decoding under TARL_TEXT_CONDITION=empty dumps;
  * cross-plan transfer: probe trained on the seen plan, tested on held-out;
  * node-level present/type decoding (canonical arms only; z_task is the
    grounded per-node truth — under empty the input slot is zeroed and
    node-level labels are unavailable by construction);
  * counterfactual divergence rates per variant (records.jsonl).

Splits are by episode index (npz file), never sharing an episode across
train/test — stricter than event_id splits because same-episode decisions
share traffic dynamics.

Usage:
  python3 dev/t34_stage1/scene_report.py --cells name=DIR [name=DIR ...] \
      --out <dir>   # writes scene_report.csv + REPORT_CARD.md
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

KIND_ID = {1: 'lane_blockage', 2: 'road_closure', 3: 'global_rain'}
KINDS = ['lane_blockage', 'road_closure', 'global_rain']


def load_cell(cell_dir):
    """Return list of episode dicts: {idx, role, events, feats(npz arrays)}."""
    cell_dir = Path(cell_dir)
    log_path = cell_dir / 'event_plan_log.jsonl'
    logs = [json.loads(l) for l in log_path.open()] if log_path.exists() else []
    npzs = sorted(cell_dir.glob('ep*.npz'))
    if len(npzs) != len(logs):
        raise RuntimeError(f'{cell_dir}: {len(npzs)} npz != {len(logs)} log lines')
    eps = []
    for i, (npz, log) in enumerate(zip(npzs, logs)):
        arr = np.load(npz)
        eps.append({'idx': i, 'role': log.get('role'),
                    'plan_episode': log.get('episode'),
                    'events': log.get('events') or [],
                    'arr': arr})
    return eps


def active_kinds(events, t):
    """Kinds active at sim time t (privileged labels from plan log)."""
    return sorted({e['kind'] for e in events
                   if e['begin'] <= t < e['end']})


def decision_frame(eps, feat_fn):
    """Build per-decision (X, kind-set, episode group) arrays for a cell."""
    Xs, ys, gs = [], [], []
    for ep in eps:
        arr = ep['arr']
        ts = arr['t']
        feats = feat_fn(arr)                       # (T, D)
        for j, t in enumerate(ts):
            Xs.append(feats[j])
            ys.append(tuple(active_kinds(ep['events'], float(t))))
            gs.append(ep['idx'])
    return np.stack(Xs), np.array(ys, dtype=object), np.asarray(gs)


def pooled(arr, key):
    """(T,N,D) -> concat(mean, max) pooled (T, 2D)."""
    x = arr[key]
    return np.concatenate([x.mean(1), x.max(1)], axis=-1).astype(np.float32)


def group_split(y, g, test_frac=0.4, seed=0):
    rng = np.random.RandomState(seed)
    uniq = np.unique(g)
    rng.shuffle(uniq)
    n_test = max(1, int(round(len(uniq) * test_frac)))
    test_g = set(uniq[:n_test])
    te = np.array([gi in test_g for gi in g])
    return ~te, te


def prf(y_true, y_pred):
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    p = tp / (tp + fp + 1e-9)
    r = tp / (tp + fn + 1e-9)
    return p, r, 2 * p * r / (p + r + 1e-9)


def fit_logreg(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0, class_weight='balanced')
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict(sc.transform(Xte))


def binary_task(X, y_bin, g, seed=0):
    """One-vs-rest binary scene-presence probe; episode-group split."""
    keep = np.ones(len(y_bin), dtype=bool)
    Xk, yk, gk = X[keep], y_bin[keep], g[keep]
    if yk.sum() < 8 or (1 - yk).sum() < 8:
        return None
    tr, te = group_split(yk, gk, seed=seed)
    if len(np.unique(yk[tr])) < 2 or len(np.unique(yk[te])) < 2:
        return None
    pred = fit_logreg(Xk[tr], yk[tr], Xk[te])
    p, r, f1 = prf(yk[te], pred)
    maj = max(float(yk[te].mean()), 1 - float(yk[te].mean()))
    return {'p': p, 'r': r, 'f1': f1, 'majority': maj,
            'n_train': int(tr.sum()), 'n_test': int(te.sum()),
            'pos_test': int(yk[te].sum())}


def multiclass_task(X, y, g, seed=0):
    tr, te = group_split(y, g, seed=seed)
    if len(te) < 10 or len(np.unique(y[tr])) < 2:
        return None
    pred = fit_logreg(X[tr], y[tr], X[te])
    per, f1s = [], []
    for c in sorted(set(y[te])):
        m = y[te] == c
        if m.sum() == 0:
            continue
        per.append(float(np.mean(pred[m] == y[te][m])))
        p, r, f1 = prf((y[te] == c).astype(int), (pred == c).astype(int))
        f1s.append(f1)
    maj = max(Counter(y[te].tolist()).values()) / len(te)
    return {'bal_acc': float(np.mean(per)), 'f1_macro': float(np.mean(f1s)),
            'majority': float(maj), 'n_test': int(len(te))}


def node_frame(eps):
    """Per-node frames: h_it node feats + z_task-derived labels (canonical)."""
    Xs, pres, typ, gs = [], [], [], []
    for ep in eps:
        arr = ep['arr']
        zt = arr['z_task']                 # (T,N,6) — grounded input truth
        h = arr['h_it']                    # (T,N,128)
        T, N, _ = h.shape
        Xs.append(h.reshape(T * N, -1))
        pres.append((zt[..., 0] > 0.5).astype(int).reshape(-1))
        typ.append(zt[..., 1].astype(int).reshape(-1))
        gs.append(np.full(T * N, ep['idx']))
    return (np.concatenate(Xs), np.concatenate(pres),
            np.concatenate(typ), np.concatenate(gs))


def counterfactual_summary(cell_dir):
    rec = Path(cell_dir) / 'probe' / 'tarl_struct_cf_records.jsonl'
    if not rec.exists():
        return {}
    rates = defaultdict(list)
    n = 0
    for line in rec.open():
        r = json.loads(line)
        n += 1
        base = np.asarray(r['variants']['canonical']['action'])
        for name, v in r['variants'].items():
            if name == 'canonical':
                continue
            rates[name].append(float(np.mean(
                np.asarray(v['action']) != base)))
    return {k: float(np.mean(v)) for k, v in rates.items()} | {'n_steps': n}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', nargs='+', required=True,
                    help='name=dir entries (dump cells)')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    feats = {'h_it': lambda a: pooled(a, 'h_it'),
             'z_post': lambda a: pooled(a, 'z_post'),
             'ob': lambda a: pooled(a, 'ob_traffic')}
    cells = {}
    frames = {}
    for spec in args.cells:
        name, d = spec.split('=', 1)
        eps = load_cell(d)
        cells[name] = {'dir': d, 'eps': eps}
        frames[name] = {fk: decision_frame(eps, fn)
                        for fk, fn in feats.items()}
        cells[name]['cf'] = counterfactual_summary(d)

    rows = []
    # ---- within-plan scene-level per-kind P/R/F1 (h_it primary) ----
    for name, fmap in frames.items():
        for fk, (X, y, g) in fmap.items():
            for kind in KINDS:
                yb = np.array([kind in yy for yy in y], dtype=int)
                m = binary_task(X, yb, g)
                if m:
                    rows.append({'cell': name, 'feature': fk,
                                 'task': f'scene_{kind}', 'split': 'in-plan',
                                 **m})
            # single-active multiclass (incl 'none')
            ys = np.array([y0[0] if len(y0) == 1 else
                           ('none' if not y0 else 'multi') for y0 in y],
                          dtype=object)
            keep = ys != 'multi'
            m = multiclass_task(X[keep], ys[keep], g[keep])
            if m:
                rows.append({'cell': name, 'feature': fk,
                             'task': 'event_kind_multiclass',
                             'split': 'in-plan', **m})

    # ---- cross-plan transfer: train on `seen`, test on held-out cells ----
    if 'seen' in frames:
        Xs, ys, gs = frames['seen']['h_it']
        for tgt in ('t3', 't4', 't3_empty'):
            if tgt not in frames:
                continue
            Xt, yt, gt = frames[tgt]['h_it']
            for kind in KINDS:
                yb_s = np.array([kind in y for y in ys], dtype=int)
                yb_t = np.array([kind in y for y in yt], dtype=int)
                if yb_s.sum() < 20 or yb_t.sum() < 8:
                    continue
                pred = fit_logreg(Xs, yb_s, Xt)
                p, r, f1 = prf(yb_t, pred)
                rows.append({'cell': f'seen→{tgt}', 'feature': 'h_it',
                             'task': f'scene_{kind}', 'split': 'cross-plan',
                             'p': p, 'r': r, 'f1': f1,
                             'majority': max(float(yb_t.mean()),
                                             1 - float(yb_t.mean())),
                             'n_train': int(len(yb_s)),
                             'n_test': int(len(yb_t)),
                             'pos_test': int(yb_t.sum())})

    # ---- node-level decoding (canonical arms only) ----
    for name, cell in cells.items():
        if 'empty' in name:
            continue
        X, pres, typ, g = node_frame(cell['eps'])
        m = binary_task(X, pres, g)
        if m:
            rows.append({'cell': name, 'feature': 'h_it_node',
                         'task': 'node_present', 'split': 'in-plan', **m})
        m = multiclass_task(X[pres == 1], typ[pres == 1], g[pres == 1])
        if m:
            rows.append({'cell': name, 'feature': 'h_it_node',
                         'task': 'node_type', 'split': 'in-plan', **m})

    import csv
    keys = sorted({k for r in rows for k in r})
    with (out / 'scene_report.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)

    # ---- report card (markdown) ----
    lines = ['# t34 stage-1 场景分类报告卡（held-out 补强）', '',
             '口径：LogReg(class_weight=balanced)，episode 级 group 切分；',
             '特征=h_it/z_post 节点维 mean+max 池化；empty 臂 z_task 输入全零', '']
    for name, cell in cells.items():
        cf = cell['cf']
        lines.append(f"## cell `{name}`  ({cell['dir']})")
        lines.append('')
        lines.append('| task | feature | split | P | R | F1 | majority | n_test |')
        lines.append('|---|---|---|---|---|---|---|---|')
        for r in rows:
            if r['cell'] != name and not r['cell'].endswith(name):
                continue
            lines.append('| {task} | {feature} | {split} | {p:.3f} | {r:.3f} '
                         '| {f1:.3f} | {majority:.3f} | {n_test} |'
                         .format(**{k: r.get(k, '') for k in
                                    ('task', 'feature', 'split', 'p', 'r',
                                     'f1', 'majority', 'n_test')}))
        if cf:
            lines.append('')
            lines.append('反事实分歧率（动作改变节点比例均值，n_steps=%d）：' %
                         cf.get('n_steps', 0))
            for k, v in cf.items():
                if k != 'n_steps':
                    lines.append(f'- {k}: {v:.4f}')
            lines.append('')
    (out / 'REPORT_CARD.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {out}/scene_report.csv ({len(rows)} rows) + REPORT_CARD.md')


if __name__ == '__main__':
    main()
