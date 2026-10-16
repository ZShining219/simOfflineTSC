#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ATT-ENTITY-004 表征探针套件 —— state_lib 上的三件套.

  ① 拥堵等级分类:   4 档标签由观测排队阈值生成 (与事件字段无关),
     对照 原始观测 vs a3s 融合表征 (h_it / z_post) vs 基线 ob.
  ② 影响预测:       fixedtime 配对仿真 (有/无事件, 同 sumo_seed+episode)
     产标签 = 事件窗内每节点排队差; probe 从 [节点表征 at 事件起始, 池化
     z_task] 预测影响量. 标签不经由编码器输入字段推导 (来自独立 rollout).
  ③ 压缩质量:       PCA 逐档降维 -> 复测 ①②, 维度↔保真曲线.

Usage: python3 tools/att004_probes.py [--cell CELL] [--out-dir DIR]
"""
import json, argparse, glob, os
import numpy as np
import yaml

LIB = 'data/output_data/analysis/att_entity_004/state_lib'
NOEV = 'data/output_data/analysis/att_entity_004/state_lib_noev'
PLAN = 'configs/events/plans/att004_state_lib_v1.yml'
Q_BINS = [0, 1, 3, 6, 1e9]   # 4 档拥堵阈值 (veh/节点): 0 / 1-2 / 3-5 / >=6
PCA_DIMS = [4, 8, 16, 32, 64]
H_ONSET_OFFSET = 10          # onset step = begin/10

def plan_events():
    """npz 索引 -> [(begin,end)]. npz0/npz11=eval(episodes 前后各一),
    npz j∈[1,10] 对应 plan.episodes[j-1]."""
    p = yaml.safe_load(open(PLAN))
    eval_evs = [(e['begin'], e['end']) for e in p['eval'].get('events', [])]
    out = {0: eval_evs, 11: eval_evs}
    for i, ep in enumerate(p['episodes']):
        out[i + 1] = [(e['begin'], e['end']) for e in ep.get('events', [])]
    return out

# ---------- ① congestion classification ----------

def node_labels(q):  # queue_per_node [T,N] -> class ids
    return np.digitize(q, Q_BINS[1:-1]).astype(int)  # 0..3

def norm_per_node(arr):  # baseline npz: per-agent rows -> [T,16]
    a = np.asarray(arr)
    if a.ndim == 3 and a.shape[0] == 1:   return a[0]          # (1,T,16)
    if a.ndim == 3 and a.shape[2] == 1:   return a[:, :, 0].T  # (16,T,1)
    if a.ndim == 2:                        return a             # already (T,16)
    raise ValueError(f'unrecognized shape {a.shape}')

def baseline_feats(d):
    """返回 {'q': [T,16], 'ob': [T,16,C], 'action': [T,16]}."""
    q = norm_per_node(d['queue_per_agent'])
    T, N = q.shape
    ob = np.asarray(d['ob'])
    if ob.ndim == 3 and ob.shape[0] == 1:
        ob = ob[0].reshape(T, N, -1)
    elif ob.ndim == 3:                      # fixedtime (16,T,12)
        ob = ob.transpose(1, 0, 2)
    elif ob.ndim == 2:
        ob = ob.reshape(T, N, -1)
    act = norm_per_node(d['action'])
    return {'q': q, 'ob': ob, 'action': act}

def clf_eval(Xtr, ytr, Xte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
    res = {}
    for name, mk in [('lin', lambda: LogisticRegression(max_iter=400)),
                     ('mlp', lambda: MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=0))]:
        m = mk(); m.fit(Xtr, ytr); pr = m.predict(Xte)
        cm = confusion_matrix(yte, pr, labels=[0,1,2,3])
        res[name] = {'acc': round(float(accuracy_score(yte, pr)), 4),
                     'macroF1': round(float(f1_score(yte, pr, average='macro', zero_division=0)), 4),
                     'per_class_recall': [round(float(cm[c, c] / max(cm[c].sum(), 1)), 3) for c in range(4)]}
    return res

def probe1(cell_dir):
    feats, ys, epids = [], [], []
    for f in sorted(glob.glob(os.path.join(cell_dir, 'ep*.npz'))):
        d = np.load(f)
        if 'h_it' in d:
            qn = d['queue_per_node']
            feats.append({'obs': d['ob_traffic'], 'h_it': d['h_it'], 'z_post': d['z_post']})
        else:
            b = baseline_feats(d); qn = b['q']
            feats.append({'ob': b['ob']})
        ys.append(node_labels(qn))
        epids.append(int(os.path.basename(f)[2:5]))
    epids = np.array(epids); te = epids % 4 == 3; tr = ~te
    out = {}
    for k in feats[0]:
        Xtr = np.concatenate([feats[i][k] for i in range(len(feats)) if tr[i]]).reshape(-1, feats[0][k].shape[-1])
        Xte = np.concatenate([feats[i][k] for i in range(len(feats)) if te[i]]).reshape(-1, feats[0][k].shape[-1])
        ytr = np.concatenate([ys[i] for i in range(len(ys)) if tr[i]]).ravel()
        yte = np.concatenate([ys[i] for i in range(len(ys)) if te[i]]).ravel()
        out[k] = clf_eval(Xtr, ytr, Xte, yte)
    # 类别分布
    out['label_dist'] = np.bincount(np.concatenate(ys).ravel(), minlength=4).tolist()
    return out

# ---------- ② impact prediction ----------

def paired_delta(ss, n_ev_eps):
    """fixedtime 有/无事件逐节点排队差 -> {(ep_idx): delta[T,N]}."""
    ev_dir = f'{LIB}/fixedtime_ss{ss}'; nv_dir = f'{NOEV}/fixedtime_ss{ss}'
    deltas = {}
    for f in sorted(glob.glob(os.path.join(ev_dir, 'ep*.npz'))):
        i = int(os.path.basename(f)[2:5])
        nf = os.path.join(nv_dir, os.path.basename(f))
        if not os.path.exists(nf): continue
        d = np.load(f); n = np.load(nf)
        deltas[i] = norm_per_node(d['queue_per_agent']) - norm_per_node(n['queue_per_agent'])
    return deltas

def probe2(cell_dir, ev_map, deltas):
    rows, labs = [], []
    for f in sorted(glob.glob(os.path.join(cell_dir, 'ep*.npz'))):
        i = int(os.path.basename(f)[2:5])
        if i not in deltas or not ev_map.get(i): continue
        begin, end = min(b for b, _ in ev_map[i]), max(e for _, e in ev_map[i])
        d = np.load(f)
        t0 = int(begin // 10)
        if t0 >= len(d['t']): continue
        delta = deltas[i]  # [T,N]
        w = (d['t'] >= begin) & (d['t'] < end)
        if w.sum() == 0: continue
        lab = delta[w].mean(axis=0)                              # [N] 窗内平均排队差
        if 'h_it' in d:
            zt = d['z_task'][t0]                                 # [N,6]
            zp = zt[zt[:, 0] > 0.5].mean(0) if (zt[:, 0] > .5).any() else zt.mean(0)
            base = d['ob_traffic'][t0]
            feats = {'obs+zt': np.concatenate([base, np.tile(zp, (16, 1))], -1),
                     'h_it+zt': np.concatenate([d['h_it'][t0], np.tile(zp, (16, 1))], -1),
                     'z_post+zt': np.concatenate([d['z_post'][t0], np.tile(zp, (16, 1))], -1)}
        else:
            feats = {'ob': baseline_feats(d)['ob'][t0]}
        rows.append((i, feats, lab))
    if not rows: return None
    out = {}
    for k in rows[0][1]:
        X = np.concatenate([r[1][k] for r in rows]); Y = np.concatenate([r[2] for r in rows])
        eps = np.array([r[0] for r in rows]).repeat(16)
        te = eps % 4 == 3; tr = ~te
        if tr.sum() < 30 or te.sum() < 10: continue
        from sklearn.linear_model import Ridge
        from sklearn.neural_network import MLPRegressor
        from sklearn.metrics import r2_score, mean_absolute_error
        from scipy.stats import spearmanr
        res = {}
        for name, mk in [('ridge', lambda: Ridge(1.0)),
                         ('mlp', lambda: MLPRegressor(hidden_layer_sizes=(64,), max_iter=400, random_state=0))]:
            m = mk(); m.fit(X[tr], Y[tr]); pr = m.predict(X[te])
            res[name] = {'r2': round(float(r2_score(Y[te], pr)), 4),
                         'mae': round(float(mean_absolute_error(Y[te], pr)), 4),
                         'spearman': round(float(spearmanr(Y[te], pr).statistic), 4)}
        out[k] = res
    out['n_rows'] = len(rows)
    return out

# ---------- ③ PCA compression ----------

def pca_eval(cell_dir, ev_map, deltas):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import f1_score, r2_score
    key = 'h_it'
    # 收集全量 h_it 训练 PCA + 复测 ①
    Xs, Ys, E = [], [], []
    for f in sorted(glob.glob(os.path.join(cell_dir, 'ep*.npz'))):
        d = np.load(f); i = int(os.path.basename(f)[2:5])
        Xs.append(d[key]); Ys.append(node_labels(d['queue_per_node'])); E += [i] * len(d[key])
    X = np.concatenate(Xs); Y = np.concatenate(Ys); E = np.array(E)
    curve = {}
    for dim in PCA_DIMS:
        if dim > X.shape[-1]: break
        tr = E % 4 != 3; te = ~tr
        p = PCA(dim).fit(X[tr].reshape(-1, X.shape[-1]))
        Xp = p.transform(X.reshape(-1, X.shape[-1])).reshape(X.shape[0], X.shape[1], dim)
        m = LogisticRegression(max_iter=400).fit(Xp[tr].reshape(-1, dim), Y[tr].ravel())
        f1 = f1_score(Y[te].ravel(), m.predict(Xp[te].reshape(-1, dim)), average='macro')
        curve[dim] = {'p1_macroF1': round(float(f1), 4)}
    # PCA on h_it -> 复测 ②
    rows = []
    for f in sorted(glob.glob(os.path.join(cell_dir, 'ep*.npz'))):
        i = int(os.path.basename(f)[2:5])
        if i not in deltas or not ev_map.get(i): continue
        begin, end = min(b for b, _ in ev_map[i]), max(e for _, e in ev_map[i])
        d = np.load(f); t0 = int(begin // 10)
        w = (d['t'] >= begin) & (d['t'] < end)
        if w.sum() == 0: continue
        zt = d['z_task'][t0]
        zp = zt[zt[:, 0] > 0.5].mean(0) if (zt[:, 0] > .5).any() else zt.mean(0)
        lab = deltas[i][w].mean(0)
        rows.append((i, d[key][t0], zp, lab))
    if rows:
        Xf = np.concatenate([r[1] for r in rows])
        Z = np.concatenate([np.tile(r[2], (16, 1)) for r in rows])
        Y = np.concatenate([r[3] for r in rows]); eps = np.array([r[0] for r in rows]).repeat(16)
        for dim in PCA_DIMS:
            if dim > Xf.shape[-1] or dim not in curve: break
            tr = eps % 4 != 3; te = ~tr
            if tr.sum() < 30 or te.sum() < 10: break
            p = PCA(dim).fit(Xf[tr]); Xp = np.concatenate([p.transform(Xf), Z], -1)
            m = Ridge(1.0).fit(Xp[tr], Y[tr]); pr = m.predict(Xp[te])
            curve[dim]['p2_r2'] = round(float(r2_score(Y[te], pr)), 4)
    return curve

# ---------- driver ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cell'); ap.add_argument('--out-dir', default='data/output_data/analysis/att_entity_004/probes')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    ev_map = plan_events()
    cells = [a.cell] if a.cell else sorted(os.listdir(LIB))
    cells = [c for c in cells if os.path.isdir(os.path.join(LIB, c))]
    # fixedtime 配对差 (按 sumo seed 缓存)
    dss = {}
    for c in cells:
        ss = c.rsplit('_ss', 1)[1]
        if ss not in dss: dss[ss] = paired_delta(ss, ev_map)
    report = {}
    for c in cells:
        ss = c.rsplit('_ss', 1)[1]; deltas = dss[ss]
        cd = os.path.join(LIB, c)
        ent = {'probe1_congestion': probe1(cd)}
        p2 = probe2(cd, ev_map, deltas)
        if p2: ent['probe2_impact'] = p2
        if c.startswith('a3s'):
            ent['probe3_pca'] = pca_eval(cd, ev_map, deltas)
        report[c] = ent
        print('done', c, flush=True)
    fname = f'probe_report_{cells[0]}.json' if a.cell else 'probe_report.json'
    with open(os.path.join(a.out_dir, fname), 'w') as f:
        json.dump({'q_bins': Q_BINS, 'cells': report}, f, indent=1)
    print('wrote', os.path.join(a.out_dir, fname))

if __name__ == '__main__':
    main()
