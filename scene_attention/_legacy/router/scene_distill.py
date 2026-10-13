"""Stage-1 scene-distillation benchmark (representation layer, decoupled
from dispatch performance).

Question: given IDENTICAL complete inputs (lane tokens + scene text +
global) and an IDENTICAL scene-understanding objective, which encoder
arm distills the scene into the condensed z most faithfully?

Protocol
  * every arm produces z in R^64 from the same tensors;
  * each arm is trained end-to-end with the same multi-task scene
    distillation loss (same epochs, batch size, optimizer);
  * after joint training the encoder is frozen and FRESH linear probes
    are refit on z -> 'lin' scores = linear readability of z;
  * episode-disjoint splits (windows of one episode never straddle
    train/test), stratified so every (phase x edge) cell has both
    seeds in train and test;
  * --held-out mode additionally removes ALL samples whose report
    names the held-out edges from training: tests whether binding is
    a generalizable name-match (structural) or memorized per-edge
    weights (which cannot generalize to a never-named edge).

Tasks on z
  rep_edge    which edge the report names (n_edges + none)
  objective   declared governance objective (active reports only)
  cong_edge   which edge is physically most congested (or none)
  edge_q      per-edge mean queue regression (n_edges-dim)
  impact      next-window queue on the REPORTED edge (reported only)
  cong_total  total network queue regression
  rep_lane    per-lane 'on the reported edge' indicator (holdout only)
  bind_mass   attention mass on reported edge's lanes (attn arms only)

Usage:
    python -m scene_attention._legacy.router.scene_distill \
        --dump output_data/dev_takeover_probe/scene_dump/scene_dump.npz \
        --out  output_data/dev_takeover_probe/scene_distill/
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.dispatch_model import (  # noqa: E402
    SceneAttention, CompositeSceneEncoder, SelfAttentionEncoder,
    ConcatEncoder,
)

ARMS = ['sga', 'attn', 'selfattn', 'concat', 'mlp']
ATN_ARMS = {'sga', 'attn'}        # arms whose w is a cross-attn map


class FlatMLPEncoder(nn.Module):
    """Strongest no-attention arm: full flat input -> MLP -> z.

    Unlike concat (mean-pooled lanes), nothing is summarised before the
    MLP sees it — the benchmark-local 'everything, no attention' arm.
    """

    def __init__(self, n_lanes, lane_dim, text_dim, global_dim,
                 d_model=64):
        super().__init__()
        in_dim = n_lanes * lane_dim + text_dim + global_dim
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 4 * d_model), nn.ReLU(),
            nn.Linear(4 * d_model, 2 * d_model), nn.ReLU(),
            nn.Linear(2 * d_model, d_model), nn.ReLU())

    def forward(self, lanes, text, glob):
        z = self.mlp(torch.cat(
            [lanes.flatten(1), text, glob], dim=-1))
        w = torch.zeros(lanes.shape[0], lanes.shape[1],
                        device=lanes.device)
        return z, w


def build_encoder(name, dims, geom):
    d = dims['d_model']
    if name == 'sga':
        return CompositeSceneEncoder(
            dims['lane'], dims['text'], dims['glob'], d,
            residual_pool='mms', geom=geom)
    if name == 'attn':
        return SceneAttention(
            dims['lane'], dims['text'], dims['glob'], d,
            residual_pool='mms', geom=geom)
    if name == 'selfattn':
        return SelfAttentionEncoder(
            dims['lane'], dims['text'], dims['glob'], d)
    if name == 'mlp':
        return FlatMLPEncoder(
            dims['n_lanes'], dims['lane'], dims['text'],
            dims['glob'], d)
    return ConcatEncoder(dims['lane'], dims['text'], dims['glob'], d)


class Distiller(nn.Module):
    """encoder -> z -> per-task linear decoder heads."""

    def __init__(self, encoder, d_model, n_edges):
        super().__init__()
        self.encoder = encoder
        self.rep_edge = nn.Linear(d_model, n_edges + 1)
        self.obj = nn.Linear(d_model, 2)
        self.cong_edge = nn.Linear(d_model, n_edges + 1)
        self.edge_q = nn.Linear(d_model, n_edges)
        self.impact = nn.Linear(d_model, 1)
        self.cong = nn.Linear(d_model, 1)

    def forward(self, batch):
        z, w = self.encoder(batch['lanes'], batch['text'],
                            batch['glob'])
        return {
            'z': z, 'w': w,
            'rep_edge': self.rep_edge(z),
            'obj': self.obj(z),
            'cong_edge': self.cong_edge(z),
            'edge_q': self.edge_q(z),
            'impact': self.impact(z).squeeze(-1),
            'cong': self.cong(z).squeeze(-1),
        }


def multitask_loss(out, y):
    ce = nn.functional.cross_entropy
    l = ce(out['rep_edge'], y['rep_edge'])
    l = l + ce(out['cong_edge'], y['cong_edge'])
    m_obj = y['active'] > 0.5
    if m_obj.any():
        l = l + 0.5 * ce(out['obj'][m_obj], y['obj'][m_obj])
    l = l + nn.functional.mse_loss(out['edge_q'], y['edge_q'])
    m_imp = torch.isfinite(y['impact'])
    if m_imp.any():
        l = l + nn.functional.mse_loss(
            out['impact'][m_imp], y['impact'][m_imp])
    l = l + 0.5 * nn.functional.mse_loss(out['cong'], y['cong'])
    return l


def tensors(d, idx):
    return {
        'lanes': torch.as_tensor(
            d['lanes'][idx].astype(np.float32)),
        'text': torch.as_tensor(d['text'][idx].astype(np.float32)),
        'glob': torch.as_tensor(d['glob'][idx].astype(np.float32)),
    }


def labels(d, idx, n_edges):
    rep = d['reported_edge_idx'][idx].astype(np.int64)
    rep = np.where(rep < 0, n_edges, rep)
    cong = d['cong_edge_idx'][idx].astype(np.int64)
    cong = np.where(cong < 0, n_edges, cong)
    return {
        'rep_edge': torch.as_tensor(rep),
        'cong_edge': torch.as_tensor(cong),
        'obj': torch.as_tensor(d['objective'][idx].astype(np.int64)),
        'active': torch.as_tensor(d['active'][idx].astype(np.float32)),
        'edge_q': torch.as_tensor(
            d['edge_q'][idx].astype(np.float32) / 10.0),
        'impact': torch.as_tensor(
            d['impact_rep'][idx].astype(np.float32) / 10.0),
        'cong': torch.as_tensor(
            d['cong_total'][idx].astype(np.float32) / 100.0),
        'rep_edge_raw': d['reported_edge_idx'][idx],
        'active_np': d['active'][idx],
        'impact_np': d['impact_rep'][idx],
    }


def train_arm(name, dims, geom, d, tr, va, te, epochs, seed):
    torch.manual_seed(seed)
    model = Distiller(build_encoder(name, dims, geom),
                      dims['d_model'], dims['n_edges'])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3,
                           weight_decay=1e-4)
    tr_idx = np.asarray(tr)
    best = {'loss': float('inf'), 'state': None, 'epoch': -1}
    for ep in range(epochs):
        model.train()
        perm = np.random.default_rng(seed + ep).permutation(len(tr_idx))
        for off in range(0, len(perm), 256):
            bi = tr_idx[perm[off:off + 256]]
            out = model(tensors(d, bi))
            loss = multitask_loss(out, labels(d, bi, dims['n_edges']))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            model.eval()
            with torch.no_grad():
                vl = multitask_loss(model(tensors(d, va)),
                                    labels(d, va, dims['n_edges']))
            if vl.item() < best['loss']:
                best = {'loss': vl.item(),
                        'state': copy.deepcopy(model.state_dict()),
                        'epoch': ep}
    if best['state'] is not None:
        model.load_state_dict(best['state'])
    model.eval()
    return model, best['epoch']


def bind_mass(out_w, d, idx, lane_edge, n_edges):
    """Attention mass on the reported edge's lanes (reported only).

    out_w rows correspond positionally to idx, not to global indices.
    """
    rep = d['reported_edge_idx'][idx]
    m = rep >= 0
    if not m.any():
        return None
    w = out_w[m]
    mass = []
    for i, e in enumerate(rep[m]):
        lanes = np.flatnonzero(lane_edge == e)
        mass.append(float(w[i, torch.as_tensor(lanes)].sum()))
    return float(np.mean(mass))


def eval_e2e(model, d, idx, dims, lane_edge):
    """Joint-decoder test metrics on idx samples."""
    n_edges = dims['n_edges']
    with torch.no_grad():
        out = model(tensors(d, idx))
        y = labels(d, idx, n_edges)
    res = {}
    res['rep_edge_acc'] = float(
        (out['rep_edge'].argmax(-1) == y['rep_edge']).float().mean())
    res['cong_edge_acc'] = float(
        (out['cong_edge'].argmax(-1) == y['cong_edge']).float().mean())
    m_obj = torch.as_tensor(y['active_np'] > 0.5)
    res['obj_acc'] = (float(
        (out['obj'][m_obj].argmax(-1) == y['obj'][m_obj])
        .float().mean()) if m_obj.any() else None)
    res['edge_q_r2'] = _r2(y['edge_q'].numpy(), out['edge_q'].numpy())
    m_imp = np.isfinite(y['impact_np'])
    res['impact_r2'] = (_r2(y['impact'].numpy()[m_imp],
                          out['impact'].numpy()[m_imp])
                       if m_imp.any() else None)
    res['cong_r2'] = _r2(y['cong'].numpy(), out['cong'].numpy())
    res['bind_mass'] = bind_mass(out['w'], d, idx, lane_edge, n_edges)
    return res, out['z'].numpy()


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean(0)) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 1e-9 else np.nan


def lin_probes(Ztr, Zte, d, tr, te, dims, lane_edge):
    """Fresh linear probes on frozen z -> linear readability scores."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import accuracy_score, r2_score
    n_edges = dims['n_edges']
    res = {}
    ytr = labels(d, tr, n_edges)
    yte = labels(d, te, n_edges)
    for name, yt in [('rep_edge', ytr['rep_edge'].numpy()),
                     ('cong_edge', ytr['cong_edge'].numpy())]:
        ye = (yte['rep_edge'] if name == 'rep_edge'
              else yte['cong_edge']).numpy()
        if len(set(yt.tolist())) < 2:
            continue
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
        m.fit(Ztr, yt)
        res[name + '_acc'] = float(accuracy_score(ye, m.predict(Zte)))
    m_obj = ytr['active_np'] > 0.5
    m_obj_te = yte['active_np'] > 0.5
    if m_obj.sum() > 20 and m_obj_te.any():
        m = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=3000, C=1.0))
        m.fit(Ztr[m_obj], ytr['obj'].numpy()[m_obj])
        res['obj_acc'] = float(accuracy_score(
            yte['obj'].numpy()[m_obj_te], m.predict(Zte[m_obj_te])))
    for name, yt, ye in [
            ('edge_q', ytr['edge_q'].numpy(), yte['edge_q'].numpy()),
            ('cong', ytr['cong'].numpy(), yte['cong'].numpy())]:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(Ztr, yt)
        res[name + '_r2'] = float(r2_score(ye, m.predict(Zte)))
    m_i = np.isfinite(ytr['impact_np'])
    m_i_te = np.isfinite(yte['impact_np'])
    if m_i.sum() > 20 and m_i_te.sum() > 5:
        m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        m.fit(Ztr[m_i], ytr['impact'].numpy()[m_i])
        res['impact_r2'] = float(r2_score(
            yte['impact'].numpy()[m_i_te], m.predict(Zte[m_i_te])))
    # rep_lane: per-lane 'on the reported edge' indicator (n_lanes-dim)
    n_lanes = dims['n_lanes']
    def rep_lane_vec(idxs):
        V = np.zeros((len(idxs), n_lanes), dtype=np.float32)
        for i, s in enumerate(idxs):
            e = d['reported_edge_idx'][s]
            if e >= 0:
                V[i, lane_edge == e] = 1.0
        return V
    m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    m.fit(Ztr, rep_lane_vec(tr))
    res['rep_lane_r2'] = float(r2_score(
        rep_lane_vec(te), m.predict(Zte)))
    return res


def flat_feats(d, idx):
    return np.concatenate([d['lanes'][idx].reshape(len(idx), -1),
                           d['text'][idx], d['glob'][idx]], axis=1)


def split_episodes(eps):
    """seed-suffix stratified episode split: *_s1 -> test."""
    uniq = sorted(set(eps.tolist()))
    tr, te = [], []
    for i, ep in enumerate(uniq):
        (te if ep.endswith('_s1') else tr).append(ep)
    tr_set, te_set = set(tr), set(te)
    tr_idx = np.array([i for i, e in enumerate(eps) if e in tr_set])
    te_idx = np.array([i for i, e in enumerate(eps) if e in te_set])
    va_eps = set(tr[::7])
    va_idx = np.array([i for i, e in enumerate(eps)
                       if e in va_eps])
    tr_idx = np.array([i for i, e in enumerate(eps)
                       if e in tr_set and e not in va_eps])
    return tr_idx, va_idx, te_idx


def run_mode(d, dims, geom, lane_edge, tr, va, te, args, tag):
    out = {}
    Ztr_raw, Zte_raw = flat_feats(d, tr), flat_feats(d, te)
    for name in ARMS:
        model, best_ep = train_arm(name, dims, geom, d, tr, va, te,
                                   args.epochs, args.seed)
        e2e, z_te = eval_e2e(model, d, te, dims, lane_edge)
        with torch.no_grad():
            z_tr = model(tensors(d, tr))['z'].numpy()
        lin = lin_probes(z_tr, z_te, d, tr, te, dims, lane_edge)
        out[name] = {'best_epoch': best_ep, 'e2e': e2e, 'lin': lin}
        print(f'[{tag}/{name}] e2e rep_edge={e2e["rep_edge_acc"]:.3f} '
              f'cong_edge={e2e["cong_edge_acc"]:.3f} '
              f'edge_q_r2={e2e["edge_q_r2"]:.3f} '
              f'bind={e2e["bind_mass"]} | '
              f'lin rep_edge={lin.get("rep_edge_acc"):.3f} '
              f'rep_lane={lin.get("rep_lane_r2"):.3f}', flush=True)
    out['raw'] = {'lin': lin_probes(
        Ztr_raw, Zte_raw, d, tr, te, dims, lane_edge)}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dump', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--held-out', default=None,
                   help='comma-separated edge names held out of '
                        'training (e.g. road_1_2_3,road_2_1_2)')
    args = p.parse_args()

    d = dict(np.load(args.dump, allow_pickle=True))
    meta = json.loads(str(d['meta']))
    n_edges, n_lanes = meta['n_edges'], meta['n_lanes']
    n_kinds = meta['n_kinds']
    lanes = meta['lanes']
    edges = meta['edges']
    lane_edge = np.array([edges.index(l.rsplit('_', 1)[0])
                          for l in lanes])
    dims = {'lane': meta['lane_dim'], 'text': meta['text_dim'],
            'glob': meta['global_dim'], 'd_model': 64,
            'n_edges': n_edges, 'n_lanes': n_lanes}
    geom = None
    if meta.get('feature_layout') == 'identity':
        geom = {'phys_dim': 3,
                'lane_id': [3, 3 + n_edges],
                'text_id': [n_kinds, n_kinds + n_edges]}
    eps = d['episodes']
    tr, va, te = split_episodes(eps)
    print(f'samples={len(eps)} episodes={len(set(eps.tolist()))} '
          f'train={len(tr)} val={len(va)} test={len(te)} '
          f'layout={meta.get("feature_layout")}', flush=True)

    results = {'n_samples': int(len(eps)),
               'edges': edges, 'main': run_mode(
                   d, dims, geom, lane_edge, tr, va, te, args, 'main')}

    if args.held_out:
        held = [e for e in args.held_out.split(',') if e]
        held_idx = [edges.index(e) for e in held]
        rep = d['reported_edge_idx']
        is_held = np.isin(rep, held_idx)
        tr_h = np.array([i for i in tr if not is_held[i]], dtype=int)
        va_h = np.array([i for i in va if not is_held[i]], dtype=int)
        te_h = np.array([i for i in te if is_held[i]], dtype=int)
        # keep some no-report windows in the held-out test set too
        te_h = np.concatenate([
            te_h, np.array([i for i in te if rep[i] < 0], dtype=int)])
        print(f'[holdout] held={held} train={len(tr_h)} '
              f'test={len(te_h)}', flush=True)
        results['held_out'] = {'edges': held}
        if len(te_h) and len(tr_h):
            results['held_out'].update(run_mode(
                d, dims, geom, lane_edge, tr_h, va_h, te_h, args,
                'holdout'))
        else:
            print('[holdout] skipped: empty train or test split')

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(f'[done] -> {out}/metrics.json')


if __name__ == '__main__':
    main()
