"""Representation-level probes for the dispatch pilot (task-book metrics).

Evaluates the three frozen encoders (attn / selfattn / concat) on the
representation axis the task book asks for, on top of — and distinct
from — the control-level P2 evaluation:

  1. incident-impact prediction:  z(t) -> next-window mean queue on the
     reported event edge's lanes (surge cells only); Ridge, R^2 / MAE.
  2. congestion-level classification: z(t) -> quartile bucket of the
     current window's mean total queue; logistic probe, acc / macro-F1.
  3. compression retention: same probes refit on the raw flattened
     inputs (lanes + text + global); retention = metric(z) / metric(raw).
  4. size / latency: parameter count + per-call forward time.

Design: ONE rollout set (4 held-out cells x 3 seeds) driven by the
frozen ``router_attn`` policy; every window snapshot is encoded by all
three encoders -> paired z on identical states, isolating the
representation from the policy. Split = leave-one-episode-out
(12 groups) so probes are scored on unseen episodes.

Outputs to --out:
  z_dump.npz       per-arm z + raw features + labels + episode meta
  metrics.json     probe scores, retention, params, latency
Usage:
    python -m dev.takeover_probe._router.probe_repr --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dev.takeover_probe.probe import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.dispatch_model import (  # noqa: E402
    DispatchRouter, collate,
)
from dev.takeover_probe._router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from dev.takeover_probe._router.eval_router import TEST_EVENTS  # noqa: E402
from hitl.live.injects import parse_injects  # noqa: E402

ARMS3 = ['attn', 'selfattn', 'concat']


def flat_raw(snap):
    """Uncompressed input view: full lane tokens + text + global."""
    return np.concatenate([
        snap['lanes'].ravel(), snap['text'], snap['global']]).astype(
            np.float32)


def window_queue(w):
    recs = w.get('records') or []
    if not recs:
        return 0.0
    return float(np.mean([r['total_queue'] for r in recs]))


def window_edge_queue(w, ev_lanes):
    recs = w.get('records') or []
    if not recs or not ev_lanes:
        return 0.0
    return float(np.mean([
        np.mean([r['lane_queue'].get(l, 0) for l in ev_lanes])
        for r in recs]))


def encode_snap(routers, snap):
    """Encode one snapshot under every frozen router -> {arm: z}."""
    import torch
    batch = collate([snap])
    out = {}
    with torch.no_grad():
        for name, r in routers.items():
            z, _w = r.encode(batch)
            out[name] = z.cpu().numpy()[0].astype(np.float32)
    return out


def collect_rollouts(args, routers, fb_kwargs):
    """Run the P2 cell grid under the attn policy; keep window snaps."""
    runtime = Path(args.runtime_dir or '/tmp/takeover_probe')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    world = None
    samples = []
    dump_meta = {}
    try:
        for ev_idx, (edge, to_edge, per, dur, onset) in enumerate(
                TEST_EVENTS):
            spec = (
                f'surge:{edge}:{to_edge}:{onset}:{dur}:{per}'
                if edge else None)
            for seed_i in range(args.n_seeds):
                sumo_seed = args.test_seed_base + ev_idx * 100 + seed_i
                injects = parse_injects([spec]) if spec else []
                if world is not None:
                    world.close()
                world = build_world(
                    args.scene, sumo_seed, runtime, out / 'logger',
                    {'t_fixed': 30, 't_min': 10,
                     'min_green_vehicle': 3, 'max_red_vehicle': 6})
                ranks = list(range(len(world.intersection_ids)))
                controllers = build_controllers(
                    dqn_s1_path=args.dqn_s1)
                world.reset()
                router = routers['attn']
                _bind_all(world, controllers, ranks,
                          needed=set(router.controller_ids))
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs)
                if not dump_meta:
                    dump_meta = {
                        'feature_layout': fb.layout,
                        'event_flag': fb.include_event_flag,
                        'n_lanes': fb.n_lanes, 'n_edges': fb.n_edges,
                        'lane_dim': fb.lane_dim, 'text_dim': fb.text_dim,
                        'global_dim': fb.global_dim,
                        'lanes': fb.lanes,
                    }
                ctrl = {k: controllers[k] for k in router.controller_ids}
                _t, _r, windows = run_router_episode(
                    world, ctrl, fb, router, ranks,
                    injects=injects, interval=args.interval,
                    window_decisions=args.window_decisions,
                    episode_seconds=args.episode_seconds,
                    epsilon=0.0, collect=False)
                ev_lanes = (
                    [l for l in fb.lanes
                     if l.rsplit('_', 1)[0] == edge] if edge else [])
                ep_id = f'e{ev_idx}_s{seed_i}'
                for wi, w in enumerate(windows):
                    snap = w['snap']
                    zs = encode_snap(routers, snap)
                    next_w = windows[wi + 1] if wi + 1 < len(windows) else None
                    samples.append({
                        'episode': ep_id, 'cell': spec or 'none',
                        'seed': sumo_seed, 't': w['t'],
                        'z': zs, 'raw': flat_raw(snap),
                        'raw_snap': snap,
                        'cong': window_queue(w),
                        'impact_next': (
                            window_edge_queue(next_w, ev_lanes)
                            if next_w is not None and ev_lanes
                            else np.nan),
                        'event_edge': edge or '',
                        'active': bool(snap['event_lane_idx']),
                    })
                print(f'[collect] {ep_id} {spec or "none"} '
                      f'windows={len(windows)}', flush=True)
    finally:
        if world is not None:
            world.close()
    return samples, dump_meta


def probe_fit(X, y, groups, kind):
    """Leave-one-episode-out CV -> per-fold scores."""
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, f1_score, r2_score, \
        mean_absolute_error
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    uniq = sorted(set(groups))
    preds = np.zeros(len(y), dtype=np.float32)
    trues = np.zeros(len(y), dtype=np.float32)
    folds = []
    for g in uniq:
        te = np.array([gi == g for gi in groups])
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0:
            continue
        if kind == 'clf':
            m = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0))
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
            folds.append({'group': g, 'acc': float(
                accuracy_score(y[te], preds[te]))})
        else:
            m = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            m.fit(X[tr], y[tr])
            preds[te] = m.predict(X[te])
            folds.append({'group': g, 'mae': float(
                mean_absolute_error(y[te], preds[te]))})
        trues[te] = y[te]
    if kind == 'clf':
        return {'acc': float(accuracy_score(trues, preds)),
                'f1': float(f1_score(trues, preds, average='macro')),
                'folds': folds}
    return {'r2': float(r2_score(trues, preds)),
            'mae': float(mean_absolute_error(trues, preds)),
            'folds': folds}


def model_stats(routers, snap):
    """Param count + encode latency, timed on a real snapshot batch."""
    import torch
    batch = collate([snap])
    stats = {}
    for name, r in routers.items():
        n_par = sum(p.numel() for p in r.encoder.parameters())
        n_all = sum(p.numel() for p in r.parameters())
        with torch.no_grad():
            r.encode(batch)                      # warmup
            t0 = time.perf_counter()
            for _ in range(200):
                r.encode(batch)
            cpu_ms = (time.perf_counter() - t0) / 200 * 1000
        stats[name] = {'encoder_params': int(n_par),
                       'router_params': int(n_all),
                       'encode_ms_cpu': round(cpu_ms, 3)}
        if torch.cuda.is_available():
            dev = torch.device('cuda')
            rc = r.to(dev)
            batch_d = {k: v.to(dev) for k, v in batch.items()
                       if torch.is_tensor(v)}
            with torch.no_grad():
                rc.encode(batch_d)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(200):
                    rc.encode(batch_d)
                torch.cuda.synchronize()
                gpu_ms = (time.perf_counter() - t0) / 200 * 1000
            stats[name]['encode_ms_cuda'] = round(gpu_ms, 3)
            rc.to('cpu')
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--attn-ckpt',
                   default='output_data/dev_takeover_probe/'
                           'router_attn_obj3/router_attn.pt')
    p.add_argument('--selfattn-ckpt',
                   default='output_data/dev_takeover_probe/'
                           'router_selfattn_obj3/router_selfattn.pt')
    p.add_argument('--concat-ckpt',
                   default='output_data/dev_takeover_probe/'
                           'router_concat_obj3/router_concat.pt')
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--test-seed-base', type=int, default=5000)
    p.add_argument('--n-seeds', type=int, default=3)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--skip-collect', action='store_true',
                   help='reuse z_dump.npz + metrics.json in --out '
                        '(refit probes only, no SUMO)')
    p.add_argument('--extra-ckpt', action='append', default=[],
                   help='NAME=CKPT extra frozen router; in --skip-collect '
                        'mode its z is re-encoded from the dumped raw '
                        'snaps (no SUMO)')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    extra_routers = {}
    for spec_kv in args.extra_ckpt:
        nm, ck = spec_kv.split('=', 1)
        extra_routers[nm], _ = DispatchRouter.load(ck)

    out = Path(args.out)
    if args.skip_collect:
        d = np.load(out / 'z_dump.npz', allow_pickle=True)
        stats = json.loads(
            (out / 'metrics.json').read_text())['model_stats']
        cong = d['cong']
        y_cls = d['y_cls']
        y_imp = d['y_impact']
        ev_mask = d['ev_mask']
        groups = [str(g) for g in d['episodes']]
        X_raw = d['raw']
        X_by = {n: d[f'z_{n}'] for n in ARMS3}
        n_samples = len(groups)
        qs = np.quantile(cong, [0.25, 0.5, 0.75])
        if extra_routers:
            from dev.takeover_probe._router.channel_ablate import (
                dump_spec, unpack, N_KINDS, SCALARS)
            if 'meta' in d.files:
                spec = dump_spec(d)
            else:   # legacy dump: flag layout, lane_dim=4 + lane indicator
                n_lanes = (X_raw.shape[1] - N_KINDS - SCALARS - 11) // 5
                spec = dict(layout='flag', n_lanes=n_lanes,
                            n_edges=None, event_flag=True, lane_dim=4,
                            text_dim=N_KINDS + n_lanes + SCALARS,
                            global_dim=11)
            for nm, r in extra_routers.items():
                zs = []
                for row in X_raw:
                    lns, txt, gl = unpack(row, spec)
                    zs.append(encode_snap({nm: r}, {
                        'lanes': lns, 'text': txt, 'global': gl,
                        'event_lane_idx': []})[nm])
                X_by[nm] = np.stack(zs)
                stats.setdefault(nm, {})
            print('re-encoded extras:', list(extra_routers))
    else:
        routers = {}
        metas = {}
        for name, ck in [('attn', args.attn_ckpt),
                         ('selfattn', args.selfattn_ckpt),
                         ('concat', args.concat_ckpt)]:
            routers[name], payload = DispatchRouter.load(ck)
            metas[name] = payload.get('meta')
        routers.update(extra_routers)

        samples, dump_meta = collect_rollouts(
            args, routers, fb_kwargs_from_meta(metas.get('attn')))

        stats = model_stats(routers, samples[0]['raw_snap'])
        print(json.dumps({'model_stats': stats}, indent=2))

        # congestion class = pooled quartile of window mean queue
        cong = np.array([s['cong'] for s in samples])
        qs = np.quantile(cong, [0.25, 0.5, 0.75])
        y_cls = np.digitize(cong, qs)

        groups = [s['episode'] for s in samples]
        ev_mask = np.array(
            [s['event_edge'] != '' and np.isfinite(s['impact_next'])
             for s in samples])
        y_imp = np.array([s['impact_next'] for s in samples])
        X_raw = np.stack([s['raw'] for s in samples])
        X_by = {n: np.stack([s['z'][n] for s in samples])
                for n in routers}
        n_samples = len(samples)
        np.savez_compressed(
            out / 'z_dump.npz',
            **{f'z_{n}': X_by[n] for n in X_by},
            raw=X_raw, y_cls=y_cls, y_impact=y_imp,
            ev_mask=ev_mask,
            episodes=np.array(groups), cong=cong,
            meta=json.dumps(dump_meta))

    g_imp = [g for i, g in enumerate(groups) if ev_mask[i]]
    metrics = {'n_samples': int(n_samples),
               'cong_quartiles': qs.tolist(),
               'model_stats': stats, 'arms': {}}
    for name in X_by:
        X = X_by[name]
        clf = probe_fit(X, y_cls, groups, 'clf')
        reg = probe_fit(X[ev_mask], y_imp[ev_mask], g_imp, 'reg')
        metrics['arms'][name] = {
            'congestion_clf': clf, 'impact_reg': reg}
    raw_clf = probe_fit(X_raw, y_cls, groups, 'clf')
    raw_reg = probe_fit(X_raw[ev_mask], y_imp[ev_mask], g_imp, 'reg')
    metrics['raw_baseline'] = {
        'congestion_clf': raw_clf, 'impact_reg': raw_reg}
    for name in X_by:
        a = metrics['arms'][name]
        a['retention'] = {
            'clf_acc': round(a['congestion_clf']['acc']
                             / max(raw_clf['acc'], 1e-9), 3),
            'reg_r2_ratio': round(a['impact_reg']['r2']
                                  / max(raw_reg['r2'], 1e-9), 3)
            if raw_reg['r2'] > 0 else None}

    (out / 'metrics.json').write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8')
    print(json.dumps(metrics['arms'], indent=2))


if __name__ == '__main__':
    main()
