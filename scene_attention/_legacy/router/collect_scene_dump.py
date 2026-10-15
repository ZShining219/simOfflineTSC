"""Scene-distillation dataset collector (stage-1 representation benchmark).

Collects (lanes, text, global) snapshots + scene-fidelity labels for the
condensed-representation comparison.  Unlike probe_repr — which dumps z
from frozen dispatch-trained routers — this dump stores raw features
plus labels so each encoder arm can be trained on the SAME scene-
understanding objective and compared on z quality alone.

Coverage design (identity layout, S2 8-edge world):

  A matched    surge on E_i, natural report names E_i       (correlated)
  B mismatched surge on E_j, text_override names E_i (i!=j)  (distractor)
  C text_only  no event, override names E_i                  (pure semantics)
  D none       no event, no report                            (control)

Text override is a legal operation here: the report channel is the
scene-semantics input, and binding must follow the NAME regardless of
physics — decorrelated report-vs-physics samples are exactly what the
held-out-edge / distractor probes need.

Labels per window (decoded from the snapshot itself, so overrides are
automatically consistent):
  reported_edge_idx  argmax of text edge block (-1 = none)
  objective          1 if report declares focus
  severity           report severity 0..3
  active             report live
  phys_event_edge_idx edge that physically surged this episode (-1)
  cong_edge_idx      edge with max window-mean queue (-1 if below thresh)
  cong_total         window-mean total queue
  impact_rep         next-window mean queue on REPORTED edge's lanes

Usage:
    python -m scene_attention._legacy.router.collect_scene_dump --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import (  # noqa: E402
    build_controllers, build_world, _bind_all, DEFAULT_DQN_S1,
)
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention.dispatch_model import DispatchRouter  # noqa: E402
from scene_attention._legacy.router.router_runtime import (  # noqa: E402
    run_router_episode,
)
from scene_attention._legacy.router.train_router import EDGE_TARGETS  # noqa: E402
from hitl.live.injects import parse_injects  # noqa: E402

CONG_THRESH = 1.0   # mean queue (veh) below this -> 'no congestion edge'


def _episodes(edges):
    """Build the A/B/C/D episode recipe.

    Returns list of dicts: {tag, inject_spec, override, controller}.
    Injectable edges come from EDGE_TARGETS; every reportable edge gets
    B/C coverage so all 8 names appear in the text channel.
    """
    rng = np.random.default_rng(7)
    eps = []
    ctrls = ['dqn_s1', 'maxpressure']
    pick = lambda i: ctrls[i % 2]
    # A: matched natural reports (only edges we can physically inject)
    for i, e in enumerate(sorted(EDGE_TARGETS)):
        for s in range(2):
            obj = 'focus' if (i + s) % 2 else 'global'
            eps.append({
                'tag': f'A_{e}_s{s}',
                'inject': (f'surge:{e}:{EDGE_TARGETS[e]}:300:600:'
                           f'{2 + s}:{obj}'),
                'override': None, 'controller': pick(i + s),
                'phys_edge': e})
    # B: mismatched — physics on a random injectable edge, report names
    # a DIFFERENT edge drawn from all edges (decorrelates report/physics)
    others = [e for e in sorted(EDGE_TARGETS)]
    for i, e in enumerate(edges):
        for s in range(2):
            cands = [x for x in others if x != e]
            phys = cands[int(rng.integers(len(cands)))]
            obj = 'focus' if (i + s) % 2 else 'global'
            eps.append({
                'tag': f'B_{e}_s{s}',
                'inject': (f'surge:{phys}:{EDGE_TARGETS[phys]}:300:600:'
                           f'2:{obj}'),
                # report names e (which is NOT the injected edge)
                'override': {'edge': e},
                'controller': pick(i + s), 'phys_edge': phys})
    # C: report only — text semantics with no physical event anywhere
    for i, e in enumerate(edges):
        for s in range(2):
            obj = 'focus' if s % 2 else 'global'
            eps.append({
                'tag': f'C_{e}_s{s}', 'inject': None,
                'override': {
                    'kind': 'surge', 'edge': e, 'active': True,
                    'recovering': False, 'severity': 1 + (i % 3),
                    'est_remaining_s': 600.0, 'onset_age_s': 60.0,
                    'objective': obj},
                'controller': pick(i + s), 'phys_edge': None})
    # D: no report
    for s in range(4):
        eps.append({'tag': f'D_none_s{s}', 'inject': None,
                    'override': None, 'controller': pick(s),
                    'phys_edge': None})
    return eps


def _decode_labels(fb, snap, w, next_w, phys_edge):
    """Per-window labels from the snapshot + window records."""
    n_k = fb.n_kinds
    edge_blk = snap['text'][n_k:n_k + fb.n_edges]
    rep_eidx = int(np.argmax(edge_blk)) if edge_blk.max() > 0.5 else -1
    scal = snap['text'][n_k + fb._text_edge_dim:]
    recs = w.get('records') or []
    edge_q = np.zeros(fb.n_edges)
    if recs:
        for r in recs:
            for l, q in (r.get('lane_queue') or {}).items():
                eidx = fb.edge_to_idx.get(l.rsplit('_', 1)[0])
                if eidx is not None:
                    edge_q[eidx] += float(q)
        edge_q /= len(recs)
    cong_eidx = (int(np.argmax(edge_q))
                 if edge_q.max() >= CONG_THRESH else -1)
    impact = np.nan
    if next_w is not None and rep_eidx >= 0:
        lanes = [l for l in fb.lanes
                 if fb.edge_to_idx[l.rsplit('_', 1)[0]] == rep_eidx]
        nr = next_w.get('records') or []
        if nr:
            impact = float(np.mean([
                np.mean([r['lane_queue'].get(l, 0.0) for l in lanes])
                for r in nr]))
    cong_total = (float(np.mean([r['total_queue'] for r in recs]))
                  if recs else 0.0)
    return {
        'reported_edge_idx': rep_eidx,
        'objective': float(scal[5]),
        'severity': float(scal[3] * 3.0),
        'active': float(scal[0]),
        'phys_event_edge_idx': (
            fb.edge_to_idx.get(phys_edge, -1) if phys_edge else -1),
        'cong_edge_idx': cong_eidx,
        'cong_total': cong_total,
        'edge_q': edge_q,
        'impact_rep': impact,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--meta-ckpt',
                   default='data/output_data/dev_takeover_probe/'
                           'router_id_attn/router_attn.pt',
                   help='identity-layout router ckpt; only its feature '
                        'meta is used to build the FeatureBuilder')
    p.add_argument('--seed-base', type=int, default=7000)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--window-decisions', type=int, default=12)
    p.add_argument('--episode-seconds', type=int, default=3600)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    _router, payload = DispatchRouter.load(args.meta_ckpt)
    fb_kwargs = fb_kwargs_from_meta(payload.get('meta'))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runtime = Path(args.runtime_dir or '/tmp/takeover_scene_dump')
    world, samples, dump_meta = None, [], None
    try:
        world = build_world(args.scene, args.seed_base, runtime,
                            out / 'logger',
                            {'t_fixed': 30, 't_min': 10,
                             'min_green_vehicle': 3,
                             'max_red_vehicle': 6})
        ranks = list(range(len(world.intersection_ids)))
        controllers = build_controllers(dqn_s1_path=args.dqn_s1)
        # first pass: need fb instance for edge list -> build once
        world.reset()
        fb_probe = FeatureBuilder(world, [], **fb_kwargs)
        edges = fb_probe.edges
        recipes = _episodes(edges)
        for ep_i, ep in enumerate(recipes):
            world.close()
            world = build_world(
                args.scene, args.seed_base + ep_i, runtime,
                out / 'logger',
                {'t_fixed': 30, 't_min': 10,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            world.reset()
            injects = parse_injects([ep['inject']]) \
                if ep['inject'] else []
            fb = FeatureBuilder(world, injects,
                                report_delay=args.report_delay,
                                **fb_kwargs)
            _bind_all(world, controllers, ranks,
                      needed={ep['controller']})
            ctrl = {ep['controller']: controllers[ep['controller']]}
            _t, _r, windows = run_router_episode(
                world, ctrl, fb, None, ranks,
                injects=injects, interval=args.interval,
                window_decisions=args.window_decisions,
                episode_seconds=args.episode_seconds,
                fixed_controller=ep['controller'],
                text_override=ep['override'], collect=False)
            for wi, w in enumerate(windows):
                snap = w['snap']
                nw = windows[wi + 1] if wi + 1 < len(windows) else None
                lab = _decode_labels(fb, snap, w, nw, ep['phys_edge'])
                samples.append({
                    'episode': ep['tag'], 't': w['t'],
                    'lanes': snap['lanes'], 'text': snap['text'],
                    'global': snap['global'], **lab})
            print(f"[collect] {ep['tag']} inject={ep['inject']} "
                  f"ovr={ep['override']} ctrl={ep['controller']} "
                  f"win={len(windows)}", flush=True)
        dump_meta = {
            'feature_layout': fb.layout,
            'event_flag': fb.include_event_flag,
            'n_lanes': fb.n_lanes, 'n_edges': fb.n_edges,
            'lane_dim': fb.lane_dim, 'text_dim': fb.text_dim,
            'global_dim': fb.global_dim, 'lanes': fb.lanes,
            'edges': fb.edges, 'n_kinds': fb.n_kinds,
        }
    finally:
        if world is not None:
            world.close()

    keys = ['reported_edge_idx', 'objective', 'severity', 'active',
            'phys_event_edge_idx', 'cong_edge_idx', 'cong_total',
            'impact_rep']
    np.savez_compressed(
        out / 'scene_dump.npz',
        lanes=np.stack([s['lanes'] for s in samples]),
        text=np.stack([s['text'] for s in samples]),
        glob=np.stack([s['global'] for s in samples]),
        edge_q=np.stack([s['edge_q'] for s in samples]),
        episodes=np.array([s['episode'] for s in samples]),
        **{k: np.array([s[k] for s in samples]) for k in keys},
        meta=json.dumps(dump_meta))
    print(f'[done] {len(samples)} windows -> {out}/scene_dump.npz')


if __name__ == '__main__':
    main()
