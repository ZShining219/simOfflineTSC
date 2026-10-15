"""Same-state counterfactual attention probe.

Runs one episode per event edge with a joint (scene-internalised) policy
driving under the TRUE report, then replays every report-active decision
tick offline: the identical lane/global state is fed to the policy three
times with only the text vector changed — true report / wrong-edge report
(lie) / no report — and we measure

  edge mass   per-edge sum of the cross-attn weights w
  w-shift     L1 distance between w distributions across conditions
  q-flip      whether argmax(q) changes when the text changes

This separates "attention follows the report location" (w moves with the
text edge) from "attention only tracks congestion content" (w stays put).
Comparing two checkpoints (e.g. sig11 no-intent vs sig12 intent-utility)
shows whether the intent utility changed the encoder's text sensitivity.

    python -m scene_attention.probe.attn_text \
        --joint name=path [--joint name2=path2] --out outdir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import build_world  # noqa: E402
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention.dispatch_model import collate  # noqa: E402
from scene_attention.signal_model import SceneJointPolicy  # noqa: E402
from scene_attention.env import (  # noqa: E402
    run_signal_episode,
)
from scene_attention.events import (  # noqa: E402
    DISCHARGE_OF, EDGE_TARGETS, FEEDER_OF, TRAIN_EDGES,
)
from hitl.live.injects import parse_injects  # noqa: E402


def edge_mass(w, fb):
    """(L,) lane weights -> {edge: mass} over the builder's edge index."""
    w = np.asarray(w, dtype=np.float64)
    return {e: float(w[fb.lane_edge_idx == i].sum())
            for i, e in enumerate(fb.edges)}


def probe_edge(world, policy, fb_meta, edge, lie_edge, args):
    spec = (f'surge:{edge}:{EDGE_TARGETS[edge]}:'
            f'{args.onset}:{args.dur}:4')
    specs = [spec]
    if args.corridor:
        # match the corridor-load training regime: unreported feeder
        # inflow + a throttled discharge link
        feeder = FEEDER_OF.get(edge)
        if feeder:
            specs.append(
                f'surge:{feeder}:{edge}:{max(0, args.onset - 120)}:'
                f'{args.dur + 120}:7:global')
        discharge = DISCHARGE_OF.get(edge)
        if discharge:
            specs.append(
                f'throttle:{discharge}:{max(0, args.onset - 60)}:'
                f'{args.dur + 60}:3.0:global')
    injects = parse_injects(specs)
    fb = FeatureBuilder(world, injects,
                        report_delay=args.report_delay,
                        **fb_meta)
    ranks = list(range(len(world.intersection_ids)))
    transitions, records = run_signal_episode(
        world, fb, ranks, injects=injects, agent=policy,
        episode_seconds=args.episode_seconds, interval=args.interval)
    world.reset()

    rows = []
    for i, tr in enumerate(transitions[:-1]):   # last is the terminal pad
        t = records[i]['t']
        fields, _ = fb.event_fields(t)
        kind = fields.get('kind', 'none')
        if kind == 'none' or not fields.get('active'):
            continue
        snap = tr['s']
        coord_edges = (fb._edge_pred.get(edge, set())
                       | fb._edge_succ.get(edge, set())) - {edge}
        lie_coord = (fb._edge_pred.get(lie_edge, set())
                     | fb._edge_succ.get(lie_edge, set())) - {lie_edge}
        # 2-hop WATCH set from the CURRENT reported fields (follows the
        # broadcast during early throttle-report windows): edges with
        # intent_watch θ > 0, minus the reported edge itself.
        th_watch = fb.intent_watch(fields)
        watch_edges = {e for e in fb.edges
                       if th_watch[fb.edge_to_idx[e]] > 0
                       and e != fields.get('edge')}

        def align(mass):
            """cosine between per-edge attention mass and θ_watch —
            the response contract: does attention land where the
            compiled intent says the governance corridor is?"""
            m = np.asarray([mass.get(e, 0.0) for e in fb.edges],
                           dtype=np.float64)
            t = np.asarray(th_watch, dtype=np.float64)
            d = float(np.linalg.norm(m) * np.linalg.norm(t))
            return float((m * t).sum() / d) if d > 0 else 0.0
        variants = {
            'true': fb.fields_text_vec(fields),
            'lie': fb.text_vec(t, override={'edge': lie_edge}),
            'none': fb.text_vec(t, override={'kind': 'none'}),
        }
        out = {}
        for name, vec in variants.items():
            batch = collate([snap])
            batch['text'] = torch.as_tensor(
                vec[None], dtype=torch.float32)
            q, w = policy.forward(batch)
            qb = policy.base(policy.base_features(batch))
            w = w.detach().cpu().numpy()[0]
            out[name] = {
                'w': w,
                'mass': edge_mass(w, fb),
                'a': int(q.argmax(dim=-1).item()),
                'res': float((q - qb).abs().max().item()),
            }
        rows.append({
            't': t, 'edge': edge, 'lie': lie_edge,
            'mass_true_on_event': out['true']['mass'][edge],
            'mass_lie_on_event': out['lie']['mass'][edge],
            'mass_none_on_event': out['none']['mass'][edge],
            'mass_lie_on_lie': out['lie']['mass'][lie_edge],
            'mass_true_on_coord': float(sum(
                out['true']['mass'][e] for e in coord_edges)),
            'mass_lie_on_coord': float(sum(
                out['lie']['mass'][e] for e in coord_edges)),
            'mass_none_on_coord': float(sum(
                out['none']['mass'][e] for e in coord_edges)),
            'mass_lie_on_liecoord': float(sum(
                out['lie']['mass'][e] for e in lie_coord)),
            'mass_true_on_liecoord': float(sum(
                out['true']['mass'][e] for e in lie_coord)),
            'mass_true_on_watch': float(sum(
                out['true']['mass'][e] for e in watch_edges)),
            'mass_lie_on_watch': float(sum(
                out['lie']['mass'][e] for e in watch_edges)),
            'mass_none_on_watch': float(sum(
                out['none']['mass'][e] for e in watch_edges)),
            'align_true': align(out['true']['mass']),
            'align_lie': align(out['lie']['mass']),
            'align_none': align(out['none']['mass']),
            'align_lie_on_liewatch': float(
                (np.asarray([out['lie']['mass'].get(e, 0.0)
                             for e in fb.edges])
                 * fb.intent_watch({'kind': 'congest',
                                    'edge': lie_edge,
                                    'severity': 3,
                                    'active': True})).sum()
                / max(1e-9,
                      float(np.linalg.norm(
                          [out['lie']['mass'].get(e, 0.0)
                           for e in fb.edges])
                          * np.linalg.norm(fb.intent_watch(
                              {'kind': 'congest', 'edge': lie_edge,
                               'severity': 3, 'active': True}))))),
            'w_l1_true_lie': float(np.abs(
                out['true']['w'] - out['lie']['w']).sum()),
            'w_l1_true_none': float(np.abs(
                out['true']['w'] - out['none']['w']).sum()),
            'flip_lie': out['true']['a'] != out['lie']['a'],
            'flip_none': out['true']['a'] != out['none']['a'],
            'res_true': out['true']['res'],
            'res_lie': out['lie']['res'],
            'res_none': out['none']['res'],
        })
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint', action='append', required=True,
                   help='name=ckpt (repeatable)')
    p.add_argument('--scene', default='S2')
    p.add_argument('--onset', type=int, default=300)
    p.add_argument('--dur', type=int, default=600)
    p.add_argument('--episode-seconds', type=int, default=1200)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--traffic-seed-base', type=int, default=9000)
    p.add_argument('--corridor', action='store_true',
                   help='pair each probe event with the unreported '
                        'feeder surge + throttled discharge link used '
                        'in corridor-load training')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    world = build_world(args.scene, args.traffic_seed_base,
                        Path('/tmp/attn_text_probe'),
                        out / 'logger',
                        {'t_fixed': args.t_fixed, 't_min': args.t_min,
                         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    for spec in args.joint:
        name, path = spec.split('=', 1)
        policy, payload = SceneJointPolicy.load(path)
        fb_meta = fb_kwargs_from_meta(payload.get('meta'))
        all_rows = []
        for i, edge in enumerate(TRAIN_EDGES):
            lie = TRAIN_EDGES[(i + 1) % len(TRAIN_EDGES)]
            rows = probe_edge(world, policy, fb_meta, edge, lie, args)
            all_rows.extend(rows)
            print(f'{name} {edge}: {len(rows)} report-active ticks')
        agg = {}
        for k in ('mass_true_on_event', 'mass_lie_on_event',
                  'mass_none_on_event', 'mass_lie_on_lie',
                  'mass_true_on_coord', 'mass_lie_on_coord',
                  'mass_none_on_coord',
                  'mass_lie_on_liecoord', 'mass_true_on_liecoord',
                  'mass_true_on_watch', 'mass_lie_on_watch',
                  'mass_none_on_watch',
                  'align_true', 'align_lie', 'align_none',
                  'align_lie_on_liewatch',
                  'w_l1_true_lie', 'w_l1_true_none',
                  'res_true', 'res_lie', 'res_none'):
            agg[k] = round(float(np.mean([r[k] for r in all_rows])), 4)
        agg['flip_lie_rate'] = round(float(np.mean(
            [r['flip_lie'] for r in all_rows])), 4)
        agg['flip_none_rate'] = round(float(np.mean(
            [r['flip_none'] for r in all_rows])), 4)
        agg['n_ticks'] = len(all_rows)
        results[name] = {'ticks': all_rows, 'summary': agg}
        print(f'{name}: {agg}')

    with open(out / 'attn_probe.json', 'w') as f:
        json.dump(results, f, indent=1)


if __name__ == '__main__':
    main()
