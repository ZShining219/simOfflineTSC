"""Attention heatmap figure for the internalised (joint) scene policy.

One traced episode driven by a SceneJointPolicy under the true report;
every recorded tick is then replayed offline with only the text vector
swapped, so all panels below share identical lane states:

  row 0  per-lane queue (physical context)
  row 1  lane×lane self-attention salience (traffic-responsive layer)
  row 2  text->lane binding w under the TRUE report
  row 3  text->lane binding w under a WRONG-edge report (lie)
  row 4  text->lane binding w with text OFF (none)
  row 5  per-edge attention mass across the three text conditions

    python -m dev.takeover_probe.joint_attn_fig \
        --joint output_data/.../agent.pt --out outdir
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

from dev.takeover_probe.probe import build_world  # noqa: E402
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.dispatch_model import collate  # noqa: E402
from dev.takeover_probe.signal_model import SceneJointPolicy  # noqa: E402
from dev.takeover_probe.signal_runtime import (  # noqa: E402
    run_signal_episode,
)
from dev.takeover_probe._router.train_router import (  # noqa: E402
    DISCHARGE_OF, EDGE_TARGETS, FEEDER_OF, TRAIN_EDGES,
)
from hitl.live.injects import parse_injects  # noqa: E402


def sa_salience(policy, lanes_t):
    """Per-lane received mass of the lane×lane self-attention layer."""
    enc = policy.encoder
    kv0 = enc.lane_proj(lanes_t)
    _, sa_w = enc.self_attn(kv0, kv0, kv0, need_weights=True)
    return sa_w[0].detach().cpu().numpy().mean(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--joint', required=True, help='joint ckpt path')
    p.add_argument('--scene', default='S2')
    p.add_argument('--event-edge', default='road_2_1_2')
    p.add_argument('--lie-edge', default='road_0_1_0')
    p.add_argument('--onset', type=int, default=300)
    p.add_argument('--dur', type=int, default=600)
    p.add_argument('--seed', type=int, default=9009)
    p.add_argument('--report-delay', type=float, default=30.0)
    p.add_argument('--episode-seconds', type=int, default=1800)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--corridor', action='store_true',
                   help='trace under the corridor-load scenario: '
                        'unreported feeder surge + throttled discharge')
    p.add_argument('--tag', default='joint scene policy',
                   help='model tag for the figure title')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    policy, payload = SceneJointPolicy.load(args.joint)
    fb_meta = fb_kwargs_from_meta(payload.get('meta'))

    world = build_world(args.scene, args.seed, Path('/tmp/joint_attn_fig'),
                        out / 'logger',
                        {'t_fixed': args.t_fixed, 't_min': args.t_min,
                         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    spec = (f'surge:{args.event_edge}:{EDGE_TARGETS[args.event_edge]}:'
            f'{args.onset}:{args.dur}:4')
    specs = [spec]
    if args.corridor:
        feeder = FEEDER_OF.get(args.event_edge)
        if feeder:
            specs.append(
                f'surge:{feeder}:{args.event_edge}:'
                f'{max(0, args.onset - 120)}:{args.dur + 120}:7:global')
        discharge = DISCHARGE_OF.get(args.event_edge)
        if discharge:
            specs.append(
                f'throttle:{discharge}:{max(0, args.onset - 60)}:'
                f'{args.dur + 60}:3.0:global')
    injects = parse_injects(specs)
    fb = FeatureBuilder(world, injects,
                        report_delay=args.report_delay, **fb_meta)
    ranks = list(range(len(world.intersection_ids)))
    transitions, records = run_signal_episode(
        world, fb, ranks, injects=injects, agent=policy,
        episode_seconds=args.episode_seconds, interval=args.interval)
    world.close()

    lanes = list(fb.lanes)
    e2i = {k: list(v) for k, v in fb.edge_to_laneidx.items()}
    ev_edge, lie_edge = args.event_edge, args.lie_edge
    coord_edges = sorted((fb._edge_pred.get(ev_edge, set())
                          | fb._edge_succ.get(ev_edge, set()))
                         - {ev_edge})
    # 2-hop WATCH set = intent_watch θ > 0 minus the event edge; the
    # part beyond 1-hop coord is the metering-source approaches.
    th_w = fb.intent_watch({'kind': 'congest', 'edge': ev_edge,
                            'severity': 3, 'active': True})
    watch_edges = sorted(e for e in fb.edges
                         if e != ev_edge
                         and th_w[fb.edge_to_idx[e]] > 0)
    meter_edges = sorted(set(watch_edges) - set(coord_edges))
    ts = [r['t'] for r in records]
    n_lanes = len(lanes)

    # replay every tick under the three text conditions
    W = {'true': [], 'lie': [], 'none': []}
    SA, Q = [], []
    for i, tr in enumerate(transitions[:-1]):
        snap = tr['s']
        batch = collate([snap])
        SA.append(sa_salience(policy, batch['lanes']))
        for name, vec in (
                ('true', snap['text']),
                ('lie', fb.text_vec(records[i]['t'],
                                    override={'edge': lie_edge})),
                ('none', fb.text_vec(records[i]['t'],
                                     override={'kind': 'none'}))):
            b = dict(batch)
            b['text'] = torch.as_tensor(vec[None], dtype=torch.float32)
            _, w = policy.forward(b)
            W[name].append(w.detach().cpu().numpy()[0])
        Q.append([records[i]['lane_queue'].get(l, 0) for l in lanes])
    SA = np.asarray(SA).T
    Qm = np.asarray(Q).T
    Wm = {k: np.asarray(v).T for k, v in W.items()}
    tt = ts[:SA.shape[1]]

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ev = parse_injects([spec])[0]
    start_t, end_t = ev['start_t'], ev['end_t']
    rep_start = start_t + args.report_delay
    bounds, centers, names = [], [], []
    for edge, idx in sorted(e2i.items(), key=lambda kv: min(kv[1])):
        idx = sorted(idx)
        bounds.append(min(idx) - 0.5)
        centers.append(float(np.mean(idx)))
        names.append(edge)
    bounds.append(max(i for v in e2i.values() for i in v) + 0.5)

    fig, axes = plt.subplots(
        6, 1, figsize=(10, 13.5), sharex=True,
        gridspec_kw={'height_ratios': [1.6, 1.8, 1.8, 1.8, 1.8, 1.5]})

    def lane_axis(ax, im=None):
        for b in bounds[1:-1]:
            ax.axhline(b, color='w', lw=0.4, alpha=0.5)
        ax.set_yticks(centers)
        ax.set_yticklabels(names, fontsize=6.5)
        for lab, edge in zip(ax.get_yticklabels(), names):
            if edge == ev_edge:
                lab.set_color('#d62728')
                lab.set_fontweight('bold')
            elif edge == lie_edge:
                lab.set_color('#e8912d')
                lab.set_fontweight('bold')
            elif edge in coord_edges:
                lab.set_color('#0d7d74')
                lab.set_fontweight('bold')
            elif edge in meter_edges:
                lab.set_color('#9467bd')
                lab.set_fontweight('bold')
        for x in (start_t, end_t):
            ax.axvline(x, color='#2ca02c', lw=0.8, ls=':')
        ax.axvline(rep_start, color='#2ca02c', lw=0.8, ls='-',
                   alpha=0.4)
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    im = axes[0].imshow(Qm, aspect='auto', origin='lower',
                        cmap='inferno', interpolation='nearest',
                        vmin=0, vmax=max(10, np.percentile(Qm, 99)),
                        extent=[tt[0], tt[-1], -0.5, n_lanes - 0.5])
    axes[0].set_title('per-lane queue (physical context)', fontsize=9)
    lane_axis(axes[0], im)

    im = axes[1].imshow(SA, aspect='auto', origin='lower',
                        cmap='inferno', interpolation='nearest',
                        vmin=0.0, vmax=0.20,
                        extent=[tt[0], tt[-1], -0.5, n_lanes - 0.5])
    axes[1].set_title('lane×lane self-attention received mass '
                      '(text-independent traffic layer)', fontsize=9)
    lane_axis(axes[1], im)

    for ax, key, ttl in (
            (axes[2], 'true',
             'text→lane binding w — TRUE report'),
            (axes[3], 'lie',
             f'text→lane binding w — WRONG report '
             f'(claims {lie_edge}, orange)'),
            (axes[4], 'none',
             'text→lane binding w — text OFF')):
        im = ax.imshow(Wm[key], aspect='auto', origin='lower',
                       cmap='inferno', interpolation='nearest',
                       vmin=0.0, vmax=0.20,
                       extent=[tt[0], tt[-1], -0.5, n_lanes - 0.5])
        ax.set_title(ttl, fontsize=9)
        lane_axis(ax, im)

    ax = axes[5]
    edge_idx = {e: np.asarray(i) for e, i in e2i.items()}

    def smooth(y, k=6):
        k = int(k)
        c = np.ones(k) / k
        return np.convolve(y, c, mode='same')

    coord_idx = np.asarray(sorted(
        i for e in coord_edges for i in e2i[e]))
    meter_idx = np.asarray(sorted(
        i for e in meter_edges for i in e2i[e]), dtype=int)
    for e, idx in edge_idx.items():
        if e in (ev_edge, lie_edge) or e in coord_edges \
                or e in meter_edges:
            continue
        ax.plot(tt, smooth(Wm['true'][idx].sum(axis=0)),
                color='#c8c8c8', lw=0.8, alpha=0.7, zorder=2)
    ax.plot([], [], color='#c8c8c8', lw=0.9, label='other edges')
    ax.plot(tt, smooth(Wm['true'][coord_idx].sum(axis=0)),
            color='#0d7d74', lw=1.8, zorder=5,
            label='corridor edges (feeder+discharge) — true')
    ax.plot(tt, smooth(Wm['none'][coord_idx].sum(axis=0)),
            color='#0d7d74', lw=1.0, ls=':', zorder=3,
            label='corridor edges — text off')
    if meter_idx.size:
        ax.plot(tt, smooth(Wm['true'][meter_idx].sum(axis=0)),
                color='#9467bd', lw=1.8, zorder=5,
                label='metering sources (2-hop watch) — true')
        ax.plot(tt, smooth(Wm['none'][meter_idx].sum(axis=0)),
                color='#9467bd', lw=1.0, ls=':', zorder=3,
                label='metering sources — text off')
    ax.plot(tt, smooth(Wm['true'][edge_idx[ev_edge]].sum(axis=0)),
            color='#1f6fb4', lw=1.8, zorder=5,
            label=f'event edge {ev_edge} — true report')
    ax.plot(tt, smooth(Wm['lie'][edge_idx[ev_edge]].sum(axis=0)),
            color='#1f6fb4', lw=1.2, ls='--', zorder=4,
            label='event edge — lie report')
    ax.plot(tt, smooth(Wm['none'][edge_idx[ev_edge]].sum(axis=0)),
            color='#7f7f7f', lw=1.2, ls=':', zorder=4,
            label='event edge — text off')
    ax.plot(tt, smooth(Wm['lie'][edge_idx[lie_edge]].sum(axis=0)),
            color='#e8912d', lw=1.8, zorder=5,
            label=f'lie edge {lie_edge} — lie report')
    ax.plot(tt, smooth(Wm['true'][edge_idx[lie_edge]].sum(axis=0)),
            color='#e8912d', lw=1.0, ls=':', zorder=3,
            label='lie edge — true report')
    for x in (start_t, end_t):
        ax.axvline(x, color='#d62728', lw=0.8, ls=':')
    ax.axvspan(start_t + args.report_delay, end_t,
               color='#2ca02c', alpha=0.06)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('simulation time (s)')
    ax.set_title('per-edge attention mass across text conditions '
                 '(report window shaded)', fontsize=9)
    ax.legend(fontsize=6.5, loc='lower right', ncol=2)
    ax.grid(alpha=0.2)

    fig.suptitle(
        f'{args.tag} | {spec}'
        + (' + corridor load' if args.corridor else '') + '\n'
        f'identical states, text swapped offline: red = event edge, '
        f'teal = feeder/discharge, purple = metering sources, '
        f'orange = falsely-reported edge', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'fig_joint_attn.png', dpi=160)
    plt.close(fig)
    json.dump({'spec': spec, 'event_edge': ev_edge,
               'lie_edge': lie_edge, 'n_ticks': len(tt)},
              open(out / 'fig_joint_attn_meta.json', 'w'), indent=1)
    print('fig ->', out / 'fig_joint_attn.png')


if __name__ == '__main__':
    main()
