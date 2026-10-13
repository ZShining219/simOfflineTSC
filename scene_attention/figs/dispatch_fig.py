"""Dispatch-gate evidence figures: one traced episode per arm
(per-tick queue + cross-attention weights + takeover audit) ->

  fig_queue.png   governance effect: queue curves with takeover window
  fig_bars.png    overall governance summary (paired-gate aggregates)
  fig_attn.png    scene-attention mechanism: text->lane weight heatmap
                  (attn arm vs blinded arm) + dispatch-key mass curve

Exploratory figure generator (dev scope); data sources:
  this run's traces   -> curves / heatmaps
  dispatch_gate_v1    -> aggregate bars
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import (  # noqa: E402
    DEFAULT_DQN_S1, build_controllers, build_world, _bind_all,
)
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention.signal_model import SceneJointPolicy  # noqa: E402
from scene_attention.env import (  # noqa: E402
    run_signal_episode,
)
from scene_attention.dispatch_specialist import (  # noqa: E402
    KeyDispatchAgent, wrap_controller,
)
from scene_attention.probe.dispatch_eval import (  # noqa: E402
    _EncoderShim, SIG9,
)
from hitl.live.injects import parse_injects  # noqa: E402


class _SAShim(_EncoderShim):
    """Encoder shim that also exposes lane×lane self-attention:
    ``last_sa`` = per-lane received attention mass (column mean of the
    LxL matrix) — the traffic-responsive layer that runs even without
    a report, complementing the text->lane binding weight w."""

    def encode(self, batch):
        enc = self.policy.encoder
        kv0 = enc.lane_proj(batch['lanes'])
        _, sa_w = enc.self_attn(kv0, kv0, kv0, need_weights=True)
        self.last_sa = sa_w[0].detach().cpu().numpy().mean(axis=0)
        return enc(batch['lanes'], batch['text'], batch['global'])


def run_arms(world, fb_meta, spec, ranks, controllers, specialist,
             binder, args):
    """One traced episode per arm on identical physics."""
    traces = {}
    # dispatch_attn / dispatch_blind
    for name, override in (('dispatch_attn', None),
                           ('dispatch_blind', {'kind': 'none'})):
        injects = parse_injects([spec])
        _bind_all(world, controllers, ranks,
                  needed=set(controllers) - {'dqn_s1'})
        controllers['dqn_s1'] = controllers['dqn_s1'].bind(
            world, ranks=ranks)
        fb = FeatureBuilder(world, injects,
                            report_delay=args.report_delay, **fb_meta)
        disp = KeyDispatchAgent(
            binder, fb, world,
            base_act=wrap_controller(controllers['dqn_s1']),
            registry={'congest': specialist}, bind_tau=args.bind_tau)
        _, recs = run_signal_episode(
            world, fb, ranks, injects, agent=disp,
            episode_seconds=args.episode_seconds,
            interval=args.interval, text_override=override)
        traces[name] = {'records': recs, 'audit': disp.audit,
                        'lanes': list(fb.lanes),
                        'edge_to_laneidx': {
                            k: list(v)
                            for k, v in fb.edge_to_laneidx.items()},
                        'event_edge': injects[0].get('edge_from')
                        or injects[0].get('edge')}
        world.reset()
    return traces


def fig_bars(gate_json, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    d = json.load(open(gate_json))
    arms = d['arms']
    order = ['dqn_s1', 'dispatch_attn']
    labels = ['base', 'Scene-Guided\nAttention']
    colors = ['#9e9e9e', '#1f6fb4']

    def pair_vals(metric):
        va, vb = [], []
        for ra, rb in zip(arms['dispatch_attn'], arms['dqn_s1']):
            if ra['event'] != rb['event']:
                continue
            xa, xb = ra[metric], rb[metric]
            if xa == xa and xb == xb:  # drop nan pairs (clear_s)
                va.append(xa)
                vb.append(xb)
        return float(np.mean(vb)), float(np.mean(va)), len(va)

    # normalized to base = 100% so bars read as 'optimized X%'
    panels = [
        ('ev_queue_int', 'Event-window queue\n(% of base)',
         'n=16 paired'),
        ('clear_s', 'Congestion clearance time\n(% of base)',
         'cells where BOTH cleared, n=9'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 4.2))
    for ax, (m, ttl, note) in zip(axes, panels):
        vb, va, n = pair_vals(m)
        norm = va / vb * 100
        ax.bar([0, 1], [100.0, norm], color=colors, width=0.5)
        ax.text(0, 101, '100%', ha='center', fontsize=9,
                color='#333333')
        ax.text(1, norm + 1, f'optimized\n{100 - norm:.1f}%',
                ha='center', fontsize=9.5, color='#1f6fb4',
                fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_title(ttl + f'\n({note})', fontsize=9)
        ax.set_ylim(0, 118)
        ax.grid(axis='y', alpha=0.2)
    fig.suptitle('Governance optimization vs deployed base '
                 '(dispatch_gate_v1, surge@4 edges)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out / 'fig_bars.png', dpi=160)
    plt.close(fig)


def fig_attn(traces, spec, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ev = parse_injects([spec])[0]
    start_t, end_t = ev['start_t'], ev['end_t']
    lanes = traces['dispatch_attn']['lanes']
    e2i = traces['dispatch_attn']['edge_to_laneidx']
    ev_edge = traces['dispatch_attn']['event_edge']

    fig, axes = plt.subplots(
        4, 1, figsize=(10, 9.6), sharex=True,
        gridspec_kw={'height_ratios': [2.4, 2.6, 2.6, 1.6]})
    # layer 1: lane self-attention — traffic-responsive, always on
    audit_sa = [r.get('sa_sal') for r in traces['dispatch_attn']['audit']]
    S = np.array([s if s is not None else np.zeros(len(lanes))
                  for s in audit_sa]).T  # lanes x ticks
    ts0 = [r['t'] for r in traces['dispatch_attn']['records']]
    im0 = axes[0].imshow(
        S, aspect='auto', origin='lower', cmap='inferno',
        interpolation='nearest', vmin=0.0, vmax=0.20,
        extent=[ts0[0], ts0[-1], -0.5, len(lanes) - 0.5])
    # shared edge grouping for all lane-indexed panels
    bounds, centers, names = [], [], []
    for edge, idx in sorted(e2i.items(), key=lambda kv: min(kv[1])):
        idx = sorted(idx)
        bounds.append(min(idx) - 0.5)
        centers.append(float(np.mean(idx)))
        names.append(edge)
    bounds.append(max(i for v in e2i.values() for i in v) + 0.5)

    def decorate(ax, im=None):
        for b in bounds[1:-1]:
            ax.axhline(b, color='w', lw=0.4, alpha=0.5)
        ax.set_yticks(centers)
        ax.set_yticklabels(names, fontsize=6.5)
        for lab, edge in zip(ax.get_yticklabels(), names):
            if edge == ev_edge:
                lab.set_color('#d62728')
                lab.set_fontweight('bold')
        for x in (start_t, end_t):
            ax.axvline(x, color='#2ca02c', lw=0.8, ls=':')
        if im is not None:
            fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)

    axes[0].set_title(
        'lane×lane self-attention — per-lane received mass\n'
        '(traffic-responsive layer, no report needed)', fontsize=9)
    decorate(axes[0], im0)
    for ax, arm, ttl in (
            (axes[1], 'dispatch_attn',
             'text->lane attention (report-driven; color clipped'
             ' at 0.20)'),
            (axes[2], 'dispatch_blind',
             'text->lane attention (blinded text; same scale)')):
        vmax = 0.20  # shared scale across both panels
        recs = traces[arm]['records']
        ts = [r['t'] for r in recs]
        W = np.array([np.asarray(r['attn_weights']).ravel()
                      if r['attn_weights'] is not None
                      else np.zeros(len(lanes))
                      for r in recs]).T  # lanes x ticks
        im = ax.imshow(W, aspect='auto', origin='lower',
                       cmap='inferno', interpolation='nearest',
                       vmin=0.0, vmax=vmax,
                       extent=[ts[0], ts[-1], -0.5, len(lanes) - 0.5])
        ax.set_title(ttl, fontsize=9)
        decorate(ax, im)
    # all-edge attention mass bundle: the reported edge separates
    # from the pack during the report window — same evidence as the
    # heatmap, shown as dynamics rather than a binary wall
    ax = axes[3]
    recs_a = traces['dispatch_attn']['records']
    tt = [r['t'] for r in recs_a]
    masses = {e: [] for e in e2i}
    for r in recs_a:
        w = np.asarray(r['attn_weights']).ravel() \
            if r['attn_weights'] is not None else np.zeros(len(lanes))
        for e, idx in e2i.items():
            masses[e].append(float(w[np.asarray(idx)].sum()))
    for e, mv in masses.items():
        if e == ev_edge:
            continue
        ax.plot(tt, mv, color='#c8c8c8', lw=0.9, alpha=0.8, zorder=2)
    ax.plot(tt, masses[ev_edge], color='#1f6fb4', lw=1.8, zorder=4,
            label=f'reported edge ({ev_edge})')
    ax.plot([], [], color='#c8c8c8', lw=0.9, label='other edges')
    audit_b = traces['dispatch_blind']['audit']
    ax.plot([r['tick'] * 10 for r in audit_b],
            [r['mass'] for r in audit_b],
            color='#7f7f7f', lw=1.0, ls='--', alpha=0.9, zorder=3,
            label='bound-edge mass (blinded)')
    ax.axhline(0.4, color='#7f7f7f', ls='--', lw=1,
               label='dispatch tau=0.4')
    audit = traces['dispatch_attn']['audit']
    ch = [r['choice'] == 'specialist' for r in audit]
    ax.fill_between(tt, 0, 1, where=ch, color='#1f6fb4', alpha=0.15,
                    label='specialist takeover')
    for x in (start_t, end_t):
        ax.axvline(x, color='#d62728', lw=0.8, ls=':')
    ax.set_ylim(0, 1.0)
    ax.set_xlabel('simulation time (s)')
    ax.set_title('per-edge attention mass & takeover', fontsize=9)
    ax.legend(fontsize=7, loc='lower right')
    ax.grid(alpha=0.2)
    fig.suptitle(
        f'Scene-attention | {spec} | reported edge {ev_edge} (red)\n'
        f'L1 self-attention tracks traffic; L2 text->lane binding'
        f' concentrates on the reported edge', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'fig_attn.png', dpi=160)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--event',
                   default='surge:road_2_1_2:road_1_2_1:300:600:4')
    p.add_argument('--seed', type=int, default=9009)
    p.add_argument('--specialist', default=SIG9)
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--bind-tau', type=float, default=0.4)
    p.add_argument('--report-delay', type=float, default=0.0)
    p.add_argument('--episode-seconds', type=int, default=1800)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--gate-json',
                   default='output_data/dev_takeover_probe/'
                           'dispatch_gate_v1/results.json')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    world = build_world(args.scene, args.seed,
                        Path('/tmp/takeover_dispatch_fig'),
                        out / 'logger',
                        {'t_fixed': args.t_fixed, 't_min': args.t_min,
                         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    ranks = list(range(len(world.intersection_ids)))
    controllers = build_controllers(dqn_s1_path=args.dqn_s1)
    specialist, s_meta = SceneJointPolicy.load(args.specialist)
    binder = _SAShim(specialist)
    fb_meta = fb_kwargs_from_meta(s_meta.get('meta'))

    traces = run_arms(world, fb_meta, args.event, ranks, controllers,
                      specialist, binder, args)
    world.close()
    (out / 'traces.json').write_text(json.dumps(
        {k: {'records': [{kk: vv for kk, vv in r.items()}
                         for r in v['records']],
             'audit': v.get('audit'),
             'lanes': v.get('lanes'),
             'edge_to_laneidx': v.get('edge_to_laneidx'),
             'event_edge': v.get('event_edge')}
         for k, v in traces.items()}, allow_nan=True))
    fig_bars(Path(args.gate_json), out)
    fig_attn(traces, args.event, out)
    print('figs ->', out)


if __name__ == '__main__':
    main()
