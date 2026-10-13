"""一句话证据图：同一场景同一模型，唯一差别是那句文本报告。

两条共享时间轴的曲线（单事件 corridor-load episode）：
  上：注意力质量——模型把绑定压在"当前播报所指边"上的比例
      （播报迁移也如实呈现：先看先活跃事件，主事件接管后跟到主事件边）；
      谎报臂画"谎报边"上的质量；无文本臂画事件边质量作对照。
  下：事件边排队——真/谎/无三臂，图例内附积分。

用法::

    python -m scene_attention.report_effect_fig \
        --policy output_data/dev_takeover_probe/sig15_corrload/agent.pt \
        --out output_data/dev_takeover_probe/evidence_fig/report_effect.png
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from scene_attention.probe.takeover import build_world  # noqa: E402
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention.signal_model import SceneJointPolicy  # noqa: E402
from scene_attention.env import run_signal_episode  # noqa: E402
from scene_attention.eval import spec_list  # noqa: E402
from scene_attention._legacy.router.train_router import (  # noqa: E402
    TRAIN_EDGES,
)
from hitl.live.injects import parse_injects  # noqa: E402

_EDGE_RE = re.compile(r'road_\d+_\d+_\d+')


def lane_idx(fb, edge):
    return sorted(fb.edge_to_laneidx.get(edge, []))


def series(records, fb, lanes_idx):
    names = [fb.lanes[i] for i in lanes_idx]
    ts = np.array([r['t'] for r in records], dtype=np.float64)
    qs = np.array([
        sum(rec['lane_queue'].get(l, 0) for l in names)
        for rec in records], dtype=np.float64)
    return ts, qs


def corridor_idx(fb, edge):
    """Watch set of a named edge: itself + feeder + discharge + 2-hop
    metering-source approaches (same topology derivation as eval)."""
    pred, succ = fb._edge_pred, fb._edge_succ
    edges = ({edge} | pred.get(edge, set()) | succ.get(edge, set())
             | {s2 for f in pred.get(edge, set())
                for s2 in pred.get(f, set())})
    return sorted(i for e in edges for i in fb.edge_to_laneidx.get(e, []))


def attn_mass_on(records, fb, edge=None, dynamic=False):
    """Mass on a fixed edge's corridor, or on whatever corridor the
    tick's report names (dynamic=True — parses the edge out of the
    rendered sentence)."""
    out, ts = [], []
    for rec in records:
        w = rec.get('attn_weights')
        if w is None:
            continue
        w = np.asarray(w).ravel()
        if dynamic:
            m = _EDGE_RE.search(rec.get('text') or '')
            if not m:
                continue
            idx = corridor_idx(fb, m.group(0))
        else:
            idx = corridor_idx(fb, edge)
        ts.append(rec['t'])
        out.append(float(w[idx].sum()) if idx else 0.0)
    return np.asarray(ts), np.asarray(out)


def _smooth(y, w=6):
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--policy', required=True)
    p.add_argument('--event', default='surge:road_1_0_1:road_1_1_2:300:600:4')
    p.add_argument('--no-corridor', action='store_true',
                   help='不挂静默 feeder surge + discharge throttle '
                        '（注意：无下游 throttle 时事件边不积队）')
    p.add_argument('--lie-edge', default=None,
                   help='谎报边；缺省取 TRAIN_EDGES 中第一条非事件边')
    p.add_argument('--seed', type=int, default=9004)
    p.add_argument('--report-delay', type=float, default=30)
    p.add_argument('--episode-seconds', type=int, default=1500)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--runtime-dir', default='/tmp/report_effect_fig')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    ev_edge = args.event.split(':')[1]
    lie_edge = args.lie_edge or next(e for e in TRAIN_EDGES if e != ev_edge)
    onset = float(args.event.split(':')[3])
    end_t = onset + float(args.event.split(':')[4])
    report_t = onset + args.report_delay

    policy, payload = SceneJointPolicy.load(args.policy)
    fb_kw = fb_kwargs_from_meta(payload.get('meta'))

    world = build_world(
        args.scene, args.seed, Path(args.runtime_dir),
        Path(args.runtime_dir) / 'logger',
        {'t_fixed': 30, 't_min': 10,
         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    ranks = list(range(len(world.intersection_ids)))

    recs, fbs = {}, {}
    for arm, override in [('true', None),
                          ('lie', {'edge': lie_edge}),
                          ('none', {'kind': 'none'})]:
        injects = parse_injects(
            spec_list(args.event, corridor=not args.no_corridor))
        fb = FeatureBuilder(world, injects, report_delay=args.report_delay,
                            **fb_kw)
        _, rec = run_signal_episode(
            world, fb, ranks, injects=injects, agent=policy,
            episode_seconds=args.episode_seconds, interval=args.interval,
            text_override=override)
        recs[arm], fbs[arm] = rec, fb
        world.reset()
    world.close()

    fb = fbs['true']
    ev_idx = lane_idx(fb, ev_edge)
    lie_idx = lane_idx(fb, lie_edge)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                             gridspec_kw={'height_ratios': [1, 1.25]})

    # ---- top: binding mass follows the edge the report names ----
    ax = axes[0]
    t, m = attn_mass_on(recs['true'], fb, dynamic=True)
    ax.plot(t, _smooth(m), color='#2ca02c', lw=2.2,
            label='true report — mass on the edge it currently names')
    # records['text'] logs the TRUE sentence; under the lie override the
    # model was shown lie_edge — plot its mass on that edge.
    t, m = attn_mass_on(recs['lie'], fb, edge=lie_edge)
    ax.plot(t, _smooth(m), color='#ff7f0e', lw=2.2,
            label=f'lie report — mass on "{lie_edge}"')
    t, m = attn_mass_on(recs['none'], fb, edge=ev_edge)
    ax.plot(t, _smooth(m), color='gray', lw=1.6, ls='--',
            label=f'no text — mass on {ev_edge}')
    ax.set_ylabel('binding mass on the\ncorridor the report names')
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, loc='center left')

    # ---- bottom: queue on the reported event edge ----
    ax = axes[1]
    for arm, color, lab in [
            ('true', '#2ca02c', 'true report'),
            ('lie', '#ff7f0e', f'lie report ({lie_edge})'),
            ('none', 'gray', 'no text')]:
        t, q = series(recs[arm], fb, ev_idx)
        qint = q.sum() * args.interval / 1000
        ax.plot(t, q, color=color, lw=0.5, alpha=0.3)
        ax.plot(t, _smooth(q), color=color, lw=2.2,
                label=f'{lab}  (∫q = {qint:.1f}k veh·s)')
    ax.set_ylabel(f'queue on reported edge\n({ev_edge})')
    ax.set_xlabel('sim time (s)')
    ax.legend(fontsize=8, loc='upper left')

    for ax in axes:
        ax.axvspan(onset, end_t, color='red', alpha=0.06)
        ax.axvline(report_t, color='k', ls='--', lw=1.2)
        ax.annotate('main-event report', (report_t + 12, 0.03),
                    xycoords=('data', 'axes fraction'), fontsize=8,
                    rotation=90, va='bottom', color='k')
    axes[0].set_title(
        'Identical traffic, identical model — only the report text differs\n'
        f'(corridor-load scenario, surge on {ev_edge} {int(onset)}–'
        f'{int(end_t)}s + silent feeder surge + discharge throttle)',
        fontsize=9.5)
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
