"""介入前后效果图：同一 episode 上，场景引导臂 vs 盲化/vanilla 底座的
被报边排队曲线。

用法::

    python -m dev.takeover_probe.plot_scene_effect \
        --event surge:road_1_0_1:road_1_1_2:300:600:4 --seed 9004 \
        --encoder output_data/dev_takeover_probe/sig5_sga/agent.pt \
        --out /tmp/scene_effect.png

两条曲线：vanilla 底座（等价于 notext 消融——绑不上播报时逐位相同）
与场景引导臂；竖线标播报到达时刻，阴影为事件窗口。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np

from dev.takeover_probe.probe import DEFAULT_DQN_S1, build_world  # noqa: E402
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.signal_model import SceneSignalAgent  # noqa: E402
from dev.takeover_probe.scene_mp import (  # noqa: E402
    SceneGuidedDQN, SceneGuidedMP, bound_dqn_agent,
)
from dev.takeover_probe.signal_runtime import run_signal_episode  # noqa: E402
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402
from tools.sumo_live_broadcast.controllers import (  # noqa: E402
    _build_inference_agent, load_qnet_payload,
)


def edge_queue_series(records, lane_names):
    ts = [r['t'] for r in records]
    qs = [sum(r['lane_queue'].get(l, 0) for l in lane_names)
          for r in records]
    return np.asarray(ts), np.asarray(qs)


def _smooth(y, w=3):
    if w <= 1:
        return y
    k = np.ones(w) / w
    return np.convolve(y, k, mode='same')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--event', required=True)
    p.add_argument('--seed', type=int, default=9004)
    p.add_argument('--encoder', required=True,
                   help='sga encoder agent.pt（Q 头不用，只用编码器）')
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--beta', type=float, default=40.0)
    p.add_argument('--gate', type=float, default=8.0)
    p.add_argument('--bind-tau', type=float, default=0.4)
    p.add_argument('--episode-seconds', type=int, default=1500)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--runtime-dir', default='/tmp/scene_effect_plot')
    p.add_argument('--out', required=True)
    args = p.parse_args()

    inj = parse_injects([args.event])[0]
    event_edge = inj['edge_from']
    onset, end_t = inj['start_t'], inj['end_t']

    world = build_world(
        args.scene, args.seed, Path(args.runtime_dir),
        Path(args.runtime_dir) / 'logger',
        {'t_fixed': 30, 't_min': 10,
         'min_green_vehicle': 3, 'max_red_vehicle': 6})
    ranks = list(range(len(world.intersection_ids)))
    enc, payload = SceneSignalAgent.load(args.encoder)
    fb_kw = fb_kwargs_from_meta(payload.get('meta'))
    dqn_payload = load_qnet_payload(args.dqn_s1)

    curves = {}
    arms = [
        ('scenedqn_sga', 'adapter+dqn', None),
        ('dqn_s1', 'vanilla_dqn', None),
        ('scenemp_sga', 'adapter+mp', None),
        ('maxpressure', 'vanilla_mp', None),
    ]
    for tag, kind, _ in arms:
        injects = parse_injects([args.event])
        fb = FeatureBuilder(world, injects, **fb_kw)
        if kind == 'adapter+dqn':
            dqn = _build_inference_agent(world, dqn_payload, rank=ranks[0])
            driver = SceneGuidedDQN(enc, fb, world, dqn, beta=args.beta,
                                    t_min=0, gate=args.gate, fade='in',
                                    bind_tau=args.bind_tau)
            _, rec = run_signal_episode(world, fb, ranks, injects,
                                        agent=driver,
                                        episode_seconds=args.episode_seconds,
                                        interval=args.interval)
        elif kind == 'adapter+mp':
            driver = SceneGuidedMP(enc, fb, world, beta=args.beta,
                                   t_min=10, gate=args.gate, fade='in',
                                   bind_tau=args.bind_tau)
            _, rec = run_signal_episode(world, fb, ranks, injects,
                                        agent=driver,
                                        episode_seconds=args.episode_seconds,
                                        interval=args.interval)
        elif kind == 'vanilla_dqn':
            dqn = _build_inference_agent(world, dqn_payload, rank=ranks[0])
            _, rec = run_signal_episode(
                world, fb, ranks, injects,
                controller=_DqnCtrl(dqn),
                episode_seconds=args.episode_seconds,
                interval=args.interval)
        else:
            from dev.takeover_probe.probe import build_controllers, _bind_all
            ctrls = build_controllers()
            _bind_all(world, ctrls, ranks, needed={'maxpressure'})
            _, rec = run_signal_episode(
                world, fb, ranks, injects,
                controller=ctrls['maxpressure'],
                episode_seconds=args.episode_seconds,
                interval=args.interval)
        lanes = [fb.lanes[i] for i in fb.edge_to_laneidx.get(event_edge, [])]
        curves[tag] = edge_queue_series(rec, lanes)
        world.reset()
    world.close()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    panels = [
        ('DQN base (dqn_s1)', 'dqn_s1', 'scenedqn_sga'),
        ('MaxPressure base', 'maxpressure', 'scenemp_sga'),
    ]
    for ax, (title, base_tag, sga_tag) in zip(axes, panels):
        ax.axvspan(onset, end_t, color='orange', alpha=0.08)
        ax.axvline(onset, color='orange', ls='--', lw=1,
                   label='report: road congest')
        tb, qb = curves[base_tag]
        ts_, qs = curves[sga_tag]
        ax.plot(tb, qb, color='gray', lw=0.7, alpha=0.35)
        ax.plot(ts_, qs, color='tab:blue', lw=0.7, alpha=0.35)
        ax.plot(tb, _smooth(qb), color='gray', lw=2.0,
                label=f'vanilla ({base_tag})')
        ax.plot(ts_, _smooth(qs), color='tab:blue', lw=2.0,
                label=f'scene-guided ({sga_tag})')
        ax.set_title(title)
        ax.set_xlabel('sim time (s)')
        ax.legend(fontsize=8, loc='upper right')
    axes[0].set_ylabel(f'queue on reported edge ({event_edge})')
    fig.suptitle(f'{args.event}  seed={args.seed}')
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f'wrote {args.out}')


class _DqnCtrl:
    """Wrap a bound inference agent as a decide(rank) controller."""

    def __init__(self, agent):
        self.agent = agent

    def decide(self, rank=0):
        from tools.sumo_live_broadcast.controllers import _inference_action
        return _inference_action(self.agent)


if __name__ == '__main__':
    main()
