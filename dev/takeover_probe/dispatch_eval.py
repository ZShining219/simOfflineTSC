"""Dispatch-gate eval: report -> attention binding key -> specialist
takeover, on the shared surge grid, against the usual baselines.

Arms (all identical physics per grid cell, ``text_override`` variants
carry the counterfactuals):

  dispatch_attn   encoder binding key -> specialist (registry['congest'])
                  else base.  THE demo arm: scene text drives an explicit
                  routing decision, not just a hidden bias.
  dispatch_oracle same dispatch with the key taken from fb.event_fields —
                  perfect report channel; the discrimination upper bound.
  dispatch_blind  dispatch_attn under text_override {'kind':'none'} —
                  no report -> key never fires -> must track the base
                  controller bit-for-bit in behaviour.
  dispatch_mp     specialist = text-free rule variant (SceneGuidedMP,
                  direction='queue', bind_gate=False) on an MP base —
                  the dispatch is the specialist's ONLY scene channel.
  joint_always    the specialist policy always controls (no dispatch
                  layer — isolates what the routing adds on top).
  fixedtime / maxpressure / dqn_s1   conventional references.

Usage:
    python -m dev.takeover_probe.dispatch_eval \
        --out output_data/dev_takeover_probe/dispatch_gate_v1
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

from dev.takeover_probe.probe import (  # noqa: E402
    DEFAULT_DQN_S1, build_controllers, build_world, _bind_all,
)
from dev.takeover_probe.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from dev.takeover_probe.signal_model import SceneJointPolicy  # noqa: E402
from dev.takeover_probe.scene_mp import SceneGuidedMP  # noqa: E402
from dev.takeover_probe.eval_signal import eval_episode  # noqa: E402
from dev.takeover_probe.dispatch_specialist import (  # noqa: E402
    KeyDispatchAgent, wrap_controller,
)
from dev.takeover_probe._router.train_router import (  # noqa: E402
    EDGE_TARGETS, TRAIN_EDGES,
)
from tools.sumo_live_broadcast.injects import parse_injects  # noqa: E402

SIG9 = 'output_data/dev_takeover_probe/sig9_joint_bc/agent.pt'


class _EncoderShim:
    """Expose a policy's encoder behind the .encode(batch) contract."""

    def __init__(self, policy):
        self.policy = policy
        self.encoder = getattr(policy, 'encoder', None)
        self.variant = getattr(policy, 'variant', None)

    def encode(self, batch):
        return self.policy.encoder(
            batch['lanes'], batch['text'], batch['global'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--specialist', default=SIG9,
                   help='joint ckpt used as the congest specialist')
    p.add_argument('--binder', default=None,
                   help='ckpt whose encoder produces the dispatch key '
                        '(default: the specialist itself)')
    p.add_argument('--bind-tau', type=float, default=0.4)
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--onsets', default='300')
    p.add_argument('--durs', default='600')
    p.add_argument('--periods', default='4')
    p.add_argument('--episodes-per-event', type=int, default=4)
    p.add_argument('--episode-seconds', type=int, default=1800)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--report-delay', type=float, default=0.0)
    p.add_argument('--traffic-seed-base', type=int, default=9000)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--clear-thresh', type=float, default=2.0)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    grids = [f'surge:{e}:{EDGE_TARGETS[e]}:{o}:{d}:{per}'
             for e in TRAIN_EDGES
             for o in (int(x) for x in args.onsets.split(',') if x)
             for d in (int(x) for x in args.durs.split(',') if x)
             for per in (int(x) for x in args.periods.split(',') if x)]

    specialist, s_meta = SceneJointPolicy.load(args.specialist)
    binder_src = SceneJointPolicy.load(args.binder)[0] \
        if args.binder else specialist
    binder = _EncoderShim(binder_src)
    fb_meta = fb_kwargs_from_meta(s_meta.get('meta'))

    results = {}
    audits = {}
    ep_counter = 0
    for spec in grids:
        for rep in range(args.episodes_per_event):
            ep_counter += 1
            seed = args.traffic_seed_base + ep_counter
            world = build_world(args.scene, seed, Path(
                args.runtime_dir or '/tmp/takeover_dispatch_eval'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ranks = list(range(len(world.intersection_ids)))
            controllers = build_controllers(dqn_s1_path=args.dqn_s1)

            # ---- dispatch arms ----------------------------------------
            for name, oracle, override in (
                    ('dispatch_attn', False, None),
                    ('dispatch_oracle', True, None),
                    ('dispatch_blind', False, {'kind': 'none'})):
                injects = parse_injects([spec])
                _bind_all(world, controllers, ranks,
                          needed=set(controllers) - {'dqn_s1'})
                controllers['dqn_s1'] = controllers['dqn_s1'].bind(
                    world, ranks=ranks)
                fb = FeatureBuilder(world, injects,
                                    report_delay=args.report_delay,
                                    **fb_meta)
                disp = KeyDispatchAgent(
                    binder, fb, world,
                    base_act=wrap_controller(controllers['dqn_s1']),
                    registry={'congest': specialist},
                    bind_tau=args.bind_tau, oracle=oracle)
                m = eval_episode(
                    world, fb, ranks, injects, agent=disp,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval, text_override=override,
                    clear_thresh=args.clear_thresh)
                m['event'] = spec
                results.setdefault(name, []).append(m)
                audits.setdefault(name, []).append(
                    {'event': spec, 'rep': rep, 'rows': disp.audit})
                world.reset()

            # ---- text-free rule specialist (dispatch is its only
            #      scene channel) --------------------------------------
            mp_spec = None
            injects = parse_injects([spec])
            _bind_all(world, controllers, ranks,
                      needed=set(controllers) - {'dqn_s1'})
            controllers['dqn_s1'] = controllers['dqn_s1'].bind(
                world, ranks=ranks)
            fb = FeatureBuilder(world, injects,
                                report_delay=args.report_delay, **fb_meta)
            mp_spec = SceneGuidedMP(
                binder, fb, world, bind_gate=False, direction='queue',
                t_min=args.t_min)
            disp = KeyDispatchAgent(
                binder, fb, world,
                base_act=wrap_controller(controllers['maxpressure']),
                registry={'congest': mp_spec}, bind_tau=args.bind_tau)
            m = eval_episode(
                world, fb, ranks, injects, agent=disp,
                episode_seconds=args.episode_seconds,
                interval=args.interval,
                clear_thresh=args.clear_thresh)
            m['event'] = spec
            results.setdefault('dispatch_mp', []).append(m)
            audits.setdefault('dispatch_mp', []).append(
                {'event': spec, 'rep': rep, 'rows': disp.audit})
            world.reset()

            # ---- controls ---------------------------------------------
            injects = parse_injects([spec])
            fb = FeatureBuilder(world, injects,
                                report_delay=args.report_delay, **fb_meta)
            m = eval_episode(
                world, fb, ranks, injects, agent=specialist,
                episode_seconds=args.episode_seconds,
                interval=args.interval,
                clear_thresh=args.clear_thresh)
            m['event'] = spec
            results.setdefault('joint_always', []).append(m)
            world.reset()

            for name in ('fixedtime', 'maxpressure', 'dqn_s1'):
                injects = parse_injects([spec])
                _bind_all(world, controllers, ranks,
                          needed=set(controllers) - {'dqn_s1'})
                controllers['dqn_s1'] = controllers['dqn_s1'].bind(
                    world, ranks=ranks)
                fb0 = FeatureBuilder(world, injects,
                                     report_delay=args.report_delay)
                m = eval_episode(
                    world, fb0, ranks, injects,
                    controller=controllers[name],
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                m['event'] = spec
                results.setdefault(name, []).append(m)
                world.reset()

            world.close()
            print(f'[{ep_counter}] {spec} rep{rep} done', flush=True)

    summary = {}
    for name, rows in results.items():
        clears = [r['clear_s'] for r in rows if r['clear_s'] == r['clear_s']]
        summary[name] = {
            'n': len(rows),
            'queue_int': round(float(np.mean([r['queue_int']
                                              for r in rows])), 1),
            'ev_queue_int': round(float(np.mean([r['ev_queue_int']
                                                 for r in rows])), 1),
            'peak_queue': round(float(np.mean([r['peak_queue']
                                               for r in rows])), 1),
            'clear_s': (round(float(np.mean(clears)), 1)
                        if clears else None),
            'clear_n': len(clears),
            'throughput': round(float(np.mean([r['throughput']
                                               for r in rows])), 1),
            'bind_mass': (round(float(np.mean(
                [r['bind_mass'] for r in rows
                 if r['bind_mass'] is not None])), 4)
                if any(r['bind_mass'] is not None for r in rows)
                else None),
        }
    (out / 'results.json').write_text(
        json.dumps({'arms': results, 'summary': summary}, indent=2,
                   allow_nan=True))
    (out / 'dispatch_audit.json').write_text(
        json.dumps(audits, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
