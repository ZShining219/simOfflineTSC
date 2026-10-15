"""Event-scenario evaluation for scene-guided signal agents.

Compares, under an identical congestion-event grid:

  learned arms   scene-guided agents (sga/attn ckpts); the
                 selfattn/concat/*_st comparison arms were removed —
                 recover from git history if their ckpts must reload
  ablation       the same sga ckpt with the text channel zeroed
                 ('no_text' — attention module present but blind)
  conventional   fixedtime / maxpressure / sotl / dqn_s1 controllers

Metrics per episode:
  queue_int      integral of total queue over the episode (veh·s)
  ev_queue_int   integral over the report-named edge's lanes
  peak_queue     max total queue
  clear_s        seconds from event end until the named edge's queue
                 returns under ``clear_thresh`` (nan = never cleared)
  throughput     vehicles completed
  bind_mass      mean cross-attn mass on the named edge's lanes while
                 the report is active (agent arms only)

Usage:
    python -m scene_attention.eval \
        --agent sga=out_sga/agent.pt --agent attn=out_attn/agent.pt \
        --builtin maxpressure,fixedtime --episodes-per-event 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_attention.probe.takeover import (  # noqa: E402
    DEFAULT_DQN_S1, build_controllers, build_world, _bind_all,
)
from scene_attention.dispatch_state import (  # noqa: E402
    FeatureBuilder, fb_kwargs_from_meta,
)
from scene_attention.signal_model import (  # noqa: E402
    SceneJointPolicy, SceneResidualPolicy, SceneSignalAgent,
)
from scene_attention.scene_mp import (  # noqa: E402
    SceneGuidedDQN, SceneGuidedMP, bound_dqn_agent,
)
from scene_attention.env import (  # noqa: E402
    run_signal_episode,
)
from scene_attention.events import (  # noqa: E402
    DISCHARGE_OF, DISTRACTOR_OF, EDGE_TARGETS, FEEDER_OF,
    TRAIN_EDGES,
)
from hitl.live.controllers import (  # noqa: E402
    inference_q_values,
)
from hitl.live.injects import (  # noqa: E402
    apply_injects, parse_injects,
)


class _JointBaseOnly:
    """Inner base net of a joint ckpt run alone — answers whether the
    attention-guided training left value inside the controller itself
    (internalisation check: base_only vs original dqn_s1)."""

    def __init__(self, policy):
        self.p = policy

    def act(self, batch, epsilon=0.0, rng=None, q_base=None):
        q = self.p.base(self.p.base_features(batch))
        return int(q.argmax(dim=-1).item()), \
            q.detach().cpu().numpy()[0].tolist(), None



def spec_list(spec, corridor=False, distractor=False):
    """Primary spec + optional unreported feeder-load surge."""
    if not spec:
        return []
    specs = [spec]
    if distractor:
        d = distractor_spec(spec)
        if d:
            specs.append(d)
    if corridor:
        parts = spec.split(':')
        if parts[0] == 'surge':
            edge, onset, dur, period = (parts[1], float(parts[3]),
                                        float(parts[4]), float(parts[5]))
            feeder = FEEDER_OF.get(edge)
            if feeder:
                specs.append(
                    f'surge:{feeder}:{edge}:{max(0.0, onset - 120)}:'
                    f'{dur + 120}:{period + 3}:global')
            discharge = DISCHARGE_OF.get(edge)
            if discharge:
                specs.append(
                    f'throttle:{discharge}:{max(0.0, onset - 60)}:'
                    f'{dur + 60}:3.0:global')
    return specs


def distractor_spec(spec):
    """Unreported surge on an edge OUTSIDE the event's watch set —
    a physically real but report-irrelevant competitor."""
    parts = spec.split(':')
    if parts[0] != 'surge':
        return None
    edge, onset, dur = parts[1], float(parts[3]), float(parts[4])
    dist = DISTRACTOR_OF.get(edge)
    target = EDGE_TARGETS.get(dist)
    if not dist or not target:
        return None
    return (f'surge:{dist}:{target}:{max(0.0, onset - 120)}:'
            f'{dur + 120}:5:global')

def eval_episode(world, fb, ranks, injects, agent=None, controller=None,
                 episode_seconds=3600, interval=10, text_override=None,
                 clear_thresh=2.0, q_base_fn=None):
    transitions, records = run_signal_episode(
        world, fb, ranks, injects=injects, agent=agent,
        controller=controller, interval=interval,
        episode_seconds=episode_seconds, text_override=text_override,
        q_base_fn=q_base_fn)
    ev = injects[0] if injects else None
    ev_edge = (ev.get('edge_from') or ev.get('edge') or '') if ev else ''
    ev_lanes = set(fb.edge_to_laneidx.get(ev_edge, []))
    end_t = ev['end_t'] if ev else episode_seconds
    queues = np.array([r['total_queue'] for r in records],
                      dtype=np.float64)
    ev_q = np.array([
        sum(rec['lane_queue'].get(fb.lanes[i], 0) for i in ev_lanes)
        for rec in records], dtype=np.float64)
    ts = np.array([r['t'] for r in records], dtype=np.float64)
    clear_s = float('nan')
    if ev is not None:
        post = [(t, q) for t, q in zip(ts, ev_q) if t >= end_t]
        for t, q in post:
            if q <= clear_thresh:
                clear_s = float(t - end_t)
                break
    # binding mass on the named edge while its report is active
    bind = []
    if ev is not None and ev_lanes:
        onset = ev['start_t'] + fb.report_delay
        idx = sorted(ev_lanes)
        for rec in records:
            w = rec.get('attn_weights')
            if w is not None and onset <= rec['t'] <= end_t:
                w = np.asarray(w).ravel()
                bind.append(float(w[idx].sum()))
    out = {
        'queue_int': float(queues.sum() * interval),
        'ev_queue_int': float(ev_q.sum() * interval),
        'peak_queue': float(queues.max() if len(queues) else 0.0),
        'clear_s': clear_s,
        'throughput': int(records[-1]['throughput']) if records else 0,
        'bind_mass': (round(float(np.mean(bind)), 4) if bind else None),
        'ticks': len(records),
    }
    # corridor-specific integrals: queue on the coordination legs —
    # feeder link, discharge links, and the metering-source approaches
    # (2-hop watch set).  Lets us see whether governance relieves or
    # merely relocates corridor pressure.
    if ev is not None:
        # topology-derived watch set — independent of the model's
        # intent_coord_w, so the metric is comparable across arms
        _pred, _succ = fb._edge_pred, fb._edge_succ
        watch_edges = sorted(
            (_pred.get(ev_edge, set()) | _succ.get(ev_edge, set())
             | {s2 for f in _pred.get(ev_edge, set())
                for s2 in _pred.get(f, set())})
            - {ev_edge})
        coord_idx = sorted(
            i for e in watch_edges
            for i in fb.edge_to_laneidx.get(e, []))
        if coord_idx:
            out['corr_queue_int'] = float(sum(
                sum(rec['lane_queue'].get(fb.lanes[i], 0)
                    for i in coord_idx)
                for rec in records) * interval)
            bind_c = []
            if ev is not None:
                onset = ev['start_t'] + fb.report_delay
                for rec in records:
                    w = rec.get('attn_weights')
                    if w is not None and onset <= rec['t'] <= end_t:
                        w = np.asarray(w).ravel()
                        bind_c.append(float(w[coord_idx].sum()))
            if bind_c:
                out['corr_bind_mass'] = round(float(np.mean(bind_c)), 4)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene', default='S2')
    p.add_argument('--agent', action='append', default=[],
                   help='NAME=PATH to an agent.pt checkpoint')
    p.add_argument('--scenemp', action='append', default=[],
                   help='NAME=PATH — SceneGuidedMP arm: encoder attention '
                        'reweights maxpressure (encoder from agent.pt, '
                        'Q head unused)')
    p.add_argument('--scenemp-beta', type=float, default=40.0)
    p.add_argument('--scenemp-gate', type=float, default=8.0,
                   help='queue scale for the attention-bonus fade')
    p.add_argument('--scenemp-fade', choices=['out', 'in', 'band'], default='in',
                   help='out = bonus fades as queue appears (anticipatory '
                        'gating, beta<0); in = bonus ramps with queue '
                        '(preferential service at wave arrival)')
    p.add_argument('--scenemp-direction',
                   choices=['auto', 'queue', 'count', 'qmix', 'attn', 'binding'],
                   default='auto')
    p.add_argument('--scenemp-bind-tau', type=float, default=0.4,
                   help='edge-mass threshold for the binding gate '
                        '(scene arms only)')
    p.add_argument('--scenemp-t-min', type=float, default=10.0)
    p.add_argument('--scenemp-notext', default=None,
                   help='PATH to agent.pt — scenemp arm with the report '
                        'channel blinded (kind=none), ablation control')
    p.add_argument('--scenemp-dose-frac', type=float, default=0.35)
    p.add_argument('--scenedqn', action='append', default=[],
                   help='NAME=PATH — SceneGuidedDQN arm: same attention '
                        'bias on the frozen DQN base (--scenedqn-dqn)')
    p.add_argument('--scenedqn-dqn', default=None,
                   help='DQN weights for scenedqn arms '
                        '(default: --dqn-s1)')
    p.add_argument('--scenedqn-notext', default=None,
                   help='PATH to agent.pt — scenedqn arm with the report '
                        'channel blinded (kind=none), ablation control')
    p.add_argument('--scenedqn-t-min', type=float, default=0.0,
                   help='min-green hold for scenedqn arms — default 0 to '
                        'match the vanilla DQN decision discipline '
                        '(no hold); scenemp keeps --scenemp-t-min')
    p.add_argument('--residual', action='append', default=[],
                   help='NAME=PATH — SceneResidualPolicy arm: trained '
                        'gated residual on the frozen DQN base '
                        '(--scenedqn-dqn supplies the weights)')
    p.add_argument('--residual-notext', default=None,
                   help='PATH — residual arm with the report channel '
                        'blinded (kind=none): with gate<tau the policy '
                        'should reproduce vanilla DQN exactly')
    p.add_argument('--joint', action='append', default=[],
                   help='NAME=PATH — SceneJointPolicy arm: controller '
                        'fine-tuned with the attention inside')
    p.add_argument('--joint-notext', default=None,
                   help='PATH — joint arm with the report channel '
                        'blinded (kind=none)')
    p.add_argument('--joint-lie', default=None,
                   help='joint ckpt under a WRONG-edge report override: '
                        'if the policy genuinely consumes the text, it '
                        'mis-serves the named edge — text-dependence proof')
    p.add_argument('--joint-base', default=None,
                   help='PATH — run the joint ckpt\'s inner base net '
                        'alone (internalisation check vs dqn_s1)')
    p.add_argument('--builtin', default='fixedtime,maxpressure',
                   help='comma-separated builtin controller ids '
                        '(+dqn_s1 for the S1 snapshot)')
    p.add_argument('--sga-notext', default=None,
                   help='PATH of an sga ckpt also run with zeroed text')
    p.add_argument('--report-delay', type=float, default=0.0)
    p.add_argument('--onsets', default='300')
    p.add_argument('--durs', default='600')
    p.add_argument('--periods', default='4')
    p.add_argument('--episodes-per-event', type=int, default=2)
    p.add_argument('--include-none', action='store_true')
    p.add_argument('--distractor', action='store_true',
                   help='also inject an unreported surge on an edge '
                        'OUTSIDE the event corridor — tests whether '
                        'attention follows the report or raw salience')
    p.add_argument('--corridor', action='store_true',
                   help='pair each event with an unreported '
                        'feeder-load surge (corridor governance)')
    p.add_argument('--episode-seconds', type=int, default=1800)
    p.add_argument('--interval', type=int, default=10)
    p.add_argument('--clear-thresh', type=float, default=2.0)
    p.add_argument('--traffic-seed-base', type=int, default=9000)
    p.add_argument('--t-fixed', type=int, default=30)
    p.add_argument('--t-min', type=int, default=10)
    p.add_argument('--runtime-dir', default=None)
    p.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    p.add_argument('--out', required=True)
    args = p.parse_args()

    rng = np.random.default_rng(7)
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    onsets = [int(x) for x in args.onsets.split(',') if x]
    durs = [int(x) for x in args.durs.split(',') if x]
    periods = [int(x) for x in args.periods.split(',') if x]

    # event grid: every injectable edge x onset/dur/period combos
    grids = []
    for edge in TRAIN_EDGES:
        for onset in onsets:
            for dur in durs:
                for period in periods:
                    grids.append(
                        f'surge:{edge}:{EDGE_TARGETS[edge]}:'
                        f'{onset}:{dur}:{period}')
    if args.include_none:
        grids.append(None)

    # load agent arms once; rebuild per-episode fb from their meta
    agents = {}
    for spec in args.agent:
        name, path = spec.split('=', 1)
        agents[name] = (SceneSignalAgent.load(path), path)
    scenemps = {}
    for spec in args.scenemp:
        name, path = spec.split('=', 1)
        scenemps[name] = (SceneSignalAgent.load(path), path)
    scenedqns = {}
    for spec in args.scenedqn:
        name, path = spec.split('=', 1)
        scenedqns[name] = (SceneSignalAgent.load(path), path)
    scenedqn_weights = args.scenedqn_dqn or args.dqn_s1
    residuals = {}
    for spec in args.residual:
        name, path = spec.split('=', 1)
        residuals[name] = (SceneResidualPolicy.load(path), path)
    joints = {}
    for spec in args.joint:
        name, path = spec.split('=', 1)
        joints[name] = (SceneJointPolicy.load(path), path)
    builtins = {c: c for c in args.builtin.split(',') if c}
    results = {name: [] for name in agents}
    for name in scenemps:
        results[f'scenemp_{name}'] = []
    for name in scenedqns:
        results[f'scenedqn_{name}'] = []
    for name in residuals:
        results[f'residual_{name}'] = []
    for name in joints:
        results[f'joint_{name}'] = []
    for name in builtins:
        results[name] = []
    if args.sga_notext:
        results['sga_notext'] = []
    if args.scenemp_notext:
        results['scenemp_notext'] = []
    if args.scenedqn_notext:
        results['scenedqn_notext'] = []
    if args.residual_notext:
        results['residual_notext'] = []
    if args.joint_notext:
        results['joint_notext'] = []
    if args.joint_lie:
        results['joint_lie'] = []
    if args.joint_base:
        results['joint_base'] = []

    ep_counter = 0
    for spec in grids:
        for rep in range(args.episodes_per_event):
            ep_counter += 1
            seed = args.traffic_seed_base + ep_counter
            world = build_world(args.scene, seed, Path(
                args.runtime_dir or '/tmp/takeover_signal_eval'),
                out / 'logger',
                {'t_fixed': args.t_fixed, 't_min': args.t_min,
                 'min_green_vehicle': 3, 'max_red_vehicle': 6})
            ranks = list(range(len(world.intersection_ids)))
            # apply_injects MUTATES inject dicts (count/next_depart/
            # applied) — every arm must get a fresh parse or later arms
            # silently run event-free episodes
            for name, ((agent, payload), path) in agents.items():
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=agent,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[name].append(metrics)
                world.reset()
            for name, ((agent, payload), path) in scenemps.items():
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                mp = SceneGuidedMP(agent, fb, world,
                                   beta=args.scenemp_beta,
                                   t_min=args.scenemp_t_min,
                                   gate=args.scenemp_gate,
                                   fade=args.scenemp_fade,
                                   direction=args.scenemp_direction,
                                   bind_tau=args.scenemp_bind_tau,
                                   dose_frac=args.scenemp_dose_frac)
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=mp,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[f'scenemp_{name}'].append(metrics)
                world.reset()
            for name, ((agent, payload), path) in scenedqns.items():
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                dqn = bound_dqn_agent(world, scenedqn_weights,
                                      rank=ranks[0])
                mp = SceneGuidedDQN(agent, fb, world, dqn,
                                    beta=args.scenemp_beta,
                                    t_min=args.scenedqn_t_min,
                                    gate=args.scenemp_gate,
                                    fade=args.scenemp_fade,
                                    direction=args.scenemp_direction,
                                    bind_tau=args.scenemp_bind_tau,
                                    dose_frac=args.scenemp_dose_frac)
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=mp,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[f'scenedqn_{name}'].append(metrics)
                world.reset()
            for name, ((policy, payload), path) in residuals.items():
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                dqn = bound_dqn_agent(world, scenedqn_weights,
                                      rank=ranks[0])
                qb_fn = (lambda dqn=dqn: inference_q_values(dqn))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=policy,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval, q_base_fn=qb_fn,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[f'residual_{name}'].append(metrics)
                world.reset()
            if args.residual_notext:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                policy, payload = SceneResidualPolicy.load(
                    args.residual_notext)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                dqn = bound_dqn_agent(world, scenedqn_weights,
                                      rank=ranks[0])
                qb_fn = (lambda dqn=dqn: inference_q_values(dqn))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=policy,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval, q_base_fn=qb_fn,
                    text_override={'kind': 'none'},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['residual_notext'].append(metrics)
                world.reset()
            for name, ((policy, payload), path) in joints.items():
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=policy,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[f'joint_{name}'].append(metrics)
                world.reset()
            if args.joint_notext:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                policy, payload = SceneJointPolicy.load(
                    args.joint_notext)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=policy,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    text_override={'kind': 'none'},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['joint_notext'].append(metrics)
                world.reset()
            if args.joint_lie:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                policy, payload = SceneJointPolicy.load(args.joint_lie)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                ev_edge = spec.split(':')[1] if spec else ''
                others = [e for e in TRAIN_EDGES if e != ev_edge]
                lie = others[0] if others else ev_edge
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=policy,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    text_override={'edge': lie},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                metrics['lie_edge'] = lie
                results['joint_lie'].append(metrics)
                world.reset()
            if args.joint_base:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                policy, payload = SceneJointPolicy.load(args.joint_base)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                metrics = eval_episode(
                    world, fb, ranks, injects,
                    agent=_JointBaseOnly(policy),
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['joint_base'].append(metrics)
                world.reset()
            if args.scenedqn_notext:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                agent, payload = SceneSignalAgent.load(
                    args.scenedqn_notext)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                dqn = bound_dqn_agent(world, scenedqn_weights,
                                      rank=ranks[0])
                mp = SceneGuidedDQN(agent, fb, world, dqn,
                                    beta=args.scenemp_beta,
                                    t_min=args.scenedqn_t_min,
                                    gate=args.scenemp_gate,
                                    fade=args.scenemp_fade,
                                    direction=args.scenemp_direction,
                                    bind_tau=args.scenemp_bind_tau,
                                    dose_frac=args.scenemp_dose_frac)
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=mp,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    text_override={'kind': 'none'},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['scenedqn_notext'].append(metrics)
                world.reset()
            if args.scenemp_notext:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                agent, payload = SceneSignalAgent.load(args.scenemp_notext)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                mp = SceneGuidedMP(agent, fb, world,
                                   beta=args.scenemp_beta,
                                   t_min=args.scenemp_t_min,
                                   gate=args.scenemp_gate,
                                   fade=args.scenemp_fade,
                                   direction=args.scenemp_direction,
                                   bind_tau=args.scenemp_bind_tau,
                                   dose_frac=args.scenemp_dose_frac)
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=mp,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    text_override={'kind': 'none'},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['scenemp_notext'].append(metrics)
                world.reset()
            if args.sga_notext:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                agent, payload = SceneSignalAgent.load(args.sga_notext)
                fb = FeatureBuilder(
                    world, injects, report_delay=args.report_delay,
                    **fb_kwargs_from_meta(payload.get('meta')))
                metrics = eval_episode(
                    world, fb, ranks, injects, agent=agent,
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    text_override={'kind': 'none'},
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results['sga_notext'].append(metrics)
                world.reset()
            controllers = build_controllers(
                dqn_s1_path=(args.dqn_s1 if 'dqn_s1' in builtins
                             else None))
            needed = set(builtins) - {'dqn_s1'}
            for name in builtins:
                injects = parse_injects(spec_list(spec, args.corridor, args.distractor)) if spec else []
                # world.reset() REBUILDS world.intersections — a controller
                # bound before an earlier arm's reset would read phase/
                # generators from a frozen Intersection.  Rebind per arm.
                _bind_all(world, controllers, ranks, needed=needed)
                if name == 'dqn_s1':
                    controllers['dqn_s1'] = controllers['dqn_s1'].bind(
                        world, ranks=ranks)
                fb0 = FeatureBuilder(world, injects,
                                     report_delay=args.report_delay)
                metrics = eval_episode(
                    world, fb0, ranks, injects,
                    controller=controllers[name],
                    episode_seconds=args.episode_seconds,
                    interval=args.interval,
                    clear_thresh=args.clear_thresh)
                metrics['event'] = spec or 'none'
                results[name].append(metrics)
                world.reset()
            world.close()
            print(f'[{ep_counter}] {spec or "none"} rep{rep} done',
                  flush=True)

    summary = {}
    for name, rows in results.items():
        if not rows:
            continue
        clears = [r['clear_s'] for r in rows
                  if r['clear_s'] == r['clear_s']]
        summary[name] = {
            'n': len(rows),
            'queue_int': round(float(np.mean(
                [r['queue_int'] for r in rows])), 1),
            'ev_queue_int': round(float(np.mean(
                [r['ev_queue_int'] for r in rows])), 1),
            'peak_queue': round(float(np.mean(
                [r['peak_queue'] for r in rows])), 1),
            'clear_s': (round(float(np.mean(clears)), 1)
                        if clears else None),
            'clear_n': len(clears),
            'throughput': round(float(np.mean(
                [r['throughput'] for r in rows])), 1),
            'bind_mass': (round(float(np.nanmean(
                [r['bind_mass'] for r in rows
                 if r['bind_mass'] is not None])), 4)
                if any(r['bind_mass'] is not None for r in rows)
                else None),
            'corr_queue_int': (round(float(np.mean(
                [r['corr_queue_int'] for r in rows
                 if 'corr_queue_int' in r])), 1)
                if any('corr_queue_int' in r for r in rows)
                else None),
            'corr_bind_mass': (round(float(np.nanmean(
                [r['corr_bind_mass'] for r in rows
                 if r.get('corr_bind_mass') is not None])), 4)
                if any(r.get('corr_bind_mass') is not None
                       for r in rows)
                else None),
        }
    (out / 'results.json').write_text(
        json.dumps({'arms': results, 'summary': summary},
                   ensure_ascii=False, indent=1))
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
