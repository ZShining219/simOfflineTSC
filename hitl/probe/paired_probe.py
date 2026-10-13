"""Paired HITL probe: same seed, same scene, student-only vs gated-expert.

探索件（hitl/probe/，登记 dev/INDEX.md）。无头跑真实的
BroadcastEngine 会话——测量面就是 recorder，所以"人在回路有效性"的
证据数据与直播录制链路完全一致（record → package → OfflineTrajectoryDataset
可直接复核）。

臂（同 seed、同场景、同 inject 调度配对）：

- ``student``  学生控制器单独控盘（默认 snap_sumohz1x1 快照）
- ``hitl``     学生控盘 + gate 命中时专家（默认 maxpressure）接管
- ``expert``   专家单独控盘——上界参考，不是 HITL 臂的一部分

每臂每 seed 一个会话，录制 transitions/proposals/interventions +
manifest；跑完收集终端三指标（travel_time/throughput/unfinished）+
过程量（queue/real_delay/waiting_time）。随后可选
``--package`` 把每个会话打成离线档案，闭合"采集→打包→可读"链路。

用法（repo 根，conda env: colight）：

    python hitl/probe/paired_probe.py \
        --scene S2 --student snap_sumohz1x1 --expert maxpressure \
        --seeds 0 1 --horizon 1200 --speed 96 \
        --inject surge:road_0_1_0:road_1_2_3:300:600:20 \
        --out output_data/hitl_probe/run1 --package

产物（``--out`` 下）：

    sessions/<session_id>/     recorder 原始输出
    archives/<dataset_id>/     打包档案（--package 时）
    runs.json                  逐 run 汇总（三指标 + gate 账本）
    probe_manifest.json        probe 配置与臂/seed 索引
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

ARMS = ('student', 'hitl', 'expert')


def _terminal_metrics(engine):
    """Harvest run-end metrics while the world is still alive."""
    world = engine.world

    def _safe(fn):
        try:
            return float(fn())
        except Exception:
            return None

    return {
        'sim_time': _safe(world.get_current_time) if world else None,
        'travel_time': _safe(world.get_average_travel_time) if world else None,
        'throughput': _safe(world.get_cur_throughput) if world else None,
        'unfinished': (
            _safe(world.get_unfinished_vehicle_count) if world else None),
        'real_delay': _safe(world.get_real_delay) if world else None,
        'waiting_time': (
            _safe(world.get_average_waiting_time) if world else None),
    }


def _ledger(session_dir):
    """Gate ledger from the recorded interventions stream."""
    path = Path(session_dir) / 'interventions.jsonl'
    queries = takeovers = agrees = 0
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get('kind') != 'gate_query':
                continue
            detail = row.get('detail') or {}
            queries += 1
            if detail.get('intervened'):
                takeovers += 1
            else:
                agrees += 1
    return {'queries': queries, 'takeovers': takeovers, 'agrees': agrees}


def _series(session_dir):
    """Per-decision metric series for plotting (from transitions.jsonl)."""
    path = Path(session_dir) / 'transitions.jsonl'
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        metrics = row.get('metrics') or {}
        jm = metrics.get('junctions') or {}
        queue = None
        if jm:
            queue = float(np.mean([
                (v or {}).get('queue', 0.0) for v in jm.values()]))
        rows.append({
            't': row.get('t1'),
            'decision_step': row.get('decision_step'),
            'episode_id': row.get('episode_id'),
            'reward': row.get('reward'),
            'queue': queue,
            'throughput': (metrics.get('throughput')),
            'real_delay': metrics.get('real_delay'),
            'waiting_time': metrics.get('waiting_time'),
            'hitl': row.get('hitl') or {},
        })
    return rows


def _inject_windows(injects):
    """Event windows (sim seconds) parsed from surge inject specs —
    the ``window`` gate's oracle schedule."""
    windows = []
    for spec in injects or []:
        parts = str(spec).split(':')
        if parts and parts[0] == 'surge' and len(parts) >= 5:
            try:
                windows.append((float(parts[3]), float(parts[4])))
            except ValueError:
                continue
    return windows


def run_arm(args, arm, seed, controllers):
    """One paired run: fresh engine session, drive to horizon, harvest."""
    from hitl.live.engine import BroadcastEngine
    from hitl.live.gate import HitlGate
    from hitl.live.recorder import HitlRecorder

    gate = None
    expert_id = None
    controller_id = args.student
    if arm == 'hitl':
        expert_id = args.expert
        budget = (
            args.gate_budget if args.gate_budget is not None
            else max(1, int(0.1 * (args.horizon / args.action_interval)))
        )
        gate = HitlGate(
            mode=args.gate, period=args.gate_period,
            queue_threshold=args.gate_queue_threshold,
            queue_growth=args.gate_queue_growth,
            budget=budget, cooldown=args.gate_cooldown,
            random_prob=args.gate_random_prob,
            margin_frac=args.gate_margin_frac,
            windows=(args.gate_window or _inject_windows(args.inject)),
            rng_seed=seed,
        )
    elif arm == 'expert':
        controller_id = args.expert

    session_id = f'{arm}_s{seed}_{args.tag}'
    recorder = HitlRecorder(
        Path(args.out) / 'sessions',
        meta={
            'session_id': session_id,
            'scene': args.scene,
            'source_config': str(args.sim_config),
            'action_interval': args.action_interval,
            'speed': args.speed,
            'sumo_seed': seed,
            'controllers': sorted(controllers),
            'inject_specs': list(args.inject or []),
            'expert_id': expert_id,
            'gate': gate.config() if gate else None,
            'probe_arm': arm,
            'student_controller': args.student,
        },
    )
    engine = BroadcastEngine(
        scene=args.scene,
        source_config=args.sim_config,
        controllers=controllers,
        default_controller_id=controller_id,
        action_interval=args.action_interval,
        speed=args.speed,
        sumo_seed=seed,
        scene_configs={args.scene: args.sim_config},
        injects=list(args.inject or []),
        recorder=recorder,
        expert_id=expert_id,
        gate=gate,
    )
    print(f'[probe] arm={arm} seed={seed} controller={controller_id} '
          f'expert={expert_id} — building world', flush=True)
    engine.start()
    engine.start_session(args.scene)
    deadline = time.time() + args.build_timeout
    try:
        while True:
            state = engine.session_state
            if state == 'error':
                raise RuntimeError(
                    f'engine session error: {engine.session_error}')
            if state == 'live' and engine.world is not None:
                sim_t = float(engine.world.get_current_time())
                if sim_t >= args.horizon:
                    break
            if time.time() > deadline:
                raise RuntimeError(
                    f'timeout at {engine.session_state} '
                    f't={sim_t if engine.world else "-"}')
            time.sleep(0.25)
        summary = _terminal_metrics(engine)
        summary.update({
            'arm': arm, 'seed': seed, 'controller': controller_id,
            'expert_id': expert_id,
            'gate': gate.config() if gate else None,
            'budget_spent': gate.total_spent() if gate else 0,
        })
    finally:
        engine.shutdown()
        time.sleep(0.3)  # let recorder flush via shutdown path
    summary['session_id'] = session_id
    summary.update(_ledger(Path(args.out) / 'sessions' / session_id))
    summary['series'] = _series(
        Path(args.out) / 'sessions' / session_id)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scene', default='S2')
    parser.add_argument('--sim-config',
                        default='configs/sim/sumohz1x1.cfg')
    parser.add_argument('--student', default='snap_sumohz1x1',
                        help='student controller id (menu name)')
    parser.add_argument('--expert', default='maxpressure',
                        help='expert seat controller id')
    parser.add_argument('--arms', nargs='+', default=list(ARMS),
                        choices=list(ARMS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0],
                        help='SUMO seeds; same seed pairs arms')
    parser.add_argument('--horizon', type=int, default=1200,
                        help='sim seconds per run')
    parser.add_argument('--action-interval', type=int, default=10)
    parser.add_argument('--speed', type=float, default=96.0)
    parser.add_argument('--inject', action='append', default=[],
                        help='e.g. surge:road_0_1_0:road_1_2_3:300:600:20')
    parser.add_argument('--gate', default='on_demand',
                        choices=['periodic', 'on_demand', 'random',
                                 'margin', 'window'])
    parser.add_argument('--gate-period', type=int, default=3)
    parser.add_argument('--gate-queue-threshold', type=float, default=None)
    parser.add_argument('--gate-queue-growth', type=float, default=5)
    parser.add_argument('--gate-random-prob', type=float, default=0.06)
    parser.add_argument('--gate-margin-frac', type=float, default=0.06)
    parser.add_argument('--gate-window', action='append', default=None,
                        type=lambda s: tuple(
                            float(x) for x in s.split('-')),
                        help='LO-HI sim-seconds; default derives from '
                             'surge inject specs; repeatable')
    parser.add_argument('--gate-budget', type=int, default=None,
                        help='queries/episode; default derives spec formula '
                             'B=floor(0.1*horizon/action_interval)')
    parser.add_argument('--gate-cooldown', type=int, default=3)
    parser.add_argument('--models-dir',
                        default=str(_REPO_ROOT / 'output_data' /
                                    'sequential'))
    parser.add_argument('--model', action='append', default=[],
                        metavar='NAME=PATH',
                        help='register an extra snapshot/checkpoint '
                             '(e.g. an early-training student); repeatable')
    parser.add_argument('--build-timeout', type=float, default=600)
    parser.add_argument('--tag', default='probe',
                        help='session-id suffix')
    parser.add_argument('--out', required=True)
    parser.add_argument('--package', action='store_true',
                        help='package every recorded session into an '
                             'archive under <out>/archives/')
    args = parser.parse_args(argv)
    args.sim_config = str(Path(args.sim_config).expanduser().resolve())
    args.out = str(Path(args.out).expanduser().resolve())

    from hitl.live.cli import (
        _parse_model_arg, build_controllers)
    ns = argparse.Namespace(
        model=[_parse_model_arg(m) for m in args.model],
        models_dir=args.models_dir, dispatch=[])
    controllers = build_controllers(ns, args.scene,
                                    inject_specs=list(args.inject or []))
    needed = set(args.arms) | {'student'}
    for name in {args.student, args.expert}:
        if name not in controllers:
            raise SystemExit(
                f'controller {name!r} not in menu: {sorted(controllers)}')
    if 'expert' in args.arms and args.expert not in controllers:
        raise SystemExit(f'expert controller {args.expert} missing')

    runs = []
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for arm in args.arms:
            runs.append(run_arm(args, arm, seed, controllers))
            s = runs[-1]
            print(f"[probe] done arm={arm} seed={seed}: "
                  f"travel_time={s.get('travel_time')} "
                  f"throughput={s.get('throughput')} "
                  f"unfinished={s.get('unfinished')} "
                  f"queries={s.get('queries')} takeovers="
                  f"{s.get('takeovers')}", flush=True)

    (Path(args.out) / 'runs.json').write_text(
        json.dumps(runs, ensure_ascii=False, indent=2), encoding='utf-8')

    probe_manifest = {
        'kind': 'hitl_paired_probe',
        'scene': args.scene,
        'sim_config': args.sim_config,
        'student': args.student,
        'expert': args.expert,
        'arms': list(args.arms),
        'seeds': list(args.seeds),
        'horizon': args.horizon,
        'action_interval': args.action_interval,
        'injects': list(args.inject or []),
        'gate': {
            'mode': args.gate, 'period': args.gate_period,
            'queue_threshold': args.gate_queue_threshold,
            'queue_growth': args.gate_queue_growth,
            'budget': (
                args.gate_budget if args.gate_budget is not None
                else max(1, int(0.1 * (args.horizon / args.action_interval)))
            ),
            'budget_formula': (
                'explicit' if args.gate_budget is not None
                else 'floor(0.1*horizon/action_interval)'),
            'cooldown': args.gate_cooldown,
            'random_prob': args.gate_random_prob,
            'margin_frac': args.gate_margin_frac,
            'windows': (args.gate_window or _inject_windows(args.inject)),
        },
        'run_count': len(runs),
    }
    (Path(args.out) / 'probe_manifest.json').write_text(
        json.dumps(probe_manifest, ensure_ascii=False, indent=2),
        encoding='utf-8')

    if args.package:
        from hitl.pipeline.package import package_session
        for run in runs:
            out_dir = (
                Path(args.out) / 'archives' /
                f"{run['arm']}_s{run['seed']}")
            try:
                report = package_session(
                    str(Path(args.out) / 'sessions' / run['session_id']),
                    str(out_dir), network=Path(args.sim_config).stem,
                    behavior_seed=run['seed'],
                    config_path=args.sim_config,
                    repo_root=_REPO_ROOT)
                print(f"[package] {run['arm']} s{run['seed']}: "
                      f"valid={report['valid']} "
                      f"transitions={report['transition_count']}",
                      flush=True)
            except Exception as exc:  # noqa: BLE001 - probe keeps going
                print(f"[package] {run['arm']} s{run['seed']} FAILED: {exc}",
                      flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
