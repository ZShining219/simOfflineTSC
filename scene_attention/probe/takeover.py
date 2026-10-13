"""Controller-takeover probe: scripted fork experiment on a SUMO scene.

探索件（非正式实验入口）。复用 hitl/live 的控制器适配器与
tools/sumo_gui_comparison 的 world 构建函数，在无 UI 的确定性循环里做：

    reference  FixedTime 全程跑一遍，逐决策记录各进口车道排队
    fork       同 seed 重跑，决策步 < k 由前缀控制器驾驶，
               >= k 切换为接管控制器，逐决策记录到 episode 结束

两臂（或多个 arm）共享同一 FixedTime 前缀——SUMO 在同 seed + 同动作序列下
是确定性的，因此分叉点仿真状态一致；脚本在 fork 处落一个车道计数指纹做校验。

用法（conda env: colight）：

    python scene_attention/probe.py reference --scene S2 --out <dir>
    python scene_attention/probe.py fork --scene S2 \
        --fork-step 60 --window 300 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sumo_gui_comparison import (  # noqa: E402
    _configure_registry,
    _make_world_config,
)
from tools.sumo_html_comparison import canonical_scene  # noqa: E402
from hitl.live.controllers import (  # noqa: E402
    build_builtin_controllers,
    build_snapshot_controller,
)
from hitl.live.engine import DEFAULT_MODEL_PARAMS  # noqa: E402
from hitl.live.injects import (  # noqa: E402
    apply_injects as _apply_injects,
    parse_injects,
)

SUBSCRIBE = [
    'lane_count', 'lane_waiting_count', 'lane_waiting_time_count',
    'throughput',
]

# plan1_dqn 正式批 S1 源模型（seed0_r2，ep400 final eval checkpoint）
DEFAULT_DQN_S1 = (
    _REPO_ROOT
    / 'output_data/tsc/sumo_dqn/sumohz1x1_config2'
    / 'p1_formal_dqn_sumohz1x1_config2_seed0_400ep_20260722_r2'
    / 'checkpoints/evaluation/episode_0400.pt'
)


def _world_config_path(scene):
    network = canonical_scene(scene)
    path = _REPO_ROOT / 'configs' / 'sim' / f'{network}.cfg'
    if not path.is_file():
        raise FileNotFoundError(f'SUMO sim config not found: {path}')
    return path


def build_world(scene, sumo_seed, runtime_dir, output_dir, model_params):
    Path(runtime_dir).mkdir(parents=True, exist_ok=True)
    sim_config = _world_config_path(scene)
    world_config = _make_world_config(sim_config, runtime_dir, 'probe', gui=False)
    world_class = _configure_registry(world_config, sumo_seed, output_dir)
    from common.registry import Registry

    Registry.mapping['model_mapping']['setting'].param = dict(model_params)
    world = world_class(str(world_config), interface='libsumo')
    world.subscribe(list(SUBSCRIBE))
    world.reset()
    return world


def build_controllers(dqn_s1_path=None, extra_models=()):
    controllers = {
        c.id: c for c in build_builtin_controllers() if c.id != 'manual'
    }
    if dqn_s1_path:
        controllers['dqn_s1'] = build_snapshot_controller(
            'dqn_s1', 'DQN-S1 ep400', str(dqn_s1_path),
            detail={'path': str(dqn_s1_path)},
        )
    for name, path in extra_models:
        controllers[name] = build_snapshot_controller(
            name, name, str(path), detail={'path': str(path)},
        )
    return controllers


def _bind_all(world, controllers, ranks, needed=None):
    wanted = set(needed) if needed else set(controllers)
    for cid, controller in controllers.items():
        if cid not in wanted:
            continue
        controller.reset()
        controller.bind(world, ranks=ranks)


def _lane_fingerprint(world):
    counts = world.get_info('lane_count') or {}
    return sorted((str(k), int(v)) for k, v in counts.items())


def run_episode(world, controllers, ranks, prefix_id, takeover_id,
                fork_step, interval, episode_seconds, window_end_t=None,
                injects=None):
    """Drive one world; returns per-decision records.

    Decisions happen every ``interval`` sim seconds. ``prefix_id`` controls
    decisions below ``fork_step``; from ``fork_step`` on, ``takeover_id``
    (None keeps the prefix controller running = untreated baseline arm).
    """
    held = {
        rank: int(inter.virtual_phase)
        for rank, inter in enumerate(world.intersections)
    }
    records = []
    fingerprint = None
    decision = 0
    next_decision_t = float(world.get_current_time())
    while True:
        now = float(world.get_current_time())
        if episode_seconds and now >= episode_seconds:
            break
        if window_end_t is not None and now >= window_end_t:
            break
        if now + 1e-6 >= next_decision_t:
            decision += 1
            active = (
                takeover_id
                if takeover_id and decision >= fork_step
                else prefix_id
            )
            controller = controllers[active]
            for rank in ranks:
                held[rank] = int(controller.decide(rank))
            if takeover_id and decision == fork_step:
                fingerprint = _lane_fingerprint(world)
            lane_queue = world.get_info('lane_waiting_count') or {}
            lane_count = world.get_info('lane_count') or {}
            records.append({
                'decision': decision,
                't': now,
                'controller': active,
                'actions': {str(r): int(held[r]) for r in ranks},
                'lane_queue': {str(k): int(v) for k, v in lane_queue.items()},
                'lane_count': {str(k): int(v) for k, v in lane_count.items()},
                'total_queue': int(sum(lane_queue.values())),
                'throughput': int(world.get_info('throughput') or 0),
            })
            next_decision_t = now + interval
        if injects:
            _apply_injects(world, injects, now)
        world.step([
            held.get(rank, int(inter.virtual_phase))
            for rank, inter in enumerate(world.intersections)
        ])
    return records, fingerprint


def detect_events(records, queue_threshold=5, persist_decisions=6,
                  max_events=5):
    """Single-lane queue >= threshold sustained for >= persist decisions."""
    lanes = sorted(
        {lane for rec in records for lane in rec['lane_queue']}
    )
    events = []
    for lane in lanes:
        run_start = None
        for i, rec in enumerate(records):
            queued = rec['lane_queue'].get(lane, 0) >= queue_threshold
            if queued and run_start is None:
                run_start = i
            elif not queued and run_start is not None:
                if i - run_start >= persist_decisions:
                    seg = records[run_start:i]
                    events.append({
                        'lane': lane,
                        'start_decision': seg[0]['decision'],
                        'start_t': seg[0]['t'],
                        'end_decision': seg[-1]['decision'],
                        'end_t': seg[-1]['t'],
                        'duration_s': seg[-1]['t'] - seg[0]['t'],
                        'peak_queue': max(
                            r['lane_queue'].get(lane, 0) for r in seg
                        ),
                    })
                run_start = None
        if run_start is not None and len(records) - run_start >= persist_decisions:
            seg = records[run_start:]
            events.append({
                'lane': lane,
                'start_decision': seg[0]['decision'],
                'start_t': seg[0]['t'],
                'end_decision': seg[-1]['decision'],
                'end_t': seg[-1]['t'],
                'duration_s': seg[-1]['t'] - seg[0]['t'],
                'peak_queue': max(
                    r['lane_queue'].get(lane, 0) for r in seg
                ),
            })
    events.sort(key=lambda e: (e['start_t'], -e['peak_queue']))
    # keep non-overlapping events in time order
    picked, last_end = [], -1.0
    for event in events:
        if event['start_t'] >= last_end:
            picked.append(event)
            last_end = event['end_t']
        if len(picked) >= max_events:
            break
    return picked


def _write_jsonl(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row) + '\n')


def cmd_reference(args, world, controllers, ranks):
    t0 = time.time()
    records, _ = run_episode(
        world, controllers, ranks,
        prefix_id=args.prefix_controller, takeover_id=None,
        fork_step=0, interval=args.interval,
        episode_seconds=args.episode_seconds,
        injects=parse_injects(args.inject),
    )
    events = detect_events(
        records, args.queue_threshold, args.persist_decisions,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / 'reference_records.jsonl', records)
    summary = {
        'scene': args.scene,
        'sumo_seed': args.sumo_seed,
        'prefix_controller': args.prefix_controller,
        'decisions': len(records),
        'episode_seconds': args.episode_seconds,
        'wall_seconds': round(time.time() - t0, 1),
        'final_throughput': records[-1]['throughput'] if records else 0,
        'event_rule': {
            'queue_threshold': args.queue_threshold,
            'persist_decisions': args.persist_decisions,
        },
        'events': events,
    }
    (out / 'reference_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps(summary['events'], indent=2, ensure_ascii=False))
    print(f"reference: {len(records)} decisions, "
          f"{len(events)} event(s), {summary['wall_seconds']}s wall")
    return summary


def cmd_fork(args, world, controllers, ranks):
    arms = list(args.arm)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    window_end = (
        args.fork_step * args.interval + args.window
        if args.window else None
    )
    results = []
    fingerprints = {}
    for arm in arms:
        world.reset()
        _bind_all(world, controllers, ranks,
                  needed={args.prefix_controller, arm})
        t0 = time.time()
        records, fingerprint = run_episode(
            world, controllers, ranks,
            prefix_id=args.prefix_controller, takeover_id=arm,
            fork_step=args.fork_step, interval=args.interval,
            episode_seconds=args.episode_seconds,
            window_end_t=window_end if args.stop_at_window_end else None,
            injects=parse_injects(args.inject),
        )
        _write_jsonl(out / f'fork_{arm}_records.jsonl', records)
        fingerprints[arm] = fingerprint
        results.append({
            'arm': arm,
            'decisions': len(records),
            'fork_step': args.fork_step,
            'prefix_controller': args.prefix_controller,
            'wall_seconds': round(time.time() - t0, 1),
            'final_throughput': records[-1]['throughput'] if records else 0,
            'final_total_queue': records[-1]['total_queue'] if records else 0,
        })
        print(f"arm {arm}: {len(records)} decisions, "
              f"{results[-1]['wall_seconds']}s wall")
    uniq = {json.dumps(v) for v in fingerprints.values() if v is not None}
    summary = {
        'scene': args.scene,
        'sumo_seed': args.sumo_seed,
        'fork_step': args.fork_step,
        'fork_t': args.fork_step * args.interval,
        'window': args.window,
        'prefix_controller': args.prefix_controller,
        'arms': results,
        'prefix_state_fingerprint_consistent': len(uniq) <= 1,
    }
    (out / 'fork_summary.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + '\n',
        encoding='utf-8',
    )
    print(json.dumps({k: v for k, v in summary.items() if k != 'arms'},
                     indent=2, ensure_ascii=False))
    if len(uniq) > 1:
        print('WARNING: prefix state fingerprints differ across arms')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(prog='takeover_probe', description=__doc__)
    parser.add_argument('mode', choices=('reference', 'fork'))
    parser.add_argument('--scene', default='S2')
    parser.add_argument('--sumo-seed', type=int, default=10000)
    parser.add_argument('--interval', type=int, default=10)
    parser.add_argument('--episode-seconds', type=int, default=3600)
    parser.add_argument('--prefix-controller', default='fixedtime')
    parser.add_argument('--fork-step', type=int, default=None)
    parser.add_argument('--window', type=int, default=300)
    parser.add_argument('--stop-at-window-end', action='store_true')
    parser.add_argument('--arm', action='append', default=[],
                        help='takeover arm; repeatable')
    parser.add_argument('--dqn-s1', default=str(DEFAULT_DQN_S1))
    parser.add_argument('--model', action='append', default=[],
                        metavar='NAME=PATH', help='extra snapshot arm')
    parser.add_argument('--t-fixed', type=int, default=30)
    parser.add_argument('--t-min', type=int, default=10)
    parser.add_argument('--inject', action='append', default=[],
                        metavar='lane_close:LANE:START:DUR',
                        help='traffic perturbation; repeatable')
    parser.add_argument('--queue-threshold', type=int, default=5)
    parser.add_argument('--persist-decisions', type=int, default=6)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    out_dir = Path(args.out).resolve()
    runtime_dir = Path(tempfile.gettempdir()) / 'takeover_probe'
    model_params = {
        **DEFAULT_MODEL_PARAMS,
        't_fixed': args.t_fixed,
        't_min': args.t_min,
    }
    world = build_world(
        args.scene, args.sumo_seed, runtime_dir, out_dir / 'logger',
        model_params,
    )
    try:
        ranks = list(range(len(world.intersection_ids)))
        extra_models = []
        for spec in args.model:
            name, _, path = spec.partition('=')
            extra_models.append((name.strip(), Path(path)))
        dqn_s1 = Path(args.dqn_s1) if args.dqn_s1 else None
        if dqn_s1 and not dqn_s1.is_file():
            raise FileNotFoundError(f'DQN-S1 checkpoint missing: {dqn_s1}')
        controllers = build_controllers(dqn_s1, extra_models)
        needed = {args.prefix_controller}
        if args.mode == 'fork':
            needed.update(args.arm)
        _bind_all(world, controllers, ranks, needed=needed)
        if args.mode == 'reference':
            return cmd_reference(args, world, controllers, ranks)
        if args.fork_step is None:
            raise SystemExit('--fork-step is required for mode=fork')
        if not args.arm:
            raise SystemExit('fork mode needs at least one --arm')
        return cmd_fork(args, world, controllers, ranks)
    finally:
        world.close()


if __name__ == '__main__':
    raise SystemExit(main())
