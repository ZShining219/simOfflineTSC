"""Command line entry for the live SUMO broadcast server.

Example:
    python -m tools.sumo_live_broadcast --scene S2 --port 8010
    python -m tools.sumo_live_broadcast --scene S4 \
        --model dhoa=output_data/.../snapshots/sumohz1x1_config3.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.sumo_html_comparison import canonical_scene  # noqa: E402

from .controllers import (  # noqa: E402
    build_builtin_controllers, build_dispatch_controller,
    build_snapshot_controller, load_qnet_payload,
)
from .engine import BroadcastEngine  # noqa: E402
from .page import build_page  # noqa: E402
from .server import BroadcastServer  # noqa: E402

SCENE_CONFIGS = {
    'S1': 'configs/sim/sumohz1x1_config2.cfg',
    'S2': 'configs/sim/sumohz1x1.cfg',
    'S3': 'configs/sim/sumohz1x1_config4.cfg',
    'S4': 'configs/sim/sumohz1x1_config3.cfg',
    'S5': 'configs/sim/sumohz4x4.cfg',
    'S6': 'configs/sim/sumohz4x4_hetero.cfg',
}

# Curated default checkpoints outside the snapshot scan root.  These are the
# formal-run final online weights (global_episode 400); override with --model.
DEFAULT_CHECKPOINT_MODELS = [
    (
        'dhoa',
        'DHOA · formal O2/SD0 ep400',
        'output_data/ha_sodqn/formal_e7705f7_20260726/'
        'P1F-DHOA-R50-O2-SD0/attempts/attempt_1/checkpoints/online/'
        'stage_04_episode_0100.pt',
    ),
    (
        'cont',
        'CONT · formal O2/SD0 ep400',
        'output_data/ha_sodqn/formal_e7705f7_20260726/'
        'CONT-FIFO-O2-SD0/attempts/attempt_1/checkpoints/online/'
        'stage_04_episode_0100.pt',
    ),
]


def _resolve_sim_config(scene, sim_config):
    if sim_config:
        path = Path(sim_config).expanduser().resolve()
    else:
        network = canonical_scene(scene)
        path = _REPO_ROOT / 'configs' / 'sim' / f'{network}.cfg'
    if not path.is_file():
        raise FileNotFoundError(f'SUMO sim config not found: {path}')
    return path


def _parse_model_arg(value):
    if '=' not in value:
        raise argparse.ArgumentTypeError(
            f'--model expects name=path, got: {value}'
        )
    name, _, path = value.partition('=')
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError('--model name must not be empty')
    return name, Path(path).expanduser().resolve()


def _discover_snapshots(models_dir, scene):
    """Scene-matched snapshot .pt files under a directory tree."""
    root = Path(models_dir).expanduser().resolve()
    if not root.is_dir():
        return []
    network = canonical_scene(scene)
    found = []
    for path in sorted(root.glob('**/snapshots/*.pt')):
        if path.stem == network:
            found.append(path)
    return found


def _snapshot_detail(path):
    """Best-effort provenance for menu labels/tooltips."""
    detail = {'path': str(path)}
    try:
        payload = load_qnet_payload(path)
        detail['kind'] = payload['kind']
        identity = payload.get('identity') or {}
        if identity:
            detail['identity'] = identity
    except Exception:
        pass
    return detail


def _snapshot_label(path, detail):
    """Human-readable label carrying the weights' provenance."""
    identity = detail.get('identity') or {}
    run = identity.get('logical_run_id')
    episode = identity.get('global_episode')
    if run:
        label = f"{path.stem} · {run}"
        if episode is not None:
            label += f" ep{episode}"
        return label
    parent = identity.get('source_parent_import')
    if parent:
        label = f'{path.stem} · 父代快照'
        if episode is not None:
            label += f' ep{episode}'
        return label
    return f'{path.stem} 快照'


def build_controllers(args, scene, inject_specs=None):
    controllers = {}
    for controller in build_builtin_controllers():
        controllers[controller.id] = controller

    snapshot_paths = _discover_snapshots(args.models_dir, scene)
    for path in snapshot_paths:
        detail = _snapshot_detail(path)
        controller_id = f'snap_{path.stem}'
        controllers.setdefault(
            controller_id,
            build_snapshot_controller(
                controller_id, _snapshot_label(path, detail), path,
                detail=detail,
            ),
        )

    for name, label, rel in DEFAULT_CHECKPOINT_MODELS:
        path = _REPO_ROOT / rel
        if not path.is_file() or name in controllers:
            continue
        controllers[name] = build_snapshot_controller(
            name, label, path, detail=_snapshot_detail(path),
        )

    for name, path in (args.model or []):
        detail = _snapshot_detail(path)
        controllers[name] = build_snapshot_controller(
            name, name, path, detail=detail,
        )

    for name, path in (getattr(args, 'dispatch', None) or []):
        controllers[name] = build_dispatch_controller(
            name, name, path, inject_specs=inject_specs,
        )

    # Keep the manual entry last in the menu regardless of insertion order.
    manual = controllers.pop('manual', None)
    if manual is not None:
        controllers['manual'] = manual
    return controllers


def _build_scene_configs(args):
    """{scene_key: config path} for the on-page simulation-package menu."""
    configs = {}
    for key, rel in SCENE_CONFIGS.items():
        path = _REPO_ROOT / rel
        if path.is_file():
            configs[key] = path
    if args.sim_config:
        # The explicit config replaces whatever entry the --scene resolves to.
        key = str(args.scene).upper()
        configs[key] = Path(args.sim_config).expanduser().resolve()
    return configs


def build_parser():
    parser = argparse.ArgumentParser(
        prog='tools.sumo_live_broadcast',
        description=__doc__,
    )
    parser.add_argument(
        '--scene', default='S2',
        help='S1-S6 label or SUMO network name (default: S2)',
    )
    parser.add_argument(
        '--sim-config', default=None,
        help='SUMO sim config json; default configs/sim/<scene>.cfg',
    )
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8010)
    parser.add_argument(
        '--speed', type=float, default=1.0,
        help='Wall-clock pacing factor; 1.0 is real time (default: 1.0)',
    )
    parser.add_argument(
        '--action-interval', type=int, default=10,
        help='Decision interval in sim seconds (default: 10)',
    )
    parser.add_argument(
        '--sumo-seed', type=int, default=None,
        help='Optional SUMO --seed for the broadcast run',
    )
    parser.add_argument(
        '--default-controller', default='fixedtime',
        help='Controller active when the page opens (default: fixedtime)',
    )
    parser.add_argument(
        '--model', action='append', type=_parse_model_arg, default=[],
        metavar='NAME=PATH',
        help='Register a Q-network snapshot/checkpoint under NAME; repeatable',
    )
    parser.add_argument(
        '--models-dir', default=str(_REPO_ROOT / 'output_data' / 'sequential'),
        help='Directory scanned for scene-matched snapshot .pt files',
    )
    parser.add_argument(
        '--dispatch', action='append', type=_parse_model_arg, default=[],
        metavar='NAME=PATH',
        help='Register a dispatch router checkpoint under NAME; repeatable',
    )
    parser.add_argument(
        '--inject', action='append', default=[],
        metavar='SPEC',
        help='Traffic perturbation, e.g. surge:FROM:TO:START:DUR:PERIOD '
             'or lane_close:LANE:START:DUR; repeatable',
    )
    parser.add_argument(
        '--record', default=None, metavar='DIR',
        help='Enable session recording (HITL data plane): per-decision '
             'transitions, proposals and interventions land under '
             'DIR/<session_id>/',
    )
    parser.add_argument(
        '--expert', default=None, metavar='NAME',
        help='Controller id for the HITL expert seat (e.g. maxpressure); '
             'answers only when the query gate fires. '
             "'human' seats the operator: gate hits raise an awaiting "
             'banner answered (or timed-out) by a real person',
    )
    parser.add_argument(
        '--hitl-human-timeout', type=float, default=30.0,
        help='Human expert seat: seconds the sim waits for an answer before '
             'resolving to silence (query stays billed; default 30)',
    )
    parser.add_argument(
        '--hitl-gate', default='off',
        choices=['off', 'periodic', 'on_demand'],
        help='Expert-query trigger mode (default: off)',
    )
    parser.add_argument(
        '--hitl-period', type=int, default=10,
        help='periodic gate: query every N eligible decisions (default 10)',
    )
    parser.add_argument(
        '--hitl-queue-threshold', type=float, default=None,
        help='on_demand gate: fire when junction incoming queue >= X',
    )
    parser.add_argument(
        '--hitl-queue-growth', type=float, default=None,
        help='on_demand gate: fire when queue grows by X vs last decision',
    )
    parser.add_argument(
        '--hitl-budget', type=int, default=36,
        help='Expert queries per episode; spec formula B=floor(0.1*N) with '
             'N=horizon/action_interval (36 for T=3600, 12 for T=1200)',
    )
    parser.add_argument(
        '--hitl-cooldown', type=int, default=3,
        help='Min decision ticks between queries (default 3)',
    )
    parser.add_argument(
        '--t-fixed', type=int, default=None,
        help='FixedTime t_fixed override (default: 30)',
    )
    parser.add_argument(
        '--t-min', type=int, default=None,
        help='MaxPressure/SOTL t_min override',
    )
    parser.add_argument(
        '--min-green-vehicle', type=int, default=None,
        help='SOTL min_green_vehicle override',
    )
    parser.add_argument(
        '--max-red-vehicle', type=int, default=None,
        help='SOTL max_red_vehicle override',
    )
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    sim_config = _resolve_sim_config(args.scene, args.sim_config)
    controllers = build_controllers(
        args, args.scene, inject_specs=list(args.inject))

    model_params = {}
    for cli_key, param_key in (
        ('t_fixed', 't_fixed'), ('t_min', 't_min'),
        ('min_green_vehicle', 'min_green_vehicle'),
        ('max_red_vehicle', 'max_red_vehicle'),
    ):
        value = getattr(args, cli_key)
        if value is not None:
            model_params[param_key] = value

    scene_configs = _build_scene_configs(args)
    gate = None
    if args.hitl_gate != 'off':
        from .gate import HitlGate
        gate = HitlGate(
            mode=args.hitl_gate, period=args.hitl_period,
            queue_threshold=args.hitl_queue_threshold,
            queue_growth=args.hitl_queue_growth,
            budget=args.hitl_budget, cooldown=args.hitl_cooldown,
        )
        if not args.expert:
            print('警告: --hitl-gate 已开启但未指定 --expert，'
                  '专家席为空，触发不会产生答案')
    recorder = None
    if args.record:
        from .recorder import HitlRecorder
        recorder = HitlRecorder(
            Path(args.record).expanduser().resolve(),
            meta={
                'scene': canonical_scene(args.scene),
                'scene_key': str(args.scene).upper(),
                'source_config': str(sim_config),
                'action_interval': int(args.action_interval),
                'speed': float(args.speed),
                'sumo_seed': args.sumo_seed,
                'controllers': sorted(controllers),
                'inject_specs': list(args.inject),
                'expert_id': (
                    'human' if args.expert == 'human' else args.expert),
                'gate': gate.config() if gate else None,
                'human_timeout': (
                    float(args.hitl_human_timeout)
                    if args.expert == 'human' else None),
            },
        )
    engine = BroadcastEngine(
        scene=canonical_scene(args.scene),
        source_config=sim_config,
        controllers=controllers,
        default_controller_id=args.default_controller,
        action_interval=args.action_interval,
        speed=args.speed,
        sumo_seed=args.sumo_seed,
        model_params=model_params,
        scene_configs=scene_configs,
        injects=list(args.inject),
        recorder=recorder,
        expert_id=('manual' if args.expert == 'human' else args.expert),
        gate=gate,
        human_timeout=args.hitl_human_timeout,
    )
    scene_key = str(args.scene).upper()
    if scene_key in scene_configs:
        engine.scene_key = scene_key
    engine.start()
    server = BroadcastServer(
        engine, build_page(), host=args.host, port=args.port,
    )
    print(f'SUMO live broadcast: {server.address}  (scene={args.scene}, '
          f'controllers={list(controllers)})')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
        engine.shutdown()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
