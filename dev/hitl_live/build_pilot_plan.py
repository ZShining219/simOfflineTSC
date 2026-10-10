#!/usr/bin/env python
"""build_pilot_plan.py — 生成 HITL C 通道 pilot 对照 plan（ha_sodqn 协议）。

臂（同一初始权重、同一 seed、同一 episode 预算，唯一差异=专家腿配比/池）：

    HITL_C0     无专家池          —— 基线（逐位等价 CONT）
    HITL_EXP    纠偏会话池         —— hitl_* 会话档案（学生主导+MP接管）
    HITL_CTRL   等量机器数据池      —— student_* 会话档案（排除"只是数据多"）
    HITL_DEMO   纯专家控盘池        —— expert_* 会话档案（示范数据对照）

产物是一个 ha_sodqn plan JSON；每个 child 通过
    python -m sequential.cli run-child --manifest PLAN \
        --logical-run-id ID --attempt-dir DIR
单独执行（探索性 pilot，不入正式台账；正式实验另走治理流程）。

用法：
    python dev/hitl_live/build_pilot_plan.py \
        --expert-pool output_data/hitl_probe/pools/hitl_ep50/expert_pool_manifest.json \
        --ctrl-pool   output_data/hitl_probe/pools/student_ep50/expert_pool_manifest.json \
        --demo-pool   output_data/hitl_probe/pools/expert_ep50/expert_pool_manifest.json \
        --episodes 20 --expert-ratio 0.25 --seed 0 \
        --out output_data/hitl_probe/pilot/plan.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from sequential.io import atomic_json, read_json, sha256_file  # noqa: E402


def _initial_state(checkpoint_path, catalog_path, network, seed):
    """Build the HA-SODQN episode-0 initial_state binding.  Digests are
    reused from the initial-state catalog when the checkpoint SHA-256
    matches; otherwise the block is emitted without digests (the agent then
    skips digest verification but still enforces episode-0/counter-zero)."""
    checkpoint_path = os.path.abspath(checkpoint_path)
    file_sha = sha256_file(checkpoint_path)
    digests = None
    if catalog_path:
        catalog = read_json(catalog_path)
        for entry in catalog.get('entries', []):
            if (entry.get('checkpoint_file_sha256') == file_sha
                    and entry.get('first_network') == network
                    and int(entry.get('training_seed', -1)) == seed):
                digests = entry.get('digests')
                break
    initial = {
        'checkpoint_episode': 0,
        'checkpoint_path': checkpoint_path,
        'network': network,
        'training_seed': int(seed),
        'algorithm_id': 'independent_dqn',
    }
    if digests:
        initial['digests'] = digests
    return initial, file_sha


def _resolve_checkpoint(pattern, catalog_path, network, seed):
    """Pick the ep0 checkpoint for (network, seed) whose sha256 matches the
    catalog entry; pattern may contain {seed} and glob wildcards."""
    import glob as _glob
    candidates = sorted(_glob.glob(pattern.format(seed=seed)))
    entry_sha = None
    if catalog_path:
        for entry in read_json(catalog_path).get('entries', []):
            if (entry.get('first_network') == network
                    and int(entry.get('training_seed', -1)) == seed):
                entry_sha = entry.get('checkpoint_file_sha256')
                break
    for path in candidates:
        if not os.path.isfile(path):
            continue
        if entry_sha is None or sha256_file(path) == entry_sha:
            return path
    raise SystemExit(
        f'no catalog-matched ep0 checkpoint for {network} seed={seed} '
        f'(candidates: {candidates})')


def _child(logical_run_id, condition, seed, network, episodes,
           initial, file_sha, expert_pool=None, expert_ratio=0.0,
           expert_capacity=5000, trainer_overrides=None):
    child = {
        'logical_run_id': logical_run_id,
        'experiment_stage': 'hitl_pilot',
        'condition': condition,
        'policy': 'ha_sodqn',
        'algorithm_id': 'independent_dqn',
        'order_id': 'O2',
        'training_seed': int(seed),
        'networks': [network],
        'stage_episodes': [int(episodes)],
        'initial_state': initial,
        'initial_checkpoint': initial['checkpoint_path'],
        'initial_checkpoint_file_sha256': file_sha,
        'archive_mode': 'NONE',
        'method': 'CONT',
        'offline_ratio': 0.0,
        'owp_capacity': 5000,
        'alignment_warmup_episodes': 10,
        'trace_replay_samples': False,
        'interface': 'libsumo',
        'status': 'planned',
    }
    if expert_pool:
        child['expert_pool_manifest'] = os.path.abspath(expert_pool)
        child['expert_ratio'] = float(expert_ratio)
        child['expert_capacity'] = int(expert_capacity)
    if trainer_overrides:
        child['trainer_overrides'] = dict(trainer_overrides)
    return child


def _demo_child(logical_run_id, condition, seed, network, episodes,
                initial, file_sha, demo_data, pretrain_actor,
                pretrain_critic, demo_steps):
    """A-channel arm: DemoAC agent, same ep0 checkpoint, policy=demo_ac.
    pretrain_actor/critic select the A0/Aπ/AQ/AπQ cell."""
    child = _child(logical_run_id, condition, seed, network, episodes,
                   initial, file_sha)
    child['policy'] = 'demo_ac'
    if pretrain_actor or pretrain_critic:
        child['demo_pretrain'] = {
            'demo_data': os.path.abspath(demo_data),
            'steps': int(demo_steps),
            'actor': bool(pretrain_actor),
            'critic': bool(pretrain_critic),
        }
    else:
        child['demo_pretrain'] = {
            'demo_data': os.path.abspath(demo_data),
            'steps': 0, 'actor': False, 'critic': False,
        }
    return child


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expert-pool', required=True,
                        help='expert_pool_manifest.json of the correction pool')
    parser.add_argument('--ctrl-pool', default=None,
                        help='equal-volume machine-data pool manifest')
    parser.add_argument('--demo-pool', default=None,
                        help='pure expert-demonstration pool manifest')
    parser.add_argument('--network', default='sumohz1x1')
    parser.add_argument('--seed', type=int, default=None,
                        help='single seed (legacy); prefer --seeds')
    parser.add_argument('--seeds', nargs='+', type=int, default=None,
                        help='training seeds; one child per seed x arm')
    parser.add_argument('--episodes', type=int, default=20)
    parser.add_argument('--expert-ratio', type=float, default=0.25)
    parser.add_argument('--expert-capacity', type=int, default=5000)
    parser.add_argument('--steps', type=int, default=1200,
                        help='train episode length (sim seconds)')
    parser.add_argument('--test-steps', type=int, default=1200,
                        help='autonomous eval episode length')
    parser.add_argument('--learning-start', type=int, default=240,
                        help='pilot-lite warmup decisions (formal=1000)')
    parser.add_argument('--initial-checkpoint', required=True,
                        help='episode-0 resumable checkpoint (.pt)')
    parser.add_argument('--initial-catalog', default=None,
                        help='initial_state_catalog.json for digest reuse')
    # --- B channel (reward shaping) arms --------------------------
    parser.add_argument('--b-arms', action='store_true',
                        help='add HITL_B1/B2 children per seed; B0=C0 '
                             'shares the C0 run (identical config)')
    parser.add_argument('--reward-lambda', type=float, default=0.1)
    parser.add_argument('--reward-calibration', default=None,
                        help='reward_calibration.json (required by B1/B2)')
    parser.add_argument('--reward-model', default=None,
                        help='reward_model.pt checkpoint (required by B2)')
    # --- B' channel (sample-time adjudication) arms ------------------
    parser.add_argument('--bprime-arms', action='store_true',
                        help='add HITL_BP children per seed scoring ALL legs '
                             'at sample time via an adjudication manifest')
    parser.add_argument('--bpc-arms', action='store_true',
                        help='add HITL_BPC children per seed: adjudication '
                             '+ expert leg (the combination legacy B could '
                             'not express)')
    parser.add_argument('--adjudication-manifest', default=None,
                        help='JSON file with the frozen judge spec: '
                             '{lambda, clip, judges: [{type, weight, mu, '
                             'sigma, ...}]} (required by --bprime-arms/'
                             '--bpc-arms)')
    parser.add_argument('--adjudication-ctx', default=None,
                        help='dump_lane_topology JSON for adjudication '
                             'features (required by --bprime-arms/--bpc-arms)')
    # --- A channel (DemoLight pretraining) arms ----------------------
    parser.add_argument('--a-arms', action='store_true',
                        help='add A0/Aπ/AQ/AπQ children per seed')
    parser.add_argument('--demo-data', default=None,
                        help='demo npz from build_demos.py')
    parser.add_argument('--demo-steps', type=int, default=2000,
                        help='pretrain gradient steps (spec default 2000)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)
    seeds = args.seeds if args.seeds else [args.seed if args.seed is not None else 0]

    children = []
    for seed in seeds:
        checkpoint = _resolve_checkpoint(
            args.initial_checkpoint, args.initial_catalog,
            args.network, seed)
        initial, file_sha = _initial_state(
            checkpoint, args.initial_catalog, args.network, seed)

        children.append(_child(
            f'HITL_C0-SD{seed}', 'HITL_C0', seed,
            args.network, args.episodes, initial, file_sha))
        children.append(_child(
            f'HITL_EXP-SD{seed}', 'HITL_EXPERT', seed,
            args.network, args.episodes, initial, file_sha,
            expert_pool=args.expert_pool, expert_ratio=args.expert_ratio,
            expert_capacity=args.expert_capacity))
        if args.ctrl_pool:
            children.append(_child(
                f'HITL_CTRL-SD{seed}', 'HITL_CTRL', seed,
                args.network, args.episodes, initial, file_sha,
                expert_pool=args.ctrl_pool, expert_ratio=args.expert_ratio,
                expert_capacity=args.expert_capacity))
        if args.demo_pool:
            children.append(_child(
                f'HITL_DEMO-SD{seed}', 'HITL_DEMO', seed,
                args.network, args.episodes, initial, file_sha,
                expert_pool=args.demo_pool, expert_ratio=args.expert_ratio,
                expert_capacity=args.expert_capacity))
        if args.b_arms:
            if not args.reward_calibration:
                raise SystemExit('--b-arms requires --reward-calibration')
            b_shared = {
                'reward_lambda': float(args.reward_lambda),
                'reward_calibration_path': os.path.abspath(
                    args.reward_calibration),
            }
            children.append(_child(
                f'HITL_B1-SD{seed}', 'HITL_B1', seed,
                args.network, args.episodes, initial, file_sha,
                trainer_overrides={**b_shared, 'reward_mode': 'b1'}))
            if not args.reward_model:
                raise SystemExit('--b-arms requires --reward-model for B2')
            children.append(_child(
                f'HITL_B2-SD{seed}', 'HITL_B2', seed,
                args.network, args.episodes, initial, file_sha,
                trainer_overrides={
                    **b_shared, 'reward_mode': 'b2',
                    'reward_model_path': os.path.abspath(args.reward_model),
                }))
        if args.bprime_arms or args.bpc_arms:
            if not args.adjudication_manifest:
                raise SystemExit(
                    '--bprime-arms/--bpc-arms require '
                    '--adjudication-manifest')
            if not args.adjudication_ctx:
                raise SystemExit(
                    '--bprime-arms/--bpc-arms require --adjudication-ctx')
            adj_spec = json.load(open(args.adjudication_manifest))
            adj_overrides = {
                'adjudication': adj_spec,
                'adjudication_ctx_path': os.path.abspath(
                    args.adjudication_ctx),
            }
        if args.bprime_arms:
            children.append(_child(
                f'HITL_BP-SD{seed}', 'HITL_BP', seed,
                args.network, args.episodes, initial, file_sha,
                trainer_overrides=dict(adj_overrides)))
        if args.bpc_arms:
            if not args.expert_pool:
                raise SystemExit('--bpc-arms requires --expert-pool')
            children.append(_child(
                f'HITL_BPC-SD{seed}', 'HITL_BPC', seed,
                args.network, args.episodes, initial, file_sha,
                expert_pool=args.expert_pool,
                expert_ratio=args.expert_ratio,
                expert_capacity=args.expert_capacity,
                trainer_overrides=dict(adj_overrides)))
        if args.a_arms:
            if not args.demo_data:
                raise SystemExit('--a-arms requires --demo-data')
            for tag, cond, pa, pc in (
                    ('A0', 'HITL_A0', False, False),
                    ('API', 'HITL_API', True, False),
                    ('AQ', 'HITL_AQ', False, True),
                    ('APIQ', 'HITL_APIQ', True, True)):
                children.append(_demo_child(
                    f'HITL_{tag}-SD{seed}', cond, seed,
                    args.network, args.episodes, initial, file_sha,
                    args.demo_data, pa, pc, args.demo_steps))

    plan = {
        'schema_version': 2,
        'protocol_id': 'ha_sodqn_b100_v1',
        'mode': 'pilot',
        'experiment_stage': 'hitl_pilot',
        'networks': [args.network],
        'trainer': {
            'stage_episodes': [int(args.episodes)],
            'steps': int(args.steps),
            'test_steps': int(args.test_steps),
            'action_interval': 10,
            'learning_start': int(args.learning_start),
            'buffer_size': 5000,
            'batch_size': 64,
            'update_model_rate': 1,
            'target_update_interval': 10,
            'evaluation_timeout_seconds': 900,
        },
        'model': {
            'learning_rate': 0.001,
            'gamma': 0.95,
            'epsilon_decay': 0.995,
            'epsilon_min': 0.01,
            'grad_clip': 5.0,
            'phase': True,
            'one_hot': True,
        },
        'children': children,
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    atomic_json(out, plan)
    print(json.dumps({
        'plan': out,
        'children': [c['logical_run_id'] for c in children],
        'initial_checkpoint': initial['checkpoint_path'],
        'initial_sha256': file_sha,
        'digests_pinned': 'digests' in initial,
    }, indent=1))


if __name__ == '__main__':
    main()
