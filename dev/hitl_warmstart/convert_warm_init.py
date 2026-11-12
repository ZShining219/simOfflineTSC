"""resumable ep-N 档 → ep0 语义的 warm-init 初始态档（探索件，登记 dev/INDEX.md）。

人在回路「已有模型+专家介入数据续训」验证链的唯一新件：sequential 的
``from_initial_state``（ha_agent.py / demo_ac.py）硬校验档内
``checkpoint_type=='resumable'``、``episode==0``、三个训练计数器为零，
但会完整载回 weights/optimizer/epsilon/RNG。本脚本把任意 resumable
档改写成满足该入口的「暖启动初始态」：

- ``episode`` 置 0、三级训练计数器（顶层 + training_counters + agents[] 内
  同名键）清零、``replay_state.items`` 置空（对齐 ep0 语义
  ``orb_initial_state: 'empty'``）；
- weights / optimizer / epsilon / python+numpy+torch RNG 原样保留——
  语义即「该策略权重+训练状态，计数从 0 起的同模型续训」；
- 追加 ``warm_source`` 溯源块（源路径/源 episode/源 sha256/转换时间），
  溯源信息写进工件本体，不构成审计歧义；
- --sidecar 输出一份可直接贴进 plan child ``initial_state`` 的 dict
  （checkpoint_path/checkpoint_file_sha256/digests，digest 复用
  infra.core 的四个摘要函数重算）。

用法（repo 根，conda env: colight）：

    python dev/hitl_warmstart/convert_warm_init.py \
        --src <epN_resumable.pt> --dst <warm_init.pt> \
        --sidecar <warm_init.json> --network sumohz1x1 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import datetime
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from infra.core import (
    online_parameter_digest, optimizer_state_digest, rng_state_digest,
    target_parameter_digest,
)
from infra.io import atomic_json, sha256_file

# 顶层与 training_counters 散装键 + agents[] 内同名键；training_counters
# 内另有 epoch/step 进度键
_COUNTER_KEYS = ('global_decision_step', 'gradient_updates', 'target_updates',
                 'epoch', 'step')


def _zero_counters(mapping):
    for key in _COUNTER_KEYS:
        if key in mapping:
            mapping[key] = 0


def _reshape_sequential(checkpoint):
    """sequential 的 agent_state 布局 → 经典 resumable 布局。

    sequential run 的 stages/committed/recovery/online 档把全状态放在
    ``agent_state`` 里（replay 用 records 键、RNG 集中在 rng_state、
    episode 在 identity.global_episode）；``from_initial_state`` 吃的是
    经典 resumable 布局（顶层 episode/training_counters/RNG +
    agents[] 数组）。这里做一次纯布局翻译，内容不改。
    """
    a = checkpoint['agent_state']
    rng = a.get('rng_state') or {}
    src_episode = int((checkpoint.get('identity') or {}).get('global_episode', -1))
    converted = {
        'checkpoint_type': 'resumable',
        'schema_version': 1,
        'episode': src_episode,
        'algorithm_id': a.get('algorithm_id', 'independent_dqn'),
        'training_counters': dict(a.get('counters') or {}),
        'agents': [{
            'rank': 0,
            'epsilon': a.get('epsilon'),
            'online_model_state_dict': a['online_model_state_dict'],
            'target_model_state_dict': a['target_model_state_dict'],
            'optimizer_state_dict': a['optimizer_state_dict'],
            'replay_state': {
                'capacity': int((a.get('replay_state') or {}).get('capacity', 5000)),
                'items': list((a.get('replay_state') or {}).get('records', [])),
            },
        }],
        'python_random_state': rng.get('python_random_state'),
        'numpy_random_state': rng.get('numpy_random_state'),
        'torch_cpu_rng_state': rng.get('torch_cpu_rng_state'),
        'torch_cuda_rng_states': rng.get('torch_cuda_rng_states'),
    }
    return converted, src_episode


def _load_state_dict_any(path):
    """取权重：online_only/快照档顶层 ``online_model_state_dict``，
    或 resumable 档 ``agents[0].online_model_state_dict``。"""
    blob = torch.load(path, map_location='cpu')
    if 'online_model_state_dict' in blob:
        return blob['online_model_state_dict']
    for agent in blob.get('agents') or ():
        if 'online_model_state_dict' in agent:
            return agent['online_model_state_dict']
    raise ValueError(f'{path}: no online_model_state_dict found')


def convert(src, dst, sidecar=None, network='sumohz1x1', seed=0,
            weights_from=None):
    src = os.path.abspath(src)
    dst = os.path.abspath(dst)
    checkpoint = torch.load(src, map_location='cpu')
    source_sha = sha256_file(src)

    if checkpoint.get('checkpoint_type') == 'resumable':
        converted = copy.deepcopy(checkpoint)
        source_episode = int(checkpoint.get('episode', -1))
    elif 'agent_state' in checkpoint:
        converted, source_episode = _reshape_sequential(checkpoint)
    else:
        raise ValueError(f'{src}: unsupported checkpoint layout '
                         f"(type={checkpoint.get('checkpoint_type')!r})")

    converted['episode'] = 0
    _zero_counters(converted)
    if isinstance(converted.get('training_counters'), dict):
        _zero_counters(converted['training_counters'])
    for agent in converted.get('agents', ()):
        _zero_counters(agent)
        replay = agent.get('replay_state')
        if isinstance(replay, dict) and 'items' in replay:
            replay['items'] = []
        if 'episodes_done' in agent:
            agent['episodes_done'] = 0
    if weights_from:
        weights_from = os.path.abspath(weights_from)
        wsd = _load_state_dict_any(weights_from)
        agent0 = converted['agents'][0]
        missing = [k for k in wsd
                   if k not in agent0['online_model_state_dict']]
        if missing or len(wsd) != len(agent0['online_model_state_dict']):
            raise ValueError(
                f'{weights_from}: state_dict keys mismatch vs {src} '
                f'(missing={missing[:4]}, n={len(wsd)} vs '
                f'{len(agent0["online_model_state_dict"])})')
        agent0['online_model_state_dict'] = copy.deepcopy(wsd)
        agent0['target_model_state_dict'] = copy.deepcopy(wsd)

    converted['warm_source'] = {
        'source_path': src,
        'source_episode': source_episode,
        'source_sha256': source_sha,
        'converted_at': datetime.datetime.now().isoformat(timespec='seconds'),
        'note': 'episode/counters/replay 归零；weights/optimizer/epsilon/RNG 保留',
    }
    if weights_from:
        converted['warm_source']['weights_override'] = {
            'path': weights_from,
            'sha256': sha256_file(weights_from),
            'note': 'online+target 权重替换为该档 online_model_state_dict'
                    '（online_only 档无 optimizer/RNG，骨架字段沿 src）',
        }

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save(converted, dst)

    agent0 = converted['agents'][0]
    entry = {
        'algorithm_id': converted.get('algorithm_id',
                                     agent0.get('algorithm_id',
                                                'independent_dqn')),
        'checkpoint_episode': 0,
        'checkpoint_path': dst,
        'checkpoint_file_sha256': sha256_file(dst),
        'network': network,
        'training_seed': int(seed),
        'warm_source': converted['warm_source'],
        'digests': {
            'online_parameter_digest': online_parameter_digest(
                agent0['online_model_state_dict']),
            'target_parameter_digest': target_parameter_digest(
                agent0['target_model_state_dict']),
            'optimizer_state_digest': optimizer_state_digest(
                agent0['optimizer_state_dict']),
            'rng_state_digest': rng_state_digest(
                converted['python_random_state'],
                converted['numpy_random_state'],
                converted['torch_cpu_rng_state'],
                converted.get('torch_cuda_rng_states'),
            ),
        },
    }
    if sidecar:
        atomic_json(sidecar, entry)
    print(f"[warm_init] {src} (ep{source_episode}) -> {dst}")
    print(f"[warm_init] sha256={entry['checkpoint_file_sha256']}")
    return entry


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True)
    parser.add_argument('--dst', required=True)
    parser.add_argument('--sidecar', default=None)
    parser.add_argument('--network', default='sumohz1x1')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--weights-from', default=None,
                        help='替换 agents[0] 的 online+target 权重为该档'
                             ' online_model_state_dict（支持 online_only'
                             ' 快照档；骨架 optimizer/epsilon/RNG 仍沿 src）')
    args = parser.parse_args(argv)
    convert(args.src, args.dst, args.sidecar, args.network, args.seed,
            args.weights_from)


if __name__ == '__main__':
    main()
