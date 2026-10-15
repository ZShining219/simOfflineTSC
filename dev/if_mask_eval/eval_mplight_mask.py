"""一次性评测脚本：原版 MPLight 在 hz4x4_rl(5816=HangZhou2) 上的 clean / 方向掩码评测。

掩码语义对齐 RobustLight 论文 Table 2：
  - sensorW   -> 西向进口车道的 lane_count 观测清零（25%）
  - sensorW,E -> 西+东进口车道清零（50%）
方向判定：进口道路的行驶向量 start->end，dx>0 记为 W（来自西侧），dx<0 记为 E。

用法（仓根目录，py310/colight 环境）：
    python dev/if_mask_eval/eval_mplight_mask.py \
        --experiment-config configs/tsc/mplight_hz4x4_rl.yml \
        --checkpoint <训练产出的 100_0.pt 路径> \
        --mask none --episodes 1

注意：ckpt/输出路径一律走 CLI 参数，不写进文件（公开仓约束）。
mask 只作用于 agent 的 lane_count 观测输入，环境与仿真本身不受影响。
"""

import argparse
import logging
import os
import sys
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
os.chdir(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import numpy as np
import torch
import yaml

import task  # noqa: F401  register tasks
import trainer  # noqa: F401  register trainers
import agent  # noqa: F401  register agents
import dataset  # noqa: F401  register datasets
from common.registry import Registry
from common import interface
from utils.logger import (build_config, merge_dicts, capture_config_sources,
                          setup_logging, reserve_run_output,
                          archive_run_config, RunStateManager)


def make_pargs(cfg_path, prefix):
    return argparse.Namespace(
        thread_num=4, ngpu='-1', prefix=prefix, seed=None, sumo_seed=None,
        debug=False, interface='libsumo', delay_type='apx', episodes=None,
        task='tsc', agent='mplight', world='cityflow',
        network='cityflow_hz4x4_rl', dataset='onfly',
        evaluation_manifest=None, evaluation_output=None,
        experiment_config=cfg_path, resume_output=None, resume_checkpoint=None)


def build_trainer(cfg_path, prefix):
    pargs = make_pargs(cfg_path, prefix)
    config, _ = build_config(pargs)
    with open(cfg_path, encoding='utf-8') as handle:
        overlay = yaml.safe_load(handle) or {}
    config, _ = merge_dicts(config, overlay)
    config['command'].update(vars(pargs))

    output_path = reserve_run_output(config)
    config['command']['output_path'] = output_path
    sources = capture_config_sources(config)
    with open(cfg_path, 'rb') as handle:
        sources['experiment_overlay.yml'] = handle.read()
    interface.Command_Setting_Interface(config)
    interface.Logger_param_Interface(config)
    interface.World_param_Interface(config, sources['simulator_source.cfg'])
    interface.Logger_path_Interface(config)
    interface.Trainer_param_Interface(config)
    interface.ModelAgent_param_Interface(config)
    archive_path = archive_run_config(
        config, sources, Registry.mapping['world_mapping']['setting'].param)
    RunStateManager(config, archive_path)

    logger = setup_logging(logging.INFO)
    return Registry.mapping['trainer_mapping']['tsc'](logger)


def direction_of_road(road):
    """按进口道路行驶向量定方向：+x=W(来自西), -x=E, +y=S, -y=N。"""
    start = road['points'][0]
    end = road['points'][-1]
    dx, dy = end['x'] - start['x'], end['y'] - start['y']
    if abs(dx) >= abs(dy):
        return 'W' if dx > 0 else 'E'
    return 'S' if dy > 0 else 'N'


def lane_direction_mask(world, wanted):
    """返回每个 (agent_idx, inter_idx) 的 12 维布尔 mask（True=屏蔽）。"""
    masks = []
    for inter in world.intersections:
        m = np.zeros(12, dtype=bool)
        cursor = 0
        for road in inter.in_roads:
            n = len(road['lanes'])
            if direction_of_road(road) in wanted:
                m[cursor:cursor + n] = True
            cursor += n
        masks.append(m)
    return np.stack(masks)  # [n_inter, 12]


def apply_mask(obs, inter_mask):
    """obs: [n_agent][n_inter, feat]；只清前 12 维 lane_count。"""
    masked = [o.copy() for o in obs]
    for o in masked:
        o[:, :12][inter_mask] = 0.0
    return masked


def run_episode(tr, inter_mask):
    obs = tr.env.reset()
    tr.metric.clear()
    for a in tr.agents:
        a.reset()
    dones = False
    i = 0
    while i < tr.test_steps:
        if i % tr.action_interval == 0:
            phases = np.stack([a.get_phase() for a in tr.agents])
            obs_in = apply_mask(obs, inter_mask) if inter_mask is not None else obs
            actions = np.stack([
                a.get_action(obs_in[k], phases[k], test=True)
                for k, a in enumerate(tr.agents)])
            rewards_list = []
            for _ in range(tr.action_interval):
                obs, rewards, dones, _ = tr.env.step(actions.flatten())
                i += 1
                rewards_list.append(np.stack(rewards))
            tr.metric.update(np.mean(rewards_list, axis=0))
            if all(dones):
                break
        else:
            i += 1
    return tr.metric.real_average_travel_time(), tr.metric.throughput(), tr.metric.queue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--experiment-config', required=True)
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--mask', choices=['none', 'W', 'WE'], default='none')
    ap.add_argument('--episodes', type=int, default=1)
    ap.add_argument('--prefix', default=None)
    args = ap.parse_args()

    wanted = {'W': {'W'}, 'WE': {'W', 'E'}, 'none': set()}[args.mask]
    prefix = args.prefix or f'if_mask_eval_{args.mask}_{int(time.time())}'

    tr = build_trainer(args.experiment_config, prefix)
    inter_mask = lane_direction_mask(tr.world, wanted) if wanted else None
    if inter_mask is not None:
        n_masked = int(inter_mask.sum())
        print(f'[mask] {args.mask}: {n_masked} lane dims masked '
              f'({n_masked}/{inter_mask.size} per step)', flush=True)

    payload = torch.load(args.checkpoint, map_location='cpu')
    state = payload['model_state_dict'] if 'model_state_dict' in payload else payload
    for ag in tr.agents:
        ag.model.load_state_dict(state)
    print(f'[ckpt] loaded {args.checkpoint}', flush=True)

    atts = []
    for ep in range(args.episodes):
        att, thr, queue = run_episode(tr, inter_mask)
        atts.append(att)
        print(f'[eval] mask={args.mask} ep={ep} ATT={att:.4f} '
              f'throughput={int(thr)} queue={queue:.4f}', flush=True)
    print(f'[eval] mask={args.mask} mean_ATT={np.mean(atts):.4f} '
          f'std={np.std(atts):.4f} n={len(atts)}', flush=True)


if __name__ == '__main__':
    main()
