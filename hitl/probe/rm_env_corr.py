#!/usr/bin/env python
"""rm_env_corr.py — 冻结奖励模型打分 vs 环境奖励相关性（对齐顾第三章 PCC 验证）。

探索件（hitl/probe/，登记 dev/INDEX.md）。她把 IRL 学出的奖励
权重与真实奖励算 PCC；我们把冻结偏好奖励模型对片段打分，与同
一片段的平均环境奖励算 Spearman/Pearson——验证"学出的信号确实
编码了交通语义而非噪声"。片段级口径（RM 本来就是按 6 决策窗训练）。

数据源：三种来源的打包池各取若干 episode（专家/学生/hitl 混合，
顺带检验相关性是否跨源成立）。

用法（repo 根，conda env: colight）：

    python hitl/probe/rm_env_corr.py \
        --out output_data/hitl_probe/reward_model_v1/rm_env_corr.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from hitl.pipeline.segments import (  # noqa: E402
    load_context, load_episode, segment_episode, step_features)

POOLS = {
    'expert': 'output_data/hitl_probe/pools/expert_ep50_big',
    'student': 'output_data/hitl_probe/pools/student_ep50_big',
    'hitl': 'output_data/hitl_probe/pools/hitl_ep50_big',
}
TOPOLOGY = 'hitl/topology/topology_sumohz1x1.json'
MODEL = 'output_data/hitl_probe/reward_model_v1/reward_model.pt'


def _episodes(archive_dir):
    index = Path(archive_dir) / 'trajectory/index.jsonl'
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    return [Path(archive_dir) / 'trajectory' / r['file'] for r in rows]


def _pearson(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _spearman(x, y):
    from scipy.stats import rankdata
    rx, ry = rankdata(x), rankdata(y)
    return _pearson(rx, ry)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--window', type=int, default=6)
    ap.add_argument('--max-episodes', type=int, default=0)
    ap.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    import torch
    from sequential.reward_model import (
        PreferenceRewardNet, predict_segment_scores)

    blob = torch.load(_REPO_ROOT / MODEL, map_location='cpu')
    model = PreferenceRewardNet(int(blob['feature_dim']))
    model.load_state_dict(blob['state_dict'])
    ctx = load_context(_REPO_ROOT / TOPOLOGY, action_interval=10.0,
                       action_dim=8)

    result = {'model': MODEL, 'window': args.window, 'sources': {}}
    all_score, all_rew = [], []
    for name, archive in POOLS.items():
        scores, rewards = [], []
        paths = _episodes(_REPO_ROOT / archive)
        if args.max_episodes:
            paths = paths[:args.max_episodes]
        for path in paths:
            ep = load_episode(path)
            feats = step_features(ep, ctx)
            for seg in segment_episode(ep, ctx, args.window):
                if not seg['valid']:
                    continue
                s = float(predict_segment_scores(
                    model, feats[seg['start']:seg['end']][None])[0])
                r = float(np.asarray(
                    ep['reward'])[seg['start']:seg['end']].mean())
                scores.append(s)
                rewards.append(r)
        result['sources'][name] = {
            'n_segments': len(scores),
            'pearson': _pearson(scores, rewards),
            'spearman': _spearman(scores, rewards),
        }
        all_score += scores
        all_rew += rewards
        print(f"{name}: n={len(scores)} "
              f"pearson={result['sources'][name]['pearson']:.3f} "
              f"spearman={result['sources'][name]['spearman']:.3f}")
    result['overall'] = {
        'n_segments': len(all_score),
        'pearson': _pearson(all_score, all_rew),
        'spearman': _spearman(all_score, all_rew),
    }
    print(f"OVERALL: n={len(all_score)} "
          f"pearson={result['overall']['pearson']:.3f} "
          f"spearman={result['overall']['spearman']:.3f}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1), encoding='utf-8')
    print('wrote', out)


if __name__ == '__main__':
    main()
