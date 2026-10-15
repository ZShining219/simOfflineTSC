#!/usr/bin/env python
"""calibrate_reward.py — 冻结 B 通道归一化统计（spec §6.3）。

对一组固定校准档案逐转移计算：
  * h_t    —— 规则教师分（B1 用）
  * f_hat_t —— 冻结奖励网络前向分（B2 用，需 --model）

产出 reward_calibration.json:
    {kind, schema_version, calibration_sources, clip,
     h:   {mu, sigma, count},
     f_hat: {mu, sigma, count}   # 仅提供 --model 时}

注意：统计只在校准集上拟合一次后冻结；B2 崩溃（sigma≈0）时
在线塑形按 spec 回退 r0。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from hitl.pipeline.segments import (  # noqa: E402
    load_context, load_episode, step_features, teacher_series)


def _episode_paths(archive_dir):
    """episode npz paths in episode order via trajectory/index.jsonl."""
    index = os.path.join(archive_dir, 'trajectory', 'index.jsonl')
    rows = []
    with open(index, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r['episode_id'])
    return [os.path.join(archive_dir, 'trajectory', r['file'])
            for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='append', required=True,
                        help='packaged archive dir (repeatable)')
    parser.add_argument('--topology', required=True,
                        help='lane topology JSON (dump_lane_topology.py)')
    parser.add_argument('--action-interval', type=float, default=10.0)
    parser.add_argument('--action-dim', type=int, default=8)
    parser.add_argument('--model', default=None,
                        help='frozen reward_model.pt (for f_hat stats)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    ctx = load_context(args.topology, args.action_interval, args.action_dim)
    h_all, f_all = [], []
    net = None
    if args.model:
        import torch
        from sequential.reward_model import PreferenceRewardNet
        blob = torch.load(args.model, map_location='cpu')
        net = PreferenceRewardNet(int(blob['feature_dim']))
        net.load_state_dict(blob['state_dict'])
        net.eval()
    sources = []
    for archive in args.archive:
        sources.append(os.path.abspath(archive))
        for npz in _episode_paths(archive):
            ep = load_episode(npz)
            h, _, _, _, _ = teacher_series(ep, ctx)
            h_all.append(h)
            if net is not None:
                import torch
                x = torch.as_tensor(step_features(ep, ctx))
                with torch.no_grad():
                    f_all.append(net(x).numpy().reshape(-1))
    h_cat = np.concatenate(h_all) if h_all else np.zeros(0)
    result = {
        'kind': 'hitl_reward_calibration',
        'schema_version': 1,
        'clip': 3.0,
        'calibration_sources': sources,
        'sources_digest': hashlib.sha256(
            json.dumps(sorted(sources)).encode()).hexdigest(),
        'h': {
            'mu': float(h_cat.mean()), 'sigma': float(h_cat.std()),
            'count': int(h_cat.size),
            'collapsed': bool(h_cat.std() < 1e-6),
        },
    }
    if net is not None:
        f_cat = np.concatenate(f_all) if f_all else np.zeros(0)
        result['f_hat'] = {
            'mu': float(f_cat.mean()), 'sigma': float(f_cat.std()),
            'count': int(f_cat.size),
            'collapsed': bool(f_cat.std() < 1e-6),
        }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(result, fh, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
