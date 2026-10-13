#!/usr/bin/env python
"""build_demos.py — A 通道示范集构建（spec §8 pilot 档）。

从打包档案抽 (obs, accepted_action, reward, next_obs, done, episode_id)
扁平数组，保持 episode 内时序连续（n-step 后缀按 episode_id 分组推导，
不打乱）。spec 的 1800 转移配额来自 5 个 SOTL 配置；本 pilot 用既有
MP 控盘池（expert_ep50）——偏差记入实验记录，不报等效。

用法：
    python hitl/pipeline/build_demos.py \
        --archive data/output_data/hitl_probe/pools/expert_ep50 \
        --out data/output_data/hitl_probe/demos/demo_expert_ep50.npz
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

from hitl.pipeline.segments import _obs_rows, load_episode  # noqa: E402


def _episode_paths(archive_dir):
    index = os.path.join(archive_dir, 'trajectory', 'index.jsonl')
    rows = []
    with open(index, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r['episode_id'])
    return [(int(r['episode_id']),
             os.path.join(archive_dir, 'trajectory', r['file']))
            for r in rows]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='append', required=True)
    parser.add_argument('--action-dim', type=int, default=8)
    parser.add_argument('--behavior-modes', nargs='*', default=None,
                        help='keep only these behavior_mode labels '
                             '(default: all executed transitions)')
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    obs_all, act_all, rew_all, nobs_all, done_all, ep_all = [], [], [], [], [], []
    sources = []
    for archive in args.archive:
        sources.append(os.path.abspath(archive))
        for ep_id, path in _episode_paths(archive):
            ep = load_episode(path)
            obs = _obs_rows(ep, args.action_dim)
            next_state = np.asarray(ep['next_state'], dtype=np.float32)
            if next_state.ndim == 3:
                next_state = next_state[:, 0, :]
            next_phase = np.asarray(
                ep.get('next_phase', ep['current_phase']), dtype=int
            ).reshape(-1)
            nxt_oh = np.zeros((len(next_phase), args.action_dim),
                              dtype=np.float32)
            for t, p in enumerate(next_phase):
                if 0 <= p < args.action_dim:
                    nxt_oh[t, p] = 1.0
            next_obs = np.concatenate([next_state, nxt_oh], axis=1)
            reward = np.asarray(ep['reward'], dtype=np.float32).reshape(-1)
            term = np.asarray(ep.get('terminated', np.zeros(len(reward))),
                              dtype=bool).reshape(-1)
            trunc = np.asarray(ep.get('truncated', np.zeros(len(reward))),
                             dtype=bool).reshape(-1)
            keep = np.ones(len(reward), dtype=bool)
            if args.behavior_modes and 'behavior_mode' in ep:
                modes = np.asarray(ep['behavior_mode']).reshape(-1)
                keep = np.isin(modes, args.behavior_modes)
            n = int(keep.sum())
            if n == 0:
                continue
            obs_all.append(obs[keep])
            act_all.append(
                np.asarray(ep['action'], dtype=np.int64).reshape(-1)[keep])
            rew_all.append(reward[keep])
            nobs_all.append(next_obs[keep])
            done_all.append((term | trunc)[keep])
            ep_all.append(np.full(n, ep_id, dtype=np.int64))
    blob = {
        'obs': np.concatenate(obs_all).astype(np.float32),
        'action': np.concatenate(act_all),
        'reward': np.concatenate(rew_all),
        'next_obs': np.concatenate(nobs_all).astype(np.float32),
        'done': np.concatenate(done_all),
        'episode_id': np.concatenate(ep_all),
    }
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
    np.savez_compressed(out, **blob)
    manifest = {
        'kind': 'hitl_demo_dataset',
        'schema_version': 1,
        'sources': sources,
        'transition_count': int(len(blob['action'])),
        'episode_count': int(len(set(blob['episode_id'].tolist()))),
        'obs_dim': int(blob['obs'].shape[1]),
        'action_dim': args.action_dim,
        'behavior_mode_filter': args.behavior_modes,
        'npz_sha256': hashlib.sha256(
            open(out, 'rb').read()).hexdigest(),
    }
    with open(out.replace('.npz', '_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
