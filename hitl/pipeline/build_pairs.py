#!/usr/bin/env python
"""build_pairs.py — B 通道数据面：打包档案 → 片段 → 确定性教师分 → 偏好对。

输入一个或多个打包档案目录（package.py 产物），输出：

    <out>/segments.npz   features [N,W,D] + H/Q/U + episode_id + archive_tag
    <out>/pairs.jsonl    spec §5.4 偏好行（clean_y/training_y/skip/delta/H_*）
    <out>/manifest.json  统计：段数/对数/平局率/来源 digest/口径声明

口径差异（对全体臂一致）：Q 用逐车道排队/c；缺逐车道向量时按标量 queue
均摊（c_l 相同则 Q 等价）；u_l 由相位服务史重建（需要 --topology）。

用法：
    python hitl/pipeline/build_pairs.py \
        --archive HITL=output_data/hitl_probe/pools/hitl_ep50 \
        --archive DEMO=output_data/hitl_probe/pools/expert_ep50 \
        --topology hitl/topology/topology_sumohz1x1.json \
        --window 6 --out output_data/hitl_probe/pairs_pilot
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
    build_pairs, label_pair, load_context, load_episode, segment_episode,
    step_features, teacher_series,
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def _episode_files(archive_dir):
    """episode npz paths in global episode order via trajectory/index.jsonl."""
    index = os.path.join(archive_dir, 'trajectory', 'index.jsonl')
    rows = []
    with open(index, encoding='utf-8') as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r['episode_id'])
    return [
        (int(r['episode_id']), os.path.join(archive_dir, 'trajectory',
                                            r['file']))
        for r in rows
    ]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', action='append', required=True,
                        metavar='TAG=DIR')
    parser.add_argument('--topology', required=True)
    parser.add_argument('--window', type=int, default=6,
                        help='segment length in decisions (default 6=60s)')
    parser.add_argument('--tie-tol', type=float, default=0.05)
    parser.add_argument('--max-pairs-per-episode', type=int, default=120)
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--T-u', type=float, default=60.0)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    archives = {}
    for spec in args.archive:
        tag, _, path = spec.partition('=')
        archives[tag] = path
    first_manifest = json.load(open(
        os.path.join(next(iter(archives.values())), 'manifest.json')))
    ctx = load_context(
        args.topology,
        action_interval=float(first_manifest['action_interval']),
        action_dim=int(first_manifest['action_dim']),
        T_u=args.T_u)

    seg_features = []
    seg_meta = []
    pairs = []
    per_archive = {}
    for tag, archive_dir in archives.items():
        ep_arrays = []
        ep_ids = []
        for ep_id, path in _episode_files(archive_dir):
            ep = load_episode(path)
            ep['episode_id'] = np.asarray([ep_id] * len(ep['decision_step']))
            ep_arrays.append(ep)
            ep_ids.append(ep_id)
        # global segment ids: continue numbering across archives
        base = len(seg_meta)
        segs, ep_pairs = build_pairs(
            ep_arrays, ctx, window=args.window, tie_tol=args.tie_tol,
            max_pairs_per_episode=args.max_pairs_per_episode,
            seed=args.seed)
        for i, seg in enumerate(segs):
            feat = step_features(
                ep_arrays[ep_ids.index(seg['episode_id'])], ctx)
            seg_features.append(feat[seg['start']:seg['end']])
            seg_meta.append({
                'archive': tag, 'episode_id': seg['episode_id'],
                'start': seg['start'], 'end': seg['end'],
                'H': seg['H'], 'Q': seg['Q'], 'U': seg['U'],
            })
        for pair in ep_pairs:
            pair['archive'] = tag
            pair['left_segment'] += base
            pair['right_segment'] += base
            pairs.append(pair)
        per_archive[tag] = {
            'episodes': len(ep_arrays),
            'segments': len(segs),
            'pairs': len(ep_pairs),
        }

    features = np.stack(seg_features).astype(np.float32) if seg_features else (
        np.zeros((0, args.window, 0), dtype=np.float32))
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    np.savez_compressed(
        os.path.join(out, 'segments.npz'),
        features=features,
        H=np.asarray([m['H'] for m in seg_meta], dtype=np.float64),
        Q=np.asarray([m['Q'] for m in seg_meta], dtype=np.float64),
        U=np.asarray([m['U'] for m in seg_meta], dtype=np.float64),
        episode_id=np.asarray([m['episode_id'] for m in seg_meta]),
        start=np.asarray([m['start'] for m in seg_meta]),
        end=np.asarray([m['end'] for m in seg_meta]),
        archive=np.asarray([m['archive'] for m in seg_meta]),
    )
    with open(os.path.join(out, 'pairs.jsonl'), 'w', encoding='utf-8') as fh:
        for row in pairs:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True)
                     + '\n')
    ties = sum(1 for p in pairs if p['clean_y'] == 0.5)
    manifest = {
        'kind': 'hitl_preference_pairs',
        'schema_version': 1,
        'teacher': 'h=-0.5*mean(q/c)-0.5*max(min(u/T_u,2))',
        'tie_tol': args.tie_tol,
        'window_decisions': args.window,
        'T_u': args.T_u,
        'seed': args.seed,
        'topology': os.path.abspath(args.topology),
        'topology_sha256': _sha256(args.topology),
        'archives': {t: os.path.abspath(p) for t, p in archives.items()},
        'archive_sha256': {
            t: _sha256(os.path.join(p, 'manifest.json'))
            for t, p in archives.items()
        },
        'per_archive': per_archive,
        'segment_count': len(seg_meta),
        'pair_count': len(pairs),
        'tie_count': ties,
        'tie_rate': (ties / len(pairs)) if pairs else 0.0,
        'feature_dim': int(features.shape[-1]),
        'approximations': [
            'q_l falls back to scalar queue / n_lanes when no per-lane '
            'queue vector was recorded (exact under equal capacities)',
            'u_l reconstructed from phase service history at decision '
            'granularity (dt=action_interval), not per-second base steps',
        ],
    }
    with open(os.path.join(out, 'manifest.json'), 'w', encoding='utf-8') as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    print(json.dumps({k: manifest[k] for k in (
        'segment_count', 'pair_count', 'tie_count', 'tie_rate',
        'feature_dim', 'per_archive')}, indent=1))


if __name__ == '__main__':
    main()
