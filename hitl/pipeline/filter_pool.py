#!/usr/bin/env python
"""filter_pool.py — 按 behavior_mode 过滤 HITL 专家池，产新池目录。

用途：C′ 通道判别臂——把纠偏池裁到只剩真纠偏转移
（默认 ``hitl_expert_took_over``），检验"伤害来自 95.5% 填充数据
稀释"假设（H3 信息密度）。产物是与 package.py 同构的只读档案：
expert_pool_manifest.json + datasets/.../manifest.json +
trajectory/episodes/*.npz + source_shards.jsonl，全部 sha256 一致，
ExpertPoolArchive(verify_hashes=True) 可直接装载。

用法：
    python hitl/pipeline/filter_pool.py \
        --src data/output_data/hitl_probe/pools/hitl_ep50 \
        --dst data/output_data/hitl_probe/pools/hitl_ep50_tookover \
        --modes hitl_expert_took_over
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EPISODE_KEYS = (
    'episode_id', 'decision_step', 'global_step', 'state', 'current_phase',
    'action', 'reward', 'next_state', 'next_phase', 'terminated', 'truncated',
    'epsilon', 'behavior_mode', 'queue', 'approximate_delay', 'real_delay',
    'throughput', 'waiting_time',
)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--src', required=True, help='源池目录（只读）')
    parser.add_argument('--dst', required=True, help='新池目录（不得已存在）')
    parser.add_argument('--modes', nargs='+',
                      default=['hitl_expert_took_over'],
                      help='保留的 behavior_mode 集合')
    args = parser.parse_args(argv)
    modes = set(args.modes)

    src = os.path.abspath(args.src)
    dst = os.path.abspath(args.dst)
    if os.path.exists(dst):
        raise SystemExit(f'dst already exists: {dst}')
    os.makedirs(os.path.join(dst, 'trajectory', 'episodes'))

    pool_manifest = json.load(
        open(os.path.join(src, 'expert_pool_manifest.json')))
    kept_total = 0
    mode_counts = Counter()
    shard_entries = []
    index_rows = []

    index = os.path.join(src, 'trajectory', 'index.jsonl')
    with open(index, encoding='utf-8') as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            src_npz = os.path.join(src, 'trajectory', row['file'])
            data = np.load(src_npz, allow_pickle=False)
            mask = np.asarray([
                str(m) in modes for m in data['behavior_mode']], dtype=bool)
            keep = int(mask.sum())
            if keep == 0:
                continue
            out = {k: data[k] for k in data.keys() if k not in EPISODE_KEYS}
            for k in EPISODE_KEYS:
                arr = np.asarray(data[k])
                out[k] = arr[mask] if arr.shape[:1] == mask.shape else arr
            dst_npz = os.path.join(dst, 'trajectory', row['file'])
            np.savez(dst_npz, **out)
            sha = _sha256(dst_npz)
            for m in out['behavior_mode']:
                mode_counts[str(m)] += 1
            kept_total += keep
            gs = out['global_step'].astype(np.int64)
            index_rows.append({
                'episode_id': int(row['episode_id']),
                'file': row['file'],
                'first_global_step': int(gs.min()),
                'last_global_step': int(gs.max()),
                'schema_version': 1,
                'sha256': sha,
                'transition_count': keep,
            })
            # source_shards entry: reuse the ORIGINAL shard entry fields,
            # re-point at the filtered npz.
            src_shards = [
                json.loads(l) for l in open(
                    os.path.join(src, 'source_shards.jsonl'))]
            entry = next(
                e for e in src_shards
                if int(e['episode_id']) == int(row['episode_id']))
            entry = dict(entry)
            entry['original_absolute_path'] = os.path.abspath(dst_npz)
            entry['transition_count'] = keep
            entry['sha256'] = sha
            entry['semantic_validation'] = dict(
                entry['semantic_validation'],
                transition_count=keep,
                first_global_step=int(gs.min()),
                last_global_step=int(gs.max()),
            )
            shard_entries.append(entry)

    if not shard_entries:
        raise SystemExit('no transitions matched the requested modes')

    with open(os.path.join(dst, 'trajectory', 'index.jsonl'), 'w') as fh:
        for r in index_rows:
            fh.write(json.dumps(r) + '\n')
    src_index_path = os.path.join(dst, 'source_shards.jsonl')
    with open(src_index_path, 'w') as fh:
        for e in shard_entries:
            fh.write(json.dumps(e) + '\n')
    src_index_sha = _sha256(src_index_path)

    # dataset manifest (sha-pinned into pool manifest)
    ds_dir = os.path.join(dst, 'datasets', pool_manifest['network'], 'hitl')
    os.makedirs(ds_dir)
    src_ds = json.load(open(os.path.join(
        src, pool_manifest['dataset_manifest']['path'])))
    ds = dict(src_ds)
    ds['expected_transition_count'] = kept_total
    ds['source_index'] = os.path.relpath(src_index_path, ds_dir)
    ds['source_index_sha256'] = src_index_sha
    ds['original_source_root'] = os.path.join(dst, 'trajectory')
    ds['episode_range'] = [min(r['episode_id'] for r in index_rows),
                           max(r['episode_id'] for r in index_rows)]
    ds_manifest_path = os.path.join(ds_dir, 'manifest.json')
    with open(ds_manifest_path, 'w') as fh:
        json.dump(ds, fh, indent=1, sort_keys=True)
    ds_sha = _sha256(ds_manifest_path)

    val = json.load(open(os.path.join(
        src, pool_manifest['dataset_manifest']['path'].replace(
            'manifest.json', 'validation.json')))) if os.path.isfile(
        os.path.join(src, pool_manifest['dataset_manifest'][
            'path'].replace('manifest.json', 'validation.json'))) else {}
    val = dict(val)
    val['resolved_source_root'] = os.path.join(dst, 'trajectory')
    val['source_index_sha256'] = src_index_sha
    val['source_shard_count'] = len(shard_entries)
    val['transition_count'] = kept_total
    with open(os.path.join(ds_dir, 'validation.json'), 'w') as fh:
        json.dump(val, fh, indent=1, sort_keys=True)

    # top-level archive manifest (passthrough, paths rebased)
    top = json.load(open(os.path.join(src, 'manifest.json')))
    top = dict(top)
    top['original_source_root'] = os.path.join(dst, 'trajectory')
    with open(os.path.join(dst, 'manifest.json'), 'w') as fh:
        json.dump(top, fh, indent=1, sort_keys=True)

    # labels index (hitl/index.json) — rebased counts
    labels_src = os.path.join(src, 'hitl', 'index.json')
    if os.path.isfile(labels_src):
        os.makedirs(os.path.join(dst, 'hitl'))
        labels = json.load(open(labels_src))
        labels = dict(labels)
        labels['authorship_counts'] = dict(mode_counts)
        labels['label_rows'] = kept_total
        labels.pop('labels_sha256', None)
        with open(os.path.join(dst, 'hitl', 'index.json'), 'w') as fh:
            json.dump(labels, fh, indent=1, sort_keys=True)

    if os.path.isdir(os.path.join(src, 'config')):
        shutil.copytree(os.path.join(src, 'config'),
                        os.path.join(dst, 'config'))

    # pool manifest: same identity, new counts/digests
    pm = dict(pool_manifest)
    pm['transition_count'] = kept_total
    pm['behavior_mode_counts'] = dict(mode_counts)
    pm['episode_ids'] = sorted({r['episode_id'] for r in index_rows})
    pm['dataset_manifest'] = {
        'path': os.path.relpath(ds_manifest_path, dst),
        'sha256': ds_sha,
    }
    pm['pool_digest'] = hashlib.sha256(
        json.dumps({
            'dataset_id': pm['dataset_id'],
            'dataset_manifest_sha256': ds_sha,
            'transition_count': kept_total,
            'modes': sorted(modes),
        }, sort_keys=True).encode()).hexdigest()
    pm['filtered_from'] = {
        'source_pool': src,
        'source_pool_digest': pool_manifest.get('pool_digest'),
        'kept_modes': sorted(modes),
    }
    with open(os.path.join(dst, 'expert_pool_manifest.json'), 'w') as fh:
        json.dump(pm, fh, indent=1, sort_keys=True)

    # Load-check through the real adapter (hash verification on).
    from sequential.expert_pool import ExpertPoolArchive
    view = ExpertPoolArchive(
        os.path.join(dst, 'expert_pool_manifest.json'),
        training_seed=0, verify_hashes=True).view()
    print(json.dumps({
        'kept_total': kept_total,
        'mode_counts': dict(mode_counts),
        'view_transition_count': view.visibility['transition_count'],
        'pool': dst,
    }, indent=1))


if __name__ == '__main__':
    main()
