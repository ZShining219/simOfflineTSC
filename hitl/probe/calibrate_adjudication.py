#!/usr/bin/env python
"""calibrate_adjudication.py — 冻结 B′ 判官面板逐判官归一化统计。

对一组固定校准池（expert_pool_manifest，repeatable）逐转移计算判官
raw 分，产出校准后的 adjudication manifest：每个判官条目补
``{mu, sigma, calibration_count}``，供 plan child
``trainer_overrides.adjudication`` 冻结使用。

统计口径与训练期腿侧打分逐位一致：
  obs = batch.observations[i]（state+one_hot，与 _apply_adjudication_leg 同）
  action / next_obs = 档案数组
  q_ratio/gap_norm = leg_features_from_view 按 (run_id,episode_id)
                     分组回放 ServiceGapTracker 重建

用法：
    python hitl/probe/calibrate_adjudication.py \
        --pool output_data/hitl_probe/pools/hitl_ep50/expert_pool_manifest.json \
        --pool output_data/hitl_probe/pools/student_ep50/expert_pool_manifest.json \
        --pool output_data/hitl_probe/pools/expert_ep50/expert_pool_manifest.json \
        --spec adjudication_manifest.json --topology topo.json \
        --out adjudication_manifest_calibrated.json
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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', action='append', required=True,
                        help='expert_pool_manifest.json (repeatable)')
    parser.add_argument('--spec', required=True,
                        help='adjudication judge spec JSON (mu/sigma '
                             'in each judge entry are overwritten)')
    parser.add_argument('--topology', required=True,
                        help='dump_lane_topology JSON')
    parser.add_argument('--action-interval', type=float, default=10.0)
    parser.add_argument('--action-dim', type=int, default=8)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    from sequential.adjudication import (
        Adjudicator, leg_features_from_view, load_topology_ctx)
    from sequential.expert_pool import ExpertPoolArchive

    ctx = load_topology_ctx(args.topology, args.action_interval)
    ctx['action_interval'] = float(args.action_interval)

    spec = json.load(open(args.spec))
    panel = Adjudicator(spec, args.action_dim)

    # Per-judge raw score accumulators, aligned to panel.judges order.
    raws = [[] for _ in panel.judges]
    sources = []
    transition_total = 0
    for pool in args.pool:
        sources.append(os.path.abspath(pool))
        view = ExpertPoolArchive(
            pool, training_seed=0, verify_hashes=False).view()
        features = leg_features_from_view(view, ctx)
        for network in view.networks:
            arrays = view.archive.datasets[network]._arrays[network]
            indices = np.asarray(view.indices_by_network[network])
            batch = view.archive.datasets[network].select_indices(
                network, indices)
            for i, tid in enumerate(batch.transition_ids):
                feat = features[str(tid)]
                feats = {
                    'obs': np.asarray(
                        batch.observations[i], dtype=np.float64),
                    'action': int(batch.actions[i]),
                    'next_obs': np.asarray(
                        batch.next_observations[i], dtype=np.float64),
                    'action_dim': args.action_dim,
                    'q_ratio': np.asarray(feat['q_ratio'], dtype=np.float64),
                    'gap_norm': np.asarray(feat['gap_norm'], dtype=np.float64),
                }
                for j, entry in enumerate(panel.judges):
                    try:
                        raws[j].append(entry['judge'].score(feats))
                    except Exception:
                        raws[j].append(np.nan)
                transition_total += 1

    cal = []
    for j, entry in enumerate(panel.judges):
        values = np.asarray(raws[j], dtype=np.float64)
        values = values[np.isfinite(values)]
        sigma = float(values.std()) if values.size else 0.0
        cal.append({
            'type': entry['type'],
            'mu': float(values.mean()) if values.size else 0.0,
            'sigma': sigma,
            'count': int(values.size),
            'nan_count': int(len(raws[j]) - values.size),
            'collapsed': bool(sigma < 1e-6),
        })
        spec['judges'][j]['mu'] = cal[-1]['mu']
        spec['judges'][j]['sigma'] = cal[-1]['sigma']
        spec['judges'][j]['calibration_count'] = cal[-1]['count']

    spec['calibration'] = {
        'kind': 'hitl_adjudication_calibration',
        'schema_version': 1,
        'sources': sources,
        'sources_digest': hashlib.sha256(
            json.dumps(sorted(sources)).encode()).hexdigest(),
        'transition_count': int(transition_total),
        'judges': cal,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.',
                exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(spec, fh, indent=1)
    print(json.dumps(spec['calibration'], indent=1))


if __name__ == '__main__':
    main()
