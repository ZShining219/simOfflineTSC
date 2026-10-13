#!/usr/bin/env python
"""train_reward.py — 训练偏好奖励模型（spec §9.1 pilot 档）。

split 按 episode 划分（防泄漏）：fit=episode 训练集、val=留出集。
产物（<out>/）：
    reward_model.pt        冻结 state_dict + feature_dim
    normalization.json     fit 集上 f 的 mu/sigma（冻结，不再重算）
    training_log.json      逐 epoch BCE + 早停记录 + 留出指标
    reward_manifest.json   配置/hash/标签分布/指标

用法：
    python hitl/pipeline/train_reward.py \
        --pairs output_data/hitl_probe/pairs_pilot \
        --out output_data/hitl_probe/reward_model_v1 --seed 11
"""

from __future__ import annotations

import argparse
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
    parser.add_argument('--pairs', required=True,
                        help='build_pairs.py output dir')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--epochs', type=int, default=30,
                        help='pilot cap (formal 100)')
    parser.add_argument('--patience', type=int, default=5,
                        help='early-stop on val log-loss (formal 10)')
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--val-episode-frac', type=float, default=0.25)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    import torch
    from sequential.reward_model import (
        PreferenceRewardNet, evaluate_pairs, pair_loss,
        predict_segment_scores)

    seg = np.load(os.path.join(args.pairs, 'segments.npz'),
                  allow_pickle=False)
    features = seg['features'].astype(np.float32)
    episodes = seg['episode_id']
    pairs = []
    with open(os.path.join(args.pairs, 'pairs.jsonl')) as fh:
        for line in fh:
            if line.strip():
                pairs.append(json.loads(line))

    rng = np.random.RandomState(args.seed)
    unique_eps = sorted(set(int(e) for e in episodes))
    rng.shuffle(unique_eps)
    n_val = max(1, int(round(len(unique_eps) * args.val_episode_frac)))
    val_eps = set(unique_eps[:n_val])
    seg_ep = {i: int(episodes[i]) for i in range(len(episodes))}
    fit_pairs = [p for p in pairs
                 if seg_ep[p['left_segment']] not in val_eps
                 and seg_ep[p['right_segment']] not in val_eps]
    val_pairs = [p for p in pairs
                 if seg_ep[p['left_segment']] in val_eps
                 or seg_ep[p['right_segment']] in val_eps]

    torch.manual_seed(args.seed)
    model = PreferenceRewardNet(int(features.shape[-1]))
    opt = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    x_all = torch.as_tensor(features)
    log = []
    best_val = float('inf')
    best_state = None
    stale = 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(len(fit_pairs))
        losses = []
        for i in range(0, len(order), args.batch):
            batch = [fit_pairs[j] for j in order[i:i + args.batch]]
            xl = x_all[[p['left_segment'] for p in batch]]
            xr = x_all[[p['right_segment'] for p in batch]]
            y = torch.as_tensor(
                [p['training_y'] for p in batch], dtype=torch.float32)
            opt.zero_grad()
            loss = pair_loss(model, xl, xr, y)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        val = evaluate_pairs(model, features, val_pairs)
        fit_bce = float(np.mean(losses)) if losses else None
        log.append({
            'epoch': epoch, 'fit_bce': fit_bce,
            'val_bce': val['bce'],
            'val_strict_acc': val['strict_accuracy'],
        })
        if val['bce'] is not None and val['bce'] < best_val - 1e-6:
            best_val = val['bce']
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # freeze normalization from the *fit* segments only (spec §6.3)
    fit_seg_ids = sorted({p['left_segment'] for p in fit_pairs}
                         | {p['right_segment'] for p in fit_pairs})
    f_fit = predict_segment_scores(model, features[fit_seg_ids])
    mu = float(np.mean(f_fit))
    sigma = float(max(np.std(f_fit), 1e-6))
    collapsed = sigma < 1e-6

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    torch.save({
        'state_dict': model.state_dict(),
        'feature_dim': int(features.shape[-1]),
        'window': int(features.shape[1]),
        'temperature': 0.25,
    }, os.path.join(out, 'reward_model.pt'))
    with open(os.path.join(out, 'normalization.json'), 'w') as fh:
        json.dump({'mu': mu, 'sigma': sigma, 'clip': 3.0}, fh, indent=1)
    val_metrics = evaluate_pairs(model, features, val_pairs)
    fit_metrics = evaluate_pairs(model, features, fit_pairs)
    with open(os.path.join(out, 'training_log.json'), 'w') as fh:
        json.dump({'epochs': log, 'early_stopped': stale >= args.patience,
                   'best_val_bce': best_val}, fh, indent=1)
    manifest = {
        'kind': 'hitl_reward_model',
        'schema_version': 1,
        'seed': args.seed,
        'pairs_dir': os.path.abspath(args.pairs),
        'feature_dim': int(features.shape[-1]),
        'window': int(features.shape[1]),
        'split': 'by_episode',
        'fit_pairs': len(fit_pairs), 'val_pairs': len(val_pairs),
        'fit_episodes': len(unique_eps) - n_val,
        'val_episodes': n_val,
        'mu': mu, 'sigma': sigma, 'reward_collapsed': collapsed,
        'fit_metrics': fit_metrics, 'val_metrics': val_metrics,
        'label_distribution': {
            'y1': sum(1 for p in pairs if p['clean_y'] == 1.0),
            'y05': sum(1 for p in pairs if p['clean_y'] == 0.5),
            'y0': sum(1 for p in pairs if p['clean_y'] == 0.0),
            'skipped': sum(1 for p in pairs if p.get('skip')),
        },
    }
    with open(os.path.join(out, 'reward_manifest.json'), 'w') as fh:
        json.dump(manifest, fh, indent=1)
    print(json.dumps({k: manifest[k] for k in (
        'fit_pairs', 'val_pairs', 'mu', 'sigma', 'reward_collapsed',
        'fit_metrics', 'val_metrics', 'label_distribution')}, indent=1))


if __name__ == '__main__':
    main()
