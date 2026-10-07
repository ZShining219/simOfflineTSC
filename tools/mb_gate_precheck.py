#!/usr/bin/env python3
"""ATT-ENTITY-004 A3 前置检查：现有 replay 的 m_B 分布与有效 LR 乘子分布。

从 resumable checkpoint 的 replay items 中取 last_obs 的文本 embedding
切片（node0，广播文本全节点一致），与 empty-text embedding 距离重建
event_active 标签（离线过渡口径，正式训练改为 transition 显式元数据）。

随后模拟 batch_size 随机抽样，给出：
  * transition 级 event 占比；minibatch 级 m_B 分布（含事件的 batch）
  * α_s = clip(R*/R̄, α_min, 1)（R̄ 取 grad_diag 实测 R_t 轨迹的逐 episode
    中位数与全期常数两种口径）
  * 有效乘子 eff = 1 − m_B(1−α_s) 在"含事件 batch"上的分布
判据（用户给定）：若多数含事件 batch 的 eff > 0.9，则 m_B 门控过弱。

用法：
    python3 tools/mb_gate_precheck.py \
        --ckpt data/output_data/tsc/sumo_tarl_attention/hz4x4/tarlp_att_s7/checkpoints/resumable/episode_0200.pt \
        --grad-diag artifacts/att_entity_003/grad_diag/att_s7.csv \
        --out artifacts/att_entity_004/mb_precheck/att_s7.json
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def empty_embedding():
    """FrozenTextEncoder('') 的 768 维向量（与训练期同一编码器）。"""
    sys_path = str(ROOT / 'reproduction' / 'tarl_tsc')
    import sys
    if sys_path not in sys.path:
        sys.path.insert(0, sys_path)
    from text.encoder import FrozenTextEncoder
    enc = FrozenTextEncoder()
    return enc(['']).squeeze(0).numpy()


def load_event_flags(ckpt_path, empty_vec):
    """返回 (event_active bool array, per-item distance array, meta)。"""
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    ag = ck['agents'][0]
    items = ag['replay_state']['items']
    dists = np.empty(len(items), dtype=np.float64)
    lane_dim = None
    for i, item in enumerate(items):
        _key, vals = item
        last_obs = np.asarray(vals[0], dtype=np.float32)  # (nodes, lane_dim+768)
        if lane_dim is None:
            lane_dim = last_obs.shape[-1] - 768
            assert lane_dim > 0, f'unexpected obs dim {last_obs.shape}'
        e = last_obs[0, lane_dim:lane_dim + 768]
        dists[i] = np.linalg.norm(e - empty_vec)
    # 双峰分割：empty 距离应聚集在 ~0。取 Otsu 式分界 = min+0.5*(p50_low,p10_high gap)
    # 稳健做法：empty 样本 dist≈0；阈值 = max(1e-3, median_of_smallest_half*10)
    srt = np.sort(dists)
    half = srt[: len(srt) // 2]
    thr = max(1e-3, float(np.median(half)) * 10 + 1e-6)
    flags = dists > thr
    meta = {'capacity': ag['replay_state']['capacity'],
            'items': len(items), 'lane_dim': int(lane_dim),
            'episode': int(ck.get('episode', -1)),
            'dist_thr': thr,
            'dist_min': float(dists.min()), 'dist_p50': float(np.median(dists)),
            'dist_max': float(dists.max())}
    return flags, dists, meta


def grad_ratio_series(csv_path):
    """grad_diag csv → {episode: median(norm_dL_ds/norm_dL_de)} + 全池中位数。"""
    per_ep = defaultdict(list)
    with open(csv_path, newline='') as h:
        for r in csv.DictReader(h):
            de, ds = float(r['norm_dL_de']), float(r['norm_dL_ds'])
            if de > 0:
                per_ep[int(r['episode'])].append(ds / de)
    med = {ep: float(np.median(v)) for ep, v in sorted(per_ep.items())}
    allr = [x for v in per_ep.values() for x in v]
    return med, float(np.median(allr))


def simulate(flags, batch_size, n_draws, alpha_fn, seed=0):
    """蒙特卡洛 minibatch m_B → eff 乘子分布。"""
    rng = np.random.default_rng(seed)
    n = len(flags)
    p = flags.mean()
    # 二项近似足够（50000 母体，128 样本无放回的差异 <1%）
    m_b = rng.binomial(batch_size, p, size=n_draws) / batch_size
    has_event = m_b > 0
    alpha = alpha_fn()
    eff = 1.0 - m_b * (1.0 - alpha)
    return {
        'transition_event_frac': float(p),
        'batch_event_frac_mean': float(m_b.mean()),
        'batch_event_frac_p50': float(np.median(m_b)),
        'batch_event_frac_p90': float(np.percentile(m_b, 90)),
        'frac_batches_with_event': float(has_event.mean()),
        'm_b_given_event_p10': float(np.percentile(m_b[has_event], 10)),
        'm_b_given_event_p50': float(np.percentile(m_b[has_event], 50)),
        'm_b_given_event_p90': float(np.percentile(m_b[has_event], 90)),
        'alpha_s': float(alpha),
        'eff_given_event_p10': float(np.percentile(eff[has_event], 10)),
        'eff_given_event_p50': float(np.percentile(eff[has_event], 50)),
        'eff_given_event_p90': float(np.percentile(eff[has_event], 90)),
        'eff_overall_p50': float(np.percentile(eff, 50)),
        'frac_event_batches_eff_gt_0.9': float((eff[has_event] > 0.9).mean()),
        'frac_all_batches_eff_gt_0.9': float((eff > 0.9).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--grad-diag', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--batch-size', type=int, default=128)
    ap.add_argument('--n-draws', type=int, default=20000)
    ap.add_argument('--r-star', type=float, default=3.0)
    ap.add_argument('--alpha-min', type=float, default=0.1)
    args = ap.parse_args()

    empty = empty_embedding()
    flags, dists, meta = load_event_flags(args.ckpt, empty)
    r_med, r_pool = grad_ratio_series(args.grad_diag)

    # α_s 两种口径：全期中位 R̄（稳态近似）与逐 episode 中位 R̄（观察动态范围）
    def alpha_of(r):
        return float(np.clip(args.r_star / r, args.alpha_min, 1.0))

    out = {'ckpt': args.ckpt, 'meta': meta,
           'R_t_median_per_episode': r_med, 'R_t_pooled_median': r_pool,
           'alpha_s_pooled': alpha_of(r_pool),
           'alpha_s_per_episode': {ep: alpha_of(r) for ep, r in r_med.items()},
           'simulations': {}}
    out['simulations']['alpha_pooled'] = simulate(
        flags, args.batch_size, args.n_draws, lambda: alpha_of(r_pool))
    for ep, r in r_med.items():
        out['simulations'][f'alpha_ep{ep}'] = simulate(
            flags, args.batch_size, args.n_draws, lambda r=r: alpha_of(r), seed=ep)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as h:
        json.dump(out, h, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False))
    print('event transition frac:', out['simulations']['alpha_pooled']['transition_event_frac'])
    print('R_t per-episode medians:', r_med, '| pooled:', round(r_pool, 2))
    for name, s in out['simulations'].items():
        print(f"{name:14s} α_s={s['alpha_s']:.3f} m_B|ev p50={s['m_b_given_event_p50']:.3f} "
              f"eff|ev p10/50/90={s['eff_given_event_p10']:.3f}/{s['eff_given_event_p50']:.3f}/{s['eff_given_event_p90']:.3f} "
              f"P(eff>0.9|ev)={s['frac_event_batches_eff_gt_0.9']:.3f}")


if __name__ == '__main__':
    main()
