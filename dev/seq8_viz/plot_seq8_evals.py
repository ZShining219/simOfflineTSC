#!/usr/bin/env python3
"""SEQ8 CONT / SELF-DHOA-R50 冻结评估可视化（seq8 pilot 已关闭，本脚本为资产分析件）

用法：python dev/seq8_viz/plot_seq8_evals.py [输出目录]（默认 data/output_data/analysis/seq8_figs/）

产出两张拼图（6行×3列，行=N1/R1/D1 各含 CONT+DHOA-R50 两臂相邻，
列=travel_time/waiting_time/queue；同列共享 y 轴便于跨图对比）：
- A_wave_all.png：训练波动折线，x=global episode(1..800)，每 stage 对当前
  场景的逐 ep 冻结评估，均值线 + SD0/SD1 min-max 带，stage 分界竖线。
- B_tri_all.png：下三角回测柱状图，簇=eval state 位置 j（S1..S8），簇内按
  时间序排 stage k≥j 末冻结 ckpt 回测（柱色=ckpt stage），双 seed 并列柱
  （SD0 实色 / SD1 半透明）。
数据源：<batch>/<run>/attempts/attempt_1/events.jsonl 的 evaluation:* 操作
  → _shared_evaluation/aliases → physical/*/attempt_1/summary.json。
"""
import json, os, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BATCHES = {
 'N1': 'data/output_data/sequential/seq8_norep_formal_20261015',
 'R1': 'data/output_data/sequential/seq8_rep_formal_20261015',
 'D1': 'data/output_data/sequential/seq8_drift_formal_20261015'}
ORDERS = {
 'N1': ['sparse_ultra','fluct_burst','lt_extreme','asym_wonly','dense_extreme','st_dom','asym_we','fluct_ramp'],
 'R1': ['sparse_low','lt_high','fluct_burst','dense_high','sparse_low','lt_high','fluct_burst','dense_high'],
 'D1': ['st_dom','sumohz1x1','lt_high','lt_extreme','lt_wsing','asym_wonly','asym_we','asym_ns']}
ARMS = {'CONT': 'CONT', 'SELF-DHOA-R50': 'DHOA'}
ROWS = [(fam, arm) for fam in ['N1', 'R1', 'D1'] for arm in ['CONT', 'SELF-DHOA-R50']]
METRICS = [('tt', 'Travel Time (s)'), ('wait', 'Waiting Time (s)'), ('queue', 'Queue (veh)')]


def harvest():
    recs = []
    for fam, bd in BATCHES.items():
        shared = os.path.join(ROOT, bd, '_shared_evaluation')
        for arm in ARMS:
            for sd in ['SD0', 'SD1']:
                run = f'{arm}-{fam}-{sd}'
                ev = os.path.join(ROOT, bd, run, 'attempts/attempt_1/events.jsonl')
                if not os.path.exists(ev):
                    print('skip missing', run); continue
                n = miss = 0
                for l in open(ev):
                    e = json.loads(l)
                    k = e.get('payload', {}).get('operation_key', '')
                    if not k.startswith('evaluation:'): continue
                    _, st, loc, net = k.split(':')
                    try:
                        a = json.load(open(e['payload']['artifact']))
                        s = json.load(open(os.path.join(
                            shared, 'physical', a['physical_key'],
                            'attempt_1', 'summary.json')))
                        recs.append(dict(run=run, fam=fam, sd=sd,
                            stage=int(st.split('_')[1]), ep=int(loc.split('_')[1]),
                            eval_net=net.replace('sumohz1x1_', ''),
                            tt=s['travel_time'], queue=s['queue'], thr=s['throughput'],
                            unfin=s['unfinished_vehicles'], wait=s['waiting_time']))
                        n += 1
                    except Exception:
                        miss += 1
                print(f'{run}: evals={n} miss={miss}')
    return recs


def main(out_dir):
    recs = harvest()
    by = defaultdict(dict)
    for r in recs:
        by[r['run']][(r['stage'], r['ep'], r['eval_net'])] = r
    os.makedirs(out_dir, exist_ok=True)

    # ---- 每列统一 y 轴：先算全 6 行该指标的 min/max ----
    def yrange(metric):
        vals = []
        for fam, arm in ROWS:
            for sd in ['SD0', 'SD1']:
                for r in by[f'{arm}-{fam}-{sd}'].values():
                    vals.append(r[metric])
        lo, hi = min(vals), max(vals)
        return lo - (hi - lo) * 0.03, hi + (hi - lo) * 0.08
    YL = {mk: yrange(mk) for mk, _ in METRICS}

    # ---- 图A：训练波动折线（6行×3列）----
    fig, axes = plt.subplots(6, 3, figsize=(19, 21), sharex=True)
    for ri, (fam, arm) in enumerate(ROWS):
        for ci, (mk, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            for k in range(1, 9):
                sc = ORDERS[fam][k-1]
                arr = np.array([[
                    (by[f'{arm}-{fam}-{sd}'].get((k, e, sc)) or {}).get(mk, np.nan)
                    for e in range(1, 101)] for sd in ['SD0', 'SD1']], dtype=float)
                gx = np.arange((k-1)*100+1, k*100+1)
                ax.plot(gx, np.nanmean(arr, axis=0), lw=1.2, color='tab:blue')
                ax.fill_between(gx, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                                color='tab:blue', alpha=0.22)
            for b in range(100, 800, 100):
                ax.axvline(b, color='gray', ls='--', lw=0.8, alpha=0.7)
            ax.set_ylim(YL[mk]); ax.set_xlim(1, 800)
            if ri == 5:
                ax.set_xlabel('global episode'); ax.set_xticks([i*100 for i in range(9)])
            if ci == 0:
                ax.set_ylabel(f'{fam}-{ARMS[arm]}\n{mlab}', fontsize=9)
                for k in range(1, 9):
                    ax.text((k-0.5)*100, YL[mk][1]*0.97, ORDERS[fam][k-1],
                            ha='center', va='top', fontsize=6, rotation=45, color='dimgray')
            else:
                ax.set_ylabel(mlab)
    fig.suptitle('SEQ8 per-episode eval on current scene  [line=mean(SD0,SD1), band=min–max; y shared per column]', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fn = os.path.join(out_dir, 'A_wave_all.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)

    # ---- 图B：下三角柱状图（6行×3列；簇=state 位置 j，簇内 ckpt k≥j 时间序）----
    PAL = plt.get_cmap('tab10')
    fig, axes = plt.subplots(6, 3, figsize=(19, 23), sharex=True)
    w = 0.38
    for ri, (fam, arm) in enumerate(ROWS):
        for ci, (mk, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            centers = []
            x0 = 0.0
            for j in range(8):
                sc = ORDERS[fam][j]
                npos = 8 - j
                for i, k in enumerate(range(j+1, 9)):
                    for si, sd in enumerate(['SD0', 'SD1']):
                        r = by[f'{arm}-{fam}-{sd}'].get((k, 100, sc))
                        if r is None: continue
                        ax.bar(x0 + i*(2*w+0.06) + si*w, r[mk], width=w,
                               color=PAL((k-1) % 10), alpha=1.0 if si == 0 else 0.45,
                               edgecolor='k', lw=0.3)
                ax.axvline(x0 - 0.35, color='gray', ls='-', lw=0.6, alpha=0.5)
                centers.append(x0 + npos*(2*w+0.06)/2)
                x0 += npos*(2*w+0.06) + 0.6
            ax.axvline(x0 - 0.3, color='gray', ls='-', lw=0.6, alpha=0.5)
            ax.set_ylim(YL[mk])
            ax.set_xticks(centers)
            ax.set_xticklabels([f'S{j+1}' for j in range(8)] if ri == 5 else [])
            if ci == 0:
                ax.set_ylabel(f'{fam}-{ARMS[arm]}\n{mlab}', fontsize=9)
            else:
                ax.set_ylabel(mlab)
            if ri == 5:
                ax.set_xlabel('eval state (bars: ckpt stage k≥j in time order, color=ckpt stage)')
            if ri == 0 and ci == 0:
                handles = [plt.Rectangle((0,0),1,1,fc='gray',alpha=1.0),
                           plt.Rectangle((0,0),1,1,fc='gray',alpha=0.45)]
                ax.legend(handles, ['SD0','SD1'], loc='upper left', fontsize=8)
    fig.suptitle('SEQ8 stage-end frozen-ckpt eval, clusters=eval state j, bars=ckpt stages k≥j in time order (y shared per column)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    fn = os.path.join(out_dir, 'B_tri_all.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data/output_data/analysis/seq8_figs')
    main(out)
