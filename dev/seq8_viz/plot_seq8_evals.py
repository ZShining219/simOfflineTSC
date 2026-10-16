#!/usr/bin/env python3
"""SEQ8 CONT / SELF-DHOA-R50 冻结评估可视化（seq8 pilot 已关闭，本脚本为资产分析件）

用法：python dev/seq8_viz/plot_seq8_evals.py [输出目录]（默认 data/output_data/analysis/seq8_figs/）

图A: 训练波动折线 x=global episode(1..800)，每 stage 对当前场景的逐 ep 冻结评估，
     y=travel_time/waiting_time/queue 三子图，均值线 + SD0/SD1 min-max 带。
图B: 下三角柱状图 x=stage 簇，簇内按场景顺序 = stage-k 末冻结 ckpt 对场景 1..k
     的回测值，双 seed 并列柱（SD0 实色 / SD1 半透明）。
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
FAM_TITLE = {'N1': 'NOREP (max-contrast, no revisit)',
             'R1': 'REP (block revisit x2, gap=4)',
             'D1': 'DRIFT (gradual drift)'}
ARMS = {'CONT': 'CONT', 'SELF-DHOA-R50': 'DHOA-R50'}
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

    # ---- 图A：训练波动折线 ----
    for arm_key, arm_tag in ARMS.items():
        for fam in ORDERS:
            fig, axes = plt.subplots(1, 3, figsize=(19, 4.6), sharex=True)
            for ax, (mk, mlab) in zip(axes, METRICS):
                for k in range(1, 9):
                    sc = ORDERS[fam][k-1]
                    arr = np.array([[
                        (by[f'{arm_key}-{fam}-{sd}'].get((k, e, sc)) or {}).get(mk, np.nan)
                        for e in range(1, 101)] for sd in ['SD0', 'SD1']], dtype=float)
                    gx = np.arange((k-1)*100+1, k*100+1)
                    ax.plot(gx, np.nanmean(arr, axis=0), lw=1.4, color='tab:blue')
                    ax.fill_between(gx, np.nanmin(arr, axis=0), np.nanmax(arr, axis=0),
                                    color='tab:blue', alpha=0.22)
                for b in range(100, 800, 100):
                    ax.axvline(b, color='gray', ls='--', lw=0.8, alpha=0.7)
                ax.set_xlim(1, 800); ax.set_ylabel(mlab); ax.set_xlabel('global episode')
                ax.set_xticks([i*100 for i in range(9)])
            for k in range(1, 9):
                axes[0].text((k-0.5)*100, axes[0].get_ylim()[1]*0.97, ORDERS[fam][k-1],
                             ha='center', va='top', fontsize=7, rotation=45, color='dimgray')
            fig.suptitle(f'{fam} ({FAM_TITLE[fam]}) — {arm_tag}: per-episode eval on current scene  [line=mean(SD0,SD1), band=min–max]')
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fn = os.path.join(out_dir, f'A_wave_{fam}_{arm_tag}.png')
            fig.savefig(fn, dpi=150); plt.close(fig); print('saved', fn)

    # ---- 图B：下三角柱状图 ----
    PAL = plt.get_cmap('tab10')
    for arm_key, arm_tag in ARMS.items():
        for fam in ORDERS:
            fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
            for ax, (mk, mlab) in zip(axes, METRICS):
                centers, w = [], 0.38
                x0 = 0.0
                for k in range(1, 9):
                    scenes = ORDERS[fam][:k]
                    for j, sc in enumerate(scenes):
                        for si, sd in enumerate(['SD0', 'SD1']):
                            r = by[f'{arm_key}-{fam}-{sd}'].get((k, 100, sc))
                            if r is None: continue
                            ax.bar(x0 + j*(2*w+0.06) + si*w, r[mk], width=w,
                                   color=PAL(j % 10), alpha=1.0 if si == 0 else 0.45,
                                   edgecolor='k', lw=0.3)
                    ax.axvline(x0 - 0.35, color='gray', ls='-', lw=0.6, alpha=0.5)
                    centers.append(x0 + len(scenes)*(2*w+0.06)/2)
                    x0 += len(scenes)*(2*w+0.06) + 0.6
                ax.axvline(x0 - 0.3, color='gray', ls='-', lw=0.6, alpha=0.5)
                ax.set_xticks(centers); ax.set_xticklabels([f'S{k}' for k in range(1, 9)])
                ax.set_xlabel('stage ckpt (bars: eval scenes in order, color=scene idx)')
                ax.set_ylabel(mlab)
            handles = [plt.Rectangle((0,0),1,1,fc='gray',alpha=1.0),
                       plt.Rectangle((0,0),1,1,fc='gray',alpha=0.45)]
            axes[0].legend(handles, ['SD0','SD1'], loc='upper left', fontsize=8)
            fig.suptitle(f'{fam} ({FAM_TITLE[fam]}) — {arm_tag}: stage-end frozen-ckpt eval on scenes 1..k (lower-triangular)')
            fig.tight_layout(rect=[0, 0, 1, 0.93])
            fn = os.path.join(out_dir, f'B_tri_{fam}_{arm_tag}.png')
            fig.savefig(fn, dpi=150); plt.close(fig); print('saved', fn)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data/output_data/analysis/seq8_figs')
    main(out)
