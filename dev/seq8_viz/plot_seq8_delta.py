#!/usr/bin/env python3
"""SEQ8 三臂遗忘「相对漂移」可视化：Δ% 柱状 / 分布 ECDF / 遗忘距离曲线

用法：python dev/seq8_viz/plot_seq8_delta.py [输出目录]（默认 data/output_data/analysis/seq8_figs/）

Δ% 口径：格(j,k)=stage k 末冻结 ckpt 对位置 j 场景的回测值，
基准=该位置自身 stage j 末回测值；Δ%=(v/b-1)*100。
R1 回访场景按位置分别计（S5 基准=st5 末，与 S1 独立）。

产出三张图：
- C_delta_bars.png：9行×3列 Δ% 柱状（同 B 图版式，y=漂移%）
- D_delta_ecdf.png：1×3 ECDF（每指标一子图，三臂各一曲线；
  读法：曲线越靠左下=遗忘越少，P(Δ≤x) 直接可读）
- E_forget_curve.png：1×3 遗忘曲线（x=stage 距离 d=k-j，
  y=该距离全部格 Δ% 均值线+min-max 带，三臂三线）

数据源同 plot_seq8_evals.py（复用其 harvest/harvest_anchor）。
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from plot_seq8_evals import harvest, harvest_anchor, ORDERS, ROWS, METRICS, ROOT

ARMS3 = ['CONT', 'SELF-DHOA-R50', 'ANCHOR']
ARM_LAB = {'CONT': 'CONT', 'SELF-DHOA-R50': 'DHOA', 'ANCHOR': 'ANCHOR'}
ARM_COL = {'CONT': 'tab:red', 'SELF-DHOA-R50': 'tab:orange', 'ANCHOR': 'tab:blue'}


def delta_table(by):
    """返回 {(fam,arm,sd): {(j,k): {mk: Δ%}}}，j/k 均 1-idx stage 位"""
    out = {}
    for fam, arm in ROWS:
        for sd in ['SD0', 'SD1']:
            d = by[f'{arm}-{fam}-{sd}']
            cells = {}
            for j in range(1, 9):
                sc = ORDERS[fam][j - 1]
                b = d.get((j, 100, sc))
                if not b:
                    continue
                for k in range(j, 9):
                    v = d.get((k, 100, sc))
                    if not v:
                        continue
                    cells[(j, k)] = {
                        mk: ((v[mk] - b[mk]) / b[mk] * 100
                             if abs(b[mk]) > 1e-9 else np.nan)
                        for mk, _ in METRICS}
            out[(fam, arm, sd)] = cells
    return out


def all_cells(dt, arm, mk):
    vals = []
    for fam in ['N1', 'R1', 'D1']:
        for sd in ['SD0', 'SD1']:
            for (j, k), m in dt[(fam, arm, sd)].items():
                if k > j and not np.isnan(m[mk]):  # 排除对角线+无效基线
                    vals.append(m[mk])
    return np.array(vals)


def fig_c(dt, out_dir):
    """Δ% 柱状：9行×3列，簇=S1..S8，簇内 ckpt k>j 时间序"""
    PAL = plt.get_cmap('tab10')
    fig, axes = plt.subplots(9, 3, figsize=(19, 30), sharex=True)
    w = 0.38
    # 同列共享 y：全 9 行同指标 Δ% 的 min/max 统一标度
    yls = {}
    for mk, _ in METRICS:
        vals = [m[mk] for c in dt.values() for m in c.values()
                if not np.isnan(m[mk])]
        # 99 分位封顶：WT/Q 小基线格会产生数千 % 离群，截断保可读性
        lo = min(min(vals), -10)
        hi = np.nanpercentile(vals, 99)
        yls[mk] = (max(lo, -np.nanpercentile(np.abs(vals), 99) * 1.05 - 8),
                   max(hi, 10) * 1.06)
    for ri, (fam, arm) in enumerate(ROWS):
        for ci, (mk, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            yl = yls[mk]
            x0 = 0.0; centers = []
            for j in range(1, 9):
                npos = 8 - j
                for i, k in enumerate(range(j + 1, 9)):
                    for si, sd in enumerate(['SD0', 'SD1']):
                        c = dt[(fam, arm, sd)].get((j, k))
                        if c is None or np.isnan(c[mk]):
                            continue
                        ax.bar(x0 + i * (2 * w + 0.06) + si * w, c[mk],
                               width=w, color=PAL((k - 1) % 10),
                               alpha=1.0 if si == 0 else 0.45,
                               edgecolor='k', lw=0.3)
                ax.axvline(x0 - 0.35, color='gray', ls='-', lw=0.6, alpha=0.5)
                centers.append(x0 + npos * (2 * w + 0.06) / 2)
                x0 += npos * (2 * w + 0.06) + 0.6
            ax.axvline(x0 - 0.3, color='gray', ls='-', lw=0.6, alpha=0.5)
            ax.axhline(0, color='k', lw=0.7)
            ax.axhline(10, color='red', ls=':', lw=0.6, alpha=0.6)
            ax.set_ylim(yl)
            ax.set_xticks(centers)
            ax.set_xticklabels([f'S{j}' for j in range(1, 9)]
                              if ri == 8 else [])
            ax.set_ylabel(f'{fam}-{ARM_LAB[arm]}\n{mlab} Δ%' if ci == 0
                          else f'{mlab} Δ%', fontsize=9)
            if ri == 8:
                ax.set_xlabel('eval state (bars: ckpt stage k>j in time order)')
    fig.suptitle('SEQ8 forgetting Δ% vs own-stage-end baseline (y shared per '
                 'column, clipped at 99th pct; dashed=+10%; color=ckpt stage; '
                 'SD0 solid / SD1 half)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fn = os.path.join(out_dir, 'C_delta_bars.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)


def fig_d(dt, out_dir):
    """ECDF：每指标一子图，三臂遗忘 Δ% 累积分布"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ci, (mk, mlab) in enumerate(METRICS):
        ax = axes[ci]
        for arm in ARMS3:
            v = np.sort(all_cells(dt, arm, mk))
            y = np.arange(1, len(v) + 1) / len(v)
            ax.plot(v, y, lw=1.8, color=ARM_COL[arm],
                    label=f'{ARM_LAB[arm]} (n={len(v)}, mean={v.mean():+.1f}%)')
        ax.axvline(0, color='k', lw=0.7)
        ax.axvline(10, color='red', ls=':', lw=0.8)
        ax.set_xlim(-60, 200)
        ax.set_xlabel(f'{mlab} forgetting Δ%')
        ax.set_ylabel('P(Δ% ≤ x)')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc='lower right')
    fig.suptitle('SEQ8 forgetting Δ% ECDF — lower-triangle cells (k>j), '
                 'pooled over N1/R1/D1 × SD0/SD1', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = os.path.join(out_dir, 'D_delta_ecdf.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)


def fig_e(dt, out_dir):
    """遗忘曲线：x=stage距离 d=k-j，y=该距离全部格 Δ% 均值+min-max 带"""
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))
    for ci, (mk, mlab) in enumerate(METRICS):
        ax = axes[ci]
        for arm in ARMS3:
            byd = defaultdict(list)
            for fam in ['N1', 'R1', 'D1']:
                for sd in ['SD0', 'SD1']:
                    for (j, k), m in dt[(fam, arm, sd)].items():
                        if k > j and not np.isnan(m[mk]):
                            byd[k - j].append(m[mk])
            ds = sorted(byd)
            mean = np.array([np.nanmean(byd[d]) for d in ds])
            lo = np.array([np.nanmin(byd[d]) for d in ds])
            hi = np.array([np.nanmax(byd[d]) for d in ds])
            ax.plot(ds, mean, 'o-', lw=1.8, ms=4, color=ARM_COL[arm],
                    label=ARM_LAB[arm])
            ax.fill_between(ds, lo, hi, color=ARM_COL[arm], alpha=0.15)
        ax.axhline(0, color='k', lw=0.7)
        ax.axhline(10, color='red', ls=':', lw=0.8)
        ax.set_xlabel('stage distance d = k − j')
        ax.set_ylabel(f'{mlab} forgetting Δ%')
        ax.set_xticks(range(1, 8))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle('SEQ8 forgetting vs stage distance '
                 '(line=mean over all cells at that distance, band=min–max)',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fn = os.path.join(out_dir, 'E_forget_curve.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)


def main(out_dir):
    recs = harvest() + harvest_anchor()
    by = defaultdict(dict)
    for r in recs:
        by[r['run']][(r['stage'], r['ep'], r['eval_net'])] = r
    dt = delta_table(by)
    os.makedirs(out_dir, exist_ok=True)
    fig_c(dt, out_dir)
    fig_d(dt, out_dir)
    fig_e(dt, out_dir)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data/output_data/analysis/seq8_figs')
    main(out)
