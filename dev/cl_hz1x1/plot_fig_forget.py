"""遗忘锐化双图（cl_hz1x1，o2 序，ATT 口径）。

F1 遗忘轨迹线图：x=回测 ckpt 阶段（st j+1..4），y=ΔATT（相对该任务
   stage-j 末 ckpt 的退化量）；面板=臂(行)×骨干(列)，线色=任务位。
F3 链末 Δ 簇柱：x 簇=历史任务 T1..T3，簇内两根柱=clear vs +P，
   y=st4 末 ckpt 的 ΔATT；面板=骨干；柱=3 seed 均值+min-max 须。

数据：复用 plot_fig_ab.load_p2()（matrix_report*.json 归一化）。
产物：artifacts/cl_hz1x1/fig_forget_curve.png / fig_forget_final.png。
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_fig_ab import (load_p2, BACKBONES, SCENES, ORDER, NSTAGE,
                         ARM_LABEL, ART)

ARMS2 = ['clear', '+P']
METRIC = 'travel_time'


def delta_series(mat, bb, arm):
    """{task_j: {ckpt_i: [seed Δ]}}，i>j。"""
    out = {}
    for j in range(1, NSTAGE):            # 任务位 1..3（T4 无后续）
        sc = SCENES[ORDER][j - 1]
        base = {sd: mat[bb][arm][sd][str(j)][sc][METRIC] for sd in mat[bb][arm]}
        out[j] = {}
        for i in range(j + 1, NSTAGE + 1):
            out[j][i] = [mat[bb][arm][sd][str(i)][sc][METRIC] - base[sd]
                         for sd in mat[bb][arm]]
    return out


def fig_forget_curve(mat):
    task_c = plt.get_cmap('Set2')
    fig, axes = plt.subplots(len(ARMS2), len(BACKBONES),
                             figsize=(11, 5.6), sharex=True, sharey=True)
    for ri, arm in enumerate(ARMS2):
        for ci, bb in enumerate(BACKBONES):
            ax = axes[ri][ci]
            ds = delta_series(mat, bb, arm)
            for j, ckpts in ds.items():
                xs = sorted(ckpts)
                arr = np.array([ckpts[i] for i in xs])     # [i, seed]
                mean = arr.mean(1)
                lo, hi = arr.min(1), arr.max(1)
                ax.fill_between(xs, lo, hi, alpha=.22, color=task_c(j - 1))
                ax.plot(xs, mean, '-o', ms=4, lw=1.4, color=task_c(j - 1),
                        label=f'T{j} ({SCENES[ORDER][j-1]})')
            ax.axhline(0, c='grey', lw=.8, ls='--')
            ax.set_xticks([2, 3, 4])
            ax.set_xticklabels([f'@st{i}' for i in (2, 3, 4)], fontsize=8)
            if ri == 0:
                ax.set_title(bb, fontsize=10)
            if ci == 0:
                ax.set_ylabel(f'{ARM_LABEL[arm]}\nΔATT vs stage-entry (s)',
                              fontsize=9)
            ax.tick_params(labelsize=8)
    axes[0][-1].legend(fontsize=8, loc='upper left')
    fig.suptitle(f'forgetting trajectory — ΔATT of historical tasks as later '
                 f'checkpoints accumulate ({ORDER}; mean±minmax, 3 seeds)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def fig_forget_final(mat):
    fig, axes = plt.subplots(1, len(BACKBONES), figsize=(11, 3.6),
                             sharey=True)
    task_x = np.arange(1, NSTAGE)          # T1..T3
    w = 0.36
    arm_c = {'clear': '#d62728', '+P': '#1f77b4'}
    for ci, bb in enumerate(BACKBONES):
        ax = axes[ci]
        ds = delta_series(mat, bb, 'clear'), delta_series(mat, bb, '+P')
        for ai, (arm, d) in enumerate(zip(ARMS2, ds)):
            means, lows, highs = [], [], []
            for j in task_x:
                cell = d[j][NSTAGE]  # 链末 ckpt 对任务 j 的逐 seed Δ
                means.append(np.mean(cell))
                lows.append(np.mean(cell) - min(cell))
                highs.append(max(cell) - np.mean(cell))
            x = task_x + (ai - .5) * w
            ax.bar(x, means, width=w * .92, color=arm_c[arm], alpha=.85,
                   label=ARM_LABEL[arm], lw=0)
            ax.errorbar(x, means, yerr=[lows, highs], fmt='none',
                        ecolor='k', elinewidth=.8, capsize=2)
        ax.axhline(0, c='grey', lw=.8, ls='--')
        ax.set_xticks(task_x)
        ax.set_xticklabels([f'T{j}\n{SCENES[ORDER][j-1]}' for j in task_x],
                           fontsize=8)
        ax.set_title(bb, fontsize=10)
        if ci == 0:
            ax.set_ylabel('ΔATT @final ckpt vs stage-entry (s)', fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].legend(fontsize=8)
    fig.suptitle(f'final-checkpoint forgetting — ΔATT on each historical task '
                 f'after last stage ({ORDER}; mean±minmax, 3 seeds)',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .93))
    return fig


def main():
    mat = load_p2()
    fig_forget_curve(mat).savefig(ART / 'fig_forget_curve.png', dpi=200)
    fig_forget_final(mat).savefig(ART / 'fig_forget_final.png', dpi=200)
    print('wrote fig_forget_curve.png / fig_forget_final.png')
    # 数字速览
    for bb in BACKBONES:
        for arm in ARMS2:
            ds = delta_series(mat, bb, arm)
            fin = {f'T{j}': np.mean(ds[j][NSTAGE]) for j in ds if NSTAGE in ds[j]}
            print(f'{bb:5s} {arm:6s} final ΔATT:', {k: round(v, 1) for k, v in fin.items()})


if __name__ == '__main__':
    main()
