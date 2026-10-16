"""fig A（训练期逐点评估 P1）+ fig B（阶段末下三角回测 P2）——cl_hz1x1 战役。

数据源：
  - P1: data/output_data/tsc/<fam>/<cur_scene>/<run>/metrics/records.jsonl
        的 EVALUATION 行（每 train episode 一次冻结协议评估）。
  - P2: artifacts/cl_hz1x1/matrix_report{,_so,_soa_soa1}.json 阶段末
        冻结 ckpt 对历史场景的下三角回测（3 eval-seed 均值）。

网格：行 = 骨干 × 臂（clear / semi / +P），列 = travel_time / queue。
序：o2（s1→c4→c2→c3）。同列 y 轴共享。
产物：artifacts/cl_hz1x1/fig_A_train.png / fig_B_retest.png +
      fig_ab_data.json（绘图用归一化数据）。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'

ORDER = 'o2'
SCENES = {'o2': ['s1', 'c4', 'c2', 'c3'], 'o3': ['c2', 'c3', 's1', 'c4']}
SCENE_DIR = {
    's1': 'sumohz1x1', 'c4': 'sumohz1x1_config4',
    'c2': 'sumohz1x1_config2', 'c3': 'sumohz1x1_config3',
}
BACKBONES = ['ddqn', 'duel', 'frap']
ARMS = ['clear', 'semi', '+P']
ARM_LABEL = {'clear': 'pure online', 'semi': 'semi-offline',
             '+P': 'semi + policy anchor'}
METRICS = [('travel_time', 'ATT (s)'), ('queue', 'Mean queue (veh)')]

# backbone → (family_dir, run-prefix per arm, matrix key per arm)
CFG = {
    'ddqn': ('sumo_dqn', {'clear': 'cl1_ddqn_clear', 'semi': 'so1_soddqn', '+P': 'soa1_soddqnp'},
             {'clear': 'ddqn_clear', 'semi': 'soddqn', '+P': 'soddqnp'}),
    'duel': ('sumo_dqn', {'clear': 'cl1_duel_clear', 'semi': 'so1_soduel', '+P': 'soa1_soduelp'},
             {'clear': 'duel_clear', 'semi': 'soduel', '+P': 'soduelp'}),
    'frap': ('sumo_frap', {'clear': 'cl1_frap_clear', 'semi': 'so1_sofrap', '+P': 'soa1_sofrapp'},
             {'clear': 'frap_clear', 'semi': 'sofrap', '+P': 'sofrapp'}),
}
SEEDS = [0, 1, 2]
NSTAGE = 4
EP_PER_STAGE = 100


def load_p1(bb, arm, seed):
    """{stage_i: {'train': {ep: m→v}, 'eval': {ep: m→v}}}。

    主矩阵 run 只在 ep{0,50,100} 跑冻结协议评估；逐 episode 曲线用
    TRAIN 行（行为策略下的 episode 指标），EVALUATION 行作锚点叠加。
    """
    fam, runpfx, _ = CFG[bb]
    scenes = SCENES[ORDER]
    out = {}
    for st in range(1, NSTAGE + 1):
        rd = OUT / fam / SCENE_DIR[scenes[st - 1]] / f'{runpfx[arm]}_{ORDER}_sd{seed}_st{st}'
        f = rd / 'metrics' / 'records.jsonl'
        tr, ev = {}, {}
        if f.exists():
            for line in open(f):
                r = json.loads(line)
                tgt = {'TRAIN': tr, 'EVALUATION': ev,
                       'FINAL_EVALUATION': ev}.get(r.get('record_type'))
                if tgt is not None:
                    ep = r['episode'] if r['record_type'] != 'FINAL_EVALUATION' \
                        else EP_PER_STAGE
                    tgt.setdefault(ep, {}).update(
                        {m: r.get(m) for m, _ in METRICS})
        out[st] = {'train': tr, 'eval': ev}
    return out


def load_p2():
    """matrix[bb][arm][seed] -> {task_j: {ckpt_i: {metric: v}}} 归一化。"""
    rep_cl = json.load(open(ART / 'matrix_report.json'))['conditions']
    rep_so = json.load(open(ART / 'matrix_report_so.json'))
    rep_soa = json.load(open(ART / 'matrix_report_soa_soa1.json'))
    mat = {bb: {a: {} for a in ARMS} for bb in BACKBONES}
    for bb in BACKBONES:
        mk = CFG[bb][2]
        for sd in SEEDS:
            chain = f'{ORDER}_sd{sd}'
            # clear（cl1 嵌套 schema：matrix[metric][i][j] 下三角）
            nested = rep_cl[mk['clear']][chain]['matrix']
            flat_cl = {}
            for i in range(1, NSTAGE + 1):
                flat_cl[str(i)] = {}
                for j, sc in enumerate(SCENES[ORDER]):
                    if j < i:
                        flat_cl[str(i)][sc] = {m: nested[m][i - 1][j] for m, _ in METRICS}
            mat[bb]['clear'][sd] = flat_cl
            for arm, rep in (('semi', rep_so), ('+P', rep_soa)):
                flat = rep[f'{mk[arm]}_{ORDER}_sd{sd}']
                mat[bb][arm][sd] = {str(i): flat[str(i)] for i in range(1, NSTAGE + 1)}
    return mat


def fig_a(data):
    rows = [(bb, a) for bb in BACKBONES for a in ARMS]
    fig, axes = plt.subplots(len(rows), len(METRICS), figsize=(11, 16),
                             sharex=True)
    scenes = SCENES[ORDER]
    for ri, (bb, arm) in enumerate(rows):
        for ci, (m, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            xs = np.arange(0, NSTAGE * EP_PER_STAGE + 1)
            curves, evalpts = [], []
            for sd in SEEDS:
                p1 = data[bb][arm][sd]['p1']
                y = np.full(len(xs), np.nan)
                ex, ey = [], []
                for st in range(1, NSTAGE + 1):
                    rec = p1.get(st, {})
                    for ep, r in rec.get('train', {}).items():
                        gi = (st - 1) * EP_PER_STAGE + ep
                        if 0 <= gi < len(y):
                            y[gi] = r.get(m, np.nan)
                    for ep, r in rec.get('eval', {}).items():
                        ex.append((st - 1) * EP_PER_STAGE + ep)
                        ey.append(r.get(m, np.nan))
                curves.append(y)
                evalpts.append((ex, ey))
            arr = np.array(curves)
            mean = np.nanmean(arr, axis=0)
            lo, hi = np.nanmin(arr, axis=0), np.nanmax(arr, axis=0)
            ax.fill_between(xs, lo, hi, alpha=.18, color='C0', lw=0)
            ax.plot(xs, mean, lw=1.1, color='C0')
            for ex, ey in evalpts:
                ax.scatter(ex, ey, s=5, color='crimson', zorder=5, lw=0)
            for st in range(1, NSTAGE):
                ax.axvline(st * EP_PER_STAGE, ls='--', c='grey', lw=.7)
            for st in range(NSTAGE):
                ax.text(st * EP_PER_STAGE + 50, ax.get_ylim()[1],
                        scenes[st], ha='center', va='top', fontsize=7,
                        color='dimgray')
            if ci == 0:
                ax.set_ylabel(f'{bb} / {ARM_LABEL[arm]}', fontsize=9)
            ax.tick_params(labelsize=7)
    for ci, (m, mlab) in enumerate(METRICS):
        axes[0][ci].set_title(mlab, fontsize=10)
        # 同列共享 y
        ymin = min(ax.get_ylim()[0] for ax in axes[:, ci])
        ymax = max(ax.get_ylim()[1] for ax in axes[:, ci])
        for ax in axes[:, ci]:
            ax.set_ylim(ymin, ymax)
    axes[-1][0].set_xlabel('global training episode')
    axes[-1][1].set_xlabel('global training episode')
    fig.suptitle(f'fig A — P1 per-episode metrics on current task '
                 f'({ORDER}: {"→".join(SCENES[ORDER])}); curve=TRAIN '
                 f'mean±range (3 seeds), red dots=frozen-protocol eval '
                 f'@ep0/50/100', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .98))
    return fig


def fig_b(mat, avg_seeds=False, fname='fig_B_retest.png', arms=ARMS):
    rows = [(bb, a) for bb in BACKBONES for a in arms]
    scenes = SCENES[ORDER]
    stage_c = plt.get_cmap('viridis', NSTAGE)
    fig, axes = plt.subplots(len(rows), len(METRICS),
                             figsize=(11, 1.75 * len(rows) + 1.5),
                             sharex=True)
    if avg_seeds:
        bw = 0.8 / NSTAGE              # 簇内每 stage 一柱
    else:
        bw = 0.8 / (NSTAGE * len(SEEDS))
    for ri, (bb, arm) in enumerate(rows):
        for ci, (m, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            for j, sc in enumerate(scenes):  # 历史任务簇
                for i in range(j + 1, NSTAGE + 1):  # 之后每个 stage 末 ckpt
                    vals = [mat[bb][arm][sd][str(i)].get(sc, {}).get(m)
                            for sd in SEEDS]
                    vals = [v for v in vals if v is not None]
                    if not vals:
                        continue
                    k = i - (j + 1)  # 簇内第几根组
                    if avg_seeds:
                        x = j + k * bw - 0.35
                        ax.bar(x, np.mean(vals), width=bw * .92,
                               color=stage_c(i - 1), alpha=.9, lw=0)
                        ax.errorbar(x, np.mean(vals),
                                    yerr=[[np.mean(vals) - min(vals)],
                                          [max(vals) - np.mean(vals)]],
                                    fmt='none', ecolor='k', elinewidth=.8,
                                    capsize=2, zorder=5)
                    else:
                        for sd, v in enumerate(vals):
                            x = j + (k * len(SEEDS) + sd) * bw - 0.35
                            ax.bar(x, v, width=bw * .92,
                                   color=stage_c(i - 1), alpha=.9, lw=0)
            ax.set_xticks(range(len(scenes)))
            ax.set_xticklabels([f'T{j+1}\n{sc}' for j, sc in
                                enumerate(scenes)], fontsize=8)
            if ci == 0:
                ax.set_ylabel(f'{bb} / {ARM_LABEL[arm]}', fontsize=9)
            ax.tick_params(labelsize=7)
    for ci, (m, mlab) in enumerate(METRICS):
        axes[0][ci].set_title(mlab, fontsize=10)
        ymin = min(ax.get_ylim()[0] for ax in axes[:, ci])
        ymax = max(ax.get_ylim()[1] for ax in axes[:, ci])
        for ax in axes[:, ci]:
            ax.set_ylim(ymin, ymax)
    # 图例：ckpt 阶段
    handles = [plt.Rectangle((0, 0), 1, 1, color=stage_c(i))
               for i in range(NSTAGE)]
    cap = ('bars = mean of 3 seeds, whiskers = min–max' if avg_seeds else
           'bars within a cluster = per-seed evals by each later stage-end '
           'ckpt (3 seeds side-by-side)')
    fig.legend(handles, [f'ckpt @st{i+1}' for i in range(NSTAGE)],
               loc='lower center', ncol=4, fontsize=9,
               title=cap, title_fontsize=8)
    fig.suptitle(f'fig B — P2 stage-end ckpt retest on historical tasks '
                 f'({ORDER}, lower-triangular)', fontsize=11)
    fig.tight_layout(rect=(0, .05, 1, .98))
    fig.savefig(ART / fname, dpi=200)
    return fig


def main():
    data = {bb: {a: {} for a in ARMS} for bb in BACKBONES}
    for bb in BACKBONES:
        for a in ARMS:
            for sd in SEEDS:
                data[bb][a][sd] = {'p1': load_p1(bb, a, sd)}
    mat = load_p2()
    for bb in BACKBONES:
        for a in ARMS:
            for sd in SEEDS:
                data[bb][a][sd]['p2'] = mat[bb][a][sd]

    with open(ART / 'fig_ab_data.json', 'w') as f:
        json.dump({'order': ORDER, 'scenes': SCENES[ORDER],
                   'backbones': BACKBONES, 'arms': ARMS, 'data': data}, f)

    n_p1 = sum(len(r) for bb in data for a in data[bb]
               for sd in data[bb][a] for st in data[bb][a][sd]['p1'].values()
               for r in st.values())
    n_p2 = sum(1 for bb in mat for a in mat[bb] for sd in mat[bb][a]
               for i in mat[bb][a][sd] for sc in mat[bb][a][sd][i])
    print(f'P1 cells: {n_p1}  P2 cells: {n_p2}')

    fig_a(data).savefig(ART / 'fig_A_train.png', dpi=200)
    fig_b(mat)
    fig_b(mat, avg_seeds=True, fname='fig_B_retest_avg.png',
          arms=['clear', '+P'])
    print('wrote fig_A_train.png / fig_B_retest{,_avg}.png')


if __name__ == '__main__':
    main()
