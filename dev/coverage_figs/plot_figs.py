#!/usr/bin/env python3
"""出图：Fig.1 覆盖热力图 / Fig.2 SACo×TQ×ATT / Fig.3 覆盖控制遗忘 ΔATT%。

- fig1_seq8_{N1,R1,D1,union}.png  seq8 需求包行×三控制器列
- fig1_hz1x1.png                 4 个 config 场景行×控制组列
  (P1-Q1..Q4 / p34-clear·fifo·fmw / ha-CONT·DHOA·RAND·COV·CQ·CQA / HITL)
- fig2_saco_tq.png               行=原始TQ / 场景内z-TQ，列=seq8·ha·plan34
- fig3_seq8.png                  三臂遗忘 Δ%（seq8，CI 按评估格）
- fig3_ha_e4.png                 HA E4 五臂 ×O1–O4（CI 按场景格×种子）

口径：覆盖 cell=ha_coverage_v1(去 network/seed)；TQ=单元均 reward；
ATT=该场景被回访/回评的 stage 末冻结评估 travel_time；Δ%=(末段回评-
自身 stage 末回评)/自身 stage 末回评×100。
跨 runtime/协议拼接处在图注声明；SELF-CQ 不纳入。
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

for _f in fm.findSystemFonts():
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
if 'Noto Sans CJK SC' in {f.name for f in fm.fontManager.ttflist}:
    plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'dev/coverage_figs/cache')
FIGS = os.path.join(ROOT, 'dev/coverage_figs/figs')

SEQ8_ORDERS = {
    'N1': ['sparse_ultra', 'fluct_burst', 'lt_extreme', 'asym_wonly',
           'dense_extreme', 'st_dom', 'asym_we', 'fluct_ramp'],
    'R1': ['sparse_low', 'lt_high', 'fluct_burst', 'dense_high',
           'sparse_low', 'lt_high', 'fluct_burst', 'dense_high'],
    'D1': ['st_dom', 'sumohz1x1', 'lt_high', 'lt_extreme',
           'lt_wsing', 'asym_wonly', 'asym_we', 'asym_ns'],
}
CONFIG_TAGS = ['base', 'config2', 'config3', 'config4']
LOAD_LABELS = ['0', '1-2', '3-6', '7-14', '15-30', '≥31', '']

C = dict(CONT='#c0392b', DHOA='#e67e22', ANCHOR='#2980b9',
         **{'P1-Q1': '#f1c40f', 'P1-Q2': '#d4ac0d', 'P1-Q3': '#b7950b',
            'P1-Q4': '#7d6608',
            'p34-clear': '#8e44ad', 'p34-fifo': '#c0392b',
            'p34-fifo_matched_wait': '#e91e63',
            'CONT-FIFO': '#c0392b', 'ha-DHOA': '#e67e22',
            'ha-RAND': '#7f8c8d', 'ha-COV': '#16a085',
            'ha-CQ': '#27ae60', 'ha-CQA': '#2980b9',
            'HITL': '#2c3e50'})
MK = dict(CONT='o', DHOA='s', ANCHOR='^', HITL='D',
          **{'P1-Q1': 'o', 'P1-Q2': 'v', 'P1-Q3': 's', 'P1-Q4': 'D',
             'p34-clear': 'o', 'p34-fifo': 's',
             'p34-fifo_matched_wait': '^',
             'CONT-FIFO': 'o', 'ha-DHOA': 's', 'ha-RAND': 'P',
             'ha-COV': 'X', 'ha-CQ': 'v', 'ha-CQA': '^'})
FIG1_COLS = ['P1-Q1', 'P1-Q2', 'P1-Q3', 'P1-Q4',
             'p34-clear', 'p34-fifo', 'p34-fifo_matched_wait',
             'CONT-FIFO', 'ha-DHOA', 'ha-RAND', 'ha-COV', 'ha-CQ',
             'ha-CQA', 'HITL']


def load_all():
    units = json.load(open(os.path.join(CACHE, 'units.json')))
    evals = json.load(open(os.path.join(CACHE, 'evals.json')))
    mz = np.load(os.path.join(CACHE, 'marginals.npz'))
    marg = {u['uid']: mz[u['uid']] for u in units if u['uid'] in mz}
    return units, evals, marg


def last_stage(units, corpus=None, order=None):
    st = [u['stage'] for u in units
          if (corpus is None or u['corpus'] == corpus)
          and (order is None or u['order'] == order)]
    return max(st) if st else 0


def eval_lookup(evals):
    d = {}
    for e in evals:
        d[(e['corpus'], e['order'], e['arm'], e['seed'],
           e['eval_stage'], e['eval_scene'])] = e
    return d


def run_scenes(units):
    """(corpus,order,arm,seed) -> {stage: scene_tag}

    注意 ANCHOR 每 stage 一个 run 目录，须按 seed 粒度聚合；
    p1/hitl 不参与评估链，合并无害。"""
    d = {}
    for u in units:
        d.setdefault((u['corpus'], u['order'], u['arm'], u['seed']),
                     {})[u['stage']] = u['scene']
    return d


def unit_retention(u, ev, scenes_by_run, last_k):
    """该单元场景在链末 last_k stage 的回评 ATT。"""
    pos = scenes_by_run.get(
        (u['corpus'], u['order'], u['arm'], u['seed']), {})
    if u['scene'] not in pos.values():
        return None
    e = ev.get((u['corpus'], u['order'], u['arm'], u['seed'],
                last_k, u['scene']))
    return e['tt'] if e else None


def pooled_marginal(units, marg, pred):
    m = None
    n = 0
    for u in units:
        if pred(u) and u['uid'] in marg:
            m = marg[u['uid']] if m is None else m + marg[u['uid']]
            n += 1
    return m, n


def heatmap(ax, mat, title):
    ax.imshow(np.log10(mat + 1), cmap='magma', aspect='auto',
              interpolation='nearest')
    ax.set_xticks(range(8))
    ax.set_xticklabels(range(8), fontsize=6)
    ax.set_yticks(range(7))
    ax.set_yticklabels(LOAD_LABELS, fontsize=6)
    ax.set_title(title, fontsize=6.5, pad=2)
    for s in ax.spines.values():
        s.set_linewidth(0.4)


def fig1_seq8(units, marg):
    for order in ['N1', 'R1', 'D1']:
        fig, axes = plt.subplots(8, 3, figsize=(7.5, 12),
                                 constrained_layout=True)
        for st in range(1, 9):
            scene = SEQ8_ORDERS[order][st - 1]
            for ci, grp in enumerate(['CONT', 'DHOA', 'ANCHOR']):
                m, n = pooled_marginal(
                    units, marg,
                    lambda u, g=grp, o=order, s=st:
                    u['corpus'] == 'seq8' and u['group'] == g
                    and u['order'] == o and u['stage'] == s)
                ax = axes[st - 1, ci]
                if m is None:
                    ax.axis('off')
                    continue
                heatmap(ax, m, f'st{st} {scene} × {grp} (n={n})'
                        if ci == 1 else f'{grp} (n={n})')
        fig.suptitle(
            f'Fig.1 seq8 {order}: 需求包×控制器 覆盖热力图'
            ' (log10 transitions; ha_coverage_v1 load_bin×action)')
        fig.savefig(os.path.join(FIGS, f'fig1_seq8_{order}.png'),
                    dpi=150)
        plt.close(fig)
        print('fig1_seq8_' + order)
    # union
    scenes = sorted({s for o in SEQ8_ORDERS.values() for s in o})
    scenes = ['sumohz1x1'] + [s for s in scenes if s != 'sumohz1x1']
    fig, axes = plt.subplots(len(scenes), 3,
                             figsize=(7.5, 1.1 * len(scenes) + 1),
                             constrained_layout=True)
    for ri, sc in enumerate(scenes):
        tag = 'base' if sc == 'sumohz1x1' else sc
        for ci, grp in enumerate(['CONT', 'DHOA', 'ANCHOR']):
            m, n = pooled_marginal(
                units, marg,
                lambda u, g=grp, t=tag:
                u['corpus'] == 'seq8' and u['group'] == g
                and u['scene'] == t)
            ax = axes[ri, ci]
            if m is None:
                ax.axis('off')
                continue
            heatmap(ax, m, f'{tag} × {grp} (n={n})')
    fig.suptitle('Fig.1 seq8 union: 14 需求包 × 控制器 (log10)')
    fig.savefig(os.path.join(FIGS, 'fig1_seq8_union.png'), dpi=150)
    plt.close(fig)
    print('fig1_seq8_union')


def fig1_hz1x1(units, marg):
    fig, axes = plt.subplots(4, len(FIG1_COLS),
                             figsize=(2.0 * len(FIG1_COLS), 7.5),
                             constrained_layout=True)
    for ri, sc in enumerate(CONFIG_TAGS):
        for ci, grp in enumerate(FIG1_COLS):
            ax = axes[ri, ci]
            m, n = pooled_marginal(
                units, marg,
                lambda u, g=grp, t=sc:
                u['group'] == g and u['scene'] == t)
            if m is None:
                ax.axis('off')
                continue
            ttl = grp if ri == 0 else ''
            heatmap(ax, m, f'{ttl}\n{sc} n={n}' if ri == 0
                    else f'{sc} n={n}')
            if ci == 0:
                ax.set_ylabel(sc, fontsize=7)
    fig.suptitle(
        'Fig.1 hz1x1: 场景×控制组 覆盖热力图 (log10; 各格按组合并, '
        'n=单元数; p34 含 b100+b400; ha 含 P1C/P1F×R25/50/75)')
    fig.savefig(os.path.join(FIGS, 'fig1_hz1x1.png'), dpi=150)
    plt.close(fig)
    print('fig1_hz1x1')


def fig2(units, evals, ev, scenes_by_run):
    """单图 jointplot：x=SACo, y=TQ(语料×场景内 z-score)，
    色=回评 ATT，marker=语料；无评估链单元(p1/hitl/末段)灰显；
    上/右边缘为各语料 SACo·TQ 分布直方图。"""
    CORP_MK = {'seq8': 'o', 'ha': 's', 'plan34': '^', 'p1': 'D',
               'hitl': 'P'}
    CORP_C = {'seq8': '#c0392b', 'ha': '#2980b9', 'plan34': '#27ae60',
              'p1': '#f39c12', 'hitl': '#2c3e50'}
    cors_of = lambda c: ('plan34' if c.startswith('plan34') else c)
    lk_of = {'seq8': 8, 'ha': 4, 'plan34_b100': 4, 'plan34_b400': 4}

    # ---- 组装点：有评估链的给 ATT，p1/hitl 仅分布层 ----
    by_cs = {}
    for u in units:
        if u['tq_mean_reward'] is not None and u['n_episodes'] >= 8:
            by_cs.setdefault((u['corpus'], u['scene']), []).append(
                u['tq_mean_reward'])
    zt = {k: (np.mean(v), np.std(v) or 1.0) for k, v in by_cs.items()}
    pts = []          # (corpus_group, saco, ztq, att_or_None)
    for u in units:
        if (u['tq_mean_reward'] is None or u['n_episodes'] < 8
                or (u['corpus'], u['scene']) not in zt):
            continue
        mu, sd = zt[(u['corpus'], u['scene'])]
        ztq = (u['tq_mean_reward'] - mu) / sd
        att = None
        lk = lk_of.get(u['corpus'])
        if lk and u['stage'] < lk:
            att = unit_retention(u, ev, scenes_by_run, lk)
        pts.append((cors_of(u['corpus']), u['cells_occupied'], ztq, att))
    atts = [a for *_, a in pts if a is not None]
    lo, hi = np.percentile(atts, [2, 98]) if atts else (0, 1)

    fig = plt.figure(figsize=(10.5, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[4.6, 1.15],
                          height_ratios=[1.15, 4.6],
                          hspace=0.05, wspace=0.05)
    ax = fig.add_subplot(gs[1, 0])
    axt = fig.add_subplot(gs[0, 0], sharex=ax)
    axr = fig.add_subplot(gs[1, 1], sharey=ax)
    cmap = plt.get_cmap('viridis')

    for cor in ['seq8', 'ha', 'plan34', 'p1', 'hitl']:
        sel = [(x, y, a) for g, x, y, a in pts if g == cor]
        if not sel:
            continue
        xs = [s[0] for s in sel]
        ys = [s[1] for s in sel]
        ae = [(x, y, a) for x, y, a in sel if a is not None]
        an = [(x, y) for x, y, a in sel if a is None]
        if ae:
            ax.scatter([p[0] for p in ae], [p[1] for p in ae],
                       c=[p[2] for p in ae], cmap=cmap, vmin=lo,
                       vmax=hi, marker=CORP_MK[cor], s=16, alpha=0.7,
                       linewidths=0,
                       label=f'{cor} (n={len(sel)})')
        if an:
            ax.scatter([p[0] for p in an], [p[1] for p in an],
                       facecolors='none', edgecolors='#95a5a6',
                       marker=CORP_MK[cor], s=14, alpha=0.55,
                       linewidths=0.6,
                       label=None if ae else f'{cor} (n={len(sel)})')
        # 边缘分布（全单元，含无评估点）
        axt.hist(xs, bins=28, density=True, histtype='step',
                 color=CORP_C[cor], lw=1.1)
        axr.hist(ys, bins=28, density=True, histtype='step',
                 orientation='horizontal', color=CORP_C[cor], lw=1.1)

    ax.set_xlabel('SACo 占用 cell 数 (ha_coverage_v1, ≤25ep/单元)')
    ax.set_ylabel('TQ (语料×场景内 z-score of 均 reward)')
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(fontsize=7, loc='lower right', framealpha=0.85,
              title='语料 (灰=无评估链)', title_fontsize=7)
    axt.set_ylabel('密度', fontsize=8)
    axr.set_xlabel('密度', fontsize=8)
    for a in (axt, axr):
        a.tick_params(labelsize=6)
        plt.setp(axt.get_xticklabels(), visible=False)
        plt.setp(axr.get_yticklabels(), visible=False)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
    fig.colorbar(sm, ax=axr, fraction=0.35, pad=0.06,
                 label='retention ATT (末 stage 回评 travel_time)')
    fig.suptitle('Fig.2 SACo×TQ 同图分布 (形=语料, 色=回评 ATT; '
                 'seq8 ANCHOR 跨 runtime; p1/hitl 无评估链灰显)', y=0.99)
    fig.savefig(os.path.join(FIGS, 'fig2_saco_tq.png'), dpi=150)
    plt.close(fig)
    print('fig2_saco_tq')


def delta_rows(units, evals, ev, scenes_by_run, corpus_pred, arms):
    """非对角 (j,k>j) 遗忘 Δ% 行: (order, arm, delta)."""
    rows = []
    for (corpus, order, arm, seed), pos in scenes_by_run.items():
        if not corpus_pred(corpus) or arm not in arms:
            continue
        for j, scene in sorted(pos.items()):
            base = ev.get((corpus, order, arm, seed, j, scene))
            if base is None:
                continue
            for k in sorted(pos):
                if k <= j:
                    continue
                e = ev.get((corpus, order, arm, seed, k, scene))
                if e is None:
                    continue
                rows.append(dict(corpus=corpus, order=order, arm=arm,
                                 seed=seed, pos=j, eval_stage=k,
                                 scene=scene,
                                 delta=(e['tt'] - base['tt'])
                                 / base['tt'] * 100.0))
    return rows


def fig3_panel(ax, rows, order, arms, colors, labels):
    mus = []
    hi = 0.0
    clip = []
    for ai, arm in enumerate(arms):
        d = [r['delta'] for r in rows
             if r['arm'] == arm and (order is None or r['order'] == order)]
        if not d:
            continue
        mu, sd = float(np.mean(d)), float(np.std(d, ddof=1))
        ci = 1.96 * sd / np.sqrt(len(d))
        ax.bar(ai, mu, color=colors.get(arm, 'k'), alpha=0.75,
               yerr=ci, capsize=3, error_kw=dict(lw=0.8))
        jit = np.linspace(-0.18, 0.18, len(d))
        ax.scatter(np.full(len(d), ai) + jit, d, s=5, c='k',
                   alpha=0.35, linewidths=0)
        hi = max(hi, mu + ci)
        clip += [x for x in d if x > 0]
        mus.append(f'{arm}: {mu:+.1f}±{ci:.1f} (n={len(d)})')
    if clip:
        cap = max(20.0, np.percentile(clip, 95) * 1.15, hi * 1.1)
        ax.set_ylim(top=cap)
    ax.axhline(0, c='gray', lw=0.6, ls='--')
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([labels.get(a, a) for a in arms], fontsize=7,
                       rotation=18)
    ax.set_title(order or 'ALL', fontsize=9)
    ax.text(0.02, 0.98, '\n'.join(mus), transform=ax.transAxes,
            fontsize=5.2, va='top', ha='left',
            bbox=dict(fc='white', ec='none', alpha=0.7))
    ax.tick_params(labelsize=7)
    ax.grid(axis='y', alpha=0.25, lw=0.4)


def fig3(units, evals, ev, scenes_by_run):
    # ---- seq8 ----
    rows = delta_rows(units, evals, ev, scenes_by_run,
                      lambda c: c == 'seq8',
                      {'CONT', 'DHOA', 'ANCHOR'})
    arms = ['CONT', 'DHOA', 'ANCHOR']
    labels = {'CONT': 'CONT\n(无重放)', 'DHOA': 'SELF-DHOA-R50',
              'ANCHOR': 'soddqnp_r25*'}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.6),
                             constrained_layout=True)
    for i, o in enumerate(['N1', 'R1', 'D1', None]):
        fig3_panel(axes[i], rows, o, arms, C, labels)
    axes[0].set_ylabel('遗忘 ΔATT% (末段回评 vs 自身 stage 末)', fontsize=8)
    fig.suptitle('Fig.3 seq8: 覆盖控制臂遗忘 Δ% +95%CI '
                 '(按场景格; *跨 runtime; n 见标题)')
    fig.savefig(os.path.join(FIGS, 'fig3_seq8.png'), dpi=150)
    plt.close(fig)
    summ = {}
    for o in ['N1', 'R1', 'D1', 'ALL']:
        for a in arms:
            d = [r['delta'] for r in rows
                 if r['arm'] == a and (o == 'ALL' or r['order'] == o)]
            if d:
                summ[f'seq8_{o}_{a}'] = dict(
                    mean=float(np.mean(d)),
                    ci95=float(1.96 * np.std(d, ddof=1)
                               / np.sqrt(len(d))), n=len(d))
    print('fig3_seq8')

    # ---- ha E4 ----
    e4 = {'CONT-FIFO', 'P1C-DHOA-R25', 'P1C-CQ-R75', 'P1C-CQA-R75',
          'P1F-CQA-R50'}
    rows_ha = delta_rows(units, evals, ev, scenes_by_run,
                         lambda c: c == 'ha', e4)
    arms_ha = ['CONT-FIFO', 'P1C-DHOA-R25', 'P1C-CQ-R75',
               'P1C-CQA-R75', 'P1F-CQA-R50']
    col_ha = dict(zip(arms_ha, ['#c0392b', '#e67e22', '#27ae60',
                                '#2980b9', '#8e44ad']))
    lab_ha = {a: a.replace('P1C-', 'C\n').replace('P1F-', 'F\n')
              for a in arms_ha}
    fig, axes = plt.subplots(1, 5, figsize=(18, 3.6),
                             constrained_layout=True)
    for i, o in enumerate(['O1', 'O2', 'O3', 'O4', None]):
        fig3_panel(axes[i], rows_ha, o, arms_ha, col_ha, lab_ha)
    axes[0].set_ylabel('遗忘 ΔATT% (末段回评 vs 自身 stage 末)',
                       fontsize=8)
    fig.suptitle('Fig.3 ha_sodqn E4: 覆盖控制臂遗忘 Δ% +95%CI '
                 '(按场景格×种子, n 见标题)')
    fig.savefig(os.path.join(FIGS, 'fig3_ha_e4.png'), dpi=150)
    plt.close(fig)
    for o in ['O1', 'O2', 'O3', 'O4', 'ALL']:
        for a in arms_ha:
            d = [r['delta'] for r in rows_ha
                 if r['arm'] == a and (o == 'ALL' or r['order'] == o)]
            if d:
                summ[f'ha_{o}_{a}'] = dict(
                    mean=float(np.mean(d)),
                    ci95=float(1.96 * np.std(d, ddof=1)
                               / np.sqrt(len(d))), n=len(d))
    print('fig3_ha_e4')
    with open(os.path.join(FIGS, 'fig3_summary.json'), 'w') as fh:
        json.dump(summ, fh, indent=1, ensure_ascii=False)


def main():
    os.makedirs(FIGS, exist_ok=True)
    units, evals, marg = load_all()
    ev = eval_lookup(evals)
    scenes_by_run = run_scenes(units)
    fig1_seq8(units, marg)
    fig1_hz1x1(units, marg)
    fig2(units, evals, ev, scenes_by_run)
    fig3(units, evals, ev, scenes_by_run)


if __name__ == '__main__':
    main()
