"""任务书佐证图组：场景引导注意力模块的感知效果与RL增强。

一图一问题，固定其余变量（顾氏纪律）。决策层一律用
"对盲化孪生的逐 case 配对增益%"（正=优），不用绝对值柱图
——绝对值把 ~9% 差压在 40k 量级上，效果看不出来：
  F1 感知-响应契约  同一状态仅换文本：注意力是否跟报告指针走
  F2 感知-浓缩表征  同一批场景快照：浓缩 z 是否保住信息
  F3 决策-主效应    配对连线 + 分指标增益分布 + 参照口径归因
  F4 决策-设计消融  意图粒度阶梯 + 非联合蒸馏编码器参照臂
  F5 决策-鲁棒边界  延迟全档/干扰/强度 下的配对增益轨迹
  F6 决策-迁移      主模型零样本到留出事件与 S1/S3/S4 需求分布

数据源（只读 results.json / 探针 JSON，不再跑仿真）：
  attn_probe_c2/attn_probe.json  probe_repr_id/metrics.json
  sig15_eval{,_main8,_modcmp,_delay{0,60,90,120},_sev,_distract,
  _holdout,_S1,_S3,_S4}/results.json
注意 sig15_eval_S{1,3,4}（主模型迁移）与 sig15_S{1,3,4}_eval
（场景内重训）是两组不同实验，本图组用前者。

用法::

    python -m scene_attention.taskbook_figs \
        --root output_data/dev_takeover_probe \
        --out  output_data/dev_takeover_probe/taskbook_figs
"""

from __future__ import annotations

import argparse
import json
from math import erfc
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

FONT_CANDIDATES = [
    Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf',
    Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'),
]
for f in FONT_CANDIDATES:
    if f.exists():
        font_manager.fontManager.addfont(str(f))
        plt.rcParams['font.family'] = \
            font_manager.FontProperties(fname=str(f)).get_name()
        break
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

C_SGA = '#2ca02c'      # 主角：场景引导注意力
C_NOTEXT = '#7f7f7f'   # 盲化孪生
C_LIE = '#ff7f0e'      # 谎报
C_AUX0 = '#9467bd'     # 无监督消融
C_REF = '#1f77b4'      # 外部基线
C_DIM = '#bbbbbb'

ARM_LABEL = {
    'joint_sga15': '场景引导注意力\n(sga15)',
    'joint_notext': '同模型·盲化文本',
    'joint_lie': '谎报文本',
    'joint_base': '联合底座',
    'joint_sga12': '单点意图\n(sig12)',
    'joint_sga16': '无监督消融\n(sga16)',
    'dqn_s1': '部署DQN',
    'maxpressure': 'MaxPressure',
    'fixedtime': '定时控制',
    'sotl': 'SOTL',
    'concat': '拼接(无注意力)',
    'selfattn': '自注意力',
    'attn': '交叉注意力',
    'sga': '场景引导\n注意力(sga)',
}


def load(root: Path, name: str):
    p = root / name / 'results.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())['arms']


def col(arms, arm, key):
    return [r[key] for r in arms[arm] if r.get(key) is not None]


def wilcoxon(a, b):
    """同 sig15_table：逐 case 配对符号秩检验（scipy-free）。"""
    d = np.array(a, dtype=float) - np.array(b, dtype=float)
    d = d[d != 0]
    if len(d) < 5:
        return None
    ranks = np.argsort(np.argsort(np.abs(d))) + 1.0
    w_plus = ranks[d > 0].sum()
    n = len(d)
    mu = n * (n + 1) / 4
    sd = (n * (n + 1) * (2 * n + 1) / 24) ** 0.5
    z = (w_plus - mu) / sd
    return z, erfc(abs(z) / 2 ** 0.5), n


def star(p):
    if p is None:
        return ''
    return '**' if p < 0.01 else ('*' if p < 0.05 else 'n.s.')


def paired_pct(a_vals, b_vals):
    """逐 case (a-b)/b 的配对改善%。返回 mean%, sem%, wilcoxon p。"""
    a, b = np.array(a_vals, dtype=float), np.array(b_vals, dtype=float)
    pct = (a - b) / b * 100.0
    w = wilcoxon(a, b)
    return pct.mean(), pct.std(ddof=1) / np.sqrt(len(pct)), \
        (w[1] if w else None)


def sem(v):
    v = np.asarray(v, dtype=float)
    return v.std(ddof=1) / np.sqrt(len(v))


def pair_by_event(arms, a, b):
    """按 (event, 第k次出现) 键配对两臂 case，不依赖遍历顺序。"""
    def keymap(rows):
        seen, m = {}, {}
        for r in rows:
            k = r.get('event', '')
            i = seen.get(k, 0)
            seen[k] = i + 1
            m[(k, i)] = r
        return m
    ma, mb = keymap(arms[a]), keymap(arms[b])
    ks = sorted(set(ma) & set(mb))
    return [ma[k] for k in ks], [mb[k] for k in ks]


def pgain(arms, a, b, key, higher_better=False):
    """逐 case 配对增益%（正 = a 优于 b）+ Wilcoxon p + 胜率。

    排队类指标 higher_better=False（降为好），吞吐 True（升为好）。
    返回 dict(pct, mean, med, win, n, p, a_raw, b_raw)。
    """
    A, B = pair_by_event(arms, a, b)
    pct, araw, braw = [], [], []
    for x, y in zip(A, B):
        va, vb = x.get(key), y.get(key)
        if not (isinstance(va, (int, float)) and
                isinstance(vb, (int, float))):
            continue
        if np.isnan(va) or np.isnan(vb) or vb == 0:
            continue
        pct.append(((va - vb) if higher_better else (vb - va)) / vb * 100)
        araw.append(va)
        braw.append(vb)
    pct = np.array(pct)
    w = wilcoxon(araw, braw)
    return {'pct': pct, 'mean': float(pct.mean()),
            'med': float(np.median(pct)), 'win': int((pct > 0).sum()),
            'n': len(pct), 'p': w[1] if w else None,
            'a_raw': np.array(araw), 'b_raw': np.array(braw)}


def annotate_bars(ax, bars, fmt='{:.0f}', dy=0.01, fs=9):
    ylim = ax.get_ylim()
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2,
                b.get_height() + (ylim[1] - ylim[0]) * dy,
                fmt.format(b.get_height()),
                ha='center', va='bottom', fontsize=fs)


# ---------------------------------------------------------------- F1
def fig1(root, out):
    """感知-响应契约：同一交通状态仅换文本，注意力是否跟报告指针走。"""
    j = json.loads((root / 'attn_probe_c2/attn_probe.json').read_text())

    def means(arm):
        ticks = j[arm]['ticks']
        return {
            'true': np.mean([t['align_true'] for t in ticks]),
            'lie': np.mean([t['align_lie_on_liewatch'] for t in ticks]),
            'none': np.mean([t['align_none'] for t in ticks]),
            'wl1': np.mean([t['w_l1_true_lie'] for t in ticks]),
            'mass_lie': j[arm]['summary'].get('mass_lie_on_lie', np.nan),
        }

    s15, s16 = means('sga15'), means('sga16')
    fl15 = sum(t['flip_none'] for t in j['sga15']['ticks']) / \
        len(j['sga15']['ticks'])
    fl16 = sum(t['flip_none'] for t in j['sga16']['ticks']) / \
        len(j['sga16']['ticks'])

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    xs = np.arange(3)
    w = 0.36
    v15 = [s15['true'], s15['lie'], s15['none']]
    v16 = [s16['true'], s16['lie'], s16['none']]
    b1 = ax.bar(xs - w / 2, v15, w, color=C_SGA,
                label='sga15（θ_watch 监督）')
    b2 = ax.bar(xs + w / 2, v16, w, color=C_AUX0,
                label='sga16（aux=0，无监督）')
    ax.axhline(0.33, ls=':', c=C_DIM, lw=1)
    ax.text(2.45, 0.335, '无监督漂移水平', fontsize=8, color='#888')
    annotate_bars(ax, b1, '{:.2f}')
    annotate_bars(ax, b2, '{:.2f}')
    ax.set_xticks(xs)
    ax.set_xticklabels([
        '真实播报\n→ 对真走廊对齐', '谎报\n→ 对谎所指走廊对齐',
        '无文本\n→ 对真走廊对齐(漂移)'])
    ax.set_ylabel('注意力权重 w 与观察集 θ_watch 的余弦对齐')
    ax.set_ylim(0, 0.52)
    ax.legend(loc='upper left', fontsize=9)
    ax.set_title('F1 感知·响应契约：同一交通状态仅换文本，注意力跟报告指针走\n'
                 '（谎报把注意力引到谎边 0.42 > 真报对齐 0.35；'
                 '撤监督后全平 ≈0.33）', fontsize=11)
    txt = (f'文本互换扰动 w-L1：sga15 {s15["wl1"]:.2f} vs sga16 '
           f'{s16["wl1"]:.2f}　|　撤文本动作翻转率：{fl15:.0%} vs {fl16:.0%}\n'
           f'谎报臂下注意力落在谎边的质量占比：{s15["mass_lie"]:.0%}')
    ax.text(0.5, -0.30, txt, transform=ax.transAxes, ha='center',
            fontsize=8.5, color='#444')
    fig.tight_layout()
    fig.savefig(out / 'f1_response_contract.png', bbox_inches='tight')
    plt.close(fig)
    return {'align_sga15': v15, 'align_sga16': v16}


# ---------------------------------------------------------------- F2
def fig2(root, out):
    """感知-浓缩表征：同一批快照上 z 的探针性能，唯一变量=编码器结构。"""
    j = json.loads((root / 'probe_repr_id/metrics.json').read_text())
    arms = ['concat', 'selfattn', 'attn', 'sga']
    mets = [('拥堵分类\naccuracy', 'congestion_clf', 'acc'),
            ('影响预测\nR²', 'impact_reg', 'r2'),
            ('压缩后分类保持率', 'retention', 'clf_acc')]

    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = np.arange(len(arms))
    n_m = len(mets)
    w = 0.24
    colors = ['#4c78a8', '#f58518', '#54a24b']
    for i, (label, sec, key) in enumerate(mets):
        vals = [j['arms'][a][sec][key] for a in arms]
        bars = ax.bar(xs + (i - (n_m - 1) / 2) * w, vals, w,
                      color=colors[i], label=label.replace('\n', ' '))
        annotate_bars(ax, bars, '{:.2f}', fs=8)
    ax.axhline(0.25, ls=':', c=C_DIM, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([ARM_LABEL[a] for a in arms])
    ax.set_ylabel('探针性能（同一批场景快照 n=360）')
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper center', fontsize=9, ncol=3,
              framealpha=0.9, edgecolor='none')
    ax.set_title('F2 感知·浓缩表征：场景引导编码器的浓缩 z 保住决策所需信息\n'
                 '（编码器仅 1~3.3 万参数、CPU 单次前向 <0.7ms；'
                 '分类为 4 分位任务，虚线=随机水平 0.25）', fontsize=11)
    fig.tight_layout()
    fig.savefig(out / 'f2_repr_quality.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- F3
def fig3(root, out):
    """决策-主效应：配对口径呈现文本通道收益（绝对柱图压不出 ~9% 差）。"""
    arms = load(root, 'sig15_eval_main8')

    fig, (axa, axb, axc) = plt.subplots(
        1, 3, figsize=(14.6, 4.4),
        gridspec_kw={'width_ratios': [1.0, 1.15, 1.25]})

    # (a) 逐 case 配对连线：盲化孪生 → sga15（走廊排队原始值）
    g = pgain(arms, 'joint_sga15', 'joint_notext', 'corr_queue_int')
    xv, yv = g['b_raw'] / 1e3, g['a_raw'] / 1e3
    for x_, y_ in zip(xv, yv):
        axa.plot([0, 1], [x_, y_], '-', color='#bbb', lw=0.9, alpha=0.75,
                 zorder=2)
    axa.plot(0, np.median(xv), 's', ms=11, color=C_NOTEXT, zorder=3)
    axa.plot(1, np.median(yv), 'o', ms=11, color=C_SGA, zorder=3)
    axa.plot([0, 1], [np.median(xv), np.median(yv)], '-', color='#333',
             lw=2.2, zorder=3)
    axa.set_xlim(-0.55, 1.55)
    axa.set_xticks([0, 1])
    axa.set_xticklabels(['同模型·盲化文本', '场景引导注意力\n(sga15)'],
                       fontsize=9.5)
    axa.set_ylabel('走廊排队积分（×10³ veh·s）')
    axa.set_title(f'(a) 逐 case 配对连线（n={g["n"]}）：'
                  f'{g["win"]}/{g["n"]} case 排队下降，中位改善 '
                  f'{g["med"]:.1f}%，Wilcoxon {star(g["p"])}',
                  fontsize=10.5)
    axa.text(0.02, np.median(xv), f'{np.median(xv):.1f}', color=C_NOTEXT,
             ha='right', va='center', fontsize=9, fontweight='bold')
    axa.text(0.98, np.median(yv), f'{np.median(yv):.1f}', color=C_SGA,
             ha='left', va='center', fontsize=9, fontweight='bold')
    axa.grid(axis='y', alpha=0.25)

    # (b) 分指标配对增益分布（sga15 vs 盲化孪生）：每点一个 case
    metrics = [('事件边排队', 'ev_queue_int', False),
               ('走廊排队', 'corr_queue_int', False),
               ('全路口排队', 'queue_int', False),
               ('峰值排队', 'peak_queue', False),
               ('吞吐量', 'throughput', True)]
    rng = np.random.default_rng(0)
    for i, (lab, k, hb) in enumerate(metrics):
        d = pgain(arms, 'joint_sga15', 'joint_notext', k,
                  higher_better=hb)
        jit = rng.uniform(-0.14, 0.14, len(d['pct']))
        axb.scatter(i + jit, d['pct'], s=13, color=C_SGA, alpha=0.45,
                    zorder=3)
        axb.plot([i - 0.22, i + 0.22], [d['med'], d['med']], color='#333',
                 lw=2.5, zorder=4)
        axb.text(i, 24.5, f'{d["med"]:+.1f}%{star(d["p"])}\n'
                 f'{d["win"]}/{d["n"]}胜', ha='center', va='top',
                 fontsize=8.3)
    axb.axhline(0, c='#555', lw=0.8)
    axb.set_xticks(range(len(metrics)))
    axb.set_xticklabels([m[0] for m in metrics], fontsize=9)
    axb.set_ylabel('对盲化孪生的配对改善（%，正=优）')
    axb.set_ylim(-15, 26)
    axb.set_title('(b) 唯一变量 = 文本通道：四个排队口径 + 吞吐\n'
                  '（点=逐 case，粗线=中位；* p<0.05 ** p<0.01）',
                  fontsize=10.5)
    axb.grid(axis='y', alpha=0.25)

    # (c) 增益口径分解：同一指标，四个参照对象
    refs = [('vs 同权重·置空文本\n(盲化孪生)', 'joint_notext'),
            ('vs 同权重·仅底座路径\n(joint_base)', 'joint_base'),
            ('vs 部署底座\n(dqn_s1)', 'dqn_s1'),
            ('vs 谎报\n(joint_lie)', 'joint_lie')]
    xs = np.arange(len(refs))
    w = 0.36
    for off, key, lab, c in [(-w / 2, 'corr_queue_int', '走廊排队',
                              C_SGA),
                             (w / 2, 'ev_queue_int', '事件边排队',
                              '#54a24b')]:
        ms, ps_, ns_ = [], [], []
        for _, ref in refs:
            d = pgain(arms, 'joint_sga15', ref, key)
            ms.append(d['mean'])
            ps_.append(d['p'])
            ns_.append((d['win'], d['n']))
        bars = axc.bar(xs + off, ms, w, color=c, label=lab)
        for b, m, p, (wn, nn) in zip(bars, ms, ps_, ns_):
            axc.text(b.get_x() + b.get_width() / 2,
                     m + (1.4 if m >= 0 else -3.4),
                     f'{m:+.1f}{star(p)}', ha='center', fontsize=7.8)
    axc.axhline(0, c='#555', lw=0.8)
    axc.set_xticks(xs)
    axc.set_xticklabels([r[0] for r in refs], fontsize=8.2)
    axc.set_ylabel('sga15 配对改善均值（%，正=优）')
    axc.set_ylim(-8, 27)
    axc.legend(fontsize=8.5, loc='upper left')
    axc.set_title('(c) 归因分解：收益按参照口径拆开\n'
                  '（底座联合微调已内化 +7.2%corr/+17.2%ev；'
                  '场景残差再 +3.7%ev）', fontsize=10.5)
    axc.grid(axis='y', alpha=0.25)

    fig.suptitle('F3 决策·主效应：corridor-load 场景下场景通道的事件治理收益'
                 '（逐 case 配对，n=20）', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out / 'f3_main_effect.png', bbox_inches='tight')
    plt.close(fig)
    return {m[0]: pgain(arms, 'joint_sga15', 'joint_notext',
                      m[1], higher_better=m[2])['mean']
            for m in metrics}


# ---------------------------------------------------------------- F4
def fig4(root, out):
    """决策-意图消融：意图编译粒度阶梯 + 同评测非联合参照臂。"""
    arms = load(root, 'sig15_eval_modcmp')

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.4),
                                 gridspec_kw={'width_ratios': [1, 1.15]})

    # (a) 意图阶梯：盲化 0 / 单点意图 / 走廊意图（对盲化配对增益 + 逐case点）
    rng = np.random.default_rng(0)
    rungs = [('盲化文本\n(参照)', 'joint_notext', C_NOTEXT),
             ('单点意图\n(sig12)', 'joint_sga12', '#e377c2'),
             ('走廊意图\n(sga15)', 'joint_sga15', C_SGA)]
    for i, (lab, arm, c) in enumerate(rungs):
        if arm == 'joint_notext':
            axa.bar(i, 0, 0.6, color=c, alpha=0.85)
            axa.text(i, 1.2, '0（参照）', ha='center', fontsize=9)
            continue
        d = pgain(arms, arm, 'joint_notext', 'corr_queue_int')
        axa.bar(i, d['mean'], 0.6, color=c, alpha=0.85)
        jit = rng.uniform(-0.16, 0.16, len(d['pct']))
        axa.scatter(i + jit, d['pct'], s=14, color='#333', alpha=0.55,
                    zorder=4)
        axa.text(i, d['mean'] + (1.8 if d['mean'] >= 0 else -4.2),
                 f'{d["mean"]:+.1f}%{star(d["p"])}\n{d["win"]}/{d["n"]}胜',
                 ha='center', fontsize=8.5)
    axa.axhline(0, c='#555', lw=0.8)
    axa.set_xticks(range(3))
    axa.set_xticklabels([r[0] for r in rungs], fontsize=9.5)
    axa.set_ylabel('走廊排队对盲化孪生的配对改善（%）')
    axa.set_ylim(-24, 16)
    p12 = pgain(arms, 'joint_sga15', 'joint_sga12', 'corr_queue_int')['p']
    axa.set_title('(a) 意图编译粒度：单点直给反而拖累（−14.4%），\n'
                  f'走廊级编译才兑现（+8.5%**，vs 单点 p={p12:.3f}）',
                  fontsize=10.5)
    axa.grid(axis='y', alpha=0.25)

    # (b) 同评测参照：同编码器不走联合蒸馏链 → 负增益
    refs = [('concat_st', '拼接编码器\n(无注意力)'),
            ('selfattn_st', '自注意力编码器'),
            ('sga_st', 'sga编码器\n(非joint)'),
            ('joint_base', '同权重\n仅底座路径'),
            ('maxpressure', 'MaxPressure')]
    xs = np.arange(len(refs))
    ms, ps_, wn = [], [], []
    for arm, _ in refs:
        d = pgain(arms, arm, 'joint_notext', 'corr_queue_int')
        ms.append(d['mean'])
        ps_.append(d['p'])
        wn.append((d['win'], d['n']))
    # 颜色按臂的角色而非符号：三个非joint编码器臂红、底座/MP为参照色
    cols = ['#d62728', '#d62728', '#d62728', C_DIM, C_REF]
    bars = axb.bar(xs, ms, 0.62, color=cols, alpha=0.8)
    for b, m, p, (w_, n_) in zip(bars, ms, ps_, wn):
        axb.text(b.get_x() + b.get_width() / 2,
                 m + (1.8 if m >= 0 else -5.5),
                 f'{m:+.0f}%\n{w_}/{n_}', ha='center', fontsize=8)
    axb.axhline(0, c='#555', lw=0.8)
    axb.set_xticks(xs)
    axb.set_xticklabels([r[1] for r in refs], fontsize=8.6)
    axb.set_ylabel('走廊排队对盲化孪生的配对改善（%）')
    axb.set_ylim(-95, 14)
    axb.set_title('(b) 同评测参照：编码器单独存在不构成收益\n'
                  '（不挂联合蒸馏链的编码器臂全线 −40%~−72%）',
                  fontsize=10.5)
    axb.grid(axis='y', alpha=0.25)

    fig.suptitle('F4 决策·设计消融：收益 = 走廊级意图编译 × 联合蒸馏训练链，'
                 '缺一不成立（同一评测 n=12）', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / 'f4_intent_ladder.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- F5
def fig5(root, out):
    """决策-鲁棒边界：报告延迟/强度/失真下，配对增益是否有界。"""
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(11.6, 4.3))

    # (a) 播报延迟 0/30/60/90/120s + 干扰播报（均为对孪生配对增益）
    delays = [(0, 'sig15_eval_delay0'), (30, 'sig15_eval'),
              (60, 'sig15_eval_delay60'), (90, 'sig15_eval_delay90'),
              (120, 'sig15_eval_delay120')]
    xs, ys, wins, pss = [], [], [], []
    for d, name in delays:
        arms = load(root, name)
        if arms is None:
            continue
        g = pgain(arms, 'joint_sga15', 'joint_notext', 'corr_queue_int')
        xs.append(d)
        ys.append(g['mean'])
        wins.append(f'{g["win"]}/{g["n"]}')
        pss.append(g['p'])
    axa.plot(xs, ys, 'o-', color=C_SGA, lw=2, ms=7,
             label='真实播报 vs 盲化孪生')
    for x, y, wn in zip(xs, ys, wins):
        axa.annotate(f'{y:+.1f}%\n({wn})', (x, y),
                     textcoords='offset points', xytext=(0, 12),
                     ha='center', fontsize=8, color=C_SGA)
    # 干扰播报：真实事件 + 无关干扰报告
    arms_d = load(root, 'sig15_eval_distract')
    gd = pgain(arms_d, 'joint_sga15', 'joint_notext', 'corr_queue_int')
    axa.plot([155], [gd['mean']], 'X', ms=11, color='#d62728',
             label=f'混入无关干扰播报（{gd["mean"]:+.1f}%）')
    axa.annotate(f'{gd["win"]}/{gd["n"]}', (155, gd['mean']),
                 textcoords='offset points', xytext=(0, 10),
                 ha='center', fontsize=8, color='#d62728')
    axa.axhline(0, c='#555', lw=0.9)
    axa.text(60, -2.6, '零线 = 盲化孪生水平', ha='center', fontsize=8,
             color='#777')
    axa.set_xlabel('播报延迟（s）｜最右点 = 干扰播报')
    axa.set_xticks(xs + [155])
    axa.set_xticklabels([f'{d}' for d in xs] + ['干扰'], fontsize=9)
    axa.set_ylabel('走廊排队配对改善（%，正=优）')
    axa.set_ylim(-5, 13)
    axa.set_title('(a) 播报迟到 0~120s 全档正收益（胜率≥8/12）；\n'
                  '内容失真时回落到盲化水平——优雅降级，不劫持',
                fontsize=10.5)
    axa.legend(fontsize=8.5, loc='upper right')
    axa.grid(alpha=0.25)

    # (b) 事件强度（period 4/2/1 → 轻/中/重；sev=min(3,4/period)）
    arms = load(root, 'sig15_eval_sev')
    pers, sev_lab = ['4', '2', '1'], ['轻\n(period4)', '中\n(period2)',
                                     '重\n(period1)']
    for arm, c, lab, sty, dy, va in [
            ('joint_sga15', C_SGA, 'sga15 vs 盲化孪生', 'o-', 11, 'bottom'),
            ('joint_lie', C_LIE, '谎报 vs 盲化孪生', 'd--', -13, 'top')]:
        A, B = pair_by_event(arms, arm, 'joint_notext')
        ms, wn = [], []
        for per in pers:
            pcts = [(r0['corr_queue_int'] - r1['corr_queue_int'])
                    / r0['corr_queue_int'] * 100
                    for r1, r0 in zip(A, B)
                    if r1['event'].split(':')[-1] == per]
            ms.append(float(np.mean(pcts)) if pcts else np.nan)
            wn.append(f'{(np.array(pcts) > 0).sum()}/{len(pcts)}')
        axb.plot(range(3), ms, sty, color=c, lw=1.8, ms=7, label=lab)
        for i, (m, w) in enumerate(zip(ms, wn)):
            axb.annotate(f'{m:+.1f}%\n({w})', (i, m),
                         textcoords='offset points', xytext=(0, dy),
                         ha='center', va=va, fontsize=8, color=c)
    axb.axhline(0, c='#555', lw=0.9)
    axb.set_xticks(range(3))
    axb.set_xticklabels(sev_lab, fontsize=9)
    axb.set_xlabel('事件强度')
    axb.set_ylabel('走廊排队配对改善（%，正=优）')
    axb.set_ylim(-6, 16)
    axb.set_title('(b) 强度剂量响应：全档为正，中轻档兑现主要收益\n'
                  '（重度事件逼近容量极限，可作用余量小）', fontsize=10.5)
    axb.legend(fontsize=8.5)
    axb.grid(alpha=0.25)

    fig.suptitle('F5 决策·鲁棒边界：报告质量退化时收益收窄但不崩溃'
                 '（门控 + 剂量封顶的机制保障）', fontsize=11.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out / 'f5_robustness.png', bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------- F6
def fig6(root, out):
    """决策-迁移：同一 sga15 权重零样本到未见场景，文本通道收益。"""
    # 用 sig15_eval_S*（主模型直接评新场景），非 sig15_S*_eval（重训版）
    cases = [('主场景\nmain8', 'sig15_eval_main8'),
             ('留出事件\nholdout', 'sig15_eval_holdout'),
             ('S1\n需求分布', 'sig15_eval_S1'),
             ('S3\n需求分布', 'sig15_eval_S3'),
             ('S4\n需求分布', 'sig15_eval_S4')]
    labels, gs = [], []
    for lab, name in cases:
        arms = load(root, name)
        if arms is None or 'joint_sga15' not in arms:
            continue
        g = pgain(arms, 'joint_sga15', 'joint_notext', 'corr_queue_int')
        labels.append(lab)
        gs.append(g)

    xs = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    cols = [C_SGA if g['mean'] > 0 else '#d62728' for g in gs]
    bars = ax.bar(xs, [g['mean'] for g in gs], 0.6, color=cols)
    rng6 = np.random.default_rng(1)
    for i, (b, g) in enumerate(zip(bars, gs)):
        ax.text(b.get_x() + b.get_width() / 2,
                g['mean'] + (0.8 if g['mean'] >= 0 else -0.8),
                f'{g["mean"]:+.1f}%{star(g["p"])}\n'
                f'{g["win"]}/{g["n"]}胜', ha='center', fontsize=9,
                va='bottom' if g['mean'] >= 0 else 'top')
        jit = rng6.uniform(-0.17, 0.17, len(g['pct']))
        ax.scatter(i + jit, g['pct'], s=11,
                   color='#333', alpha=0.4, zorder=4)
    ax.axhline(0, c='#555', lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel('走廊排队对盲化孪生的配对改善（%）')
    ax.set_ylim(-14, 28)
    ax.set_title('F6 决策·迁移：同一 sga15 权重在留出事件与未见需求分布上\n'
                 '文本通道收益方向保持（S4 事件边口径 −13%，走廊口径为正）',
                 fontsize=11)
    ax.grid(axis='y', alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / 'f6_transfer.png', bbox_inches='tight')
    plt.close(fig)
    return {l.replace('\n', ''): g['mean'] for l, g in zip(labels, gs)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default='output_data/dev_takeover_probe')
    ap.add_argument('--out', default='output_data/dev_takeover_probe/'
                                    'taskbook_figs')
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig1(root, out)
    fig2(root, out)
    m3 = fig3(root, out)
    fig4(root, out)
    fig5(root, out)
    m6 = fig6(root, out)
    print(json.dumps({'out': str(out), 'f3_paired_pct': m3,
                      'f6_paired_pct': m6}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
