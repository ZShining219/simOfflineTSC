"""HITL 汇报组图（探索件，登记 dev/INDEX.md）。

按顾诗怡论文的呈现逻辑组织：一图一事、对比对象唯一锚定 C0。

Fig1 「优化」主证据：主场景（bc-tyc）训练曲线——
      C0(不进人) vs APIQB/B2/EXPB(三条进人通道代表臂)，
      5 seed mean±std，tt/thr/unfin 三联。
      仿论文图4.3（完整版 vs 简化版训练曲线）的叙事位。

Fig2 「信号质量」：冻结奖励模型的偏好预测准确率——
      in-domain(bc-tyc 留出集 94.6%) vs 零样本三场景
      (qc/kn/sb 各 2400 对)。仿论文表4.3。

数据：pilot5/_shared_evaluation 冻结 eval；plan 的
run_id→condition 映射从 plan_A/plan_merged/plan_r2 合并。

用法（repo 根，conda env: colight）：

    python hitl/figs/deck_figs.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

from hitl.probe import paired_bootstrap as pb  # noqa: E402

ROOT = _REPO_ROOT / 'output_data/hitl_probe'
PILOT = ROOT / 'pilot5'
OUT = PILOT / 'figs_deck'

MAIN_ARMS = {   # 对比对象唯一锚点 C0 + 三通道代表臂
    'HITL_C0': ('C0 (no human)', '#7f7f7f'),
    'HITL_APIQB': ('A: demo-pretrain', '#1f77b4'),
    'HITL_B2': ('B: RM shaping', '#2ca02c'),
    'HITL_EXPB': ('C: expert pool', '#d62728'),
}
SYS_ARMS = {    # 系统集成消融：集成系统 vs 单形态注入 vs 无外部数据
    'HITL_C0': ('无外部数据注入（从头自学）', '#7f7f7f'),
    'HITL_FULL': ('集成人在回路系统（示范+偏好双通道）', '#000000'),
    'HITL_APIQB': ('仅示范轨迹注入', '#1f77b4'),
    'HITL_B2': ('仅偏好奖励塑形注入', '#2ca02c'),
    'HITL_EXPB': ('仅纠偏转移回放注入', '#d62728'),
}
METRICS = [('travel_time', '平均旅行时间（s，越小越好）'),
           ('throughput', '吞吐量（veh，越大越好）'),
           ('unfinished_vehicles', '滞留车辆数（越小越好）')]
PSEUDO_ARMS = {  # 外部数据源替换验证：同一注入管线，唯一变量=外部轨迹由谁产出
    'HITL_C0': ('无外部数据注入（从头自学）', '#7f7f7f'),
    'HITL_APIQB': ('外部轨迹注入：虚拟专家产出（1462条）', '#1f77b4'),
    'HITL_APISD': ('外部轨迹注入：训练中学生自产（1213条）', '#9467bd'),
    'HITL_APISW': ('外部轨迹注入：未训练学生自产（1217条）', '#d62728'),
}


def _cjk_font():
    from matplotlib import font_manager
    p = Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf'
    if p.exists():
        font_manager.fontManager.addfont(str(p))
        plt.rcParams['font.family'] = 'Noto Sans CJK SC'
        plt.rcParams['axes.unicode_minus'] = False


def merged_plan_map():
    m = {}
    for f in ['plan_A.json', 'plan_merged.json', 'plan_r2.json',
              'plan_r4.json']:
        p = json.loads((PILOT / f).read_text())
        for c in p['children']:
            m[c['logical_run_id']] = c['condition']
    return m


def collect_main(rows):
    """(condition,seed)->{ep:metric} 只保留主场景臂。"""
    series = {}
    for r in rows:
        if r['condition'] not in MAIN_ARMS:
            continue
        series.setdefault((r['condition'], r['seed']), {})[r['episode']] = \
            r['metrics']
    return series


SCENES_MS = [  # (tag, 显示名, eval 根, plan 过滤路径列表)
    ('bc', 'bc-tyc 2021veh', str(PILOT), None),
    ('qc', 'qc-yn 1417veh', str(ROOT / 'multiscene'), 'QC'),
    ('kn', 'kn-hz 743veh', str(ROOT / 'multiscene'), 'KN'),
    ('sb', 'sb-sx 1671veh', str(ROOT / 'multiscene'), 'SB'),
]


def fig3(rows, ms_rows):
    """4 场景 × 3 指标训练曲线总图（论文图4.3 版式）。"""
    _cjk_font()
    fig, axes = plt.subplots(4, 3, figsize=(15, 13))
    dpe = 120
    for i, (tag, title, _, flt) in enumerate(SCENES_MS):
        src = rows if tag == 'bc' else ms_rows
        series = {}
        for r in src:
            if flt and f'-{flt}-' not in r['run_id']:
                continue
            if r['condition'] not in ('HITL_C0', 'HITL_FULL'):
                continue
            series.setdefault((r['condition'], r['seed']), {})[
                r['episode']] = r['metrics']
        for j, (metric, ylabel) in enumerate(METRICS):
            ax = axes[i][j]
            for cond in ('HITL_C0', 'HITL_FULL'):
                label, color = SYS_ARMS[cond]
                pts = {}
                for (c, s), eps in series.items():
                    if c != cond:
                        continue
                    pts[s] = {e: m.get(metric) for e, m
                              in sorted(eps.items())
                              if m.get(metric) is not None}
                if not pts:
                    continue
                all_eps = sorted({e for p in pts.values() for e in p})
                xs, ys, ss = [], [], []
                for e in all_eps:
                    vals = [p[e] for p in pts.values() if e in p]
                    if not vals:
                        continue
                    xs.append(e * dpe)
                    ys.append(np.mean(vals))
                    ss.append(np.std(vals)
                              / max(len(vals), 1) ** 0.5)
                xs, ys, ss = map(np.asarray, (xs, ys, ss))
                lw = 2.2 if cond == 'HITL_FULL' else 1.2
                ax.plot(xs, ys, marker='o', ms=2.5, color=color, lw=lw)
                ax.fill_between(xs, ys - ss, ys + ss, color=color,
                                alpha=0.15)
            ax.set_xlim(0, 2500)
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(f'{title}\n{ylabel.split(" (")[0]}',
                              fontsize=8)
            else:
                ax.set_ylabel(ylabel.split(' (')[0], fontsize=8)
            if i == 3:
                ax.set_xlabel('训练决策数', fontsize=8)
    handles = [plt.Line2D([], [], color=SYS_ARMS[c][1],
                        label=SYS_ARMS[c][0])
               for c in ('HITL_C0', 'HITL_FULL')]
    fig.legend(handles=handles, loc='upper right', fontsize=9,
               ncol=2, bbox_to_anchor=(0.99, 1.005))
    fig.suptitle('集成人在回路系统 vs 无外部数据注入：四场景逐集冻结评测'
                 '（5 seed，阴影=SEM；场景本地 ep0 初始）', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT / 'fig3_curves_all_scenes.png', dpi=150)
    print('wrote', OUT / 'fig3_curves_all_scenes.png')


def fig0(rows):
    """系统集成消融（头图）：集成系统 vs 单形态注入 vs 无外部数据，
    主场景。"""
    _cjk_font()
    series = {}
    for r in rows:
        if r['condition'] not in SYS_ARMS:
            continue
        series.setdefault((r['condition'], r['seed']), {})[
            r['episode']] = r['metrics']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    dpe = 120
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for cond, (label, color) in SYS_ARMS.items():
            pts = {}
            for (c, s), eps in series.items():
                if c != cond:
                    continue
                pts[s] = {e: m.get(metric) for e, m in sorted(eps.items())
                          if m.get(metric) is not None}
            if not pts:
                continue
            all_eps = sorted({e for p in pts.values() for e in p})
            xs, ys, ss = [], [], []
            for e in all_eps:
                vals = [p[e] for p in pts.values() if e in p]
                if not vals:
                    continue
                xs.append(e * dpe)
                ys.append(np.mean(vals))
                ss.append(np.std(vals) / max(len(vals), 1) ** 0.5)
            xs, ys, ss = map(np.asarray, (xs, ys, ss))
            lw = 2.0 if cond == 'HITL_FULL' else 1.2
            ax.plot(xs, ys, marker='o', ms=2.5, color=color, lw=lw,
                    label=label)
            ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.12)
        ax.set_xlim(0, 2500)
        ax.set_xlabel('训练决策数（每次10s）')
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('主场景 bc-tyc：集成系统 vs 单形态注入 vs 无外部数据'
                 '（冻结评测，5 seed±SEM）',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig0_system.png', dpi=150)
    print('wrote', OUT / 'fig0_system.png')


def fig2():
    zs = json.loads((ROOT / 'multiscene/rm_zeroshot_acc.json').read_text())
    rm = json.loads((ROOT / 'reward_model_v1/reward_manifest.json')
                    .read_text())
    in_acc = rm['val_metrics']['strict_accuracy']
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    names = ['bc-tyc\n(holdout)', 'qc-yn\n(0-shot)', 'kn-hz\n(0-shot)',
             'sb-sx\n(0-shot)']
    accs = [in_acc, zs['qc']['strict_accuracy'],
            zs['kn']['strict_accuracy'], zs['sb']['strict_accuracy']]
    colors = ['#1f77b4', '#2ca02c', '#2ca02c', '#2ca02c']
    bars = ax.bar(names, [a * 100 for a in accs], color=colors, width=0.55)
    for b, a in zip(bars, accs):
        ax.text(b.get_x() + b.get_width() / 2, a * 100 + 1,
                f'{a * 100:.1f}%', ha='center', fontsize=10)
    ax.axhline(50, color='gray', ls='--', lw=1)
    ax.text(3.35, 51, 'chance', fontsize=8, color='gray')
    ax.set_ylim(0, 105)
    ax.set_ylabel('pairwise preference accuracy (%)')
    ax.set_title('Frozen reward model: preference prediction accuracy\n'
                 '(2400 pairs/scene, rule-teacher labels, zero-shot transfer)')
    fig.tight_layout()
    fig.savefig(OUT / 'fig2_rm_acc.png', dpi=150)
    print('wrote', OUT / 'fig2_rm_acc.png')


def fig4_pseudo(rows):
    """外部数据源替换验证：同一注入管线（demo→双预训），外部轨迹
    由谁产出不影响收益——证明管线对来源通用，真人=换源不换管线。"""
    _cjk_font()
    series = {}
    for r in rows:
        if r['condition'] not in PSEUDO_ARMS:
            continue
        series.setdefault((r['condition'], r['seed']), {})[
            r['episode']] = r['metrics']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    dpe = 120
    for ax, (metric, ylabel) in zip(axes, METRICS):
        for cond, (label, color) in PSEUDO_ARMS.items():
            pts = {}
            for (c, s), eps in series.items():
                if c != cond:
                    continue
                pts[s] = {e: m.get(metric) for e, m in sorted(eps.items())
                          if m.get(metric) is not None}
            if not pts:
                continue
            all_eps = sorted({e for p in pts.values() for e in p})
            xs, ys, ss = [], [], []
            for e in all_eps:
                vals = [p[e] for p in pts.values() if e in p]
                if not vals:
                    continue
                xs.append(e * dpe)
                ys.append(np.mean(vals))
                ss.append(np.std(vals) / max(len(vals), 1) ** 0.5)
            xs, ys, ss = map(np.asarray, (xs, ys, ss))
            ax.plot(xs, ys, marker='o', ms=2.5, color=color, lw=1.6,
                    label=label)
            ax.fill_between(xs, ys - ss, ys + ss, color=color, alpha=0.12)
        ax.set_xlim(0, 2500)
        ax.set_xlabel('训练决策数（每次10s）')
        ax.set_ylabel(ylabel.split(' (')[0])
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')
    fig.suptitle('外部数据源替换验证：管线忠实注入源质量——'
                 '好源等效、坏源有害（主场景 bc-tyc，5 seed±SEM）',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / 'fig4_demo_source.png', dpi=150)
    print('wrote', OUT / 'fig4_demo_source.png')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan_map = merged_plan_map()
    # collect 按 plan children 的 run_id 过滤，需合并三份 plan
    children = []
    for f in ['plan_A.json', 'plan_merged.json', 'plan_r2.json',
              'plan_r4.json', 'plan_pseudo.json', 'plan_weak.json']:
        children += json.loads((PILOT / f).read_text())['children']
    tmp = OUT / '_merged_plan.json'
    tmp.write_text(json.dumps({'children': children}))
    rows = pb.collect(str(PILOT), str(tmp))
    for r in rows:
        r['condition'] = plan_map.get(r['run_id'], r['condition'])
    # r3 多场景 eval（run_id 含 -QC-/-KN-/-SB-）+ r4ms FULL 臂
    ms_children = []
    for f in ['plan_r3.json', 'plan_r4ms.json']:
        ms_children += json.loads((PILOT / f).read_text())['children']
    ms_map = {c['logical_run_id']: c['condition'] for c in ms_children}
    t3 = OUT / '_r3_plan.json'
    t3.write_text(json.dumps({'children': ms_children}))
    ms_rows = pb.collect(str(ROOT / 'multiscene'), str(t3))
    for r in ms_rows:
        r['condition'] = ms_map.get(r['run_id'], r['condition'])
    series = collect_main(rows)
    print('arms found:', {c: sorted(s for (cc, s) in series if cc == c)
                          for c in MAIN_ARMS})
    fig0(rows)
    fig2()
    fig3(rows, ms_rows)
    fig4_pseudo(rows)


if __name__ == '__main__':
    main()
