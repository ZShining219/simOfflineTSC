"""人在回路（HITL）模块架构图（中文，探索件，登记 dev/INDEX.md）。

呈现口径：一个系统三层——
  在线会话层：人只出现在"专家席"；gate 决定何时问人，
    recorder 全量审计（transitions/proposals/interventions）。
  数据资产层：同一份录制分层成三种人源数据（demo / 片段+偏好 /
    纠偏转移），package.py 打包成可复用池。
  训练层：三条消费路径（A 双预训 / B 奖励塑形 / C 第三回放腿）
    作用于同一个 HASODQNAgent；训练出的策略可回到会话层当
    新学生——人在回路是可持续闭环，不是一次性注入。

用法（repo 根，conda env: colight）：

    python hitl/figs/hitl_arch_fig.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch

FONT = Path.home() / '.local/share/fonts/NotoSansCJKsc-Regular.otf'
if FONT.exists():
    font_manager.fontManager.addfont(str(FONT))
    plt.rcParams['font.family'] = \
        font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams['axes.unicode_minus'] = False

BLUE = ('#eaf1f8', '#4a7fbf')      # 仿真/策略
ORANGE = ('#fdf3e0', '#c8902a')    # 人（专家席）
PURPLE = ('#f3e8f7', '#8a4fa3')    # gate/编排
GREEN = ('#e7f4e8', '#2e8b43')     # 人源数据资产
YELLOW = ('#fff8e0', '#c8a02a')    # 训练通道
GRAY = ('#f2f2f2', '#777')


def box(ax, x0, y0, x1, y1, fc, ec, title, body, title_fs=10.5,
        body_fs=8.4, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.3,rounding_size=0.9',
        fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x0 + 1.4, y1 - 0.9, title, fontsize=title_fs,
            fontweight='bold', color=ec, va='top', zorder=3)
    if body:
        ax.text(x0 + 1.4, y1 - 3.4, body, fontsize=body_fs,
                color='#222', va='top', zorder=3, linespacing=1.45)


def arrow(ax, xy, xytext, color='#444', lw=1.5, ls='-', label=None,
          lab_xy=None, lab_fs=8.0, rad=0.0, zorder=4):
    ax.annotate('', xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=zorder)
    if label and lab_xy:
        ax.text(*lab_xy, label, fontsize=lab_fs, color=color,
                ha='center', zorder=zorder,
                bbox=dict(fc='white', ec='none', pad=1.0))


def main():
    out = Path('output_data/hitl_probe/pilot5/figs_deck')
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 104)
    ax.set_ylim(0, 74)
    ax.axis('off')

    ax.text(52, 72.2, '人在回路交通信号控制模块（HITL-live）架构',
            ha='center', fontsize=15, fontweight='bold')
    ax.text(52, 70.2, '人只出现在一个位置：专家席　|　一份会话录制'
            ' → 分层成三种人源数据 → 三条训练消费路径　|　'
            '冻结评测闭环，训练后策略可回会话层当新学生',
            ha='center', fontsize=10, color='#444')

    # ============ 层一：在线会话（左，虚线包框） ============
    ax.add_patch(FancyBboxPatch(
        (1.5, 30), 40.5, 37,
        boxstyle='round,pad=0.3,rounding_size=1.2',
        fc='#fbfaf7', ec='#888', lw=1.5, ls=(0, (6, 3)), zorder=1))
    ax.text(22, 67.0, '在线会话层（paired_probe / 直播引擎）',
            fontsize=10.5, fontweight='bold', color='#666',
            ha='center', zorder=3,
            bbox=dict(fc='white', ec='none', pad=2))

    box(ax, 4, 55, 19, 64.5, *BLUE, 'SUMO 世界',
        'libsumo 实时会话\n'
        'horizon 1200s · 10s/决策\n'
        'surge 注入造事件', body_fs=8.0)
    box(ax, 4, 41.5, 19, 53, *BLUE, '学生控制器',
        '冻结 ep50 DQN 快照\n'
        '逐步提出动作提案', body_fs=8.0)
    box(ax, 4, 30.5, 19, 39.5, *PURPLE, 'Gate 触发器',
        'on_demand：排队增长\n'
        '触发问人 · 预算 36/ep\n'
        '冷却 · 全程记账', body_fs=8.0)
    box(ax, 24, 41.5, 40, 53, *ORANGE, '专家席（人=虚拟专家）',
        'MaxPressure 代人\n'
        '（真人模式：冻结仿真\n'
        '  横幅等回答，超时降级）\n'
        '回答：接管 / 同意', body_fs=8.0)
    box(ax, 24, 30.5, 40, 39.5, *GRAY, 'Recorder 审计',
        'transitions + proposals\n'
        '+ interventions.jsonl\n'
        '逐决策可溯源', body_fs=8.0)

    # 会话层内流：世界→学生观测；学生→提案；gate 监听提案；
    # gate→专家席问；专家→accepted action 回世界；全落 recorder
    arrow(ax, (13, 55), (13, 53.2), color='#4a7fbf', label='观测',
          lab_xy=(10.5, 54.4))
    arrow(ax, (17, 47), (23.8, 47), color='#c8902a',
          label='gate 命中时\n送提案', lab_xy=(20.6, 49.4))
    arrow(ax, (17, 35), (23.8, 35), color='#8a4fa3')
    arrow(ax, (12, 41.5), (12, 39.7), color='#8a4fa3')
    arrow(ax, (31, 53), (15, 55.2), color='#c8902a',
          label='接管/同意', lab_xy=(26, 54.6))
    arrow(ax, (31, 41.5), (31, 39.7), color='#888')

    # ============ 层二：数据资产（中） ============
    ax.add_patch(FancyBboxPatch(
        (44.5, 30), 25, 37,
        boxstyle='round,pad=0.3,rounding_size=1.2',
        fc='#f6fbf6', ec='#2e8b43', lw=1.5, ls=(0, (6, 3)), zorder=1))
    ax.text(57, 67.0, '数据资产层（package.py）',
            fontsize=10.5, fontweight='bold', color='#2e8b43',
            ha='center', zorder=3,
            bbox=dict(fc='white', ec='none', pad=2))

    box(ax, 47, 54, 68, 64, *GREEN, '打包档案（trajectory 分片）',
        'expert 控盘池 1462 转移\n'
        'hitl 纠偏池 2190（含128条接管标注）\n'
        'student 对照池 1213', body_fs=8.0)
    box(ax, 47, 44.5, 68, 52.5, *GREEN, '① 示范集 demo.npz',
        'expert 转移扁平化\n'
        'episode 内时序连续', body_fs=8.0)
    box(ax, 47, 37, 68, 43, *GREEN, '② 片段+偏好对',
        '6决策窗切段 → 教师分排序\n'
        '→ RM 偏好对（720+2400×3）', body_fs=8.0)
    box(ax, 47, 30, 68, 35.5, *GREEN, '③ 纠偏转移池',
        'expert_pool_manifest\n'
        '（agreed/took_over 标注）', body_fs=8.0)

    arrow(ax, (40.3, 35), (46.8, 40), color='#888',
          label='一份录制', lab_xy=(43.6, 41.5))
    ax.text(57.5, 28.2, '同一份会话录制 → 三种人源数据',
            fontsize=8.5, color='#2e8b43', ha='center', style='italic')

    # ============ 层三：训练（右） ============
    ax.add_patch(FancyBboxPatch(
        (71.5, 30), 30.5, 37,
        boxstyle='round,pad=0.3,rounding_size=1.2',
        fc='#fdfcf3', ec='#c8a02a', lw=1.5, ls=(0, (6, 3)), zorder=1))
    ax.text(87, 67.0, '训练层（sequential / HASODQNAgent）',
            fontsize=10.5, fontweight='bold', color='#a08030',
            ha='center', zorder=3,
            bbox=dict(fc='white', ec='none', pad=2))

    box(ax, 74, 54, 100, 64, *YELLOW, 'A：示范双预训',
        'demo → actor+critic 同躯干\n'
        '2000 步 BC+TD → ep0 即教师级\n'
        '（主承重通道：跨场景全胜）', body_fs=8.0)
    box(ax, 74, 44.5, 100, 52.5, *YELLOW, 'B：奖励塑形（RM 冻结）',
        "r' = r0 + λ·clip((f−μ)/σ,±3)\n"
        'f 由偏好对学的奖励模型\n'
        '零样本迁移 79~83% 准确率', body_fs=8.0)
    box(ax, 74, 35.5, 100, 43, *YELLOW, 'C：专家回放腿',
        '转移按 expert_ratio 混入\n'
        '（静态池上限=中性，已查明）', body_fs=8.0)
    box(ax, 74, 30, 100, 34, *BLUE, '→ 冻结评测 / 新学生',
        '20ep 训练 → 逐ep冻结eval\n'
        '策略可回会话层迭代', body_fs=8.0)

    # 资产→通道箭头（颜色一一对应）
    arrow(ax, (68.2, 48.5), (73.8, 57), color='#2e8b43')
    arrow(ax, (68.2, 40), (73.8, 48), color='#2e8b43')
    arrow(ax, (68.2, 32.5), (73.8, 38), color='#2e8b43')

    # ============ 底条：证据链 ============
    box(ax, 4, 3, 100, 26, *GRAY, '证据链（每一环可审计/可统计）', '',
        title_fs=10)
    ev = [
        ('① 进入', '干预账本：\nqueries/takeover\n预算克制'),
        ('② 可信', 'RM 偏好准确率\n94.6% / 零样本\n79~83%'),
        ('③ 优化', 'FULL vs C0\n4场景16指标\n全显著'),
        ('④ 消融', '单通道臂定位\n承重结构\n(A>B>C)'),
        ('⑤ 代价', 'demo 1462 条\n+干预预算\n极少消耗'),
        ('⑥ 边界', 'kn 零触发 /\nC 静态池中性\n（如实呈现）'),
    ]
    x = 6
    for t, sub in ev:
        ax.text(x + 6.5, 20.5, t, fontsize=9.5, fontweight='bold',
                color='#555', ha='center', zorder=3)
        ax.text(x + 6.5, 11.5, sub, fontsize=8, color='#333',
                ha='center', zorder=3, linespacing=1.35)
        x += 16
        if x < 92:
            ax.annotate('', xy=(x - 0.5, 15), xytext=(x - 3, 15),
                        arrowprops=dict(arrowstyle='-|>', color='#aaa',
                                        lw=1.4), zorder=3)

    # 闭环虚线：新学生回会话层
    ax.annotate('', xy=(19.5, 44), xytext=(87, 29.8),
                arrowprops=dict(arrowstyle='-|>', color='#4a7fbf',
                                lw=1.4, linestyle=(0, (5, 4)),
                                connectionstyle='arc3,rad=0.28'),
                zorder=3)
    ax.text(58, 24.5, '闭环：训练后策略可作为新学生重新进入人在回路会话'
            '（迭代采集→再训练）', fontsize=8.5, color='#4a7fbf',
            ha='center', style='italic', zorder=3,
            bbox=dict(fc='white', ec='none', pad=1.2))

    fig.tight_layout()
    fig.savefig(out / 'hitl_arch.png', dpi=150,
                bbox_inches='tight')
    print('wrote', out / 'hitl_arch.png')


if __name__ == '__main__':
    main()
