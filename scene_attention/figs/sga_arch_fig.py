"""现行场景引导注意力架构图（sga15 / SceneJointPolicy，中文）。

相对 arch_fig/design_intent_attn.png（router 时代英文版）的修正：
  z = fuse3(ctx, 原始交通残差, text_proj)   —— 不是 sum(w·lane_token)
  底座 dqn_s1 在图内（ckpt 初始化 + --base-lr 微调），非冻结外挂
  增加训练链路条：adapter 教师 → 赢家过滤 → 加权 CE 蒸馏 → 原装 reward TD
  明确设计纪律：绑定只做门控/审计，不做转向

用法::

    python -m scene_attention.sga_arch_fig \
        --out output_data/dev_takeover_probe/taskbook_figs
"""

from __future__ import annotations

import argparse
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

# 配色：与 taskbook_figs 图组一致
BLUE = ('#eaf1f8', '#4a7fbf')      # 世界/策略
ORANGE = ('#fdf3e0', '#c8902a')    # 外部播报
PURPLE = ('#f3e8f7', '#8a4fa3')    # 意图编译
GREEN = ('#e7f4e8', '#2e8b43')     # sga 编码器（主角）
YELLOW = ('#fff8e0', '#c8a02a')    # 训练链路
GRAY = '#777'


def box(ax, x0, y0, x1, y1, fc, ec, title, body, title_fs=10.5,
        body_fs=8.6, lw=1.4):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle='round,pad=0.3,rounding_size=0.9',
        fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x0 + 1.4, y1 - 0.9, title, fontsize=title_fs,
            fontweight='bold', color=ec, va='top', zorder=3)
    if body:
        ax.text(x0 + 1.4, y1 - 3.6, body, fontsize=body_fs,
                color='#222', va='top', zorder=3, linespacing=1.45)


def arrow(ax, xy, xytext, color='#444', lw=1.5, ls='-', label=None,
          lab_xy=None, lab_fs=8.5, rad=0.0, zorder=4):
    ax.annotate('', xy=xy, xytext=xytext,
                arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                linestyle=ls, shrinkA=0, shrinkB=0,
                                connectionstyle=f'arc3,rad={rad}'),
                zorder=zorder)
    if label and lab_xy:
        ax.text(*lab_xy, label, fontsize=lab_fs, color=color,
                ha='center', zorder=zorder)


def polyline(ax, pts, color='#999', lw=1.2, ls='--', arrow_end=True):
    """折线（可带末端箭头）。"""
    for i in range(len(pts) - 2):
        ax.plot([pts[i][0], pts[i + 1][0]], [pts[i][1], pts[i + 1][1]],
                color=color, lw=lw, ls=ls, zorder=3)
    if arrow_end:
        ax.annotate('', xy=pts[-1], xytext=pts[-2],
                    arrowprops=dict(arrowstyle='-|>', color=color, lw=lw,
                                    linestyle=ls, shrinkA=0, shrinkB=0),
                    zorder=3)
    else:
        ax.plot([pts[-2][0], pts[-1][0]], [pts[-2][1], pts[-1][1]],
                color=color, lw=lw, ls=ls, zorder=3)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='output_data/dev_takeover_probe/'
                                    'taskbook_figs')
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15.5, 9.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 68)
    ax.axis('off')

    ax.text(50, 66.8, '场景引导注意力信号控制器（SceneJointPolicy · sga15）'
            '当前架构', ha='center', fontsize=15, fontweight='bold')
    ax.text(50, 64.9, '外部输入仅两项：SUMO 观测 + 现象级场景播报　|　'
            '播报入 SGA 两条通路：text_vec / 意图编译（θ_reward·θ_watch）'
            '　|　门控有界残差：绑定不足 → 恒等底座',
            ha='center', fontsize=10, color='#444')

    # ---------------- 左列：仅两项外部输入 ----------------
    box(ax, 2, 46, 24, 62, *BLUE, 'SUMO 世界（观测）',
        '车道 token ×8 边：排队/在途/等待\n'
        '　 + 边身份 one-hot（与播报共索引）\n'
        '全局：相位 · 时刻 · 总排队\n'
        'corridor-load 注入：surge(事件边)\n'
        '　 + surge(feeder) + throttle(discharge)')
    box(ax, 2, 30, 24, 44, *ORANGE, '现象级场景播报（外部接口）',
        '现象级 · 无控制指令 · 延迟 ~30s\n'
        'kind / edge / severity / onset /\n'
        '预计剩余时长 · objective\n'
        '例："congestion: road_2_1_0, 强度3"')

    # ---------------- SGA 边界（机制主体，虚线包框） ----------------
    ax.add_patch(FancyBboxPatch(
        (27.5, 15), 40, 47.4,
        boxstyle='round,pad=0.3,rounding_size=1.2',
        fc='#f6fbf6', ec='#2e8b43', lw=1.6, ls=(0, (6, 3)), zorder=1))
    ax.text(47.5, 62.7, '场景引导注意力机制（SGA）', fontsize=11.5,
            fontweight='bold', color='#2e8b43', ha='center', va='center',
            zorder=3, bbox=dict(fc='white', ec='none', pad=2.5))

    # 播报 → 两条通路（均穿边界进入机制内部）
    arrow(ax, (30.7, 46), (24.3, 39), color='#c8902a')
    ax.text(27.0, 43.9, '通路① text_vec', fontsize=7.8, color='#c8902a',
            rotation=47, ha='center', zorder=4,
            bbox=dict(fc='white', ec='none', pad=1.2))
    arrow(ax, (30.7, 28.5), (24.3, 33), color='#8a4fa3')
    ax.text(28.4, 30.4, '通路② 播报字段', fontsize=7.8, color='#8a4fa3',
            rotation=-35, ha='center', zorder=4,
            bbox=dict(fc='white', ec='none', pad=1.2))

    # 意图编译：SGA 内部规则模块（训练通路）
    box(ax, 31, 17, 50, 31.5, *PURPLE, '意图编译（内部规则，可审计）',
        'θ_reward = sev/3·事件边\n'
        '　 + coord_w·(feeder+discharge)\n'
        '　 → r_scene = −mean(q) − λ·Σ(q·θ)/Σθ\n'
        'θ_watch = θ + 0.6·coord_w·2-hop 来路\n'
        '　 → aux：加权 CE(w, θ_watch)\n'
        '（feeder/discharge 拓扑取自 SUMO）',
        title_fs=9.5, body_fs=7.6)

    # 编码器：L1→L2 是模块递进；残差/text 是并行旁支
    box(ax, 31, 34, 66, 60.5, *GREEN, 'CompositeSceneEncoder',
        '【递进主干】\n'
        ' lane_proj → L1 车道自注意力 → kv′\n'
        ' L2 文本→车道交叉注意力（q=text_proj）\n'
        ' logits += bind_gain·(lane_id·text_edge)\n'
        '   ——身份绑定：播报 = 事件位置唯一来源\n'
        '【并行旁支】原始残差 mms(物理维)+global｜text_proj\n'
        ' z = fuse3( ctx, 残差, text )：64 维浓缩表征\n'
        '副产物 w∈R^L：只作门控/审计，不做转向',
        body_fs=8.2)

    # ---------------- 右列：策略 ----------------
    box(ax, 70, 50, 96, 62, *BLUE, '底座 dqn_s1（图内）',
        'ckpt 初始化 + --base-lr 微调\n'
        '输入：入口 lane_count + 相位 one-hot\n'
        '→ q_base')
    box(ax, 70, 30, 96, 47, *BLUE, '门控有界残差策略头',
        'Q = q_base\n'
        '　 + dose(0.35)·spread(q_base)\n'
        '　 · gate · tanh(δ(z))\n'
        'gate = ramp(最大边绑定质量, τ=0.4)\n'
        'δ 零初始化 → 出生残差恒 0')
    box(ax, 70, 18, 96, 27, *BLUE, '动作',
        'argmax_a Q → SUMO 信号相位\n'
        '（最短绿时约束）')

    # ---------------- 底条：训练链路 ----------------
    box(ax, 2, 2, 96, 13, *YELLOW, '训练链路（教师蒸馏 → 原装目标微调）',
        '', title_fs=10)
    steps = [
        ('① adapter 教师\nSceneGuidedDQN', '冻结底座+注意力\n偏置偏转，跑事件局'),
        ('② 赢家过滤', '仅留跑赢原始\ndqn_s1 的 episode'),
        ('③ 加权 CE 蒸馏', '偏离 tick ×3 权重\n→ sig9_joint_bc'),
        ('④ 原装 reward\nTD 微调', '与 dqn_s1 同收敛\n目标 → sig15'),
    ]
    x = 5
    for t, sub in steps:
        ax.text(x + 8, 10.0, t, fontsize=9, fontweight='bold',
                color='#7a5c00', ha='center', va='center', zorder=3,
                linespacing=1.3)
        ax.text(x + 8, 5.6, sub, fontsize=8, color='#333', ha='center',
                va='center', zorder=3, linespacing=1.3)
        x += 17
        if x < 75:
            ax.annotate('', xy=(x - 0.5, 7.5), xytext=(x - 3.2, 7.5),
                        arrowprops=dict(arrowstyle='-|>', color='#a08030',
                                        lw=1.6), zorder=3)
    ax.text(80.5, 7.5,
        '为何不直接 RL：\n从零学不会驾驶（sig6）\n场景 reward 联训灾难性\n'
        '遗忘 +9%（sig8）\n冻结残差仅 +0.7%（sig7）',
        fontsize=8, color='#7a3b00', va='center', zorder=3,
        linespacing=1.35)
    ax.plot([78.5, 78.5], [2.5, 12.5], color='#d8c080', lw=1, zorder=3)

    # ---------------- 数据流箭头 ----------------
    # SUMO → 编码器：lane tokens + global（穿 SGA 左边界）
    arrow(ax, (30.7, 52), (24.3, 52), color='#4a7fbf')
    ax.text(27.5, 53.8, 'lane tokens + global', fontsize=8,
            color='#4a7fbf', ha='center', zorder=4,
            bbox=dict(fc='white', ec='none', pad=1.2))
    # aux 监督：意图编译 θ_watch → 编码器 w（边界内部）
    arrow(ax, (44, 33.7), (44, 31.8), color='#8a4fa3', ls='--', lw=1.3)
    ax.text(46, 32.7, 'aux', fontsize=7.5, color='#8a4fa3')
    # θ_reward → 策略头 r_scene（出右边界，进 TD 更新）
    polyline(ax, [(50.3, 23), (66.8, 23), (66.8, 31), (69.7, 31)],
             color='#8a4fa3')
    ax.text(58.5, 24.8, 'θ_reward → r_scene（进 TD 更新）', fontsize=8,
            color='#8a4fa3', ha='center')
    # 底座特征：A1 顶 → 上方越过 SGA 边界 → C1 顶（不经 SGA；
    # 输入构成 C1 框内已注明，此处不再放标签以免与副标题碰撞）
    polyline(ax, [(14, 62.3), (14, 63.8), (83, 63.8), (83, 62.3)],
             color='#4a7fbf')
    # SGA 出边界 → 策略头：z 与 w 两条
    arrow(ax, (69.7, 44), (66.3, 52), color='#2e8b43')
    ax.text(67.9, 49.2, 'z', fontsize=9.5, color='#2e8b43',
            ha='center', zorder=4,
            bbox=dict(fc='white', ec='none', pad=1.2))
    arrow(ax, (69.7, 36.5), (66.3, 40), color='#2e8b43')
    ax.text(67.9, 37.2, 'w → gate', fontsize=8.5, color='#2e8b43',
            ha='center', zorder=4,
            bbox=dict(fc='white', ec='none', pad=1.2))
    # q_base 下行、argmax 下行
    arrow(ax, (83, 47.3), (83, 49.7), color='#4a7fbf')
    ax.text(88.5, 48.5, 'q_base', fontsize=8.5, color='#4a7fbf')
    arrow(ax, (83, 27.3), (83, 29.7), color='#4a7fbf')
    ax.text(88.8, 28.5, 'argmax', fontsize=8.5, color='#4a7fbf')
    # 闭环：动作 → SUMO（经画布最左 x=1 通道折返，避开所有箱体）
    polyline(ax, [(69.7, 20), (64, 20), (64, 14.2), (1, 14.2),
                  (1, 55), (1.6, 55)], color='#999')
    ax.text(32, 14.2, '相位 → 下一状态（闭环）', fontsize=8, color='#999',
            ha='center', va='center',
            bbox=dict(fc='white', ec='none', pad=1.2))

    fig.savefig(out / 'f0_architecture.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('written:', out / 'f0_architecture.png')


if __name__ == '__main__':
    main()
