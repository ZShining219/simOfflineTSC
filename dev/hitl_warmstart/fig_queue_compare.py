#!/usr/bin/env python3
"""warmloop 三臂 queue 轨迹对比图。

三组曲线（同一 S2 场景 + surge 注入协议，逐决策过程量）：

  A0     M0，优化前控制器（warmloop/eval_m0/sd*，student 臂 25 会话）
  expert MaxPressure 虚拟专家单独控盘（warmloop/eval_expert，expert 臂）
  A1     M1，M0 经 C 通道专家池续训后的控制器（warmloop/eval_m1/sd*）

数据源为 paired_probe 的 runs.json['series']（probe 内建逐决策过程量，
t/queue/throughput，决策间隔 10s）。三组臂的 series 插值对齐到统一
时间网格，绘制 mean±SEM。终端 travel_time 均值作为角标注出。

用法（仓库根，colight 环境）::

    python dev/hitl_warmstart/fig_queue_compare.py \
        --root data/output_data/hitl_probe/warmloop \
        --out  data/output_data/hitl_probe/warmloop/figs/fig_queue_three_arm.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# (label, color, runs.json glob 相对 --root)
GROUPS = [
    ('A0: M0 (baseline)', '#888888', 'eval_m0/sd*/runs.json'),
    ('A1: C-channel round 1', '#c23b22', 'eval_m1/sd*/runs.json'),
    ('A2: A-channel round', '#2e8b57', 'eval_a2/sd*/runs.json'),
    ('A3: C+A iterative rounds', '#17becf', 'eval_a4/sd*/runs.json'),
    ('expert: MaxPressure (virtual)', '#1f6fb4', 'eval_expert/runs.json'),
]

EVENT_WINDOW = (300.0, 600.0)  # surge 注入窗（probe 协议固定）


def _load_group_runs(root: Path, pattern: str) -> list[dict]:
    runs: list[dict] = []
    for f in sorted(root.glob(pattern)):
        runs.extend(json.loads(f.read_text(encoding='utf-8')))
    return runs


def _series_arrays(runs: list[dict]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    ts, qs = [], []
    for r in runs:
        rows = sorted(r.get('series') or [],
                      key=lambda x: x.get('decision_step') or 0)
        t = [float(x['t']) for x in rows if x.get('queue') is not None]
        q = [float(x['queue']) for x in rows if x.get('queue') is not None]
        if t:
            ts.append(np.asarray(t))
            qs.append(np.asarray(q))
    return ts, qs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', required=True, help='warmloop 输出根目录')
    ap.add_argument('--out', required=True, help='输出 png 路径')
    args = ap.parse_args()
    root = Path(args.root)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.axvspan(*EVENT_WINDOW, color='#f4c542', alpha=0.15,
               label='injected surge window')

    tt_note = []
    for label, color, pattern in GROUPS:
        runs = _load_group_runs(root, pattern)
        ts, qs = _series_arrays(runs)
        if not ts:
            print(f'[WARN] no series for {label} ({len(runs)} runs)')
            continue
        grid = np.unique(np.concatenate(ts))
        mat = np.stack([np.interp(grid, t, q) for t, q in zip(ts, qs)])
        mean = mat.mean(axis=0)
        sem = mat.std(axis=0, ddof=1) / np.sqrt(len(mat))
        ax.plot(grid, mean, color=color, lw=1.9, label=f'{label} (n={len(mat)})')
        ax.fill_between(grid, mean - sem, mean + sem, color=color, alpha=0.15)
        tts = [float(r['travel_time']) for r in runs
               if r.get('travel_time') is not None]
        if tts:
            tt_note.append(f"{label.split(':')[0]} tt={np.mean(tts):.1f}s")

    ax.set_xlabel('simulation time (s)')
    ax.set_ylabel('mean queued vehicles')
    ax.set_title('Queue trajectory: controller iterative HITL expert-data '
                 'training vs MaxPressure expert')
    ax.legend(loc='best', fontsize=8)
    ax.text(0.99, 0.02, '  '.join(tt_note), transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8, color='#444444')
    fig.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'[OK] wrote {out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
