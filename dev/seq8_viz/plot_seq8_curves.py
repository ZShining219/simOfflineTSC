#!/usr/bin/env python3
"""SEQ8 当前场景适应保持：仿真内性能曲线对比（三臂）

用法：python dev/seq8_viz/plot_seq8_curves.py [输出目录]

口径：每条链（fam×seed）取 stage k 末冻结 ckpt 对当前场景的一次完整
0–3600s 评估仿真，得到逐步性能曲线（10s 决策网格，360 点）；8 个 stage
的曲线按仿真时刻逐点平均 → 每 arm 每序列一条"平均仿真内曲线"。
若三臂当前场景适应能力一致，三条曲线应重合。

指标（两侧同口径，WT 不入图——见下）：
- travel_time(t)：t 时刻已到达车辆的行程时间均值（world.vehicles 累计
  均值；CONT/DHOA 侧由 tripinfo.xml 按 arrival≤t 的 duration 累计均值
  逐步重建，语义等价）；
- queue(t)：路口排队车道计数快照之和（decisions.jsonl 'queue' =
  agent.get_queue()；ANCHOR 侧 = mean(ag.get_queue())，单路口同值）；
- throughput(t)：累计到达车辆数（decisions throughput_cumulative；
  ANCHOR 侧 len(world.vehicles)）。
- waiting_time 排除：cl 侧为"当前在网车辆累积等待"快照语义，
  sequential 侧 tripinfo 只能重建"已到达车辆"口径，两义不可比。

数据源：
- CONT/DHOA（sequential runtime）：<batch>/<run>/attempts/attempt_1/
  events.jsonl 的 evaluation:stage_k:local_100:<net> 操作 → alias →
  physical/*/attempt_1/{decisions.jsonl,tripinfo.xml}（对角线格 = stage
  k 末 ckpt × 第 k 个场景）；
- ANCHOR：s8fs_* 补跑批（run_queue_seq8_stepeval.json，48 对角线格，
  eval_step_log=true）的 evaluation/steps_ep0000.jsonl。
  注：跨 runtime 对比沿用既有 caveat。

产出：G_adapt_curve.png（3 行序列族 × 3 列指标；每格三臂曲线，
线=两 seed 各自的 8-stage 平均曲线再取均值，带=seed 极差）。
"""
import json, os, sys
import xml.etree.ElementTree as ET
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
S8F_GLOB = os.path.join(ROOT, 'data/output_data/tsc/sumo_dqn')
ARMC = {'CONT': 'tab:red', 'DHOA': 'tab:green', 'ANCHOR': 'tab:blue'}
GRID = np.arange(10, 3601, 10)  # 360 decision steps
METRICS = [('tt', 'Avg Travel Time of arrived vehs (s)'),
           ('queue', 'Queue length (veh)'),
           ('thr', 'Cumulative Throughput (veh)')]


def net_key(net):
    return net if net.startswith('sumohz1x1') else 'sumohz1x1_' + net


def trace_seq(phys_dir):
    """CONT/DHOA 单格：decisions.jsonl 逐步 queue/thr + tripinfo 重建 TT(t)"""
    dj = os.path.join(phys_dir, 'decisions.jsonl')
    tj = os.path.join(phys_dir, 'tripinfo.xml')
    if not os.path.exists(dj) or not os.path.exists(tj):
        return None
    times, queue, thr = [], [], []
    for l in open(dj):
        r = json.loads(l)
        if r.get('record_type') != 'DECISION_METRICS':
            continue
        times.append(r['simulation_time_seconds'])
        queue.append(r['queue'])
        thr.append(r['throughput_cumulative'])
    # tripinfo：按到达时刻排序，累计平均 duration → 在决策时刻采样（阶梯）
    arr = []
    for obj in ET.parse(tj).getroot().iter('tripinfo'):
        arr.append((float(obj.get('arrival')), float(obj.get('duration'))))
    arr.sort()
    ats = np.array([a for a, _ in arr])
    csum = np.cumsum([d for _, d in arr])
    tt = []
    for t in times:
        n = int(np.searchsorted(ats, t, side='right'))
        tt.append(csum[n - 1] / n if n else np.nan)
    return {'t': np.array(times), 'tt': np.array(tt), 'queue': np.array(queue),
            'thr': np.array(thr)}


def trace_anchor(diag_dir):
    sf = os.path.join(diag_dir, 'evaluation/steps_ep0000.jsonl')
    if not os.path.exists(sf):
        return None
    times, tt, queue, thr = [], [], [], []
    for l in open(sf):
        r = json.loads(l)
        times.append(r['sim_time']); tt.append(r['travel_time'])
        queue.append(r['queue']); thr.append(r['throughput'])
    return {'t': np.array(times), 'tt': np.array(tt), 'queue': np.array(queue),
            'thr': np.array(thr)}


def harvest():
    """tr[arm][fam][sd] = list of 8 stage traces（每条 = 一个对角线格）"""
    tr = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # CONT / DHOA
    for fam, bd in BATCHES.items():
        shared = os.path.join(ROOT, bd, '_shared_evaluation')
        for arm in ['CONT', 'SELF-DHOA-R50']:
            for sd in ['SD0', 'SD1']:
                ev = os.path.join(ROOT, bd, arm + '-' + fam + '-' + sd,
                                  'attempts/attempt_1/events.jsonl')
                if not os.path.exists(ev):
                    print('missing', ev); continue
                want = {'evaluation:stage_%d:local_100:%s'
                        % (k, net_key(ORDERS[fam][k - 1])): k
                        for k in range(1, 9)}
                found = 0
                for l in open(ev):
                    e = json.loads(l)
                    k = e.get('payload', {}).get('operation_key', '')
                    if k not in want:
                        continue
                    a = json.load(open(e['payload']['artifact']))
                    phys = os.path.join(shared, 'physical',
                                        a['physical_key'], 'attempt_1')
                    t = trace_seq(phys)
                    if t is not None:
                        tr['DHOA' if arm == 'SELF-DHOA-R50' else 'CONT'
                           ][fam][sd].append((want[k], t))
                        found += 1
                print(f'{arm}-{fam}-{sd}: diag cells={found}')
    # ANCHOR s8fs 补跑批
    import glob
    for fam in ORDERS:
        for sd in ['SD0', 'SD1']:
            n = 0
            for k in range(1, 9):
                for d in glob.glob(os.path.join(
                        S8F_GLOB, '*', f's8fs_{fam}_{sd.lower()}_st{k}diag')):
                    t = trace_anchor(d)
                    if t is not None:
                        tr['ANCHOR'][fam][sd].append((k, t)); n += 1
            print(f'ANCHOR-{fam}-{sd}: diag cells={n}')
    return tr


def main(out_dir):
    tr = harvest()
    fig, axes = plt.subplots(3, 3, figsize=(19, 13), sharex=True)
    for ri, fam in enumerate(['N1', 'R1', 'D1']):
        for ci, (mk, mlab) in enumerate(METRICS):
            ax = axes[ri][ci]
            for ai, arm in enumerate(['CONT', 'DHOA', 'ANCHOR']):
                seed_curves = []
                for sd in ['SD0', 'SD1']:
                    curves = [t[mk] for _, t in sorted(tr[arm][fam][sd])]
                    curves = [c for c in curves if len(c) == len(GRID)]
                    if curves:
                        seed_curves.append(np.nanmean(
                            np.stack(curves), axis=0))
                if not seed_curves:
                    continue
                arr = np.stack(seed_curves)          # (nseed, 360)
                mu = np.nanmean(arr, axis=0)
                ax.plot(GRID, mu, lw=1.6, color=ARMC[arm], label=arm)
                ax.fill_between(GRID, np.nanmin(arr, axis=0),
                                np.nanmax(arr, axis=0),
                                color=ARMC[arm], alpha=0.18)
                ax.annotate(f'{arm} end={mu[-1]:.1f}',
                            xy=(GRID[-1], mu[-1]), xytext=(4, 8 - 8 * ai),
                            textcoords='offset points', fontsize=7,
                            color=ARMC[arm], va='center')
            ax.set_xlim(0, 3600)
            ax.grid(alpha=0.3)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=9)
            if ci == 0:
                ax.set_ylabel(f'{fam}\n{mlab}', fontsize=9)
            else:
                ax.set_ylabel(mlab)
            if ri == 2:
                ax.set_xlabel('simulation time (s)')
    fig.suptitle('SEQ8 current-scene adaptation preserved: within-sim '
                 'performance curves, avg over 8 stage-end ckpts '
                 '[line=mean(SD0,SD1 of stage-avg), band=seed min–max]',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fn = os.path.join(out_dir, 'G_adapt_curve.png')
    fig.savefig(fn, dpi=140); plt.close(fig); print('saved', fn)


if __name__ == '__main__':
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        ROOT, 'data/output_data/analysis/seq8_figs')
    os.makedirs(out, exist_ok=True)
    main(out)
