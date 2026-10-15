"""量化证据四联图：从已收割的 results.json / attn_probe.json 汇总
sig15 系列的模块有效性证据，不再跑仿真。

面板:
  A E1 主表:   各臂 事件边/走廊腿 排队积分（corridor-load 网格）
  B E5 契约:   sga15 vs sga16 的注意力-报告对齐与文本敏感度
  C E7 迁移:   各场景"文本通道收益%"（零样本 vs 在训, corr_queue_int）
  D E2 选型:   文本注入机制对比（事件边排队）

用法::

    python -m scene_attention.evidence_fig \
        --root data/output_data/dev_takeover_probe \
        --out data/output_data/dev_takeover_probe/evidence_fig/evidence.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def load_eval(root: Path, name: str) -> dict:
    return json.loads((root / name / 'results.json').read_text())['summary']


def load_probe(root: Path, name: str) -> dict:
    return json.loads((root / name / 'attn_probe.json').read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='data/output_data/dev_takeover_probe')
    p.add_argument('--out', required=True)
    args = p.parse_args()
    root = Path(args.root)

    e1 = load_eval(root, 'sig15_eval')
    e2 = load_eval(root, 'sig15_eval_modcmp')
    probe = load_probe(root, 'attn_probe_c2')
    zs = {s: load_eval(root, f'sig15_eval_{s}') for s in ('S1', 'S3', 'S4')}
    ins = {s: load_eval(root, f'sig15_{s}_eval') for s in ('S1', 'S3', 'S4')}

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # ---------- A: E1 main table ----------
    ax = axes[0][0]
    order = ['joint_sga15', 'joint_notext', 'joint_lie', 'joint_base',
             'maxpressure', 'fixedtime']
    labels = ['sga15\n(report)', 'same model\nno text', 'same model\nlie',
              'base path\nonly', 'max-\npressure', 'fixed\ntime']
    ev = [e1[a]['ev_queue_int'] / 1000 for a in order]
    cq = [e1[a]['corr_queue_int'] / 1000 for a in order]
    x = np.arange(len(order))
    b1 = ax.bar(x - 0.2, ev, 0.38, color='#d62728', label='reported edge queue')
    b2 = ax.bar(x + 0.2, cq, 0.38, color='#1f77b4',
                label='corridor legs queue')
    for b in list(b1) + list(b2):
        ax.annotate(f'{b.get_height():.0f}', (b.get_x() + b.get_width() / 2,
                    b.get_height()), ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x, labels, fontsize=8)
    ax.set_ylabel('queue integral (x1000 veh·s)')
    ax.set_title('A. Corridor governance (E1): text arm is best on the\n'
                 'corridor the report names  [S2, 12 cases]', fontsize=9)
    ax.legend(fontsize=8)
    ax.annotate('report vs no-text:\ncorridor -8.6%', (0.5, 0.97),
                xycoords='axes fraction', ha='center', va='top',
                fontsize=8, color='#1f77b4')

    # ---------- B: E5 contract ----------
    ax = axes[0][1]
    metrics = [('align_true', 'attention\naligns w/\ntrue corridor'),
               ('align_lie_on_liewatch', 'attention\naligns w/\nLIED corridor'),
               ('w_l1_true_lie', 'attention\nsensitivity\nto text swap'),
               ('flip_none_rate', 'action flip\nrate when\ntext off')]
    xv = np.arange(len(metrics))
    for i, (model, color, lab) in enumerate(
            [('sga15', '#2ca02c', 'sga15 (watch-supervised)'),
             ('sga16', '#9467bd', 'sga16 (aux=0)')]):
        vals = [probe[model]['summary'][k] for k, _ in metrics]
        bars = ax.bar(xv + (i - 0.5) * 0.36, vals, 0.34, color=color,
                      label=lab)
        for b in bars:
            ax.annotate(f'{b.get_height():.2f}',
                        (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha='center', va='bottom', fontsize=7)
    ax.set_xticks(xv, [l for _, l in metrics], fontsize=8)
    ax.set_ylim(0, 0.62)
    ax.set_title('B. Response contract (E5): aux supervision makes\n'
                 'attention follow the REPORT, not just the physics', fontsize=9)
    ax.legend(fontsize=8)

    # ---------- C: E7 transfer ----------
    ax = axes[1][0]
    scenes = ['S2', 'S1', 'S3', 'S4']

    def benefit(d, a_true, a_nt):
        return 100 * (d[a_nt]['corr_queue_int']
                      - d[a_true]['corr_queue_int']) / d[a_nt]['corr_queue_int']

    zs_v = [benefit(e1, 'joint_sga15', 'joint_notext')]
    zs_v += [benefit(zs[s], 'joint_sga15', 'joint_notext') for s in scenes[1:]]
    tr_v = [zs_v[0]]
    tr_v += [benefit(ins[s], f'joint_sga_{s}', 'joint_notext')
             for s in scenes[1:]]
    x = np.arange(len(scenes))
    ax.bar(x - 0.2, zs_v, 0.38, color='#17becf',
           label='zero-shot (S2-trained)')
    ax.bar(x + 0.2, tr_v, 0.38, color='#2ca02c',
           label='trained in-scene')
    for xi, (a, b) in enumerate(zip(zs_v, tr_v)):
        ax.annotate(f'{a:.1f}', (xi - 0.2, a), ha='center', va='bottom'
                    if a >= 0 else 'top', fontsize=8)
        if xi > 0:
            ax.annotate(f'{b:.1f}', (xi + 0.2, b), ha='center',
                        va='bottom' if b >= 0 else 'top', fontsize=8)
    ax.axhline(0, color='k', lw=0.8)
    ax.annotate('S2 = training scene\n(same bar, in-domain)', (0, -4),
                ha='center', va='top', fontsize=7, color='gray')
    ax.set_xticks(x, scenes)
    ax.set_ylabel('text-channel benefit on corridor queue (%)')
    ax.set_title('C. Cross-network transfer (E7): report benefit survives\n'
                 'on unseen topologies; in-scene training widens it', fontsize=9)
    ax.legend(fontsize=8)

    # ---------- D: E2 mechanism selection ----------
    ax = axes[1][1]
    order2 = ['joint_sga15', 'sig12_single', 'sga_bare', 'selfattn_st',
              'concat_st', 'fixedtime']
    labs2 = ['scene attn\n+ intent\n(sga15)', 'scene attn\nsingle-edge\n(sig12)',
             'bare attn\nscorer', 'self-attn\n(no bind)', 'concat\ninto state',
             'fixed\ntime']
    keys2 = {'joint_sga15': 'joint_sga15', 'sig12_single': 'joint_sga12',
             'sga_bare': 'sga_st', 'selfattn_st': 'selfattn_st',
             'concat_st': 'concat_st', 'fixedtime': 'fixedtime'}
    avail = [k for k in order2 if keys2[k] in e2]
    xv = np.arange(len(avail))
    evv = [e2[keys2[k]]['ev_queue_int'] / 1000 for k in avail]
    cov = [e2[keys2[k]]['corr_queue_int'] / 1000 for k in avail]
    ax.bar(xv - 0.2, evv, 0.38, color='#d62728')
    ax.bar(xv + 0.2, cov, 0.38, color='#1f77b4')
    for xi, (a, b) in enumerate(zip(evv, cov)):
        ax.annotate(f'{a:.0f}', (xi - 0.2, a), ha='center', va='bottom',
                    fontsize=7)
        ax.annotate(f'{b:.0f}', (xi + 0.2, b), ha='center', va='bottom',
                    fontsize=7)
    ax.set_xticks(xv, [labs2[order2.index(k)] for k in avail], fontsize=8)
    ax.set_ylabel('queue integral (x1000 veh·s)')
    ax.set_title('D. Mechanism selection (E2): naive text injection\n'
                 'collapses under corridor load; internalized attn wins',
                 fontsize=9)
    fig.suptitle('Scene-guided attention: quantified evidence (corridor-load '
                 'scenario, 12 cases/grid)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
