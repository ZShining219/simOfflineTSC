#!/usr/bin/env python3
"""Aggregate ATT-ENTITY-003 stage-1 gate-opening diagnostics.

Reads each diagnostic run's sga_branch.jsonl (per-update head norm / gradient
norms / event-row counts), event_plan_log.jsonl (per-episode schedule hash +
label) and optional counterfactual probe outputs, then writes:

  stage1_summary.csv    one row per run: final head_norm, event-row coverage,
                        grad-norm stats, probe deltas
  figures/*.png         head-norm trajectory, grad norms, event coverage
  stage1_analysis.md    facts + interpretation-ready summary

Usage:
    python3 tools/analyze_flx_stage1.py \
        --runs sga_flx_diag60_s7 sga_flx_diag60_s17 sga_flx_diag60_s27 \
        --run-root data/output_data/tsc/sumo_sga_flx_colight/hz4x4 \
        --out data/output_data/analysis/att_entity_003/stage1
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def branch_stats(run_dir):
    """Per-update branch diagnostics from sga_branch.jsonl."""
    path = Path(run_dir) / 'sga_branch.jsonl'
    if not path.exists():
        return None
    rows = load_jsonl(path)
    head_norm = np.array([r.get('head_norm', np.nan) for r in rows])
    head_grad = np.array([r.get('head_grad_norm') if r.get('head_grad_norm')
                          is not None else np.nan for r in rows])
    inner_grad = np.array([r.get('inner_grad_norm') if r.get('inner_grad_norm')
                           is not None else np.nan for r in rows])
    event_rows = np.array([r.get('event_rows', 0) for r in rows])
    batch = np.array([r.get('batch', np.nan) for r in rows])
    has_event = event_rows > 0
    return {
        'updates': len(rows),
        'update_index': np.arange(len(rows)),
        'head_norm': head_norm,
        'head_grad': head_grad,
        'inner_grad': inner_grad,
        'event_rows': event_rows,
        'batch': batch,
        'frac_event_updates': float(has_event.mean()) if len(rows) else 0.0,
        'frac_head_grad_nonzero': float(np.mean(head_grad > 0)) if len(rows) else 0.0,
        'frac_inner_grad_nonzero': float(np.mean(inner_grad > 0)) if len(rows) else 0.0,
        'head_norm_final': float(head_norm[-1]) if len(rows) else np.nan,
        'head_norm_max': float(np.nanmax(head_norm)) if len(rows) else np.nan,
    }


def plan_log_stats(run_dir):
    """Per-episode plan bindings from event_plan_log.jsonl."""
    path = Path(run_dir) / 'event_plan_log.jsonl'
    if not path.exists():
        return None
    rows = load_jsonl(path)
    labels = {}
    hashes = set()
    for r in rows:
        labels[r.get('label', '?')] = labels.get(r.get('label', '?'), 0) + 1
        if 'schedule_sha256' in r:
            hashes.add(r['schedule_sha256'])
    return {'episodes': len(rows), 'labels': labels,
            'distinct_schedules': len(hashes)}


VARIANTS = ('correct', 'wrong_location', 'zero_g')


def probe_deltas(probe_dir):
    """Aggregate counterfactual probe records -> per (run, ep, variant) deltas.

    Each record embeds every variant's Q/action for the same decision, so a
    single `correct`-driven probe suffices for fixed-state comparisons.
    Returns {(run, ep): {variant: (q_l1_all, act_diff_all,
                                    q_l1_event, act_diff_event)}}.
    """
    out = {}
    root = Path(probe_dir)
    if not root.exists():
        return out
    for rec in sorted(root.glob('*_correct/counterfactual_records.jsonl')):
        run = rec.parent.name.rsplit('_ep', 1)[0]
        ep = int(rec.parent.name.rsplit('_ep', 1)[1].split('_')[0])
        q_l1 = {v: [] for v in VARIANTS}
        act_diff = {v: [] for v in VARIANTS}
        q_l1_ev = {v: [] for v in VARIANTS}
        act_diff_ev = {v: [] for v in VARIANTS}
        for r in load_jsonl(rec):
            vs = r.get('variants', {})
            if 'correct' not in vs:
                continue
            qc = np.asarray(vs['correct']['q'])
            ac = np.asarray(vs['correct']['action'])
            has_event = r.get('event_count', 0) > 0
            for v in VARIANTS:
                if v not in vs:
                    continue
                qv = np.asarray(vs[v]['q'])
                av = np.asarray(vs[v]['action'])
                q_l1[v].append(np.abs(qv - qc).mean())
                act_diff[v].append(float((av != ac).mean()))
                if has_event:
                    q_l1_ev[v].append(np.abs(qv - qc).mean())
                    act_diff_ev[v].append(float((av != ac).mean()))
        out[(run, ep)] = {v: (float(np.mean(q_l1[v])),
                              float(np.mean(act_diff[v])),
                              float(np.mean(q_l1_ev[v])) if q_l1_ev[v]
                              else float('nan'),
                              float(np.mean(act_diff_ev[v]))
                              if act_diff_ev[v] else float('nan'))
                          for v in VARIANTS}
    return out


def plot_probes(deltas, out_dir):
    if not deltas:
        return
    fig_dir = Path(out_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs = sorted({k[0] for k in deltas})
    eps = sorted({k[1] for k in deltas})
    variants = [v for v in VARIANTS if v != 'correct']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    for run in runs:
        for v in variants:
            ys_q, ys_a = [], []
            for ep in eps:
                d = deltas.get((run, ep), {}).get(v, (np.nan,) * 4)
                ys_q.append(d[2]); ys_a.append(d[3])
            style = '-' if v == 'wrong_location' else '--'
            axes[0].plot(eps, ys_q, style, marker='o', ms=3,
                         label=f'{run}:{v}', lw=1)
            axes[1].plot(eps, ys_a, style, marker='o', ms=3,
                         label=f'{run}:{v}', lw=1)
    axes[0].set_title('counterfactual |Q variant - Q correct| (event steps)')
    axes[0].set_xlabel('checkpoint episode'); axes[0].set_ylabel('mean L1')
    axes[1].set_title('action-change rate vs correct (event steps)')
    axes[1].set_xlabel('checkpoint episode')
    axes[1].set_ylabel('fraction of nodes w/ different argmax')
    for ax in axes:
        ax.legend(fontsize=6); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'counterfactual_deltas.png', dpi=140)
    plt.close(fig)


def plot_branch(branch_data, out_dir):
    fig_dir = Path(out_dir) / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for tag, st in branch_data.items():
        x = st['update_index']
        axes[0].plot(x, st['head_norm'], label=tag, lw=1)
        axes[1].semilogy(x, np.where(st['head_grad'] > 0,
                                     st['head_grad'], np.nan),
                         label=tag, lw=0.6, alpha=0.8)
        axes[2].semilogy(x, np.where(st['inner_grad'] > 0,
                                     st['inner_grad'], np.nan),
                         label=tag, lw=0.6, alpha=0.8)
    axes[0].set_title('branch head norm (zero-init output layer)')
    axes[0].set_xlabel('update'); axes[0].set_ylabel('||W_out||_F')
    axes[1].set_title('head grad norm (event batches only)')
    axes[1].set_xlabel('update'); axes[1].set_ylabel('||grad W_out||')
    axes[2].set_title('inner-branch grad norm')
    axes[2].set_xlabel('update'); axes[2].set_ylabel('||grad inner||')
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'branch_opening.png', dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    tags = list(branch_data)
    cov = [branch_data[t]['frac_event_updates'] for t in tags]
    ax.bar(tags, cov)
    ax.set_ylabel('fraction of updates with event rows')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / 'event_coverage.png', dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', required=True)
    parser.add_argument('--run-root',
                        default='data/output_data/tsc/sumo_sga_flx_colight/hz4x4')
    parser.add_argument('--probe-dir', default=None)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_data, plan_data, summary = {}, {}, []
    for tag in args.runs:
        run_dir = Path(args.run_root) / tag
        b = branch_stats(run_dir)
        p = plan_log_stats(run_dir)
        if b:
            branch_data[tag] = b
        if p:
            plan_data[tag] = p
        if b:
            summary.append({
                'run': tag,
                'updates': b['updates'],
                'head_norm_final': round(b['head_norm_final'], 4),
                'head_norm_max': round(b['head_norm_max'], 4),
                'frac_event_updates': round(b['frac_event_updates'], 3),
                'frac_head_grad_nonzero': round(b['frac_head_grad_nonzero'], 3),
                'frac_inner_grad_nonzero': round(b['frac_inner_grad_nonzero'], 3),
                'plan_episodes': p['episodes'] if p else '',
                'plan_labels': json.dumps(p['labels']) if p else '',
            })

    if branch_data:
        plot_branch(branch_data, out_dir)

    csv_path = out_dir / 'stage1_summary.csv'
    if summary:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)

    deltas = probe_deltas(args.probe_dir) if args.probe_dir else {}
    if deltas:
        plot_probes(deltas, out_dir)
        with open(out_dir / 'counterfactual_deltas.csv', 'w',
                  newline='') as f:
            w = csv.writer(f)
            w.writerow(['run', 'checkpoint_ep', 'variant', 'q_l1_all',
                        'act_diff_all', 'q_l1_event', 'act_diff_event'])
            for (run, ep), by_var in sorted(deltas.items()):
                for v, vals in by_var.items():
                    w.writerow([run, ep, v] + [round(x, 6) for x in vals])

    (out_dir / 'stage1_summary.json').write_text(json.dumps({
        'runs': summary,
        'counterfactual_deltas': {
            f'{run}|ep{ep}|{v}': vals
            for (run, ep), by_var in deltas.items()
            for v, vals in by_var.items()},
    }, indent=2))

    lines = ['# ATT-ENTITY-003 阶段一开门诊断', '']
    for row in summary:
        lines.append(f"- {row['run']}: head_norm {row['head_norm_final']} "
                     f"(max {row['head_norm_max']}), event-updates "
                     f"{row['frac_event_updates']}, head-grad>0 "
                     f"{row['frac_head_grad_nonzero']}, inner-grad>0 "
                     f"{row['frac_inner_grad_nonzero']}")
    if deltas:
        lines += ['', '## counterfactual deltas (event steps)',
                  'run | ep | variant | q_l1 | act_diff', '---|---|---|---|---']
        for (run, ep), by_var in sorted(deltas.items()):
            for v, vals in by_var.items():
                if v == 'correct':
                    continue
                lines.append(f'{run} | {ep} | {v} | {vals[2]:.4f} | '
                             f'{vals[3]:.4f}')
    (out_dir / 'stage1_analysis.md').write_text('\n'.join(lines) + '\n')
    print(f'wrote {csv_path} + figures under {out_dir}')


if __name__ == '__main__':
    main()
