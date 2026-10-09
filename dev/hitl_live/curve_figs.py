#!/usr/bin/env python
"""curve_figs.py — HITL C-channel pilot learning-effect figures.

Reads a 4-arm pilot output root produced by build_pilot_plan.py +
sequential_run.py run-child:

    <root>/runs/<LOGICAL_RUN_ID>/attempt_001/           (per-arm attempt dirs)
    <root>/_shared_evaluation/physical/*/attempt_*/     (autonomous evals)

Each evaluation attempt's request.json carries `snapshot_path` (contains the
logical_run_id) and `checkpoint_episode`; the committed summary.json carries
the frozen-autonomous metrics.  Figures:

    fig1_learning_curves.png  travel_time vs actual decision count, per arm,
                              teacher (MaxPressure) reference band
    fig2_aulc_final.png       AULC over decisions + final-episode metrics
    fig3_delta_vs_c0.png      per-episode delta vs C0 baseline
                              (isolates expert data from "more data")
    fig4_expert_leg.png       expert leg consumption: counts, unique usage,
                              loss_expert trajectory

Usage:
    python dev/hitl_live/curve_figs.py --root output_data/hitl_probe/pilot \
        --teacher-tt 83.72 --out output_data/hitl_probe/pilot/figs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METRICS = ('travel_time', 'throughput', 'unfinished_vehicles')
ARM_STYLE = {
    'HITL_C0': dict(color='#7f7f7f', label='C0 baseline (online only)'),
    'HITL_EXPERT': dict(color='#d62728', label='C1 expert pool (HITL)'),
    'HITL_CTRL': dict(color='#1f77b4', label='CTRL equal machine data'),
    'HITL_DEMO': dict(color='#9467bd', label='DEMO pure expert pool'),
    'HITL_B1': dict(color='#2ca02c', label='B1 rule-teacher shaping'),
    'HITL_B2': dict(color='#ff7f0e', label='B2 reward-model shaping'),
    'HITL_A0': dict(color='#8c564b', label='A0 DemoAC no pretrain'),
    'HITL_API': dict(color='#e377c2', label='Aπ actor pretrain'),
    'HITL_AQ': dict(color='#17becf', label='AQ critic pretrain'),
    'HITL_APIQ': dict(color='#bcbd22', label='AπQ actor+critic pretrain'),
}


def _seed_of(run_id):
    for part in str(run_id).split('-'):
        if part.startswith('SD'):
            return int(part[2:])
    return None


def _arm_of(request):
    snapshot = request.get('snapshot_path', '')
    for part in Path(snapshot).parts:
        if part.startswith('HITL_'):
            return part
    return None


def _condition_of(plan, logical_run_id):
    for child in plan.get('children', []):
        if child.get('logical_run_id') == logical_run_id:
            return child.get('condition', logical_run_id.rsplit('-', 1)[0])
    return logical_run_id.rsplit('-', 1)[0]


def collect(root, plan_path=None):
    """Per-run rows: (arm/condition, seed, episode) -> eval summary metrics.

    Snapshot-dedup aliasing: while weights are still identical to the ep0
    checkpoint the evaluator dedups the shared snapshot to the first
    requesting run, so other runs legitimately lack early episodes.  Fill
    leading episodes 1..min-1 with the earliest evaluated value for the
    same seed (same frozen snapshot), matching paired_bootstrap.py."""
    if plan_path is None:
        plan_path = os.path.join(root, 'plan_4arm.json')
    plan = json.load(open(plan_path)) if os.path.isfile(plan_path) else {}
    per_run = {}
    pattern = os.path.join(
        root, '_shared_evaluation', 'physical', '*', 'attempt_*')
    for attempt_dir in glob.glob(pattern):
        req_path = os.path.join(attempt_dir, 'request.json')
        sum_path = os.path.join(attempt_dir, 'summary.json')
        if not (os.path.isfile(req_path) and os.path.isfile(sum_path)):
            continue
        request = json.load(open(req_path))
        summary = json.load(open(sum_path))
        arm = _arm_of(request)
        if arm is None:
            continue
        episode = int(request.get('checkpoint_episode', -1))
        per_run.setdefault(arm, {})[episode] = summary
    # earliest evaluated value per seed (the ep0-identical snapshot)
    seed_early = {}
    for arm, pts in per_run.items():
        seed = _seed_of(arm)
        early = min(pts)
        if seed not in seed_early or early < seed_early[seed][0]:
            seed_early[seed] = (early, pts[early])
    rows = []
    for arm, pts in per_run.items():
        seed = _seed_of(arm)
        if seed in seed_early:
            early_val = seed_early[seed][1]
            for ep in range(1, min(pts)):
                pts.setdefault(ep, early_val)
        # interior same-snapshot aliases (e.g. ep2 == ep1 weights before
        # learning_start) — forward-fill with last evaluated value
        for ep in range(min(pts) + 1, max(pts) + 1):
            if ep not in pts:
                prev = max(e for e in pts if e < ep)
                pts[ep] = pts[prev]
        for ep, summary in pts.items():
            rows.append({
                'arm': arm,
                'condition': _condition_of(plan, arm),
                'seed': seed,
                'episode': ep,
                'metrics': summary,
            })
    return rows


def _curve(rows, condition, metric):
    """Per-episode (mean, std) across runs of the same condition."""
    by_ep = {}
    for r in rows:
        if r['condition'] == condition and metric in r['metrics']:
            by_ep.setdefault(r['episode'], []).append(
                float(r['metrics'][metric]))
    if not by_ep:
        return np.array([]), np.array([]), np.array([])
    xs = sorted(by_ep)
    mean = np.asarray([np.mean(by_ep[e]) for e in xs])
    std = np.asarray([np.std(by_ep[e]) for e in xs])
    return np.asarray(xs), mean, std


def _load_diagnostics(root, arm):
    pattern = os.path.join(
        root, 'runs', arm, 'attempt_001', 'replay_diagnostics',
        'stage_*_episode_*.json')
    out = []
    for path in sorted(glob.glob(pattern)):
        with open(path) as fh:
            out.append(json.load(fh))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, help='pilot output root')
    parser.add_argument('--plan', default=None,
                        help='plan json for run_id -> condition mapping')
    parser.add_argument('--teacher-tt', type=float, default=None,
                        help='MaxPressure-alone travel time reference')
    parser.add_argument('--decisions-per-episode', type=int, default=120)
    parser.add_argument('--out', required=True)
    args = parser.parse_args(argv)

    rows = collect(args.root, args.plan)
    if not rows:
        raise SystemExit(f'no evaluations found under {args.root}')
    conditions = [c for c in ARM_STYLE if any(r['condition'] == c for r in rows)]
    for r in rows:  # fallback for unknown condition names
        if r['condition'] not in ARM_STYLE:
            ARM_STYLE[r['condition']] = dict(
                color='#2ca02c', label=r['condition'])
            if r['condition'] not in conditions:
                conditions.append(r['condition'])
    os.makedirs(args.out, exist_ok=True)
    dpe = args.decisions_per_episode

    # fig1: learning curves (travel_time vs decisions)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    for cond in conditions:
        x, y, s = _curve(rows, cond, 'travel_time')
        if len(x):
            ax.plot(x * dpe, y, marker='o', ms=3,
                    color=ARM_STYLE[cond]['color'],
                    label=ARM_STYLE[cond]['label'])
            ax.fill_between(x * dpe, y - s, y + s,
                            color=ARM_STYLE[cond]['color'], alpha=0.15)
    if args.teacher_tt is not None:
        ax.axhline(args.teacher_tt, color='#d62728', ls='--', lw=1, alpha=0.6)
        ax.text(0.01, args.teacher_tt, f'MP expert alone ({args.teacher_tt:.1f}s)',
                transform=ax.get_yaxis_transform(), fontsize=8, va='bottom')
    ax.set_xlabel('Training decisions (control steps)')
    ax.set_ylabel('Autonomous travel time (s)')
    ax.set_title('Frozen autonomous eval after each episode')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for cond in conditions:
        x, y, s = _curve(rows, cond, 'unfinished_vehicles')
        if len(x):
            ax.plot(x * dpe, y, marker='o', ms=3,
                    color=ARM_STYLE[cond]['color'], label=cond)
            ax.fill_between(x * dpe, y - s, y + s,
                            color=ARM_STYLE[cond]['color'], alpha=0.15)
    ax.set_xlabel('Training decisions (control steps)')
    ax.set_ylabel('Unfinished vehicles')
    ax.set_title('Unfinished vehicles vs decisions')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig1_learning_curves.png'), dpi=150)

    # fig2: AULC + final metrics
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    aulc = {}
    for cond in conditions:
        x, y, _ = _curve(rows, cond, 'travel_time')
        if len(x) > 1:
            trapz = getattr(np, 'trapezoid', None) or np.trapz
            aulc[cond] = float(trapz(y, x * dpe) / ((x[-1] - x[0]) * dpe))
    ax = axes[0]
    labels = list(aulc)
    ax.bar(range(len(labels)), [aulc[c] for c in labels],
           color=[ARM_STYLE[c]['color'] for c in labels])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Mean travel time over training (AULC)')
    ax.set_title('AULC (lower = learned faster)')
    ax.grid(alpha=0.3, axis='y')
    for i, m in enumerate(METRICS):
        ax = axes[i + 1]
        vals = {}
        for cond in conditions:
            x, y, _ = _curve(rows, cond, m)
            if len(x):
                vals[cond] = y[-1]
        ax.bar(range(len(vals)), list(vals.values()),
               color=[ARM_STYLE[c]['color'] for c in vals])
        ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(list(vals), rotation=25, ha='right', fontsize=8)
        ax.set_title(f'Final {m}')
        ax.grid(alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig2_aulc_final.png'), dpi=150)

    # fig3: seed-paired delta vs C0 (per-seed difference, then mean±std)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for i, m in enumerate(METRICS):
        ax = axes[i]
        c0_by_seed = {}
        for r in rows:
            if r['condition'] == 'HITL_C0' and m in r['metrics']:
                c0_by_seed.setdefault(r['seed'], {})[r['episode']] = float(
                    r['metrics'][m])
        for cond in conditions:
            if cond == 'HITL_C0':
                continue
            paired = {}
            for r in rows:
                if (r['condition'] != cond or m not in r['metrics']
                        or r['seed'] not in c0_by_seed):
                    continue
                base = c0_by_seed[r['seed']].get(r['episode'])
                if base is None:
                    continue
                paired.setdefault(r['episode'], []).append(
                    float(r['metrics'][m]) - base)
            if paired:
                px = sorted(paired)
                pm = np.asarray([np.mean(paired[e]) for e in px])
                ps = np.asarray([np.std(paired[e]) for e in px])
                ax.plot(np.asarray(px) * dpe, pm, marker='o', ms=3,
                        color=ARM_STYLE[cond]['color'],
                        label=ARM_STYLE[cond]['label'])
                ax.fill_between(np.asarray(px) * dpe, pm - ps, pm + ps,
                                color=ARM_STYLE[cond]['color'], alpha=0.15)
        ax.axhline(0, color='#7f7f7f', lw=1)
        ax.set_xlabel('Training decisions')
        ax.set_ylabel(f'delta {m} vs C0')
        ax.set_title(f'{m}: improvement over baseline (negative=better)'
                     if m != 'throughput' else
                     f'{m}: improvement over baseline (positive=better)')
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig3_delta_vs_c0.png'), dpi=150)

    # fig4: expert leg consumption (HITL_EXPERT arm diagnostics)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for arm in ('HITL_EXP-SD0', 'HITL_DEMO-SD0', 'HITL_CTRL-SD0'):
        diags = _load_diagnostics(args.root, arm)
        if not diags:
            continue
        ep = [d['global_episode'] for d in diags]
        exp_used = [d.get('expert_sampler_sample_count', 0) for d in diags]
        uniq = [d.get('expert_unique_samples_used', 0) for d in diags]
        axes[0].plot(ep, exp_used, marker='o', ms=3,
                     label=f'{arm} consumed')
        axes[1].plot(ep, uniq, marker='o', ms=3, label=arm)
        losses = []
        updates = []
        for d in diags:
            for w in d.get('sampling_windows', []):
                updates.append(w['gradient_updates'])
                losses.append((w.get('loss_expert'), w.get('loss_online')))
        if updates:
            lx = [a for a, _ in losses]
            lo = [b for _, b in losses]
            axes[2].plot(updates, lx, label=f'{arm} loss_expert', alpha=0.8)
            axes[2].plot(updates, lo, '--', label=f'{arm} loss_online',
                         alpha=0.6)
    axes[0].set_xlabel('global episode')
    axes[0].set_ylabel('expert samples consumed')
    axes[0].set_title('Expert transitions drawn (cumulative)')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel('global episode')
    axes[1].set_ylabel('unique expert transitions')
    axes[1].set_title('Pool coverage (of ~243 packaged)')
    axes[1].grid(alpha=0.3)
    axes[2].set_xlabel('gradient update #')
    axes[2].set_ylabel('TD loss')
    axes[2].set_title('Branch losses (expert vs online)')
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, 'fig4_expert_leg.png'), dpi=150)

    summary = {
        'conditions': conditions,
        'points_per_condition': {
            c: sum(1 for r in rows if r['condition'] == c) for c in conditions
        },
        'aulc_travel_time': aulc,
        'final': {
            c: {m: _curve(rows, c, m)[1][-1] for m in METRICS
                if len(_curve(rows, c, m)[1])}
            for c in conditions
        },
    }
    with open(os.path.join(args.out, 'learning_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == '__main__':
    main()
