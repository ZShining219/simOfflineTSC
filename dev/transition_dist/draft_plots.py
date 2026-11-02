#!/usr/bin/env python3
"""Draft: dataset-level transition characterization plots (hz1x1 family).

Three figures over trajectory NPZ assets (schema A, training trajectories):
  fig1_state_dist.png   P(total queue at decision point) per dataset
  fig2_cond_action.png  P(action | queue bin, dataset)
  fig3_deltaq.png       P(Delta q | queue bin, action, dataset)

Queue semantics: npz 'queue' field = instantaneous total lane_waiting_count
over 8 incoming lanes, recorded post-step at decision i -> it is the queue
observed at decision point i+1. So per transition i>=1:
    q_pre = queue[i-1], action = action[i], Delta q = queue[i] - queue[i-1]

Datasets are (demand scene, run) pairs; all runs are hz1x1 / 8-phase.
Usage: python dev/transition_dist/draft_plots.py
"""
import csv
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TSC = os.path.join(ROOT, 'data/output_data/tsc/sumo_dqn')
OUT = os.path.join(ROOT, 'dev/transition_dist/figs')

# label -> (glob pattern for run dir, episode stride)
DATASETS = [
    ('base',       'sumohz1x1/p1_formal_dqn_sumohz1x1_seed0_400ep_20260722_r2',        4),
    ('cfg2',       'sumohz1x1_config2/p1_formal_dqn_sumohz1x1_config2_seed0_400ep_20260722_r2', 4),
    ('cfg3',       'sumohz1x1_config3/p1_formal_dqn_sumohz1x1_config3_seed0_400ep_20260722_r2', 4),
    ('cfg4',       'sumohz1x1_config4/p1_formal_dqn_sumohz1x1_config4_seed0_400ep_20260722_r2', 4),
    ('dense_high', 'sumohz1x1_dense_high/s8f_soddqnp_R1_sd0_st8',            1),
    ('sparse_low', 'sumohz1x1_sparse_low/s8f_soddqnp_R1_sd0_st5',            1),
    ('asym_we',    'sumohz1x1_asym_we/s8f_soddqnp_D1_sd0_st7',               1),
    ('lt_high',    'sumohz1x1_lt_high/s8f_soddqnp_R1_sd0_st6',               1),
]

MIN_CELL_N = 50


def load_run(pattern, stride):
    runs = sorted(glob.glob(os.path.join(TSC, pattern)))
    if not runs:
        raise FileNotFoundError(pattern)
    run = runs[0]
    files = sorted(glob.glob(os.path.join(run, 'trajectory/episodes/*.npz')))[::stride]
    qs, qs_next, acts, eps, modes = [], [], [], [], []
    for f in files:
        with np.load(f, allow_pickle=False) as d:
            q = d['queue'].reshape(-1).astype(np.float64)
            a = d['action'].reshape(-1).astype(np.int64)
            qs.append(q[:-1])          # queue at decision points 1..359
            qs_next.append(q[1:])      # post-transition queue
            acts.append(a[1:])
            eps.append(float(np.mean(d['epsilon'])))
            modes.append(str(d['behavior_mode'][-1]))
    q_pre = np.concatenate(qs)
    q_post = np.concatenate(qs_next)
    return {
        'run': os.path.basename(run),
        'n_ep': len(files),
        'q_pre': q_pre,
        'dq': q_post - q_pre,
        'q_hist': np.concatenate([np.concatenate(qs), np.concatenate(qs_next)[-1:]]),
        'act': np.concatenate(acts),
        'eps_mean': float(np.mean(eps)),
        'mode': modes[-1] if modes else '?',
    }


def main():
    data = {label: load_run(p, s) for label, p, s in DATASETS}
    labels = [l for l, _, _ in DATASETS]
    colors = plt.get_cmap('tab10').colors

    pooled_q = np.concatenate([v['q_pre'] for v in data.values()])
    qs = np.quantile(pooled_q, [0.33, 0.66, 0.99, 0.995])
    print(f'pooled queue: mean={pooled_q.mean():.2f} p50={np.median(pooled_q):.0f} '
          f'p90={np.quantile(pooled_q,0.9):.0f} p99={qs[2]:.0f} max={pooled_q.max():.0f}')

    # fixed, interpretable congestion bins
    edges = [0, 5, 20, np.inf]
    bin_names = ['q<5', '5<=q<20', 'q>=20']

    print(f'{"dataset":12s} {"run":44s} {"n":>7s} {"eps":>4s} {"q_mean":>7s} '
          f'{"q_p90":>6s} {"q_max":>6s} {"eps_mean":>8s} bin_counts')
    summary_rows = []
    for l in labels:
        v = data[l]
        bc = [int(((v['q_pre'] >= lo) & (v['q_pre'] < hi)).sum())
              for lo, hi in zip(edges[:-1], edges[1:])]
        print(f'{l:12s} {v["run"]:44s} {len(v["q_pre"]):7d} {v["n_ep"]:4d} '
              f'{v["q_pre"].mean():7.2f} {np.quantile(v["q_pre"],0.9):6.0f} '
              f'{v["q_pre"].max():6.0f} {v["eps_mean"]:8.3f} {bc}')
        summary_rows.append({'dataset': l, 'run': v['run'], 'n_transitions': len(v['q_pre']),
                             'n_episodes': v['n_ep'], 'queue_mean': round(float(v['q_pre'].mean()), 3),
                             'queue_p90': round(float(np.quantile(v['q_pre'], 0.9)), 1),
                             'queue_max': int(v['q_pre'].max()), 'epsilon_mean': round(v['eps_mean'], 3),
                             'bin_counts': ';'.join(map(str, bc))})
    with open(os.path.join(OUT, 'datasets_summary.csv'), 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0])); w.writeheader(); w.writerows(summary_rows)

    # ---------------- fig 1: queue marginal ----------------
    xmax = float(np.quantile(pooled_q, 0.995))
    bins = np.arange(0, xmax + 2, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5), sharex=True)
    for ax, log in zip(axes, (False, True)):
        for i, l in enumerate(labels):
            ax.hist(data[l]['q_hist'], bins=bins, density=True, histtype='step',
                    lw=1.4, color=colors[i], label=f"{l} ({data[l]['run'][:28]})")
        if log:
            ax.set_yscale('log'); ax.set_ylim(bottom=1e-5)
        ax.set_xlabel('total queue at decision point (veh)')
        ax.set_ylabel('density')
        ax.set_title('linear' if not log else 'log-y')
    axes[0].legend(fontsize=7)
    fig.suptitle('Fig1  P(total queue) per dataset')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig1_state_dist.png'), dpi=150)
    plt.close(fig)

    # ---------------- fig 2: P(action | queue bin, dataset) ----------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
    for b, ax in enumerate(axes):
        lo, hi = edges[b], edges[b + 1]
        M = np.full((len(labels), 8), np.nan)
        ns = []
        for r, l in enumerate(labels):
            v = data[l]
            m = (v['q_pre'] >= lo) & (v['q_pre'] < hi)
            ns.append(int(m.sum()))
            if m.sum() < MIN_CELL_N:
                continue
            cnt = np.bincount(v['act'][m], minlength=8)
            M[r] = cnt / cnt.sum()
        im = ax.imshow(np.ma.masked_invalid(M), aspect='auto', vmin=0, vmax=1, cmap='viridis')
        for r in range(len(labels)):
            for c in range(8):
                if np.isnan(M[r, c]):
                    continue
                ax.text(c, r, f'{M[r, c] * 100:.0f}', ha='center', va='center',
                        fontsize=7, color='w' if M[r, c] < 0.55 else 'k')
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels([f'{l} (n={n})' for l, n in zip(labels, ns)], fontsize=8)
        ax.set_xticks(range(8)); ax.set_xlabel('action (phase id)')
        ax.set_title(f'bin {bin_names[b]}')
    fig.colorbar(im, ax=axes, label='P(action | bin, dataset)', shrink=0.85)
    fig.suptitle('Fig2  conditional action distribution')
    fig.tight_layout(); fig.savefig(os.path.join(OUT, 'fig2_cond_action.png'), dpi=150)
    plt.close(fig)

    # ---------------- fig 3: P(Delta q | bin, action, dataset) ----------------
    pooled_dq = np.concatenate([v['dq'] for v in data.values()])
    dx = float(np.quantile(np.abs(pooled_dq), 0.999))
    dx = min(dx, 60.0)
    dbins = np.linspace(-dx, dx, 61)
    fig, axes = plt.subplots(3, 8, figsize=(22, 8), sharex=True, sharey=True)
    for b in range(3):
        for a in range(8):
            ax = axes[b, a]
            for i, l in enumerate(labels):
                v = data[l]
                m = (v['q_pre'] >= edges[b]) & (v['q_pre'] < edges[b + 1]) & (v['act'] == a)
                if m.sum() < MIN_CELL_N:
                    continue
                ax.hist(v['dq'][m], bins=dbins, density=True, histtype='step',
                        lw=1.0, color=colors[i])
            ax.axvline(0, color='k', lw=0.5, alpha=0.4)
            if b == 0:
                ax.set_title(f'a={a}', fontsize=9)
            if a == 0:
                ax.set_ylabel(bin_names[b], fontsize=9)
            ax.tick_params(labelsize=7)
    handles = [plt.Line2D([], [], color=colors[i], label=l) for i, l in enumerate(labels)]
    fig.legend(handles=handles, loc='lower center', ncol=len(labels), fontsize=8)
    fig.suptitle('Fig3  P(Delta q | queue bin, action) per dataset '
                 f'(cells with n<{MIN_CELL_N} omitted; Delta q over 10s interval)')
    fig.supxlabel('Delta q = q_post - q_pre (veh)')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(OUT, 'fig3_deltaq.png'), dpi=150)
    plt.close(fig)

    print('figs written to', OUT)


if __name__ == '__main__':
    main()
