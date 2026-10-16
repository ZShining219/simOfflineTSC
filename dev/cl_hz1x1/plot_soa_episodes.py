"""Per-episode training-curve panels for the cl_hz1x1 anchor campaign.

Stage-level reports only show stage-end checkpoint evals; this script reads
each training run's metrics/records.jsonl (TRAIN records carry per-episode
travel_time / queue / reward_mean) and draws the 100-episode learning curve
inside every stage, replay-only vs +P-anchor overlaid per backbone.

Grid: rows = backbone (ddqn, duel, frap), cols = stage (st1..st4 on o2).
Two lines per panel: so1 replay-only (grey) and soa1 +P (orange), mean over
seeds 0-2 with min-max band.  Answers at episode resolution:
  * does +P slow new-scene learning (plasticity, finer than diagonal ΔATT)?
  * how large is the stage-switch shock on the first episodes of a stage?

Output: artifacts/cl_hz1x1/fig_soa_episodes.png

Run: python dev/cl_hz1x1/plot_soa_episodes.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'data' / 'output_data' / 'tsc'
FIG = ROOT / 'artifacts' / 'cl_hz1x1' / 'fig_soa_episodes.png'

# backbone -> (agent_dir, replay-arm run prefix, +P-arm run prefix)
BACKBONES = {
    'ddqn': ('sumo_dqn', 'so1_soddqn', 'soa1_soddqnp'),
    'duel': ('sumo_dqn', 'so1_soduel', 'soa1_soduelp'),
    'frap': ('sumo_frap', 'so1_sofrap', 'soa1_sofrapp'),
    'vdqn': ('sumo_dqn', 'soav_sodqn', 'soav_sodqnp'),
}
ORDER = ['sumohz1x1', 'sumohz1x1_config4',
         'sumohz1x1_config2', 'sumohz1x1_config3']
SCENE_TAG = {'sumohz1x1': 's1', 'sumohz1x1_config4': 'c4',
             'sumohz1x1_config2': 'c2', 'sumohz1x1_config3': 'c3'}
SEEDS = [0, 1, 2]
MAX_EP = 100


def train_curve(agent_dir, net, run_id):
    """(episode->train ATT, ep0 eval ATT) for one stage run."""
    path = OUT / agent_dir / net / run_id / 'metrics' / 'records.jsonl'
    if not path.exists():
        return {}, None
    curve, ep0_eval = {}, None
    for line in path.read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        tt = row.get('travel_time')
        if tt is None:
            continue
        if row.get('record_type') == 'TRAIN':
            curve[int(row['episode'])] = float(tt)
        elif row.get('record_type') == 'EVALUATION' \
                and int(row.get('episode', -1)) == 0:
            ep0_eval = float(tt)
    return curve, ep0_eval


def rolling(values, window=5):
    out = np.full_like(values, np.nan, dtype=float)
    valid = ~np.isnan(values)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo:i + 1]
        chunk = chunk[~np.isnan(chunk)]
        if valid[i] and len(chunk):
            out[i] = float(np.mean(chunk))
    return out


def main():
    fig, axes = plt.subplots(len(BACKBONES), len(ORDER),
                             figsize=(16, 9))
    for r, (bb, (agent_dir, rep_prefix, anc_prefix)) in enumerate(
            BACKBONES.items()):
        for c, net in enumerate(ORDER):
            ax = axes[r][c]
            for prefix, color, label in (
                    (rep_prefix, '#888888', 'replay-only'),
                    (anc_prefix, '#d95f02', '+P anchor')):
                series, ep0s = [], []
                for sd in SEEDS:
                    rid = f'{prefix}_o2_sd{sd}_st{c + 1}'
                    curve, ep0 = train_curve(agent_dir, net, rid)
                    arr = np.full(MAX_EP + 1, np.nan)
                    for ep, tt in curve.items():
                        if 0 <= ep <= MAX_EP:
                            arr[ep] = tt
                    series.append(arr)
                    if ep0 is not None:
                        ep0s.append(ep0)
                series = np.stack(series)
                mean = np.nanmean(series[:, 1:], axis=0)
                smean = np.concatenate([[np.nan], rolling(mean)])
                x = np.arange(MAX_EP + 1)
                with np.errstate(all='ignore'):
                    ax.fill_between(
                        x, np.nanmin(series, axis=0),
                        np.nanmax(series, axis=0),
                        color=color, alpha=0.15, linewidth=0)
                ax.plot(x, smean, color=color, lw=1.8, label=label)
                if ep0s:
                    ax.plot(0, float(np.mean(ep0s)), 'o', color=color,
                            ms=4)
            ax.set_title(f'train {SCENE_TAG[net]} (st{c + 1})', fontsize=9)
            ax.grid(alpha=0.3)
            ax.tick_params(labelsize=8)
            if r == len(BACKBONES) - 1:
                ax.set_xlabel('episode', fontsize=8)
            if c == 0:
                ax.set_ylabel(f'{bb}\nepisode ATT (s)', fontsize=8)
            if r == 0 and c == 0:
                ax.legend(fontsize=8, loc='upper right')
    fig.suptitle(
        'cl_hz1x1 o2 — per-episode training ATT by stage '
        '(seed mean, rolling-5; band = seed min/max)', fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIG, dpi=150)
    print(f'wrote {FIG}')


if __name__ == '__main__':
    sys.exit(main())
