"""Paper motivation figure: why semi-offline replay (problem -> mechanism).

One representative composite, all data already on disk:

  (a) PCA partition: real (state,phase) pairs the controller experienced,
      sampled from the so1_soddqn_o2_sd0 stage archives, coloured by
      scene_id -> different scenes occupy different regions of the state
      space (experience distributions differ across scenes).
  (b) Value/policy drift: |Q-Q_base|_1 (and greedy flip, dashed) measured
      on the FIXED s1 reference pool along the o2 chain, pure-online vs
      semi-offline -> historical value drifts in both arms; semi-offline
      compresses its magnitude (ddqn sd0).
  (c) Distribution cover: KDE of PC1 per scene plus the stage-4 buffer
      union -> the semi-offline buffer covers the multi-modal union, not
      only the current scene.
  (d) Consequence: frozen-eval ATT on the old scene s1 along the chain
      (drift_s1_tt.json, eval seeds 101-103, mean over seeds with range
      band) -> drift hurts the pure-online arm; semi-offline holds.

Run: python dev/cl_hz1x1/plot_motivation.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

OUT = ROOT / 'data' / 'output_data' / 'tsc'
ART = ROOT / 'artifacts' / 'cl_hz1x1'
ARM_DIR = 'sumo_dqn'
ARM_PREF = 'so1_soddqn_o2_sd0'
O2_NET = {1: 'sumohz1x1', 2: 'sumohz1x1_config4',
          3: 'sumohz1x1_config2', 4: 'sumohz1x1_config3'}
SCENE_TAG = {1: 's1', 2: 'c4', 3: 'c2', 4: 'c3'}
SCENE_COL = {'s1': '#1f77b4', 'c4': '#9467bd', 'c2': '#ff7f0e', 'c3': '#2ca02c'}
PER_SCENE = 800
RNG = np.random.RandomState(0)


def archive_features(st):
    """features(obs,phase) of real transitions from stage-st archive."""
    feats, acts = [], []
    for f in sorted(glob.glob(
            str(OUT / ARM_DIR / O2_NET[st]
                / f'{ARM_PREF}_st{st}' / 'history_archive' / 'episode_*.pt')
            ))[::10]:
        for t in torch.load(f, map_location='cpu', weights_only=False):
            s = np.asarray(t.state, dtype=np.float32)
            if s.sum() == 0:
                continue
            ph = np.zeros(8, dtype=np.float32)
            ph[int(np.asarray(t.phase).flat[0]) % 8] = 1.0
            feats.append(np.concatenate([s, ph]))
            acts.append(int(t.action))
    feats = np.asarray(feats, dtype=np.float32)
    sel = RNG.choice(len(feats), min(PER_SCENE, len(feats)), replace=False)
    return feats[sel], np.asarray(acts)[sel]


def pca_fit(X, k=2):
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:k].T, vt[:k]


def kde(xs, grid, bw=None):
    xs = np.asarray(xs)
    bw = bw or 1.06 * xs.std() * len(xs) ** -0.2
    k = np.exp(-0.5 * ((grid[:, None] - xs[None, :]) / bw) ** 2)
    return k.sum(1) / (len(xs) * bw * np.sqrt(2 * np.pi))


def panel_pca(ax, proj, tags):
    for sc in ('s1', 'c4', 'c2', 'c3'):
        m = tags == sc
        ax.scatter(proj[m, 0], proj[m, 1], s=4, alpha=0.35,
                   color=SCENE_COL[sc], label=sc, rasterized=True)
    ax.set_title('(a) experienced (state,phase) pairs\n'
                 'PCA, coloured by scene', fontsize=9)
    ax.legend(fontsize=7, markerscale=3, loc='best')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')


def panel_drift(ax):
    rows = json.loads((ART / 'drift_s1_rows.json').read_text())
    xs_lbl = ['st2e0', 'st2e50', 'st2e100', 'st3e50', 'st3e100',
              'st4e50', 'st4e100']
    xs = np.arange(len(xs_lbl))
    cmap = {'ddqn_clear': ('#d62728', 'pure online'),
            'soddqn': ('#1f77b4', 'semi-offline')}
    for arm, (col, lab) in cmap.items():
        for sd in (0, 1, 2):
            ys = [next((r['q_l1'] for r in rows
                        if r['arm'] == arm and r['dataset'] == 'pool'
                        and r['seed'] == sd
                        and r['stage'] == int(x[2]) and r['ep'] == int(x[4:])),
                       np.nan)
                  for x in xs_lbl]
            ax.plot(xs, ys, 'o-', color=col, ms=3,
                    lw=1.6 if sd == 0 else 0.8,
                    alpha=1.0 if sd == 0 else 0.35,
                    label=lab if sd == 0 else None)
    ax.set_title('(b) historical-value drift on fixed s1 pool\n'
                 '|Q(ckpt) - Q(stage-1 teacher)|1 along the chain', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs_lbl, fontsize=7)
    ax.set_ylabel('mean |ΔQ|')
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)


def panel_cover(ax, pc1, tags):
    grid = np.linspace(pc1.min(), pc1.max(), 300)
    for sc in ('s1', 'c4', 'c2', 'c3'):
        ax.plot(grid, kde(pc1[tags == sc], grid), color=SCENE_COL[sc],
                lw=1.2, label=sc)
    ax.plot(grid, kde(pc1, grid), 'k-', lw=2.2,
            label='so buffer (union)')
    ax.fill_between(grid, kde(pc1, grid), color='k', alpha=0.08)
    ax.set_title('(c) marginal density on PC1\n'
                 'semi-offline buffer covers the union', fontsize=9)
    ax.set_xlabel('PC1')
    ax.set_ylabel('density')
    ax.legend(fontsize=7)


def panel_tt(ax):
    rows = json.loads((ART / 'drift_s1_tt.json').read_text())
    xs_lbl = ['st1', 'st2', 'st3', 'st4']
    xs = np.arange(4)
    cmap = {'ddqn_clear': ('#d62728', 'pure online'),
            'soddqn': ('#1f77b4', 'semi-offline')}
    for arm, (col, lab) in cmap.items():
        mu, lo, hi = [], [], []
        for st in (1, 2, 3, 4):
            vals = [r['tt'] for r in rows
                    if r['arm'] == arm and r['stage'] == st and r['ep'] == 100]
            mu.append(np.mean(vals))
            lo.append(np.min(vals))
            hi.append(np.max(vals))
        ax.plot(xs, mu, 'o-', color=col, lw=1.8, ms=5, label=lab)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.15)
    ax.set_title('(d) consequence: frozen eval ATT on old scene s1\n'
                 '(stage-end ckpts, eval seeds 101-103)', fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(xs_lbl)
    ax.set_xlabel('chain position (stage end)')
    ax.set_ylabel('eval ATT (s)')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)


def main():
    feats, tags, acts = [], [], []
    for st in (1, 2, 3, 4):
        f, a = archive_features(st)
        feats.append(f)
        tags += [SCENE_TAG[st]] * len(f)
    F = np.concatenate(feats)
    tags = np.asarray(tags)
    proj, comp = pca_fit(F)
    ev = None
    Xc = F - F.mean(0)
    sv = np.linalg.svd(Xc, full_matrices=False, compute_uv=False)
    ev = sv ** 2 / (sv ** 2).sum()
    print(f'PCA var explained: {ev[:2]}')
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    panel_pca(axes[0], proj, tags)
    panel_drift(axes[1])
    panel_cover(axes[2], proj[:, 0], tags)
    panel_tt(axes[3])
    fig.suptitle('Semi-offline replay under continual learning: experience '
                 'distributions differ, historical value drifts, replay '
                 'covers and contains', fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(ART / 'fig_cl_motivation.png', dpi=200)
    print('wrote fig_cl_motivation.png')


if __name__ == '__main__':
    main()
