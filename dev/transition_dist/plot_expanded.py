#!/usr/bin/env python3
"""Expanded transition-distribution figures over the cached corpus.

Reads dev/transition_dist/cache/<group>__<label>.npz + datasets_index.json
produced by build_cache.py.

Queue metric: q_hat = -2/3 * reward = interval-mean total incoming-lane
waiting vehicles (uniform across both trajectory schemas).  Schema-A units
also carry qi = instantaneous decision-point queue.

Figures:
  fig1_hero_ridgeline.png  per-scene pooled p(q_hat), 16 demand rows sorted
                           by median -> "datasets occupy different regimes"
  fig1b_scene_source.png   within-scene x source-group ridgeline (scenes
                           with >=2 source groups): same demand, different
                           collection pipelines still differ
  fig1c_density_heatmap.png  scene x queue-bin log-density companion
  fig2_cond_action.png     P(action | queue bin, scene) — scene rows
  fig3_deltaq.png          P(dq | bin, action) small multiples, scenes split
                           into light/heavy panels by median
"""
import glob
import json
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CACHE = os.path.join(ROOT, 'dev/transition_dist/cache')
FIGS = os.path.join(ROOT, 'dev/transition_dist/figs')

GROUP_TAG = {'A_train': 'train', 'ha_sodqn': 'ha', 'seq8': 'seq8',
             'plan34': 'p34', 'hitl': 'hitl', 'hitl_ms': 'hitl',
             'p1freeze': 'p1fz'}
EXCLUDE_POOL = {'p1freeze'}          # reproduction copies -> no double count
MIN_ROW_N = 2000


def short(net):
    return 'base' if net == 'sumohz1x1' else net.replace('sumohz1x1_', '')


def load_units():
    idx = json.load(open(os.path.join(CACHE, 'datasets_index.json')))
    units = []
    for r in idx:
        p = os.path.join(CACHE, f"{r['uid']}.npz")
        if not os.path.isfile(p):
            continue
        d = np.load(p)
        units.append({**r, 'q': d['q'], 'qpre': d['qpre'],
                      'dq': d['dq'], 'a': d['a'],
                      'qi': d['qi'] if 'qi' in d.files else None})
    return units


def ridgeline(ax, rows, xmax, xlabel, cmap_by='median'):
    """rows: list of (label, values, color) top->bottom."""
    bins = np.arange(0, xmax + 1.0, 1.0)
    centers = 0.5 * (bins[:-1] + bins[1:])
    h = 1.0
    for i, (label, vals, color, n) in enumerate(rows):
        dens, _ = np.histogram(vals, bins=bins, density=True)
        dens = np.convolve(dens, np.ones(5) / 5, mode='same')
        y = (len(rows) - 1 - i) * h  # rows[0] drawn at top
        ax.fill_between(centers, y, y + dens / dens.max() * 0.9 * h,
                        color=color, alpha=0.75, lw=0)
        ax.plot(centers, y + dens / dens.max() * 0.9 * h, color=color, lw=0.8)
        ax.text(-1.5, y + 0.35 * h, f'{label}  (n={n // 1000}k)',
                ha='right', va='center', fontsize=8)
        med = np.median(vals)
        ax.plot([med, med], [y, y + 0.9 * h], color='k', lw=0.8, alpha=0.6)
    ax.set_ylim(-0.3 * h, len(rows) * h)
    ax.set_xlim(-0.02 * xmax, xmax)
    ax.set_yticks([])
    ax.set_xlabel(xlabel)
    ax.spines[['left', 'top', 'right']].set_visible(False)


def main():
    units = load_units()
    print(f'{len(units)} dataset units loaded')
    os.makedirs(FIGS, exist_ok=True)

    # ---------- per-scene pools (exclude reproduction copies) ----------
    scenes = {}
    for u in units:
        if u['group'] in EXCLUDE_POOL or u['scene'] == 'unknown':
            continue
        scenes.setdefault(u['scene'], []).append(u)
    order = sorted(scenes, key=lambda s: -np.median(
        np.concatenate([u['q'] for u in scenes[s]])))
    pooled = {s: np.concatenate([u['q'] for u in scenes[s]]) for s in order}
    pooled_pre = {s: np.concatenate([u['qpre'] for u in scenes[s]]) for s in order}
    pooled_dq = {s: np.concatenate([u['dq'] for u in scenes[s]]) for s in order}
    pooled_a = {s: np.concatenate([u['a'] for u in scenes[s]]) for s in order}

    xmax = float(np.quantile(np.concatenate(list(pooled.values())), 0.998))
    xmax = min(xmax, 110.0)

    # ================= fig1 hero: scene ridgeline + heatmap =================
    meds = {s: float(np.median(v)) for s, v in pooled.items()}
    norm = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=max(meds.values()))
    cmap = plt.get_cmap('RdYlGn_r')
    scene_color = {s: plt.get_cmap('tab20').colors[i % 20]
                   for i, s in enumerate(order)}
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.5),
                           gridspec_kw={'width_ratios': [1.15, 1]})
    rows = [(f'{short(s)}', pooled[s], cmap(norm(meds[s])), len(pooled[s]))
            for s in order]
    ridgeline(axes[0], rows, xmax,
              'interval-mean total queue  $\\hat{q}=-2r/3$  (veh)')
    axes[0].set_title('P(queue) per demand scene — all sources pooled', fontsize=10)

    ax = axes[1]
    bins = np.arange(0, xmax + 2, 2.0)
    H = np.zeros((len(order), len(bins) - 1))
    for i, s in enumerate(order):
        H[i], _ = np.histogram(pooled[s], bins=bins, density=True)
    ax.imshow(H + 1e-6, aspect='auto', norm=LogNorm(vmin=1e-4, vmax=H.max()),
              extent=[0, xmax, len(order), 0], cmap='magma')
    ax.set_yticks(np.arange(len(order)) + 0.5)
    ax.set_yticklabels([short(s) for s in order], fontsize=8)
    ax.set_xlabel('interval-mean total queue (veh)')
    ax.set_title('log-density heatmap (coverage gaps = dark)', fontsize=10)
    fig.suptitle('Fig1  state distribution — 16 hz1x1 demand scenes, '
                 f'{sum(len(v) for v in pooled.values()) // 1000}k transitions')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1_hero_ridgeline.png'), dpi=150)
    plt.close(fig)

    # ===== fig1c: ECDF overlay — all scenes on one axis =================
    fig, ax = plt.subplots(figsize=(11, 6.5))
    for s in order:
        v = np.sort(pooled[s])
        stride = max(1, len(v) // 5000)
        v, y = v[::stride], np.arange(1, len(v) + 1)[::stride] / len(v)
        ax.plot(v, y, lw=1.4, color=cmap(norm(meds[s])))
        ax.plot([meds[s]], [0.5], 'o', ms=4, color=cmap(norm(meds[s])))
    for s in order[:2]:  # only the well-separated heavy scenes get tags
        ax.annotate(short(s), (meds[s], 0.5), xytext=(6, 0),
                    textcoords='offset points', fontsize=8,
                    color=cmap(norm(meds[s])))
    ax.axhline(0.5, color='k', lw=0.5, alpha=0.4)
    ax.axhline(0.9, color='k', lw=0.5, alpha=0.4, ls=':')
    ax.text(xmax, 0.5, ' median', ha='right', va='bottom', fontsize=7,
            color='0.4')
    ax.text(xmax, 0.9, ' p90', ha='right', va='bottom', fontsize=7,
            color='0.4')
    handles = [plt.Line2D([], [], color=cmap(norm(meds[s])),
                          label=f'{short(s)} (med={meds[s]:.0f})')
               for s in order]
    ax.legend(handles=handles, loc='lower right', fontsize=7, ncol=2,
              framealpha=0.9)
    ax.set_xlim(0, xmax); ax.set_ylim(0, 1.02)
    ax.set_xlabel('interval-mean total queue  $\\hat{q}=-2r/3$  (veh)')
    ax.set_ylabel('P(queue <= x)')
    ax.set_title('Fig1c  ECDF — all 16 demand scenes on one axis '
                 '(dots = medians)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1c_ecdf.png'), dpi=150)
    plt.close(fig)

    # ===== fig1d: letter-value (boxen) plot — quantiles, no density =====
    fig, ax = plt.subplots(figsize=(12, 7.5))
    LV = [(0.25, 0.75), (0.125, 0.875), (0.0625, 0.9375), (0.03125, 0.96875)]
    LW = [0.80, 0.62, 0.45, 0.30]
    for i, s in enumerate(order):
        y = len(order) - 1 - i
        v = pooled[s]
        c = cmap(norm(meds[s]))
        for (lo, hi), w in zip(LV, LW):
            ql, qh = np.quantile(v, [lo, hi])
            ax.barh(y, qh - ql, left=ql, height=w, color=c,
                    edgecolor='k', lw=0.5, zorder=3)
        ax.plot([meds[s], meds[s]], [y - 0.42, y + 0.42], color='k',
                lw=1.4, zorder=4)
        qlo, qhi = np.quantile(v, [0.96875, 0.995])
        ax.plot([qlo, qhi], [y, y], color=c, lw=1.2, zorder=2)
        ax.plot(np.quantile(v, 0.999), y, '|', color=c, ms=9, mew=1.6,
                zorder=4)
        ax.text(-1.5, y, f'{short(s)}  (n={len(v) // 1000}k)', ha='right',
                va='center', fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(-0.02 * xmax, xmax)
    ax.set_xlabel('interval-mean total queue  $\\hat{q}=-2r/3$  (veh)')
    ax.set_title('Fig1d  letter-value plot — boxes = quartile/octile/... '
                 'down to p97, whisker = p99.5, tick = p99.9')
    ax.spines[['left', 'top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1d_boxen.png'), dpi=150)
    plt.close(fig)

    # ===== fig1e: density overlay per demand family (shared x) ==========
    by_short = {short(s): s for s in order}
    FAM = [('dense',     ['dense_extreme', 'dense_high']),
           ('fluct',     ['fluct_burst', 'fluct_ramp']),
           ('base/misc', ['base', 'st_dom']),
           ('left-turn', ['lt_high', 'lt_wsing', 'lt_extreme']),
           ('asym',      ['asym_we', 'asym_ns', 'asym_wonly']),
           ('config/sparse', ['config4', 'config2', 'sparse_low',
                              'config3', 'sparse_ultra'])]
    xb = np.arange(0, xmax + 1.0, 1.0)
    xc = 0.5 * (xb[:-1] + xb[1:])
    fig, axes = plt.subplots(2, 3, figsize=(17, 7.5), sharex=True)
    for (fname, members), ax in zip(FAM, axes.ravel()):
        for ms in sorted(members, key=lambda m: -meds[by_short[m]]):
            s = by_short[ms]
            dens, _ = np.histogram(pooled[s], bins=xb, density=True)
            dens = np.convolve(dens, np.ones(5) / 5, mode='same')
            ax.plot(xc, dens, lw=1.6, color=scene_color[s],
                    label=f'{ms} (med={meds[s]:.0f}, n={len(pooled[s])//1000}k)')
        ax.set_title(fname, fontsize=10)
        ax.set_yscale('log'); ax.set_ylim(5e-5, 3)
        ax.legend(fontsize=7, loc='upper right')
        ax.tick_params(labelsize=8)
    fig.suptitle('Fig1e  P(queue) density overlay within demand families '
                 '(log-y; shared x)')
    fig.supxlabel('interval-mean total queue  $\\hat{q}=-2r/3$  (veh)')
    fig.supylabel('density')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1e_density_family.png'), dpi=150)
    plt.close(fig)

    # ===== fig1f: small multiples — each scene vs all others (ghost) ====
    all_dens = {}
    for s in order:
        dens, _ = np.histogram(pooled[s], bins=xb, density=True)
        all_dens[s] = np.convolve(dens, np.ones(5) / 5, mode='same')
    fig, axes = plt.subplots(3, 6, figsize=(19, 8.5), sharex=True)
    for i, s in enumerate(order):
        ax = axes.ravel()[i]
        for g in order:
            if g == s:
                continue
            ax.plot(xc, all_dens[g], lw=0.6, color='0.75', zorder=1)
        ax.fill_between(xc, all_dens[s], color=scene_color[s], alpha=0.6,
                        zorder=2)
        ax.plot(xc, all_dens[s], color=scene_color[s], lw=1.3, zorder=3)
        ax.plot([meds[s], meds[s]], [0, all_dens[s].max()], color='k',
                lw=0.8, alpha=0.6)
        ax.set_title(f'{short(s)}  med={meds[s]:.0f} n={len(pooled[s])//1000}k',
                     fontsize=8)
        ax.tick_params(labelsize=6)
    for i in range(len(order), axes.size):
        axes.ravel()[i].axis('off')
    fig.suptitle('Fig1f  each scene (color) vs all other scenes (gray) — '
                 'shared x, per-panel y')
    fig.supxlabel('interval-mean total queue (veh)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1f_density_ghost.png'), dpi=150)
    plt.close(fig)

    # ===== fig1b: within-scene, source-group comparison (uniform q_hat) =====
    multi = {}
    for u in units:
        if u['group'] in EXCLUDE_POOL or u['scene'] == 'unknown':
            continue
        multi.setdefault(u['scene'], {}).setdefault(
            GROUP_TAG.get(u['group'], u['group']), []).append(u)
    multi = {s: g for s, g in multi.items() if len(g) >= 2}
    g_order = sorted(multi, key=lambda s: -np.median(pooled[s]))
    gcolors = plt.get_cmap('tab10').colors
    panels = [s for s in g_order]
    ncol = 4
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(16, 2.6 * nrow),
                             sharex=True)
    axes = np.atleast_1d(axes).ravel()
    xb = np.arange(0, xmax + 1.0, 1.0)
    tags_all = sorted({t for s in multi for t in multi[s]})
    for pi, s in enumerate(panels):
        ax = axes[pi]
        for gi, tag in enumerate(tags_all):
            if tag not in multi[s]:
                continue
            v = np.concatenate([u['q'] for u in multi[s][tag]])
            dens, _ = np.histogram(v, bins=xb, density=True)
            dens = np.convolve(dens, np.ones(3) / 3, mode='same')
            ax.plot(0.5 * (xb[:-1] + xb[1:]), dens, lw=1.2,
                    color=gcolors[gi], label=f'{tag} (n={len(v)//1000}k)')
        ax.set_title(f'{short(s)}', fontsize=9)
        ax.set_yscale('log'); ax.set_ylim(1e-4, 2)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6, loc='upper right')
    for pi in range(len(panels), len(axes)):
        axes[pi].axis('off')
    fig.suptitle('Fig1b  same demand scene, different collection pipelines '
                 '(log-y density)')
    fig.supxlabel('interval-mean total queue (veh)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig1b_scene_source.png'), dpi=150)
    plt.close(fig)

    # ================= fig2: P(action | queue bin, scene) =================
    edges = [0, 5, 20, np.inf]
    bnames = ['q<5', '5<=q<20', 'q>=20']
    nsc = len(order)
    fig, axes = plt.subplots(1, 3, figsize=(20, 0.38 * nsc + 2.4), sharey=True)
    for b, ax in enumerate(axes):
        M = np.full((nsc, 8), np.nan)
        ns = []
        for r, s in enumerate(order):
            m = (pooled_pre[s] >= edges[b]) & (pooled_pre[s] < edges[b + 1])
            ns.append(int(m.sum()))
            if m.sum() < 200:
                continue
            cnt = np.bincount(pooled_a[s][m], minlength=8)
            M[r] = cnt / cnt.sum()
        im = ax.imshow(np.ma.masked_invalid(M), aspect='auto', vmin=0, vmax=1,
                       cmap='viridis')
        for r in range(nsc):
            for c in range(8):
                if np.isnan(M[r, c]):
                    continue
                ax.text(c, r, f'{M[r, c] * 100:.0f}', ha='center', va='center',
                        fontsize=6.5,
                        color='w' if M[r, c] < 0.55 else 'k')
            if ns[r] >= 1:
                ax.text(8.15, r, f'{ns[r] / 1000:.0f}k', va='center',
                        fontsize=6.5, color='0.35', clip_on=False)
        ax.set_yticks(range(nsc))
        ax.set_yticklabels([short(s) for s in order], fontsize=8)
        ax.set_xlim(-0.5, 9.2)
        ax.set_xticks(range(8)); ax.set_xlabel('action (phase id)')
        ax.set_title(f'bin {bnames[b]}  (right margin: n)')
    fig.colorbar(im, ax=axes, label='P(action | bin, scene)', shrink=0.5)
    fig.suptitle('Fig2  conditional action distribution per scene pool')
    fig.savefig(os.path.join(FIGS, 'fig2_cond_action.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ---------- fig2b: 100% stacked horizontal bars (no heatmap) ----------
    acolors = plt.get_cmap('tab10').colors[:8]
    fig, axes = plt.subplots(1, 3, figsize=(19, 0.42 * nsc + 2.2),
                             sharey=True)
    for b, ax in enumerate(axes):
        for r, s in enumerate(order):
            y = nsc - 1 - r
            m = (pooled_pre[s] >= edges[b]) & (pooled_pre[s] < edges[b + 1])
            nsr = int(m.sum())
            if nsr < 200:
                continue
            cnt = np.bincount(pooled_a[s][m], minlength=8) / nsr * 100
            left = 0.0
            for a in range(8):
                ax.barh(y, cnt[a], left=left, height=0.82,
                        color=acolors[a], lw=0)
                if cnt[a] > 7:
                    ax.text(left + cnt[a] / 2, y, f'a{a}', ha='center',
                            va='center', fontsize=5.5, color='w')
                left += cnt[a]
            ax.text(102, y, f'{nsr // 1000}k', va='center', fontsize=6.5,
                    color='0.35')
        ax.set_xlim(0, 110); ax.set_ylim(-0.6, nsc - 0.4)
        ax.set_yticks(range(nsc))
        ax.set_yticklabels([short(order[nsc - 1 - i]) for i in range(nsc)],
                         fontsize=8)
        ax.set_xlabel('P(action | bin, scene)  %')
        ax.set_title(f'bin {bnames[b]}', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=acolors[a], label=f'a{a}')
               for a in range(8)]
    fig.legend(handles=handles, loc='lower center', ncol=8, fontsize=8)
    fig.suptitle('Fig2b  action fingerprint — 100% stacked bars per scene')
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(os.path.join(FIGS, 'fig2b_action_bars.png'), dpi=150)
    plt.close(fig)

    # ================= fig3: dq | bin, action — light vs heavy scenes ======
    light = [s for s in order if meds[s] < 8]
    heavy = [s for s in order if meds[s] >= 8]
    for tag, sset in (('light', light), ('heavy', heavy)):
        if not sset:
            continue
        fig, axes = plt.subplots(3, 8, figsize=(22, 8), sharex=True,
                                 sharey=True)
        dbins = np.linspace(-15, 15, 61)
        cset = {s: scene_color[s] for s in sset}
        for b in range(3):
            for a in range(8):
                ax = axes[b, a]
                nmax = 0
                for s in sset:
                    m = (pooled_pre[s] >= edges[b]) & \
                        (pooled_pre[s] < edges[b + 1]) & (pooled_a[s] == a)
                    if m.sum() < 100:
                        continue
                    nmax = max(nmax, int(m.sum()))
                    ax.hist(pooled_dq[s][m], bins=dbins, density=True,
                            histtype='step', lw=1.0, color=cset[s])
                ax.text(0.98, 0.95, f'n<={nmax // 1000}k', transform=ax.transAxes,
                        ha='right', va='top', fontsize=6, color='0.4')
                ax.axvline(0, color='k', lw=0.5, alpha=0.4)
                if b == 0:
                    ax.set_title(f'a={a}', fontsize=9)
                if a == 0:
                    ax.set_ylabel(bnames[b], fontsize=9)
                ax.tick_params(labelsize=7)
        handles = [plt.Line2D([], [], color=cset[s], label=short(s))
                   for s in sset]
        fig.legend(handles=handles, loc='lower center',
                   ncol=min(len(sset), 8), fontsize=8)
        fig.suptitle(f'Fig3-{tag}  P(dq | queue bin, action) — {tag} scenes '
                     '(median>=8 / <8 split)')
        fig.supxlabel('dq = q_post - q_pre (veh, per 10s interval)', y=0.055)
        fig.tight_layout(rect=[0, 0.07, 1, 1])
        fig.savefig(os.path.join(FIGS, f'fig3_deltaq_{tag}.png'), dpi=150)
        plt.close(fig)

    # ============ fig3c: mean dq heatmap  scene x (bin, action) =========
    ncol = 24
    Mm = np.full((nsc, ncol), np.nan)   # mean
    Ms = np.full((nsc, ncol), np.nan)   # std
    for r, s in enumerate(order):
        for b in range(3):
            for a in range(8):
                m = (pooled_pre[s] >= edges[b]) & \
                    (pooled_pre[s] < edges[b + 1]) & (pooled_a[s] == a)
                if m.sum() < 100:
                    continue
                Mm[r, b * 8 + a] = pooled_dq[s][m].mean()
                Ms[r, b * 8 + a] = pooled_dq[s][m].std()
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.2), sharey=True)
    vmax = float(np.nanmax(np.abs(Mm)))
    labels = [f'{bnames[b]}\na={a}' for b in range(3) for a in range(8)]
    for ax, MM, ttl, cm in (
            (axes[0], Mm, 'mean dq  (veh / 10s interval)', 'RdYlGn_r'),
            (axes[1], Ms, 'std dq', 'magma')):
        if cm == 'RdYlGn_r':
            im = ax.imshow(np.ma.masked_invalid(MM), aspect='auto',
                           vmin=-vmax, vmax=vmax, cmap=cm)
        else:
            im = ax.imshow(np.ma.masked_invalid(MM), aspect='auto',
                           vmin=0, cmap=cm)
        for b in (1, 2):
            ax.axvline(b * 8 - 0.5, color='w', lw=2)
        for r in range(nsc):
            for c in range(ncol):
                if np.isnan(MM[r, c]):
                    continue
                ax.text(c, r, f'{MM[r, c]:.1f}', ha='center', va='center',
                        fontsize=5, color='w' if abs(MM[r, c]) > 0.6 *
                        (vmax if cm == 'RdYlGn_r' else np.nanmax(Ms)) else 'k')
        ax.set_xticks(range(ncol))
        ax.set_xticklabels(labels, fontsize=5.5)
        ax.set_title(ttl, fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.6)
    axes[0].set_yticks(range(nsc))
    axes[0].set_yticklabels([short(s) for s in order], fontsize=8)
    fig.suptitle('Fig3c  E[dq] / std[dq] per (queue bin, action) — '
                 'all scenes, one map (blank = n<100)')
    fig.savefig(os.path.join(FIGS, 'fig3c_dq_heatmap.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)

    # ============ fig3d: profile lines over the 24 condition cells ========
    fig, ax = plt.subplots(figsize=(16, 6.8))
    x = np.arange(ncol)
    for b in (0, 1, 2):
        ax.axvspan(b * 8 - 0.5, b * 8 + 7.5,
                   color='0.92' if b % 2 == 0 else '1.0', zorder=0)
    for r, s in enumerate(order):
        ax.plot(x, Mm[r], marker='o', ms=3, lw=1.1,
                color=scene_color[s], label=short(s), zorder=3)
    ax.axhline(0, color='k', lw=0.7, alpha=0.5)
    ytop = np.nanmax(Mm) * 1.08
    ax.set_ylim(None, ytop)
    for b in (0, 1, 2):
        ax.text(b * 8 + 3.5, ytop * 0.97, bnames[b], ha='center',
                va='top', fontsize=10, color='0.4')
    ax.set_xticks(x)
    ax.set_xticklabels([f'a{a}' for a in range(8)] * 3, fontsize=8)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylabel('mean dq  (veh / 10s interval)')
    ax.legend(loc='upper left', fontsize=7, ncol=2, framealpha=0.9)
    ax.set_title('Fig3d  consequence profile — mean dq per (bin, action); '
                 'shaded bands = queue bins')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig3d_dq_profiles.png'), dpi=150)
    plt.close(fig)

    # ---------------- console summary ----------------
    print(f'\n{"scene":16s} {"n_k":>6s} {"med":>5s} {"p90":>5s} {"mean":>6s} '
          f'sources')
    for s in order:
        srcs = sorted({GROUP_TAG.get(u["group"], u["group"])
                       for u in scenes[s]})
        print(f'{short(s):16s} {len(pooled[s]) // 1000:6d} '
              f'{meds[s]:5.0f} {np.quantile(pooled[s], 0.9):5.0f} '
              f'{pooled[s].mean():6.2f} {"|".join(srcs)}')


if __name__ == '__main__':
    main()
