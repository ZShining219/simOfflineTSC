#!/usr/bin/env python3
"""Cluster-structure evidence: same network, different state occupancies.

Claim under test: on the same hz1x1 network, different demand profiles push
the system into separable clusters of the 8-lane state space.

Data: schema-A trajectories only (tsc/sumo_dqn/<scene>/<run>/trajectory/
episodes/*.npz) — 'state' is the instantaneous 8-lane vehicle-count vector
at each decision point; uniform semantics across all scenes.

Outputs (dev/transition_dist/figs/):
  fig4a_pca_scatter.png    PCA-2D of lane states, colored by family / scene
  fig4b_membership.png     k-means cluster -> scene membership (stacked bars)
  fig4c_mixture.png        pooled scalar-queue density decomposed by family
"""
import glob
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, 'data/output_data')
FIGS = os.path.join(ROOT, 'dev/transition_dist/figs')

RUNS_PER_SCENE = 4
EPS_PER_RUN = 6
CAP_SCENE = 14000          # max rows per scene
SEED = 0

FAMILY = {
    'dense_extreme': 'dense', 'dense_high': 'dense',
    'fluct_burst': 'fluct', 'fluct_ramp': 'fluct',
    'base': 'base/misc', 'st_dom': 'base/misc',
    'lt_high': 'left-turn', 'lt_wsing': 'left-turn', 'lt_extreme': 'left-turn',
    'asym_we': 'asym', 'asym_ns': 'asym', 'asym_wonly': 'asym',
    'config2': 'config/sparse', 'config3': 'config/sparse',
    'config4': 'config/sparse', 'sparse_low': 'config/sparse',
    'sparse_ultra': 'config/sparse',
}
FAM_ORDER = ['dense', 'fluct', 'base/misc', 'left-turn', 'asym',
             'config/sparse']


def short(net):
    return 'base' if net == 'sumohz1x1' else net.replace('sumohz1x1_', '')


def take_stride(xs, cap):
    return xs if len(xs) <= cap else [xs[int(i * len(xs) / cap)]
                                      for i in range(cap)]


def gauss_smooth(H, sigma=1.2):
    """separable box approximation of gaussian blur for 2D histograms."""
    w = int(3 * sigma) * 2 + 1
    k = np.ones(w) / w
    for _ in range(3):
        H = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 0, H)
        H = np.apply_along_axis(lambda r: np.convolve(r, k, 'same'), 1, H)
    return H


def kmeans(X, k, iters=25, seed=0):
    rng = np.random.default_rng(seed)
    # kmeans++ init
    cent = [X[rng.integers(len(X))]]
    for _ in range(k - 1):
        d2 = np.min(((X[:, None] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
        cent.append(X[rng.choice(len(X), p=d2 / d2.sum())])
    C = np.array(cent)
    for _ in range(iters):
        lab = ((X[:, None] - C[None]) ** 2).sum(-1).argmin(1)
        new = np.array([X[lab == j].mean(0) if (lab == j).any()
                        else X[rng.integers(len(X))] for j in range(k)])
        if np.allclose(new, C):
            break
        C = new
    return lab, C


def main():
    rng = np.random.default_rng(SEED)
    scenes = {}
    for sdir in sorted(glob.glob(os.path.join(OUT, 'tsc/sumo_dqn/*/'))):
        scene = os.path.basename(sdir.rstrip('/'))
        rows = []
        runs = [r for r in sorted(os.listdir(sdir))
                if glob.glob(os.path.join(
                    sdir, r, 'trajectory/episodes/*.npz'))]
        for run in take_stride(runs, RUNS_PER_SCENE):
            eps = sorted(glob.glob(os.path.join(
                sdir, run, 'trajectory/episodes/*.npz')))
            for f in take_stride(eps, EPS_PER_RUN):
                try:
                    st = np.load(f)['state'].reshape(-1, 8)
                    rows.append(st.astype(np.float32))
                except Exception as e:
                    print('  skip', f, e)
        if rows:
            X = np.concatenate(rows)
            if len(X) > CAP_SCENE:
                X = X[rng.choice(len(X), CAP_SCENE, replace=False)]
            scenes[scene] = X
            print(f'{short(scene):14s} {len(X):6d} rows  '
                  f'total-mean={X.sum(1).mean():6.1f}')

    order = sorted(scenes, key=lambda s: -np.median(scenes[s].sum(1)))
    Xall = np.concatenate([scenes[s] for s in order])
    labels = np.concatenate([[i] * len(scenes[s])
                             for i, s in enumerate(order)])
    fams = np.array([FAMILY.get(short(order[i]), '?') for i in labels])
    q_tot = Xall.sum(1)

    # ---- PCA (standardized) ----
    mu, sd = Xall.mean(0), Xall.std(0) + 1e-6
    Z = (Xall - mu) / sd
    U, Sv, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    PC = U[:, :3] * Sv[:3]                    # scores
    ev = Sv ** 2 / (Sv ** 2).sum()
    print('PCA explained var PC1..3:', np.round(ev[:3], 3))
    print('PC1 loadings:', np.round(Vt[0], 2))
    print('PC2 loadings:', np.round(Vt[1], 2))

    scene_color = {s: plt.get_cmap('tab20').colors[i % 20]
                   for i, s in enumerate(order)}
    fam_color = dict(zip(FAM_ORDER, plt.get_cmap('Set2').colors))

    # ---- fig4a: PCA scatter + family covariance ellipses ----
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    sub = rng.choice(len(PC), min(60000, len(PC)), replace=False)
    for ax, mode in ((axes[0], 'family'), (axes[1], 'scene'),
                     (axes[2], 'queue')):
        if mode == 'family':
            for f_ in FAM_ORDER:
                m = sub[fams[sub] == f_]
                ax.scatter(PC[m, 0], PC[m, 1], s=3, alpha=0.25,
                           color=fam_color[f_], label=f_, lw=0)
        elif mode == 'scene':
            for i, s in enumerate(order):
                m = sub[labels[sub] == i]
                ax.scatter(PC[m, 0], PC[m, 1], s=3, alpha=0.25,
                           color=scene_color[s], lw=0)
        else:
            sc = ax.scatter(PC[sub, 0], PC[sub, 1], s=3, alpha=0.25,
                            c=np.log1p(q_tot[sub]), cmap='viridis', lw=0)
            fig.colorbar(sc, ax=ax, label='log(1+lane count sum)', shrink=0.8)
        ax.set_xlabel(f'PC1 ≈ total count ({ev[0] * 100:.0f}%)')
        ax.set_ylabel(f'PC2 ≈ direction contrast ({ev[1] * 100:.0f}%)')
        ax.set_title(mode, fontsize=10)
    # family 68% ellipses over the full data (not just subsample)
    from matplotlib.patches import Ellipse
    ax = axes[0]
    for f_ in FAM_ORDER:
        m = fams == f_
        P = PC[m, :2]
        mu_f = P.mean(0)
        vals, vecs = np.linalg.eigh(np.cov(P.T))
        ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
        w, h = 2 * np.sqrt(vals * 2.30)          # 68% ellipse
        ax.add_patch(Ellipse(mu_f, w, h, angle=ang, fill=False,
                             ec=fam_color[f_], lw=2.2, zorder=5))
        ax.annotate(f_, mu_f, fontsize=9, weight='bold',
                    color=fam_color[f_], zorder=6,
                    ha='center', va='center',
                    path_effects=[pe.withStroke(
                        linewidth=3, foreground='w')])
    axes[0].legend(fontsize=8, markerscale=6)
    fig.suptitle('Fig4a  lane-state PCA — do demand families occupy '
                 'separable state clusters?')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig4a_pca_scatter.png'), dpi=150)
    plt.close(fig)

    # ---- fig4a2: family 2D-density contours over pooled background ------
    xlim = np.quantile(PC[:, 0], [0.005, 0.995])
    ylim = np.quantile(PC[:, 1], [0.005, 0.995])
    gx = np.linspace(*xlim, 80)
    gy = np.linspace(*ylim, 80)
    Hall, _, _ = np.histogram2d(PC[:, 0], PC[:, 1], bins=[gx, gy])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    for f_, ax in zip(FAM_ORDER, axes.ravel()):
        m = fams == f_
        H, _, _ = np.histogram2d(PC[m, 0], PC[m, 1], bins=[gx, gy])
        ax.pcolormesh(gx, gy, np.log10(Hall.T + 1), cmap='Greys',
                      alpha=0.35, shading='auto', zorder=1)
        Hs = gauss_smooth(H.T / max(H.sum(), 1))
        cs = np.sort(Hs.ravel())[::-1]
        cs /= cs.sum()
        lv50 = cs[np.searchsorted(np.cumsum(cs), 0.5)]
        lv90 = cs[np.searchsorted(np.cumsum(cs), 0.9)]
        ax.contour(gx[:-1], gy[:-1], Hs, levels=[lv90, lv50],
                   colors=[fam_color[f_], fam_color[f_]],
                   linewidths=[1.0, 2.4], zorder=3)
        n = m.sum()
        ax.set_title(f'{f_}  (n={n // 1000}k, {100 * m.mean():.0f}% of pool)',
                     fontsize=10, color=fam_color[f_])
        ax.tick_params(labelsize=8)
    fig.suptitle('Fig4a2  per-family 50%/90% density contours over pooled '
                 'state cloud — cluster separability')
    fig.supxlabel(f'PC1 ≈ total count ({ev[0] * 100:.0f}%)')
    fig.supylabel(f'PC2 ≈ direction contrast ({ev[1] * 100:.0f}%)')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig4a2_family_contour.png'), dpi=150)
    plt.close(fig)

    # ---- kmeans on standardized state ----
    K = 6
    lab, C = kmeans(Z, K, seed=SEED)
    ctot = C.sum(1)          # cluster "size" proxy in standardized space
    corder = np.argsort(ctot)[::-1]
    remap = {o: i for i, o in enumerate(corder)}
    lab = np.array([remap[l] for l in lab])
    # interpretable cluster stat: mean raw total count
    cl_q = np.array([q_tot[lab == j].mean() for j in range(K)])
    cl_pc = np.array([PC[lab == j].mean(0)[:2] for j in range(K)])
    print('cluster mean total-count:', np.round(cl_q, 1))

    # ---- fig4b: cluster -> scene membership ----
    ccolors = plt.get_cmap('Set1').colors
    fig, ax = plt.subplots(figsize=(13, 7))
    M = np.zeros((len(order), K))
    for i in range(len(order)):
        for j in range(K):
            M[i, j] = ((labels == i) & (lab == j)).sum()
    M /= M.sum(1, keepdims=True)
    for i, s in enumerate(order):
        y = len(order) - 1 - i
        left = 0
        for j in range(K):
            ax.barh(y, M[i, j] * 100, left=left, height=0.8,
                    color=ccolors[j % 9], lw=0)
            left += M[i, j] * 100
        ax.text(-2, y, short(s), ha='right', va='center', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ccolors[j % 9],
                            label=f'C{j} (q̄={cl_q[j]:.0f}, '
                                  f'PC=({cl_pc[j,0]:.1f},{cl_pc[j,1]:.1f}))')
               for j in range(K)]
    ax.legend(handles=handles, loc='center left', bbox_to_anchor=(1.0, 0.5),
              fontsize=8)
    ax.set_yticks([]); ax.set_xlim(0, 100)
    ax.set_xlabel('share of scene transitions in cluster  (%)')
    ax.set_title('Fig4b  k-means (k=6) on 8-lane states → which scenes '
                 'populate which cluster')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig4b_membership.png'), dpi=150)
    plt.close(fig)

    # ---- fig4c: pooled scalar queue mixture, decomposed by family ----
    xmax = float(np.quantile(q_tot, 0.995))
    xb = np.arange(0, xmax + 2, 2.0)
    xc = 0.5 * (xb[:-1] + xb[1:])
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    B = np.zeros((len(FAM_ORDER), len(xc)))
    for fi, f_ in enumerate(FAM_ORDER):
        m = fams == f_
        B[fi], _ = np.histogram(q_tot[m], bins=xb, density=True)
        B[fi] *= m.mean()          # weight by family share of pooled mass
    bottom = np.zeros(len(xc))
    for fi, f_ in enumerate(FAM_ORDER):
        ax.fill_between(xc, bottom, bottom + B[fi], color=fam_color[f_],
                        alpha=0.8, lw=0, label=f_)
        bottom += B[fi]
    ax.plot(xc, bottom, color='k', lw=1.4)
    ax.set_xlabel('instantaneous 8-lane vehicle count  sum(state)  (veh)')
    ax.set_ylabel('density contribution')
    ax.set_title('Fig4c  pooled state distribution decomposed by demand '
                 'family (stacked)')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig4c_mixture.png'), dpi=150)
    plt.close(fig)

    np.savez_compressed(os.path.join(FIGS, 'state_pca_cache.npz'),
                        pc=PC[:, :2].astype(np.float32),
                        labels=labels, q=q_tot.astype(np.float32),
                        scenes=np.array([short(s) for s in order]))

    # ============ fig5: SEMANTIC axes instead of PCA ====================
    # lane order at intersection_1_1 (SUMO lane index: 0=THRU, 1=LT —
    # verified vs net.xml dirs and empirical lane means; roadnet.json's
    # laneLinks use its own index order, not SUMO's):
    #   dims 0,1 = W approach (road_0_1_0) [THRU, LT]
    #   dims 2,3 = S approach (road_1_0_1) [THRU, LT]
    #   dims 4,5 = E approach (road_2_1_2) [THRU, LT]
    #   dims 6,7 = N approach (road_1_2_3) [THRU, LT]
    ew = Xall[:, [0, 1, 4, 5]].sum(1)          # W+E approaches (EW axis)
    ns = Xall[:, [2, 3, 6, 7]].sum(1)          # S+N approaches (NS axis)
    ltv = Xall[:, [1, 3, 5, 7]].sum(1)         # left-turn lanes (odd dims)
    tot = q_tot + 1e-6
    imb = (ew - ns) / tot                      # + = EW-loaded, - = NS-loaded
    lts = ltv / tot                            # left-turn share
    appr = np.stack([Xall[:, i:i + 2].sum(1) for i in range(0, 8, 2)], 1)
    dom = appr.max(1) / tot                    # dominant-approach share

    def plane(ax_pairs, X_, Y_, xlab, ylab, ttl):
        """overlay scatter + family 68% ellipses on a semantic plane."""
        fig, ax = plt.subplots(figsize=(11, 8))
        for f_ in FAM_ORDER:
            m = fams == f_
            sm = rng.choice(np.where(m)[0], min(12000, m.sum()),
                            replace=False)
            ax.scatter(X_[sm], Y_[sm], s=3, alpha=0.18,
                       color=fam_color[f_], lw=0)
            P = np.stack([X_[m], Y_[m]], 1)
            mu_f = P.mean(0)
            vals, vecs = np.linalg.eigh(np.cov(P.T))
            ang = np.degrees(np.arctan2(vecs[1, 1], vecs[0, 1]))
            w, h = 2 * np.sqrt(vals * 2.30)
            ax.add_patch(Ellipse(mu_f, w, h, angle=ang, fill=False,
                                 ec=fam_color[f_], lw=2.4, zorder=5))
            ax.annotate(f_, mu_f, fontsize=10, weight='bold', zorder=6,
                        color=fam_color[f_], ha='center', va='center',
                        path_effects=[pe.withStroke(linewidth=3, foreground='w')])
        for i, s in enumerate(order):        # per-scene centroids
            m = labels == i
            ax.plot(np.median(X_[m]), np.median(Y_[m]), 's', ms=5,
                    color=scene_color[s], mec='k', mew=0.6, zorder=7)
            ax.annotate(short(s), (np.median(X_[m]), np.median(Y_[m])),
                        xytext=(4, 4), textcoords='offset points',
                        fontsize=6.5, color='0.25', zorder=7)
        ax.set_xlabel(xlab); ax.set_ylabel(ylab)
        ax.set_title(ttl, fontsize=11)
        handles = [plt.Line2D([], [], marker='o', ls='', color=fam_color[f_],
                              label=f_) for f_ in FAM_ORDER]
        ax.legend(handles=handles, fontsize=9, loc='best', framealpha=0.9)
        fig.tight_layout()
        fig.savefig(os.path.join(FIGS, fname_), dpi=150)
        plt.close(fig)

    from matplotlib.patches import Ellipse
    fname_ = 'fig5a_imb_lt.png'
    plane(None, imb, lts,
          'axis imbalance  (EW−NS)/total   [+ = W/E approach heavy]',
          'left-turn lane share  LT/total',
          'Fig5a  demand fingerprint plane — directional demand families '
          'separate by construction')
    fname_ = 'fig5b_sev_imb.png'
    plane(None, imb, np.log1p(q_tot),
          'axis imbalance  (EW−NS)/total   [+ = W/E approach heavy]',
          'log(1+total lane count)  (severity)',
          'Fig5b  severity × imbalance — asym/config families split '
          'horizontally, dense/base top')
    fname_ = 'fig5c_sev_lt.png'
    plane(None, lts, np.log1p(q_tot),
          'left-turn lane share  LT/total',
          'log(1+total lane count)  (severity)',
          'Fig5c  severity × LT share — lt family occupies high-share band')

    # ---- fig5d: named semantic regimes -> per-scene occupancy bars -----
    sev = np.digitize(q_tot, [20, 45])          # 0 L / 1 M / 2 H
    dirn = np.digitize(imb, [-0.15, 0.15])      # 0 NS / 1 bal / 2 EW
    trn = (lts > 0.18).astype(int)              # 0 thru-ish / 1 LT-heavy
    rname = {s * 6 + d * 2 + t:
             f'{"L M H"[s]}·{"NS" if d == 0 else "bal" if d == 1 else "EW"}'
             + ('·LT+' if t else '')
             for s in range(3) for d in range(3) for t in range(2)}
    reg_id = sev * 6 + dirn * 2 + trn
    # order regimes by mean total count desc for readability
    rstat = {r: q_tot[reg_id == r].mean() for r in np.unique(reg_id)}
    r_sorted = sorted(rstat, key=lambda r: -rstat[r])
    ridx = {r: i for i, r in enumerate(r_sorted)}
    KR = len(r_sorted)
    M = np.zeros((len(order), KR))
    for i in range(len(order)):
        for r in np.unique(reg_id):
            M[i, ridx[r]] = ((labels == i) & (reg_id == r)).sum()
    M /= M.sum(1, keepdims=True)
    cR = plt.get_cmap('tab20').colors
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, s in enumerate(order):
        y = len(order) - 1 - i
        left = 0
        for j in range(KR):
            v = M[i, j] * 100
            ax.barh(y, v, left=left, height=0.8, color=cR[j % 20], lw=0)
            if v > 6:
                ax.text(left + v / 2, y, f'{v:.0f}', ha='center',
                        va='center', fontsize=5.5, color='w')
            left += v
        ax.text(-2, y, short(s), ha='right', va='center', fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=cR[j % 20],
                            label=rname[r_sorted[j]])
               for j in range(KR)]
    ax.legend(handles=handles, loc='center left', fontsize=7, ncol=1,
              bbox_to_anchor=(1.0, 0.5), title='regime (sev·dir·turn)')
    ax.set_yticks([]); ax.set_xlim(0, 100)
    ax.set_xlabel('share of scene transitions in named regime  (%)')
    ax.set_title('Fig5d  semantic regime occupancy — sev{L,M,H} × '
                 'dir{NS,bal,EW} × turn{·,LT+}')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, 'fig5d_regime_bars.png'), dpi=150,
                bbox_inches='tight')
    plt.close(fig)
    print('done ->', FIGS)


if __name__ == '__main__':
    main()
