"""P1: explore natural mode structure of recovered windows for vocabulary design.

Reads p0 windows.csv.gz, builds an interpretable feature vector, KMeans over a
few k values, and prints per-cluster centroids in readable terms so the mode
vocabulary can be drafted from real data.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DIRS = ('N', 'E', 'S', 'W')
FEATURES = [
    'total_mean',           # level
    'N_share', 'E_share', 'S_share', 'W_share',
    'ns_share',
    'total_slope',          # overall trend (standardised t axis)
    'N_slope', 'E_slope', 'S_slope', 'W_slope',
]


def summarise_cluster(df):
    out = {
        'n': len(df),
        'total_mean': df['total_mean'].mean(),
        'dominant_dir': df['dominant_dir'].value_counts(normalize=True).round(2).to_dict(),
        'ns_share': df['ns_share'].mean(),
        'total_slope': df['total_slope'].mean(),
        'dir_means': {d: round(df[f'{d}_mean'].mean(), 1) for d in DIRS},
        'dir_slopes': {d: round(df[f'{d}_slope'].mean(), 3) for d in DIRS},
        'top_phase': df['dominant_phase'].mode()[0],
        'scenes': df['network'].str.replace('sumohz1x1', 'hz').value_counts(normalize=True).round(2).to_dict(),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--table', default='data/output_data/scene_semantics/p0/windows.csv.gz')
    ap.add_argument('--ks', type=int, nargs='+', default=[8, 12])
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    X = df[FEATURES].to_numpy(dtype=np.float64)
    # quantile-normalise level, z-score the rest
    Xz = X.copy()
    for j, name in enumerate(FEATURES):
        col = Xz[:, j]
        if name == 'total_mean':
            ranks = col.argsort().argsort() / (len(col) - 1)
            Xz[:, j] = ranks
        else:
            Xz[:, j] = (col - col.mean()) / (col.std() + 1e-9)
    # cap slope magnitudes so trend contributes but does not dominate
    slope_cols = [FEATURES.index(c) for c in FEATURES if c.endswith('slope')]
    Xz[:, slope_cols] = np.clip(Xz[:, slope_cols], -3, 3)

    from sklearn.cluster import KMeans
    for k in args.ks:
        km = KMeans(n_clusters=k, n_init=8, random_state=args.seed).fit(Xz)
        df['cluster'] = km.labels_
        print(f'\n================ k={k} ================')
        order = df.groupby('cluster').size().sort_values(ascending=False).index
        for c in order:
            s = summarise_cluster(df[df['cluster'] == c])
            print(f"\n-- cluster {c}  n={s['n']} ({s['n']/len(df)*100:.1f}%)")
            print(f"   level total_mean={s['total_mean']:.1f}  ns_share={s['ns_share']:.2f}  "
                  f"slope={s['total_slope']:+.3f}  top_phase={s['top_phase']}")
            print(f"   dir_means={s['dir_means']}")
            print(f"   dir_slopes={s['dir_slopes']}")
            print(f"   dominant={s['dominant_dir']}")
            print(f"   scenes={s['scenes']}")


if __name__ == '__main__':
    main()
