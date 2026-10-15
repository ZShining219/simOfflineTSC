"""P0: recover observation windows from Plan 1 trajectories and extract stats.

Reads the p1 whitelist, samples episodes inside the Q1 range (episodes 1-100),
slides a fixed-length decision window and records per-window statistics built
ONLY from the 16-dim policy observation (8 incoming lane counts + phase).

Lane order in ``state`` (verified identical across all four hz1x1 nets):
    [N_straight, N_left, E_straight, E_left, S_straight, S_left, W_straight, W_left]
where N/S/E/W is the approach side (compass position of the upstream junction).

Output: one row per window (csv.gz), plus a text summary of distributions.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OLD_PREFIX = '/projects/simOfflineTSC'

DIRECTIONS = ('N', 'E', 'S', 'W')
# lane index -> approach; each approach contributes through+left lanes
DIR_LANES = {'N': (0, 1), 'E': (2, 3), 'S': (4, 5), 'W': (6, 7)}


def load_whitelist(path):
    df = pd.read_csv(path)
    df['run_path'] = df['run_path'].str.replace(OLD_PREFIX, str(ROOT), regex=False)
    return df


def episode_file(run_dir, episode):
    return Path(run_dir) / 'trajectory' / 'episodes' / f'episode_{episode:04d}.npz'


def sample_episodes(run_dir, episodes_per_run, lo=1, hi=100, seed=0):
    available = sorted(
        int(re.search(r'episode_(\d+)', p.name).group(1))
        for p in (Path(run_dir) / 'trajectory' / 'episodes').glob('*.npz')
    )
    in_range = [e for e in available if lo <= e <= hi]
    rng = np.random.RandomState(seed)
    picked = sorted(rng.choice(in_range, size=min(episodes_per_run, len(in_range)),
                               replace=False).tolist())
    return picked, len(available), len(in_range)


def window_stats(state, phase, start, window):
    """state: [T,8] lane counts; phase: [T] green-action index (0..7)."""
    s = state[start:start + window]          # [W,8]
    ph = phase[start:start + window]         # [W]
    t = np.arange(window, dtype=np.float64)
    t = (t - t.mean()) / max(t.std(), 1e-9)
    out = {}
    dir_means = {}
    for i, lane in enumerate(('N_s', 'N_l', 'E_s', 'E_l', 'S_s', 'S_l', 'W_s', 'W_l')):
        out[f'{lane}_mean'] = float(s[:, i].mean())
    for d in DIRECTIONS:
        lanes = DIR_LANES[d]
        series = s[:, lanes].sum(axis=1)     # vehicles on the approach's 2 lanes
        dir_means[d] = float(series.mean())
        out[f'{d}_mean'] = dir_means[d]
        out[f'{d}_max'] = float(series.max())
        out[f'{d}_slope'] = float(np.dot(t, series - series.mean()) / window)
    means = np.array([dir_means[d] for d in DIRECTIONS])
    total = float(means.sum())
    out['total_mean'] = total
    out['total_max'] = float(s.sum(axis=1).max())
    totals = s.sum(axis=1)
    out['total_slope'] = float(np.dot(t, totals - totals.mean()) / window)
    ns = means[0] + means[2]
    ew = means[1] + means[3]
    out['ns_share'] = float(ns / total) if total > 0 else 0.5
    out['ew_share'] = float(ew / total) if total > 0 else 0.5
    shares = means / total if total > 0 else np.full(4, 0.25)
    for i, d in enumerate(DIRECTIONS):
        out[f'{d}_share'] = float(shares[i])
    out['dominant_dir'] = DIRECTIONS[int(np.argmax(shares))]
    out['max_dir_share'] = float(shares.max())
    out['dir_imbalance'] = float(shares.max() - shares.min())
    counts = np.bincount(ph.astype(int), minlength=8)[:8]
    for a in range(8):
        out[f'phase{a}_share'] = float(counts[a] / window)
    out['dominant_phase'] = int(counts.argmax())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--whitelist',
                    default='data/output_data/analysis/plan1/'
                            'p1_formal_20_trajectory_whitelist_20260722.csv')
    ap.add_argument('--episodes-per-run', type=int, default=10)
    ap.add_argument('--window', type=int, default=18, help='decisions (10s each)')
    ap.add_argument('--stride', type=int, default=6)
    ap.add_argument('--sample-seed', type=int, default=0)
    ap.add_argument('--out', default='data/output_data/scene_semantics/p0')
    args = ap.parse_args()

    rows, skipped = [], []
    wl = load_whitelist(args.whitelist)
    for _, r in wl.iterrows():
        picked, n_total, n_q1 = sample_episodes(
            r['run_path'], args.episodes_per_run, seed=args.sample_seed)
        if not picked:
            skipped.append((r['run_path'], n_total, n_q1))
            continue
        for ep in picked:
            z = np.load(episode_file(r['run_path'], ep))
            state = z['state'].reshape(z['state'].shape[0], -1)     # [360,8]
            phase = z['current_phase'].reshape(-1)                  # [360]
            T = state.shape[0]
            for start in range(0, T - args.window + 1, args.stride):
                st = window_stats(state, phase, start, args.window)
                st.update({
                    'network': r['network'],
                    'behavior_seed': int(r['behavior_training_seed']),
                    'episode': ep,
                    'win_start': start,
                    'win_end': start + args.window,
                })
                rows.append(st)

    df = pd.DataFrame(rows)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / 'windows.csv.gz'
    df.to_csv(table_path, index=False, compression='gzip')

    meta = {
        'whitelist': str(args.whitelist),
        'episodes_per_run': args.episodes_per_run,
        'window_decisions': args.window,
        'window_seconds': args.window * 10,
        'stride_decisions': args.stride,
        'q1_range': [1, 100],
        'n_windows': len(df),
        'n_runs': int(df[['network', 'behavior_seed']].drop_duplicates().shape[0]),
        'skipped_runs': skipped,
        'state_fields_used': ['state', 'current_phase'],
        'lane_order': 'N_s,N_l,E_s,E_l,S_s,S_l,W_s,W_l',
    }
    with open(out_dir / 'manifest.json', 'w') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # ---- summary printed for review ----
    pd.set_option('display.width', 160)
    print(f'windows: {len(df)}  -> {table_path}')
    print('\n== total_mean quantiles (per scene) ==')
    print(df.groupby('network')['total_mean'].describe(
        percentiles=[.1, .25, .5, .75, .9]).round(2))
    print('\n== dominant direction histogram (share %) ==')
    print((df.groupby(['network', 'dominant_dir']).size()
             .unstack(fill_value=0)
             .pipe(lambda x: (x.T / x.T.sum()).T * 100).round(1)))
    print('\n== dir_imbalance quantiles ==')
    print(df['dir_imbalance'].describe(percentiles=[.25, .5, .75, .9]).round(3))
    print('\n== dominant phase histogram (all scenes) ==')
    print(df['dominant_phase'].value_counts().sort_index().to_dict())


if __name__ == '__main__':
    sys.exit(main())
