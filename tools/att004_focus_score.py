#!/usr/bin/env python3
"""Focus-set scoring over the ATT-ENTITY-004 state library.

Per artifacts/att_entity_004/virtual_expert_focus.md: for each dump episode we
identify localized bound nodes (z_task present & type in {blockage, closure}),
build the node-level focus set F = bound nodes + their grid neighbours +
event edge endpoints, and measure where the text-conditioned attention mass
(fusion_w) actually goes, plus behaviour-level traces (action flips, queue).

Baselines without fusion_w contribute only behaviour-level rows.
Output: JSONL per cell + a summary table under the output dir.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GRID = 4  # hz4x4: idx = (x-1)*4 + (y-1)
LOCAL_TYPES = {1.0, 2.0}  # lane_blockage / road_closure; 3 = global_rain


def neighbors(idx):
    x, y = idx // GRID + 1, idx % GRID + 1
    out = []
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx, ny = x + dx, y + dy
        if 1 <= nx <= GRID and 1 <= ny <= GRID:
            out.append((nx - 1) * GRID + (ny - 1))
    return out


def score_a3s_episode(npz_path):
    z = np.load(npz_path)
    zt, fw = z['z_task'], z['fusion_w']       # (T,16,6), (T,1,16,16)
    t = z['t']
    bound_mask = (zt[..., 0] > 0.5) & np.isin(zt[..., 1], list(LOCAL_TYPES))
    rain_mask = (zt[..., 0] > 0.5) & (zt[..., 1] == 3.0)
    out = []
    prev_action = None
    for k in range(len(t)):
        bound = np.where(bound_mask[k])[0]
        w = fw[k, 0]                         # (16,16): text-token i over traffic j
        rec = {'t': float(t[k]), 'n_bound': int(len(bound)),
               'rain_active': bool(rain_mask[k].any())}
        if len(bound):
            F = sorted(set(bound.tolist()) |
                       {n for b in bound for n in neighbors(int(b))})
            mass = w[bound][:, F].sum(-1)    # (n_bound,)
            rec.update(focus_set=F, focus_mass=float(mass.mean()),
                       focus_ratio=float(mass.mean() / (len(F) / 16.0)),
                       focus_bound_only=float(w[bound][:, bound].sum(-1).mean()))
        act = z['action'][k]
        if prev_action is not None:
            rec['action_flips'] = int((act != prev_action).sum())
        prev_action = act
        qn = z['queue_per_node'][k]
        rec['queue_total'] = float(qn.sum())
        if len(bound):
            rec['queue_focus'] = float(qn[F].sum())
            rec['queue_nonfocus'] = float(qn.sum() - qn[F].sum())
        out.append(rec)
    return out


def score_baseline_episode(npz_path):
    z = np.load(npz_path)
    t = z['t']
    out = []
    prev = None
    for k in range(len(t)):
        act = z['action'][:, k].reshape(16, -1)[:, 0]
        rec = {'t': float(t[k]),
               'queue_total': float(np.nansum(z['queue_per_agent'][:, k]))}
        if prev is not None:
            rec['action_flips'] = int((act != prev).sum())
        prev = act
        out.append(rec)
    return out


def roles_from_plan_log(dump_dir):
    """Episode index -> (role, label-ish summary)."""
    log = dump_dir / 'event_plan_log.jsonl'
    roles = []
    if log.exists():
        for line in log.read_text().splitlines():
            d = json.loads(line)
            kinds = '+'.join(e['kind'] for e in d.get('events', [])) or 'normal'
            roles.append((d.get('role'), kinds))
    return roles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--state_lib', default='data/output_data/analysis/att_entity_004/state_lib')
    ap.add_argument('--output', default='data/output_data/analysis/att_entity_004/focus_score')
    args = ap.parse_args()
    lib = (ROOT / args.state_lib).resolve()
    out_dir = (ROOT / args.output).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []
    for cell_dir in sorted(p for p in lib.iterdir()
                           if p.is_dir() and not p.name.startswith('_')):
        roles = roles_from_plan_log(cell_dir)
        is_a3s = (cell_dir / 'ep000.npz').exists() and 'fusion_w' in np.load(
            cell_dir / 'ep000.npz').files
        recs_path = out_dir / f'{cell_dir.name}.jsonl'
        ep_stats = []
        with recs_path.open('w') as fh:
            for npz in sorted(cell_dir.glob('ep*.npz')):
                ep_idx = int(npz.stem[2:])
                role, kinds = roles[ep_idx] if ep_idx < len(roles) else ('?', '?')
                rows = (score_a3s_episode(npz) if is_a3s
                        else score_baseline_episode(npz))
                fm = [r.get('focus_mass') for r in rows if 'focus_mass' in r]
                fr = [r.get('focus_ratio') for r in rows if 'focus_ratio' in r]
                for r in rows:
                    fh.write(json.dumps({'cell': cell_dir.name, 'ep': ep_idx,
                                         'role': role, 'kinds': kinds, **r}) + '\n')
                ep_stats.append({'ep': ep_idx, 'role': role, 'kinds': kinds,
                                 'event_steps': sum(1 for r in rows
                                                    if r.get('n_bound')),
                                 'focus_mass_mean': float(np.mean(fm)) if fm else None,
                                 'focus_ratio_mean': float(np.mean(fr)) if fr else None})
        # aggregate: event vs normal split
        ev = [s['focus_ratio_mean'] for s in ep_stats
              if s['event_steps'] > 0 and s['focus_ratio_mean'] is not None]
        summary_rows.append({'cell': cell_dir.name, 'a3s': is_a3s,
                             'episodes': len(ep_stats),
                             'event_episodes': sum(1 for s in ep_stats
                                                   if s['event_steps'] > 0),
                             'focus_ratio_event_mean':
                                 float(np.mean(ev)) if ev else None})
    (out_dir / 'focus_summary.json').write_text(
        json.dumps(summary_rows, indent=1))
    print(json.dumps(summary_rows, indent=1))


if __name__ == '__main__':
    main()
