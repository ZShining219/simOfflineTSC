#!/usr/bin/env python3
"""ATT-ENTITY-004 live status: queue + per-run episode + ledger summary.

Usage: python3.10 tools/att004_status.py [--state artifacts/att_entity_004/run_state_wave1]
"""
import argparse
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/output_data/tsc/sumo_tarl_attention/hz4x4'
LOGS = None


def episode_of(run_id):
    log = LOGS / (run_id.replace('/', '_') + '.log')
    if not log.exists():
        return '-'
    eps = [l for l in log.read_bytes().decode('utf-8', 'replace').splitlines()
           if 'episode:' in l]
    if not eps:
        return '-'
    return eps[-1].split('episode:')[1].split(',')[0].strip()


def ledger_summary(prefix):
    path = OUT / prefix / 'grad_ledger.jsonl'
    if not path.exists():
        return None
    rows = [json.loads(l) for l in path.open()]
    rt = np.array([r['r_t'] for r in rows if r['r_t']])
    al = np.array([r['alpha_s'] for r in rows if r['alpha_s'] is not None])
    g = sum(1 for r in rows if r['gated'])
    q = np.array_split(rt, min(4, max(1, len(rt) // 50))) if len(rt) else []
    tp = sum(r.get('aux_present_tp', 0) for r in rows)
    fp = sum(r.get('aux_present_fp', 0) for r in rows)
    fn = sum(r.get('aux_present_fn', 0) for r in rows)
    ta = [r['aux_type_acc'] for r in rows if 'aux_type_acc' in r]
    return {
        'steps': len(rows),
        'r_t_p50': float(np.median(rt)) if len(rt) else None,
        'r_t_traj': [round(float(np.median(c)), 2) for c in q],
        'r_ema': rows[-1]['r_ema'],
        'alpha_min': float(al.min()) if al.size else None,
        'frac_gated': g / len(rows) if rows else 0,
        'm_b': float(np.mean([r['m_b'] for r in rows])),
        'aux_P': tp / (tp + fp) if tp + fp else None,
        'aux_R': tp / (tp + fn) if tp + fn else None,
        'aux_type_acc': float(np.mean(ta[-200:])) if ta else None,
        'aux_td_loss': rows[-1].get('aux_td_loss'),
        'aux_loss': rows[-1].get('aux_loss'),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--state',
                    default=str(ROOT / 'artifacts/att_entity_004/run_state_wave1'))
    args = ap.parse_args()
    global LOGS
    STATE = Path(args.state) / 'run_manifest.json'
    LOGS = Path(args.state) / 'logs'
    manifest = json.loads(STATE.read_text())['tasks']
    print(f'{"run":22s} {"status":9s} {"ep":9s} {"steps":>6s} '
          f'{"r_t_p50":>8s} {"r_ema":>6s} {"a_min":>6s} {"gated%":>6s} m_b')
    for t in manifest:
        rid, status = t['run_id'], t['status']
        ep = episode_of(rid)
        s = ledger_summary(t['output_path'].rsplit('/', 1)[-1])
        if s:
            amin = (f'{s["alpha_min"]:.2f}' if s['alpha_min'] is not None
                    else '-')
            extra = ''
            if s['aux_loss'] is not None:
                f1 = (2 * s['aux_P'] * s['aux_R'] / (s['aux_P'] + s['aux_R'])
                      if s['aux_P'] and s['aux_R'] else None)
                extra = (f'  P/R/F1={s["aux_P"] or 0:.2f}/{s["aux_R"] or 0:.2f}/'
                         f'{f1 or 0:.2f} type={s["aux_type_acc"] or 0:.2f}'
                         f' td={s["aux_td_loss"] or 0:.2f}'
                         f' sem={s["aux_loss"] or 0:.2f}')
            print(f'{rid:22s} {status:9s} {ep:9s} {s["steps"]:6d} '
                  f'{s["r_t_p50"] or 0:8.2f} {s["r_ema"] or 0:6.2f} '
                  f'{amin:>6s} {s["frac_gated"]:6.1%} {s["m_b"]:.2f}{extra} '
                  f'traj={s["r_t_traj"]}')
        else:
            print(f'{rid:22s} {status:9s} {ep:9s}      -')


if __name__ == '__main__':
    main()
