#!/usr/bin/env python3
"""Lag-window (pre-symptom) utility analysis for the attention line (att004).

Question under test: does the text channel deliver *pre-symptom* value —
i.e. does the policy act differently, at the right nodes, after an event is
announced but before sensors can see its effect (queue formation)?

Data per eval_v2 dump dir (tools/run_tarl_struct_cf.py --dump_dir):
    ep*.npz: t[360], ob_traffic[360,16,12], z_task[360,16,6], action[360,16],
             queue_now[360], queue_per_node[360,16]
    probe/tarl_struct_cf_records.jsonl: per decision, per-variant argmax
             actions on the SAME live state (canonical side only — the emp
             run's records are structurally all-zero and carry no contrast).
    event_plan_log.jsonl: per plan episode, event begin/end/kind/targets.

Per event e (aligned dump<->plan by detected offset):
  bound_nodes   nodes with a present z_task row of matching kind in window
  feeder_node   road_closure only: index of edge.from_junction if signalized
  down_node     road_closure only: index of edge.to_junction if signalized
  dead_feats    feature indices at feeder whose lanelinks turn into the closed
                edge ("dead demand": vehicles there wait for reopening)
  symptom+/-    nodes whose emp queue rises/falls vs own pre-window median
  t_sym         first sustained (3 steps) crossing of summed + symptom queue
  t_sym_dead    first sustained rise of feeder dead-lane count (ob_traffic)
  t_sym_down    first sustained drop of the downstream node's queue
  divergence    cf: fraction of decisions where variants.empty argmax !=
                canonical argmax, per node group, per phase band
  rollout div   post-guard npz action mismatch can-vs-emp, and time of first
                mismatch (clean causal onset: identical states until then)
  foreign       same cf divergence but for the OOD 'foreign' variant —
                specificity check (does ANY text flip actions, or the right one)
  dead metrics  share of feeder incoming lane_count parked on dead lanes
                (the demand a text-aware controller should discount)
  dq bands      mean queue difference (can-emp) on +symptom nodes in
                [begin,t_sym) / [t_sym,end) / [end,end+300]

Outputs: events.csv, summary.json, lag_report.md under --out.

Usage:
    python3 dev/lag_utility/lag_analysis.py \
        --name fixG_s7 --can-dump .../eval_v2/ev2_fixG_s7_can \
        --emp-dump .../eval_v2/ev2_fixG_s7_emp \
        --phase-map dev/lag_utility/hz4x4_phase_map.json --out DIR
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np

KIND_CODE = {'lane_blockage': 1, 'road_closure': 2, 'global_rain': 3}
ACTION_DT = 10.0
DELTA_Q = 2.0          # min absolute queue change (veh) for a symptom node
SUSTAIN = 3            # consecutive steps needed to call a symptom
PRE_BASE_S = 300.0     # pre-event baseline window
POST_S = 300.0         # recovery window
PRE_SYM_CAP_S = 180.0  # cap the "pre-symptom" band for divergence stats


def load_plan(dump_dir):
    path = Path(dump_dir) / 'event_plan_log.jsonl'
    events = {}
    for line in path.read_text().splitlines():
        r = json.loads(line)
        if r.get('role') == 'eval':
            continue
        events[r['episode']] = r.get('events') or []
    return events


def dump_windows(dump_dir):
    out = {}
    for p in sorted(Path(dump_dir).glob('ep*.npz')):
        idx = int(p.stem[2:])
        d = np.load(p)
        z, t = d['z_task'], d['t']
        present = np.abs(z).sum(axis=(1, 2)) > 0
        out[idx] = None if not present.any() else (float(t[present].min()),
                                                   float(t[present].max()))
    return out


def align_offset(plan_events, wins):
    pw = {ep: (min(e['begin'] for e in evs), max(e['end'] for e in evs))
          for ep, evs in plan_events.items() if evs}
    best, hits = 0, -1
    for off in (0, 1, 2, -1):
        h = sum(1 for ep, w in pw.items()
                if wins.get(ep + off) is not None and abs(wins[ep + off][0] - w[0]) <= 20)
        if h > hits:
            best, hits = off, h
    return best, hits


def load_probe_blocks(probe_jsonl):
    probe_jsonl = Path(probe_jsonl)
    if not probe_jsonl.exists():
        return None
    blocks, cur, last = [], [], -1.0
    for line in probe_jsonl.read_text().splitlines():
        r = json.loads(line)
        if r['simulation_time'] < last and cur:
            blocks.append(cur)
            cur = []
        cur.append(r)
        last = r['simulation_time']
    if cur:
        blocks.append(cur)
    return blocks


def sustained_cross(sig, mask_t, t, base, lim, direction=+1):
    """First time in mask_t where direction*(sig-base) > lim for SUSTAIN steps."""
    d = direction * (sig - base)
    for i in np.where(mask_t)[0]:
        if i + SUSTAIN <= len(d) and (d[i:i + SUSTAIN] > lim).all():
            return float(t[i])
    return float('nan')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--can-dump', required=True)
    ap.add_argument('--emp-dump', required=True)
    ap.add_argument('--phase-map', required=True)
    ap.add_argument('--plan', default='configs/events/plans/hz4x4_random_v1.yml',
                    help='plan yaml carrying edge_id/lane_id targets')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    import yaml
    plan_yaml = yaml.safe_load(Path(args.plan).read_text())
    targets = {}
    for ep in plan_yaml['episodes']:
        for e in ep.get('events') or []:
            targets[e['event_id']] = e

    pmap = json.loads(Path(args.phase_map).read_text())
    node_of = {nid: v['intersection_index'] for nid, v in pmap['nodes'].items()}
    node_name = {v['intersection_index']: k for k, v in pmap['nodes'].items()}
    lane_pos = {v['intersection_index']: {l: i for i, l in
                                          enumerate(v['in_lane_order'])}
                for v in pmap['nodes'].values()}
    phases_of = {v['intersection_index']: v['phases']
                 for v in pmap['nodes'].values()}
    edges = pmap['edges']

    plan = load_plan(args.can_dump)
    wins = dump_windows(args.can_dump)
    offset, hits = align_offset(plan, wins)
    probe = load_probe_blocks(Path(args.can_dump) / 'probe'
                              / 'tarl_struct_cf_records.jsonl')

    rows = []
    for plan_ep, evs in sorted(plan.items()):
        if not evs:
            continue
        dep = plan_ep + offset
        npz_c = Path(args.can_dump) / f'ep{dep:03d}.npz'
        npz_e = Path(args.emp_dump) / f'ep{dep:03d}.npz'
        if not npz_c.exists() or not npz_e.exists():
            continue
        if probe is not None and dep >= len(probe):
            continue
        dc, de = np.load(npz_c), np.load(npz_e)
        t = np.asarray(dc['t'], float)
        qc = np.asarray(dc['queue_per_node'], float)
        qe = np.asarray(de['queue_per_node'], float)
        oc = np.asarray(dc['ob_traffic'], float)
        oe = np.asarray(de['ob_traffic'], float)
        zc = np.asarray(dc['z_task'])
        act_c = np.asarray(dc['action'])
        act_e = np.asarray(de['action'])
        block = probe[dep] if probe is not None else None

        # rollout-level first divergence (post-guard applied actions): before
        # the first mismatch the two runs see identical states, so the onset
        # is causally attributable to the text channel alone.
        diff = (act_c != act_e)
        first_any = diff.any(axis=1).nonzero()[0]
        t_first_div = float(t[first_any[0]]) if first_any.size else float('nan')
        first_per_node = {}
        for n in range(act_c.shape[1]):
            idx = diff[:, n].nonzero()[0]
            if idx.size:
                first_per_node[n] = float(t[idx[0]])

        for ev in evs:
            kind = ev['kind']
            b, en = ev['begin'], ev['end']
            win = (t >= b) & (t <= en)
            if not win.any():
                continue
            kc = KIND_CODE[kind]
            bound = sorted({int(n) for n in np.where(
                ((zc[:, :, 0] > 0.5) & (zc[:, :, 1] == kc))[win].any(axis=0))[0]})
            feeder = down = None
            dead_feats = []
            n_ph_serve_e = None
            if kind == 'road_closure':
                eid = targets[ev['event_id']]['edge_id']
                feeder = node_of.get(edges[eid]['from'])
                down = node_of.get(edges[eid]['to'])
                edge_lanes = set(edges[eid]['lanes'])
                if feeder is not None:
                    ph = phases_of.get(feeder, [])
                    hits_l = set()
                    for links in ph:
                        for fr, to in links:
                            if to in edge_lanes:
                                hits_l.add(fr)
                    lp = lane_pos.get(feeder, {})
                    dead_feats = sorted({lp[l] for l in hits_l if l in lp})
                    n_ph_serve_e = int(sum(
                        any(to in edge_lanes for _, to in links)
                        for links in ph))

            pre = (t >= b - PRE_BASE_S) & (t < b)
            post = (t > en) & (t <= en + POST_S)
            base = np.median(qe[pre], axis=0) if pre.any() else np.zeros(16)
            sd = qe[pre].std(axis=0) if pre.any() else np.zeros(16)
            dq = qe[win].mean(axis=0) - base
            thr = np.maximum(DELTA_Q, 2 * sd)
            sym_pos = sorted(np.where(dq > thr)[0].tolist())
            sym_neg = sorted(np.where(dq < -thr)[0].tolist())

            # symptom clocks are constrained to the event window `win`;
            # a later crossing is reported separately (post) so that a
            # dissipation wave after reopening is not mistaken for symptom
            # onset.
            t_sym = float('nan')
            if sym_pos:
                sig = qe[:, sym_pos].sum(axis=1)
                bs = base[sym_pos].sum()
                lim = max(3.0, float(2 * qe[pre][:, sym_pos].sum(axis=1).std()))
                t_sym = sustained_cross(sig, win, t, bs, lim, +1)
            t_sym_dead = float('nan')
            t_sym_dead_post = float('nan')
            if dead_feats:
                sig = oe[:, feeder, dead_feats].sum(axis=1)
                ob_pre = oe[pre][:, feeder, dead_feats].sum(axis=1)
                bs = float(np.median(ob_pre)) if ob_pre.size else 0.0
                lim = max(3.0, float(2 * ob_pre.std()))
                t_sym_dead = sustained_cross(sig, win, t, bs, lim, +1)
                t_sym_dead_post = sustained_cross(sig, post, t, bs, lim, +1)
            t_sym_down = float('nan')
            if down is not None:
                sig = qe[:, down]
                sd_dn = float(sd[down])
                t_sym_down = sustained_cross(
                    sig, win, t, base[down], max(DELTA_Q, 2 * sd_dn), -1)

            pre_end = b if math.isnan(t_sym) else min(t_sym, b + PRE_SYM_CAP_S)
            pre_sym = (t >= b) & (t < pre_end)
            pre_ev = (t >= b - 120.0) & (t < b)

            def diverge(mask_t, node_set, variant='empty'):
                if block is None or not node_set:
                    return float('nan')
                vals = []
                for i in np.where(mask_t)[0]:
                    if i >= len(block):
                        break
                    rec = block[i]
                    va = np.asarray(rec['variants'][variant]['action'])
                    ca = np.asarray(rec['variants']['canonical']['action'])
                    vals.append([(va[n] != ca[n]) for n in node_set])
                return (float(np.mean(vals)) * len(node_set)
                        if vals else 0.0)

            def rollout_div(mask_t, node_set):
                if not node_set:
                    return float('nan')
                idx = np.where(mask_t)[0]
                if not idx.size:
                    return float('nan')
                d = diff[idx][:, node_set]
                return float(d.mean()) * len(node_set)

            def guard_fired(mask_t):
                if block is None:
                    return 0, 0
                n = tot = 0
                for i in np.where(mask_t)[0]:
                    if i >= len(block):
                        break
                    ca = np.asarray(block[i]['variants']['canonical']['action'])
                    n += int((act_c[i] != ca).sum())
                    tot += 16
                return n, tot

            def dead_stats():
                """Dead-lane demand share at the feeder on the empty rollout."""
                out = {'dead_share_emp_inwin': float('nan'),
                       'dead_peak_emp_inwin': float('nan'),
                       'dead_mean_emp_inwin': float('nan'),
                       'dq_dead_inwin': float('nan')}
                if not dead_feats:
                    return out
                w = np.where(win)[0]
                tot = oe[w][:, feeder, :].sum(axis=1)
                ded = oe[w][:, feeder, dead_feats].sum(axis=1)
                out.update({
                    'dead_share_emp_inwin': float((ded / np.maximum(tot, 1)).mean()),
                    'dead_peak_emp_inwin': float(ded.max()),
                    'dead_mean_emp_inwin': float(ded.mean()),
                    'dq_dead_inwin': float(
                        (oc[:, feeder, dead_feats].sum(axis=1)
                         - oe[:, feeder, dead_feats].sum(axis=1))[w].mean()),
                })
                return out

            gf_w, gt_w = guard_fired(win)
            row = {
                'episode': plan_ep, 'kind': kind, 'event_id': ev['event_id'],
                'begin': b, 'end': en,
                'bound_nodes': bound, 'feeder_node': feeder, 'down_node': down,
                'dead_feats': dead_feats, 'n_phases_serving_edge': n_ph_serve_e,
                'sym_pos': sym_pos, 'sym_neg': sym_neg,
                't_sym_minus_begin': None if math.isnan(t_sym) else t_sym - b,
                't_dead_minus_begin': (None if math.isnan(t_sym_dead)
                                       else t_sym_dead - b),
                't_dead_post_minus_end': (None if math.isnan(t_sym_dead_post)
                                          else t_sym_dead_post - en),
                't_down_minus_begin': (None if math.isnan(t_sym_down)
                                       else t_sym_down - b),
                'bound_is_symptom': bool(set(bound) & set(sym_pos)),
                'feeder_is_bound': (feeder in bound) if feeder is not None else None,
                'feeder_is_symptom': (feeder in sym_pos) if feeder is not None else None,
                't_first_div_minus_begin': (None if math.isnan(t_first_div)
                                            else t_first_div - b),
                't_first_div_feeder_minus_begin': (
                    first_per_node[feeder] - b
                    if feeder in first_per_node else None),
                'div_bound_presym': diverge(pre_sym, bound),
                'div_bound_win': diverge(win, bound),
                'div_bound_preev': diverge(pre_ev, bound),
                'div_symp_presym': diverge(pre_sym, sym_pos),
                'div_all_presym': diverge(pre_sym, list(range(16))),
                'div_feeder_presym': (diverge(pre_sym, [feeder])
                                      if feeder is not None else float('nan')),
                'div_foreign_bound_presym': diverge(pre_sym, bound, 'foreign'),
                'div_foreign_all_presym': diverge(pre_sym, list(range(16)),
                                                'foreign'),
                'rdiv_feeder_presym': rollout_div(pre_sym, [feeder])
                                      if feeder is not None else float('nan'),
                'rdiv_feeder_win': rollout_div(win, [feeder])
                                   if feeder is not None else float('nan'),
                'rdiv_all_presym': rollout_div(pre_sym, list(range(16))),
                'guard_fired_inwin': gf_w, 'guard_total_inwin': gt_w,
                'dq_symp_presym': (float((qc - qe)[pre_sym][:, sym_pos].mean())
                                   if sym_pos and pre_sym.any() else float('nan')),
                'dq_symp_inwin': (float((qc - qe)[win][:, sym_pos].mean())
                                  if sym_pos else float('nan')),
                'dq_symp_post': (float((qc - qe)[post][:, sym_pos].mean())
                                 if sym_pos and post.any() else float('nan')),
                'dq_feeder_inwin': (float((qc - qe)[win][:, feeder].mean())
                                    if feeder is not None else float('nan')),
                'dq_feeder_post': (float((qc - qe)[post][:, feeder].mean())
                                   if feeder is not None and post.any()
                                   else float('nan')),
                'dq_all_inwin': float((qc - qe)[win].mean()),
                'dq_all_post': (float((qc - qe)[post].mean())
                                if post.any() else float('nan')),
            }
            row.update(dead_stats())
            rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import csv as _csv
    keys = list(rows[0].keys()) if rows else []
    with (out / 'events.csv').open('w', newline='') as fh:
        w = _csv.writer(fh)
        w.writerow(keys)
        for r in rows:
            w.writerow([json.dumps(r[k]) if isinstance(r[k], (list, dict))
                        else ('' if r[k] is None or
                              (isinstance(r[k], float) and math.isnan(r[k]))
                              else r[k]) for k in keys])

    def agg(vals):
        v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
        if not v:
            return None
        v = np.asarray(v, float)
        return {'n': int(v.size), 'mean': float(v.mean()),
                'median': float(np.median(v)), 'sd': float(v.std(ddof=1)) if v.size > 1 else 0.0}

    by_kind = {}
    for kind in KIND_CODE:
        sub = [r for r in rows if r['kind'] == kind]
        by_kind[kind] = {
            'n_events': len(sub),
            'symptom_lag_s': agg([r['t_sym_minus_begin'] for r in sub]),
            'dead_lag_inwin_s': agg([r['t_dead_minus_begin'] for r in sub]),
            'dead_lag_post_s': agg([r['t_dead_post_minus_end'] for r in sub]),
            'down_lag_s': agg([r['t_down_minus_begin'] for r in sub]),
            't_first_div': agg([r['t_first_div_minus_begin'] for r in sub]),
            'div_bound_presym': agg([r['div_bound_presym'] for r in sub]),
            'div_bound_inwin': agg([r['div_bound_win'] for r in sub]),
            'div_bound_preev': agg([r['div_bound_preev'] for r in sub]),
            'div_symp_presym': agg([r['div_symp_presym'] for r in sub]),
            'div_all_presym': agg([r['div_all_presym'] for r in sub]),
            'div_foreign_bound_presym': agg([r['div_foreign_bound_presym']
                                             for r in sub]),
            'rdiv_all_presym': agg([r['rdiv_all_presym'] for r in sub]),
            'dq_symp_presym': agg([r['dq_symp_presym'] for r in sub]),
            'dq_symp_inwin': agg([r['dq_symp_inwin'] for r in sub]),
            'dq_symp_post': agg([r['dq_symp_post'] for r in sub]),
            'dq_all_inwin': agg([r['dq_all_inwin'] for r in sub]),
            'dq_all_post': agg([r['dq_all_post'] for r in sub]),
        }
    closures = [r for r in rows if r['kind'] == 'road_closure'
                and r['feeder_node'] is not None]
    bind_gap = [r for r in rows if r['kind'] == 'road_closure'
                and not r['bound_nodes']]
    summary = {
        'name': args.name, 'offset': offset, 'align_hits': hits,
        'n_events': len(rows), 'by_kind': by_kind,
        'closures_with_controlled_feeder': len(closures),
        'closures_bound_to_feeder': sum(1 for r in closures if r['feeder_is_bound']),
        'closures_feeder_symptomatic': sum(1 for r in closures if r['feeder_is_symptom']),
        'closures_no_bound_node': len(bind_gap),
        'closure_dead_share_emp': agg([r.get('dead_share_emp_inwin')
                                       for r in closures]),
        'closure_dead_peak_emp': agg([r.get('dead_peak_emp_inwin')
                                      for r in closures]),
        'closure_dq_dead_inwin': agg([r.get('dq_dead_inwin') for r in closures]),
        'closure_dq_feeder_inwin': agg([r['dq_feeder_inwin'] for r in closures]),
        'closure_dq_feeder_post': agg([r['dq_feeder_post'] for r in closures]),
        'rdiv_feeder_presym': agg([r['rdiv_feeder_presym'] for r in closures]),
        'rdiv_feeder_inwin': agg([r['rdiv_feeder_win'] for r in closures]),
        'div_feeder_presym_cf': agg([r['div_feeder_presym'] for r in closures]),
        't_first_div_feeder': agg([r['t_first_div_feeder_minus_begin']
                                   for r in closures]),
        'guard_share_inwin': agg([r['guard_fired_inwin'] / r['guard_total_inwin']
                                  for r in rows if r['guard_total_inwin']]),
    }
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))

    md = [f'# lag-window analysis — {args.name}', '',
          f'- events analysed: {len(rows)} (dump/plan offset {offset}, {hits} hits)',
          f'- closures with controlled feeder: {summary["closures_with_controlled_feeder"]}, '
          f'of which feeder bound: {summary["closures_bound_to_feeder"]}, '
          f'feeder symptomatic: {summary["closures_feeder_symptomatic"]}; '
          f'closures with no bound node: {summary["closures_no_bound_node"]}',
          '']
    for kind, s in by_kind.items():
        md += [f'## {kind} (n={s["n_events"]})', '',
               '| metric | n | mean | median | sd |', '|---|---|---|---|---|']
        for k, v in s.items():
            if k == 'n_events' or v is None:
                continue
            md.append(f'| {k} | {v["n"]} | {v["mean"]:+.3f} | {v["median"]:+.3f} | {v["sd"]:.3f} |')
        md.append('')
    md += ['## closures: feeder-side detail', '',
           f'- dead-demand share at feeder (emp, in-window): {json.dumps(summary["closure_dead_share_emp"])}',
           f'- dead-demand peak (emp, in-window, veh on dead lanes): {json.dumps(summary["closure_dead_peak_emp"])}',
           f'- dq feeder in-window (can-emp): {json.dumps(summary["closure_dq_feeder_inwin"])}',
           f'- dq feeder post-window (can-emp): {json.dumps(summary["closure_dq_feeder_post"])}',
           f'- rollout divergence at feeder pre-symptom: {json.dumps(summary["rdiv_feeder_presym"])}',
           f'- rollout divergence at feeder in-window: {json.dumps(summary["rdiv_feeder_inwin"])}',
           f'- first rollout divergence at feeder (s after begin): {json.dumps(summary["t_first_div_feeder"])}',
           f'- guard-fired share in window: {json.dumps(summary["guard_share_inwin"])}', '']
    (out / 'lag_report.md').write_text('\n'.join(md))
    print(json.dumps(summary, indent=1)[:2500])
    print(f'\nwrote {out}/events.csv summary.json lag_report.md')


if __name__ == '__main__':
    main()
