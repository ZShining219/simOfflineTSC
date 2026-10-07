#!/usr/bin/env python3
"""w34_2 P2 — offline binding audit for eval_v2 plan episodes.

Replays ReportGrounder + TextEntityBinder semantics OFFLINE (no simulation):
for every event in plan episodes 0..39 of hz4x4_random_v1 it classifies the
z_task delivery outcome as one of

    bound          target lanes observed by >=1 controlled node (lane-level)
    fallback_bind  road_closure unbindable at lane level but bound upstream
                   by closure_binding='nearest_upstream' (closure_bind arm)
    gap            road_closure whose lanes no controlled node observes and
                   the legacy binder drops it (canonical/fixG/fixR/a3s arms)
    masked         bound report loses its node to an earlier-report_id event
                   sharing the node during overlap (first-report-wins)

Node observation layout approximation: a controlled node observes the motor-
vehicle, non-internal lanes of edges whose to_junction is that node — the
same lane set LaneVehicleGenerator(in_only) builds for hz4x4.

Then joins eval_v2_paired per-episode diffs and re-splits canonical-minus-empty
by event kind x delivery class, and by delivered-vs-gapped episodes.

Outputs: analysis/att_entity_004/p2_binding_audit/{audit.json,audit.md}
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

NET = ROOT / 'data/raw_data/hangzhou_4x4_gudang_18041610_1h/hangzhou_4x4_gudang_18041610_1h.net.xml'
PLAN = ROOT / 'configs/events/plans/hz4x4_random_v1.yml'
PAIRED_CSV = ROOT / 'data/output_data/analysis/att_entity_004/eval_v2_paired/paired_episodes.csv'
CLOSURE_LOG = ROOT / 'data/output_data/tsc/sumo_tarl_attention/hz4x4/att004_closure_bind_s7/closure_bind_log.jsonl'
OUT = ROOT / 'data/output_data/analysis/att_entity_004/p2_binding_audit'

MAX_EP = 40           # eval_v2 uses plan episodes 0..39
MAX_HOPS = 4


def load_catalog():
    """Replicate SumoEventRuntime's lane catalog without a Schedule."""
    import sumolib
    net = sumolib.net.readNet(str(NET), withInternal=True)
    tls = tuple(sorted(t.getID() for t in net.getTrafficLights()))
    lanes = {}
    turn_names = {'l': 'left', 'r': 'right', 's': 'straight', 't': 'uturn', 'L': 'left', 'R': 'right', 's': 'straight', 'T': 'uturn'}
    motor = ('passenger', 'taxi', 'bus', 'coach', 'delivery', 'truck',
             'trailer', 'motorcycle', 'moped', 'evehicle', 'private',
             'emergency', 'authority', 'army', 'vip', 'custom1', 'custom2')
    for edge in net.getEdges(withInternal=True):
        for lane in edge.getLanes():
            shape = lane.getShape()
            dx, dy = shape[-1][0]-shape[0][0], shape[-1][1]-shape[0][1]
            direction = (('eastbound' if dx > 0 else 'westbound') if abs(dx) > abs(dy)
                         else ('northbound' if dy > 0 else 'southbound'))
            lanes[lane.getID()] = {
                'lane_id': lane.getID(), 'edge_id': edge.getID(),
                'lane_index': lane.getIndex(), 'length': lane.getLength(),
                'from_junction': edge.getFromNode().getID(),
                'to_junction': edge.getToNode().getID(),
                'direction': direction,
                'movements': sorted({turn_names.get(c.getDirection(), c.getDirection())
                                     for c in lane.getOutgoing()}),
                'internal': edge.isSpecial(),
                'motor_vehicle_lane': any(lane.allows(c) for c in motor),
            }
    return tls, lanes


def load_plan():
    import yaml
    data = yaml.safe_load(PLAN.read_text())
    return {row['episode']: row['events'] for row in data['episodes']}


def audit(plan, tls, lanes):
    tls = set(tls)
    obs_lanes = {lid for lid, l in lanes.items()
                 if l['motor_vehicle_lane'] and not l['internal']
                 and l['to_junction'] in tls}
    lane_node = {lid: l['to_junction'] for lid, l in lanes.items()
                 if lid in obs_lanes}
    # upstream topology for the fallback check
    edge_from = {l['edge_id']: l['from_junction'] for l in lanes.values()}
    preds = {}
    for l in lanes.values():
        preds.setdefault(l['to_junction'], set()).add(l['from_junction'])
    rank = {n: i for i, n in enumerate(sorted(tls))}

    def nearest_upstream(edge):
        start = edge_from.get(edge)
        if start is None:
            return None
        seen, frontier = set(), {start}
        for _ in range(MAX_HOPS):
            hits = [j for j in frontier if j in rank]
            if hits:
                return min(hits, key=lambda j: rank[j])
            seen |= frontier
            nxt = set()
            for j in frontier:
                nxt |= preds.get(j, set())
            frontier = nxt - seen
            if not frontier:
                return None
        return None

    episodes = {}
    for ep in range(MAX_EP):
        events = plan.get(ep) or []
        rows = []
        for e in events:
            kind = e['kind']
            rec = {'event_id': e['event_id'], 'kind': kind,
                   'begin': e['begin'], 'end': e['end']}
            if kind == 'global_rain':
                rec.update(class_='bound', nodes=sorted(tls),
                           detail='global scope')
            elif kind == 'lane_blockage':
                node = lane_node.get(e['lane_id'])
                if node is not None:
                    rec.update(class_='bound', nodes=[node],
                               detail=f"lane->{node}")
                else:
                    rec.update(class_='gap', nodes=[],
                               detail=f"lane {e['lane_id']} unobserved")
            elif kind == 'road_closure':
                edge_lanes = [lid for lid, l in lanes.items()
                              if l['edge_id'] == e['edge_id']
                              and l['motor_vehicle_lane']]
                nodes = sorted({lane_node[l] for l in edge_lanes
                                if l in lane_node})
                if nodes:
                    rec.update(class_='bound', nodes=nodes,
                               detail=f"{len(edge_lanes)} lanes -> {nodes}")
                else:
                    up = nearest_upstream(e['edge_id'])
                    rec.update(class_='gap', nodes=[], upstream=up,
                               detail=f"unobserved edge; upstream={up}")
            rows.append(rec)
        # masking: same-node overlap between active reports; sorted report_id wins
        masked = set()
        evs = [r for r in rows if r['class_'] in ('bound',)]
        for i, a in enumerate(evs):
            for b in evs[i+1:]:
                if a['begin'] < b['end'] and b['begin'] < a['end']:
                    shared = set(a['nodes']) & set(b['nodes'])
                    if shared:
                        loser = max(a['event_id'], b['event_id'])
                        masked.add(loser)
        for r in rows:
            if r['event_id'] in masked:
                r['class_'] = 'masked'
                r['detail'] += ' (masked by earlier report_id)'
        episodes[ep] = rows
    return episodes, obs_lanes


def main():
    tls, lanes = load_catalog()
    plan = load_plan()
    audit_rows, obs_lanes = audit(plan, tls, lanes)

    # closure_bind training-run fallback evidence (what the arm actually bound)
    closure_log = []
    if CLOSURE_LOG.exists():
        seen = set()
        for line in CLOSURE_LOG.read_text().splitlines():
            r = json.loads(line)
            k = (r['report_id'], r.get('edge_id'), r.get('bound_node'))
            if k not in seen:
                seen.add(k)
                closure_log.append(r)

    # join paired diffs: record episode = plan episode + 1
    diffs = defaultdict(list)
    for row in csv.DictReader(PAIRED_CSV.open()):
        plan_ep = int(row['episode']) - 1
        rows = audit_rows.get(plan_ep, [])
        kinds = {r['kind'] for r in rows}
        classes = {r['class_'] for r in rows}
        for metric in ('travel_time', 'queue', 'win_queue_in', 'win_queue_out'):
            v = row.get(metric)
            if v:
                key = (metric,
                       '+'.join(sorted(kinds)) or 'no_event',
                       'has_gap' if 'gap' in classes else
                       ('has_masked' if 'masked' in classes else
                        ('all_bound' if classes else 'no_event')))
                diffs[key].append((row['pair'], float(v)))

    strata = {}
    for (metric, kind, cls), vals in sorted(diffs.items()):
        xs = [v for _, v in vals]
        strata.setdefault(kind, {}).setdefault(cls, {})[metric] = {
            'n': len(xs), 'mean': sum(xs)/len(xs),
            'pairs': sorted({p for p, _ in vals})}

    # per-episode delivery table
    table = []
    for ep, rows in audit_rows.items():
        for r in rows:
            table.append({'episode': ep, **r})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'audit.json').write_text(json.dumps(
        {'episodes': table, 'strata': strata,
         'closure_bind_fallbacks': closure_log,
         'observed_lane_count': len(obs_lanes)}, indent=1))

    md = ['# P2 binding audit — eval_v2 plan episodes 0..39', '']
    md.append(f'nodes={len(tls)} observed_lanes={len(obs_lanes)}')
    md.append('')
    md.append('## per-event delivery')
    md.append('| ep | event | kind | class | nodes | detail |')
    md.append('|---|---|---|---|---|---|')
    for r in table:
        md.append(f"| {r['episode']} | {r['event_id']} | {r['kind']} | "
                  f"{r['class_']} | {','.join(r.get('nodes') or [])} | {r['detail']} |")
    md.append('')
    md.append('## closure_bind training-run fallback evidence')
    for r in closure_log[:10]:
        md.append(f"- {r['report_id']} edge={r.get('edge_id')} -> {r.get('bound_node')}")
    md.append('')
    md.append('## canonical-minus-empty stratified by kind x class')
    md.append('| kind | class | metric | n | mean diff |')
    md.append('|---|---|---|---|---|')
    for kind, clsd in strata.items():
        for cls, mets in clsd.items():
            for metric, s in mets.items():
                md.append(f"| {kind} | {cls} | {metric} | {s['n']} | {s['mean']:+.4f} |")
    (OUT / 'audit.md').write_text('\n'.join(md) + '\n')
    print(f'wrote {OUT}/audit.{{json,md}}; events={len(table)}; '
          f'classes={json.dumps({c: sum(1 for r in table if r["class_"]==c) for c in ("bound","gap","masked")}, sort_keys=True)}')


if __name__ == '__main__':
    main()
