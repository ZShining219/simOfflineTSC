"""Traffic-perturbation injection + event broadcast for SUMO worlds.

Shared by ``dev/takeover_probe`` (offline fork probe) and the live
``BroadcastEngine`` (interactive broadcast).  All specs are plain strings so
the same schedule can drive headless runs and the UI.

Supported specs
---------------
``lane_close:LANE:START:DUR``
    Close LANE to all vehicle classes between START and START+DUR sim
    seconds (reopened afterwards).
``surge:FROM_EDGE:TO_EDGE:START:DUR:PERIOD[:OBJECTIVE]``
    Inject one extra vehicle every PERIOD sim seconds departing on
    FROM_EDGE routed toward TO_EDGE during [START, START+DUR).
    TO_EDGE is a preference hint; if unreachable the first reachable edge
    is used instead.  Optional OBJECTIVE declares the governance
    priority reported with the event: ``global`` (default, overall
    network flow) or ``focus`` (drain the incident approach first).

Each inject produces an :class:`EventSpec` that doubles as the broadcast
source: the event *report* becomes available ``report_delay`` seconds
after physical onset, describing only what is knowable at report time
(type / location / severity / estimated duration) — never the resulting
queue or control outcome.
"""

from __future__ import annotations


def parse_injects(specs):
    """Parse --inject spec strings into mutable schedule dicts."""
    injects = []
    for spec in specs or []:
        kind, _, rest = str(spec).partition(':')
        if kind == 'lane_close':
            lane, _, times = rest.partition(':')
            start_s, _, dur_s = times.partition(':')
            injects.append({
                'kind': kind, 'lane': str(lane),
                'edge': _edge_of_lane(str(lane)),
                'start_t': float(start_s),
                'end_t': float(start_s) + float(dur_s),
                'applied': False, 'reverted': False,
            })
        elif kind == 'surge':
            parts = rest.split(':')
            if len(parts) not in (5, 6):
                raise ValueError(
                    f'surge needs FROM:TO:START:DUR:PERIOD[:OBJECTIVE], '
                    f'got: {spec}')
            objective = parts[5] if len(parts) == 6 else 'global'
            if objective not in ('global', 'focus'):
                raise ValueError(f'unknown objective: {objective}')
            injects.append({
                'kind': kind,
                'edge_from': parts[0], 'edge_to': parts[1],
                'start_t': float(parts[2]),
                'end_t': float(parts[2]) + float(parts[3]),
                'period': float(parts[4]),
                'objective': objective,
                'route_id': None, 'next_depart': float(parts[2]),
                'count': 0,
            })
        else:
            raise ValueError(f'unsupported inject kind: {kind}')
    return injects


def _edge_of_lane(lane):
    """'road_0_1_0_1' -> 'road_0_1_0' (drop trailing lane index)."""
    return lane.rsplit('_', 1)[0] if lane.rsplit('_', 1)[-1].isdigit() else lane


def apply_injects(world, injects, now):
    """Advance pending injects for a libsumo-backed ``world_sumo.World``."""
    for inj in injects:
        if inj['kind'] == 'surge':
            if not inj['start_t'] <= now < inj['end_t']:
                continue
            if now + 1e-6 < inj['next_depart']:
                continue
            if inj['route_id'] is None:
                edges = None
                for cand in [inj['edge_to']] + sorted(
                        e for e in world.eng.edge.getIDList()
                        if not str(e).startswith(':')):
                    try:
                        route = world.eng.simulation.findRoute(
                            inj['edge_from'], cand)
                    except Exception:
                        continue
                    if list(route.edges):
                        edges = list(route.edges)
                        inj['edge_to_resolved'] = cand
                        break
                if not edges:
                    # e.g. inject spec written for a different scene —
                    # mark dead rather than killing the session
                    inj['route_id'] = 'INVALID'
                    continue
                inj['route_id'] = f"INJ_ROUTE_{inj['edge_from']}"
                world.eng.route.add(inj['route_id'], edges)
            if inj['route_id'] == 'INVALID':
                continue
            vid = f"INJ_{inj['edge_from']}_{inj['count']}"
            world.eng.vehicle.add(vid, inj['route_id'], departLane='best')
            inj['count'] += 1
            inj['next_depart'] = now + inj['period']
            continue
        if not inj['applied'] and now >= inj['start_t']:
            if inj['kind'] == 'lane_close':
                try:
                    world.eng.lane.setAllowed(inj['lane'], [])
                except Exception:
                    pass  # lane absent in this scene — skip
            inj['applied'] = True
        elif inj['applied'] and not inj['reverted'] and now >= inj['end_t']:
            if inj['kind'] == 'lane_close':
                try:
                    world.eng.lane.setAllowed(inj['lane'], ['all'])
                except Exception:
                    pass
            inj['reverted'] = True


# ---------------------------------------------------------------------------
# Event broadcast: what an incident report can honestly say at report time
# ---------------------------------------------------------------------------

EVENT_KIND_LABEL = {
    'surge': '突发车流',
    'lane_close': '占道封闭',
    'none': '正常交通',
}
EVENT_KINDS = list(EVENT_KIND_LABEL)


def event_specs(injects):
    """Normalize inject dicts into event records for text generation."""
    events = []
    for inj in injects or []:
        edge = inj.get('edge') or inj.get('edge_from') or ''
        if inj['kind'] == 'surge':
            severity = min(3, max(1, round(4.0 / inj['period'])))
        else:
            severity = 2
        events.append({
            'kind': inj['kind'],
            'edge': edge,
            'start_t': inj['start_t'],
            'end_t': inj['end_t'],
            'severity': severity,
            'est_duration_s': inj['end_t'] - inj['start_t'],
            'objective': inj.get('objective', 'global'),
        })
    return events


def active_event(events, now, report_delay=30.0, recovery_tail=120.0):
    """Event report available at ``now``.

    The report exists only after onset + report_delay; it carries onset
    time, location, severity and estimated duration — never outcomes.
    Between end and end+recovery_tail the report degrades to '恢复中'.
    Returns (fields_dict, sentence).
    """
    live = [e for e in events if e['start_t'] + report_delay <= now]
    if live:
        # With concurrent events the broadcast names the prioritised
        # (focus-declared) incident first; otherwise the newest one.
        focus_live = [e for e in live if e.get('objective') == 'focus']
        ev = max(focus_live or live, key=lambda e: e['start_t'])
        remaining = max(0.0, ev['end_t'] - now)
        recovering = now >= ev['end_t']
        fields = {
            'kind': 'none' if (not recovering and now >= ev['end_t'])
                    else ev['kind'],
            'edge': ev['edge'],
            'active': not (now >= ev['end_t'] + recovery_tail),
            'recovering': recovering,
            'onset_age_s': max(0.0, now - ev['start_t']),
            'severity': ev['severity'],
            'est_remaining_s': remaining if not recovering else 0.0,
            'objective': ev.get('objective', 'global'),
        }
        if now >= ev['end_t'] + recovery_tail:
            return _normal_report()
        label = EVENT_KIND_LABEL[ev['kind']]
        obj_hint = (
            '，调度要求：优先消化该进口积压'
            if fields['objective'] == 'focus'
            else '，调度要求：保障全网通行效率')
        if recovering:
            sentence = (
                f"{label}已结束：{ev['edge']}，残余排队消散中{obj_hint}"
            )
        else:
            sentence = (
                f"{label}：{ev['edge']} 进口，已持续 "
                f"{fields['onset_age_s']:.0f}s，预计再持续 "
                f"{remaining:.0f}s，强度等级 {ev['severity']}{obj_hint}"
            )
        return fields, sentence
    return _normal_report()


def _normal_report():
    return ({
        'kind': 'none', 'edge': '', 'active': False,
        'recovering': False, 'onset_age_s': 0.0,
        'severity': 0, 'est_remaining_s': 0.0,
        'objective': 'global',
    }, '正常交通：无事件播报')
