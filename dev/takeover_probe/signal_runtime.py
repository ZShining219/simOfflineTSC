"""Phase-level episode driver for scene-guided signal agents.

Single-layer counterpart of ``router_runtime``: every ``interval``
seconds the agent (or a builtin controller) picks ONE signal phase,
held until the next decision tick.  No child controllers, no window
abstraction — the scene encoder's z feeds the Q head directly.

``agent``      : SceneSignalAgent — snapshot -> q -> phase.
``controller`` : builtin/snapshot controller with ``decide(rank)`` —
                 the conventional-policy comparison arms and the
                 "attention OFF" reference (they see no scene text).

Returns (transitions, records):
  transitions  (s, a, r, s', done) per decision tick — DQN updates
  records      per-tick rows (phase/queue/throughput/text/attn) for
               metrics and audit
"""

from __future__ import annotations

import numpy as np

from hitl.live.injects import apply_injects


def run_signal_episode(world, fb, ranks, injects=None, agent=None,
                       controller=None, interval=10, episode_seconds=3600,
                       epsilon=0.0, rng=None, text_override=None,
                       reward_scale=50.0, record_snaps=False,
                       switch_penalty=0.0, q_base_fn=None,
                       orig_reward=False, scene_w=0.0):
    """``record_snaps``: take snapshots + emit transitions even when a
    builtin ``controller`` drives — used to collect teacher
    demonstrations for BC warm-start.  ``switch_penalty``: per phase-change
    reward cost (yellow/all-red time is real capacity loss — without it a
    learned policy can collapse into every-tick toggling).
    ``q_base_fn``: callable -> per-phase values of a frozen base
    controller at the current tick (residual policies ride on it);
    sampled once per decision tick and stored as ``qb``/``qbp`` inside
    each transition.
    ``orig_reward``: replace the probe shaping with the deployed DQN's
    own objective — -12 x mean lane_waiting_count over the intersection's
    incoming lanes, sampled at the tick boundary (LaneVehicleGenerator
    in_only=True, average='all', negative=True, x12).  Keeps the TD
    target identical to what the base converged on, so fine-tuning does
    not rewrite its Q calibration.
    ``scene_w``: SIGMA-style intent utility — while a report is live the
    internal intent vector (``fb.intent_vec``) adds
    ``-scene_w * mean(queue over intent lanes, theta-weighted)`` to the
    reward.  The report's edge identity then defines WHERE the extra
    penalty applies: a policy can only optimise the term by consuming
    the text channel, so scene conditioning stops being optional."""
    injects = injects or []
    held = {
        rank: int(inter.virtual_phase)
        for rank, inter in enumerate(world.intersections)
    }
    transitions, records = [], []
    next_decision_t = float(world.get_current_time())
    open_tick = None          # {'snap','a','q','w','t'}
    interval_queue = 0.0
    interval_switches = 0.0
    in_lanes = []
    if orig_reward and ranks:
        inter = world.intersections[ranks[0]]
        in_lanes = [l for r in inter.in_roads
                    for l in inter.road_lane_mapping.get(r, [])]

    def _reward(now):
        if orig_reward:
            lwc = world.get_info('lane_waiting_count') or {}
            vals = [float(lwc.get(l, 0.0)) for l in in_lanes]
            r = -12.0 * float(np.mean(vals)) if vals else 0.0
        else:
            r = -(interval_queue / max(1.0, now - open_tick['t'])) \
                / reward_scale - interval_switches
        if scene_w:
            fields_now, _ = fb.event_fields(now)
            th = fb.intent_vec(fields_now)
            if th.any():
                # theta-normalised mean queue over intent lanes:
                # named edge weight = 1, corridor edges = coord_w —
                # identical to the v1 edge-mean when coord_w == 0.
                w_lane = (th / th.max())[fb.lane_edge_idx]
                lq = world.get_info('lane_waiting_count') or {}
                q = np.array([lq.get(l, 0.0) for l in fb.lanes])
                r -= scene_w * float(
                    (q * w_lane).sum() / w_lane.sum())
        return r
    while True:
        now = float(world.get_current_time())
        if episode_seconds and now >= episode_seconds:
            break
        if now + 1e-6 >= next_decision_t:
            # close the previous tick: r over its interval, s' = now
            snap = fb.snapshot(now, text_override=text_override) \
                if (agent is not None or record_snaps) else None
            qb = q_base_fn() if q_base_fn is not None else None
            if open_tick is not None:
                r = _reward(now)
                if agent is not None or record_snaps:
                    transitions.append({
                        's': open_tick['snap'], 'a': open_tick['a'],
                        'r': r, 'sp': snap, 'done': False,
                        'qb': open_tick['qb'], 'qbp': qb,
                    })
                open_tick['r'] = r
            # decide for the new tick
            qvals, attn_w, sentence = None, None, None
            prev_held = held[ranks[0]] if ranks else 0
            if agent is not None:
                from dev.takeover_probe.dispatch_model import collate
                batch = collate([snap])
                a, qvals, attn_w = agent.act(
                    batch, epsilon=epsilon, rng=rng, q_base=qb)
                _, sentence = fb.event_fields(now)
                for rank in ranks:
                    held[rank] = a
            else:
                a = int(controller.decide(0))
                for rank in ranks:
                    held[rank] = a
            interval_switches = switch_penalty if a != prev_held else 0.0
            lane_queue = world.get_info('lane_waiting_count') or {}
            records.append({
                't': now,
                'phase': int(a),
                'total_queue': int(sum(lane_queue.values())),
                'lane_queue': {str(k): int(v)
                               for k, v in lane_queue.items()},
                'throughput': int(world.get_info('throughput') or 0),
                'text': sentence,
                'q_values': qvals,
                'attn_weights': (
                    attn_w.detach().cpu().numpy().ravel().tolist()
                    if attn_w is not None else None),
            })
            open_tick = {'snap': snap, 'a': a, 't': now, 'qb': qb}
            interval_queue = 0.0
            next_decision_t = now + interval
        lane_queue = world.get_info('lane_waiting_count') or {}
        interval_queue += float(sum(lane_queue.values()))
        apply_injects(world, injects, now)
        world.step([
            held.get(rank, int(inter.virtual_phase))
            for rank, inter in enumerate(world.intersections)
        ])
    if open_tick is not None and (agent is not None or record_snaps):
        r = _reward(now)
        transitions.append({
            's': open_tick['snap'], 'a': open_tick['a'], 'r': r,
            'sp': open_tick['snap'], 'done': True,
            'qb': open_tick['qb'], 'qbp': open_tick['qb'],
        })
    return transitions, records
