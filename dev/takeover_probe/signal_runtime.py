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

from tools.sumo_live_broadcast.injects import apply_injects


def run_signal_episode(world, fb, ranks, injects=None, agent=None,
                       controller=None, interval=10, episode_seconds=3600,
                       epsilon=0.0, rng=None, text_override=None,
                       reward_scale=50.0):
    injects = injects or []
    held = {
        rank: int(inter.virtual_phase)
        for rank, inter in enumerate(world.intersections)
    }
    transitions, records = [], []
    next_decision_t = float(world.get_current_time())
    open_tick = None          # {'snap','a','q','w','t'}
    interval_queue = 0.0
    while True:
        now = float(world.get_current_time())
        if episode_seconds and now >= episode_seconds:
            break
        if now + 1e-6 >= next_decision_t:
            # close the previous tick: r over its interval, s' = now
            snap = fb.snapshot(now, text_override=text_override) \
                if agent is not None else None
            if open_tick is not None:
                r = -(interval_queue / max(1.0, now - open_tick['t'])) \
                    / reward_scale
                if agent is not None:
                    transitions.append({
                        's': open_tick['snap'], 'a': open_tick['a'],
                        'r': r, 'sp': snap, 'done': False,
                    })
                open_tick['r'] = r
            # decide for the new tick
            qvals, attn_w, sentence = None, None, None
            if agent is not None:
                from dev.takeover_probe.dispatch_model import collate
                batch = collate([snap])
                a, qvals, attn_w = agent.act(
                    batch, epsilon=epsilon, rng=rng)
                _, sentence = fb.event_fields(now)
                for rank in ranks:
                    held[rank] = a
            else:
                a = int(controller.decide(0))
                for rank in ranks:
                    held[rank] = a
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
            open_tick = {'snap': snap, 'a': a, 't': now}
            interval_queue = 0.0
            next_decision_t = now + interval
        lane_queue = world.get_info('lane_waiting_count') or {}
        interval_queue += float(sum(lane_queue.values()))
        apply_injects(world, injects, now)
        world.step([
            held.get(rank, int(inter.virtual_phase))
            for rank, inter in enumerate(world.intersections)
        ])
    if open_tick is not None and open_tick['snap'] is not None:
        r = -(interval_queue / max(1.0, interval)) / reward_scale
        transitions.append({
            's': open_tick['snap'], 'a': open_tick['a'], 'r': r,
            'sp': open_tick['snap'], 'done': True,
        })
    return transitions, records
