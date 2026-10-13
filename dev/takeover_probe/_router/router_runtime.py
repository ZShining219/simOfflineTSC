"""Windowed dispatch runtime shared by training / evaluation / broadcast.

A router decision happens every ``window_decisions`` underlying control
decisions (10 s each -> 120 s governance window).  The chosen controller
then owns every decision inside the window — the router never mixes
per-step actions, satisfying the "sustained takeover" constraint.

Reward (per window):
    - mean_total_queue / 10        congestion pressure, global
    - 0.5 * switched               switching cost (hysteresis)
    - 0.2 * max_lane_queue_at_end  terminal backlog penalty: prevents
                                   "clears early, leaves a pit" myopia
"""

from __future__ import annotations

import numpy as np

from hitl.live.injects import apply_injects


def _window_reward(records, switched, priority_lanes=None):
    """Window reward follows the *declared* objective.

    ``priority_lanes`` (lane names) set => 'focus' governance: congestion
    is scored on the incident-approach lanes.  Otherwise global.
    """
    if not records:
        return 0.0
    if priority_lanes:
        pri = [
            float(np.mean([
                r['lane_queue'].get(l, 0) for l in priority_lanes]))
            for r in records]
        mean_cong = float(np.mean(pri))
    else:
        mean_cong = float(np.mean([r['total_queue'] for r in records]))
    max_lane_end = max(records[-1]['lane_queue'].values() or [0])
    return (-mean_cong / 10.0
            - 0.5 * float(switched)
            - 0.2 * float(max_lane_end))


def run_router_episode(world, controllers, fb, router, ranks,
                       injects=None, interval=10, window_decisions=12,
                       episode_seconds=3600, epsilon=0.0, rng=None,
                       fixed_controller=None, text_override=None,
                       priority_lanes=None, collect=True,
                       force_choices=None):
    """Run one episode under router (or fixed) control.

    Returns (transitions, records, windows):
      transitions  (s, a, r, s', done) per window, for SMDP TD updates
      records      per-decision rows (queue/action/active controller)
      windows      per-window audit log incl. text sentence + q-values
    """
    injects = injects or []
    ctrl_ids = list(controllers)
    if router is not None:
        ctrl_ids = router.controller_ids
    held = {
        rank: int(inter.virtual_phase)
        for rank, inter in enumerate(world.intersections)
    }
    records, transitions, windows = [], [], []
    decision = 0
    next_decision_t = float(world.get_current_time())
    window_open = None
    active = fixed_controller or ctrl_ids[0]
    prev_arm = None
    while True:
        now = float(world.get_current_time())
        if episode_seconds and now >= episode_seconds:
            break
        if now + 1e-6 >= next_decision_t:
            decision += 1
            if (decision - 1) % window_decisions == 0:
                snap = fb.snapshot(now, text_override=text_override)
                if window_open is not None:
                    reward = _window_reward(
                        window_open['records'], window_open['switched'],
                        priority_lanes=priority_lanes)
                    transitions.append({
                        's': window_open['snap'], 'a': window_open['arm_idx'],
                        'r': reward, 'sp': snap, 'done': False,
                    })
                    window_open['reward'] = reward
                    windows.append(window_open)
                if router is None:
                    arm_idx = ctrl_ids.index(active)
                    qvals, attn_w = None, None
                else:
                    from dev.takeover_probe.dispatch_model import collate
                    batch = collate([snap])
                    arm_idx, qvals, attn_w = router.act(
                        batch, epsilon=epsilon, rng=rng)
                new_active = ctrl_ids[arm_idx]
                if force_choices and now in force_choices:
                    # diagnostic fork: override the applied controller for
                    # this window while keeping the router's logged
                    # q_values for comparison
                    new_active = force_choices[now]
                    arm_idx = ctrl_ids.index(new_active)
                switched = prev_arm is not None and new_active != prev_arm
                _, sentence = fb.event_fields(now)
                window_open = {
                    't': now, 'decision': decision,
                    'arm_idx': arm_idx, 'controller': new_active,
                    'switched': switched, 'snap': snap,
                    'records': [], 'text': sentence,
                    'q_values': qvals,
                    'attn_top_lane': (
                        fb.lanes[int(np.argmax(
                            attn_w.detach().cpu().numpy()))]
                        if attn_w is not None else None),
                    'attn_weights': (
                        attn_w.detach().cpu().numpy().ravel().tolist()
                        if attn_w is not None else None),
                    'lane_names': list(fb.lanes),
                }
                active = new_active
                prev_arm = new_active
            controller = controllers[active]
            for rank in ranks:
                held[rank] = int(controller.decide(rank))
            lane_queue = world.get_info('lane_waiting_count') or {}
            row = {
                'decision': decision, 't': now,
                'controller': active,
                'actions': {str(r): int(held[r]) for r in ranks},
                'lane_queue': {
                    str(k): int(v) for k, v in lane_queue.items()},
                'total_queue': int(sum(lane_queue.values())),
                'throughput': int(world.get_info('throughput') or 0),
            }
            records.append(row)
            if window_open is not None:
                window_open['records'].append(row)
            next_decision_t = now + interval
        apply_injects(world, injects, now)
        world.step([
            held.get(rank, int(inter.virtual_phase))
            for rank, inter in enumerate(world.intersections)
        ])
    if window_open is not None:
        reward = _window_reward(
            window_open['records'], window_open['switched'],
            priority_lanes=priority_lanes)
        transitions.append({
            's': window_open['snap'], 'a': window_open['arm_idx'],
            'r': reward, 'sp': window_open['snap'], 'done': True,
        })
        window_open['reward'] = reward
        windows.append(window_open)
    if not collect:
        transitions = []
    return transitions, records, windows
