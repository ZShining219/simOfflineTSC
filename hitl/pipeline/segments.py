#!/usr/bin/env python
"""segments.py — B 通道片段切割 + 确定性教师分（对齐 spec §6.2/§9.1）。

数据源 = 打包档案的 trajectory npz（每决策一行）。教师分：

    Q_t = mean_l(q_l / c_l)          q_l = 逐车道排队；c_l = 车道容量
    U_t = max_l min(u_l / T_u, 2)    u_l = 距上次被相位服务的时间（服务缺口）
    h_t = -0.5 * Q_t - 0.5 * U_t
    片段教师分 H = mean_t(h_t)

逐车道近似口径（对全部臂一致，差异已文档化）：
  * q_l 无逐车道记录时用标量 queue 均摊（各 c_l 相同则数学等价于真值）
  * u_l 由相位服务史重建（相位→服务车道映射来自 dump_lane_topology.py）

配对：片段 A/B → Δ=H_A−H_B；|Δ|≤tie_tol → y=0.5（平局）；否则 y∈{0,1}。
无完整有效区间的片段不参与（episode 边界/截断窗丢弃）。
"""

from __future__ import annotations

import json
import os

import numpy as np


def load_episode(npz_path):
    """Load one packaged episode npz into a plain dict of arrays."""
    data = np.load(npz_path, allow_pickle=False)
    return {key: data[key] for key in data.files}


def service_gap_series(current_phase, phase_served, dt, lane_occupancy):
    """Reconstruct per-lane service gap u_l (seconds) at each decision.

    Spec §5.x: u_l grows by dt only while the lane has waiting demand
    (n_l > 0) AND none of its movements is currently permitted; it resets
    to 0 when served or empty.

    current_phase: (T,) int active green action index per decision.
    phase_served: list over action index -> iterable of in-lane indices.
    lane_occupancy: (T, L) per-lane vehicle counts (state block).
    """
    phases = np.asarray(current_phase, dtype=int).reshape(-1)
    occ = np.asarray(lane_occupancy, dtype=np.float64)
    n_lanes = max(
        (i for served in phase_served for i in served), default=-1) + 1
    n_lanes = max(n_lanes, occ.shape[1] if occ.ndim == 2 else 1)
    gaps = np.zeros((len(phases), n_lanes), dtype=np.float64)
    current = np.zeros(n_lanes, dtype=np.float64)
    for t, phase in enumerate(phases):
        served = np.zeros(n_lanes, dtype=bool)
        if 0 <= phase < len(phase_served):
            for lane in phase_served[phase]:
                if lane < n_lanes:
                    served[lane] = True
        has_demand = (
            occ[t] > 0 if occ.ndim == 2 and t < occ.shape[0]
            else np.ones(n_lanes, dtype=bool))
        current = np.where(
            (~served) & has_demand, current + dt, 0.0)
        gaps[t] = current
    return gaps


def _lane_occupancy(ep, n_lanes):
    """state rows are (T, 1, L): per-in-lane vehicle counts."""
    state = np.asarray(ep['state'], dtype=np.float64)
    if state.ndim == 3:
        state = state[:, 0, :]
    if state.ndim == 2 and state.shape[1] >= n_lanes:
        return state[:, :n_lanes]
    return np.zeros((state.shape[0] if state.ndim else 0, n_lanes))


def _obs_rows(ep, action_dim):
    """obs = state(lanes) + one_hot(current_phase) — the network input."""
    state = np.asarray(ep['state'], dtype=np.float64)
    if state.ndim == 3:
        state = state[:, 0, :]
    phase = np.asarray(ep['current_phase'], dtype=int).reshape(-1)
    one_hot = np.zeros((len(phase), action_dim), dtype=np.float64)
    for t, p in enumerate(phase):
        if 0 <= p < action_dim:
            one_hot[t, p] = 1.0
    return np.concatenate([state, one_hot], axis=1)


def _lane_queues(ep, n_lanes):
    """Per-lane queue counts, preferring an explicitly recorded vector."""
    scalar = np.asarray(ep.get('queue', []), dtype=np.float64).reshape(-1)
    if 'queue_by_lane' in ep:
        rows = np.asarray(ep['queue_by_lane'], dtype=np.float64)
        if rows.ndim == 2 and rows.shape[1] == n_lanes:
            return rows
    if scalar.size:
        # uniform capacities -> mean_l(q_l/c_l) = total/(n_lanes*c); keep a
        # uniform per-lane split so both Q and RM features stay consistent.
        return np.repeat((scalar / n_lanes)[:, None], n_lanes, axis=1)
    # last resort: vehicle counts in the state block
    return _lane_occupancy(ep, n_lanes)


def teacher_series(ep, ctx):
    """Per-decision teacher score h_t and its components."""
    q = _lane_queues(ep, len(ctx['lane_capacity']))
    cap = np.asarray(ctx['lane_capacity'], dtype=np.float64)
    q_ratio = q / cap[None, :]
    Q = q_ratio.mean(axis=1)
    gaps = service_gap_series(
        ep['current_phase'], ctx['phase_served_lanes'],
        float(ctx['action_interval']),
        _lane_occupancy(ep, len(ctx['lane_capacity'])))
    U = np.minimum(gaps / float(ctx['T_u']), 2.0).max(axis=1)
    h = -0.5 * Q - 0.5 * U
    return h, Q, U, q_ratio, gaps


def step_features(ep, ctx):
    """Reward-model input x per spec §6.2 (no teacher score / label fields):

        [obs | one_hot(accepted_action) | next_obs |
         queue_by_lane/c | service_gap_by_lane/T_u]
        (dims = 2d + P + 2L)
    """
    action_dim = int(ctx['action_dim'])
    obs = _obs_rows(ep, action_dim)
    action = np.asarray(ep['action'], dtype=int).reshape(-1)
    one_hot = np.zeros((len(action), action_dim), dtype=np.float64)
    for t, a in enumerate(action):
        if 0 <= a < action_dim:
            one_hot[t, a] = 1.0
    next_phase = np.asarray(ep.get('next_phase', ep['current_phase']),
                            dtype=int).reshape(-1)
    next_state = np.asarray(ep['next_state'], dtype=np.float64)
    if next_state.ndim == 3:
        next_state = next_state[:, 0, :]
    next_one_hot = np.zeros((len(next_phase), action_dim), dtype=np.float64)
    for t, p in enumerate(next_phase):
        if 0 <= p < action_dim:
            next_one_hot[t, p] = 1.0
    next_obs = np.concatenate([next_state, next_one_hot], axis=1)
    _, _, _, q_ratio, gaps = teacher_series(ep, ctx)
    gap_norm = np.minimum(gaps / float(ctx['T_u']), 2.0)
    feat = np.concatenate(
        [obs, one_hot, next_obs, q_ratio, gap_norm], axis=1)
    return feat.astype(np.float32)


def segment_episode(ep, ctx, window):
    """Split one episode into non-overlapping full windows.

    Returns a list of dicts: {episode_id, start, end, n, valid}.
    A window is valid only when every step inside is usable (no truncation
    flag inside the window; the final step may legitimately truncate).
    """
    n = int(len(ep['decision_step']))
    out = []
    for start in range(0, n - window + 1, window):
        end = start + window
        trunc = np.asarray(ep['truncated']).reshape(-1)
        inner_trunc = bool(trunc[start:end - 1].any())
        out.append({
            'episode_id': int(ep['episode_id'][0])
            if np.asarray(ep['episode_id']).size else None,
            'start': start, 'end': end, 'n': n,
            'valid': not inner_trunc,
        })
    return out


def label_pair(H_left, H_right, tie_tol=0.05):
    """y=1 left better, 0 right better, 0.5 tie."""
    delta = float(H_left - H_right)
    if abs(delta) <= tie_tol:
        return 0.5, delta
    return (1.0 if delta > 0 else 0.0), delta


def build_pairs(episode_arrays, ctx, window=6, tie_tol=0.05,
                max_pairs_per_episode=120, seed=11):
    """Cut every episode into windows, score each, emit preference pairs.

    Pairing: all unordered window pairs inside an episode (counterfactual
    same-context comparison), subsampled to max_pairs_per_episode.
    """
    rng = np.random.RandomState(int(seed))
    segments = []
    for ep in episode_arrays:
        h, Q, U, _, _ = teacher_series(ep, ctx)
        for seg in segment_episode(ep, ctx, window):
            if not seg['valid']:
                continue
            s = dict(seg)
            s['H'] = float(h[seg['start']:seg['end']].mean())
            s['Q'] = float(Q[seg['start']:seg['end']].mean())
            s['U'] = float(U[seg['start']:seg['end']].mean())
            segments.append(s)
    pairs = []
    by_episode = {}
    for i, seg in enumerate(segments):
        by_episode.setdefault(seg['episode_id'], []).append(i)
    for ep_id, idxs in sorted(by_episode.items(), key=lambda kv: str(kv[0])):
        cand = [(a, b) for x, a in enumerate(idxs) for b in idxs[x + 1:]]
        if len(cand) > max_pairs_per_episode:
            pick = rng.choice(
                len(cand), size=max_pairs_per_episode, replace=False)
            cand = [cand[i] for i in sorted(pick)]
        for left, right in cand:
            y, delta = label_pair(
                segments[left]['H'], segments[right]['H'], tie_tol)
            pairs.append({
                'episode_id': ep_id,
                'left_segment': left, 'right_segment': right,
                'H_left': segments[left]['H'],
                'H_right': segments[right]['H'],
                'delta': delta, 'clean_y': y, 'training_y': y,
                'skip': False, 'y': y,
            })
    return segments, pairs


def load_context(topology_path, action_interval, action_dim, T_u=60.0):
    """Spec constants: c_l = max(1, floor(effective_length/7.5m)); T_u=60s."""
    import math
    topo = json.load(open(topology_path))
    lanes = topo['in_lanes']
    cap = [
        max(1.0, math.floor(float(topo['lane_length_m'][l]) / 7.5))
        for l in lanes
    ]
    return {
        'in_lanes': lanes,
        'lane_capacity': cap,
        'phase_served_lanes': [
            [int(i) for i in served]
            for served in topo['phase_served_lanes']
        ],
        'action_interval': float(action_interval),
        'action_dim': int(action_dim),
        'T_u': float(T_u),
    }
