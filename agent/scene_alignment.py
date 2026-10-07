"""CAREL-style event-traffic alignment auxiliary objective (ATT-ENTITY-003 stage 3).

Structure borrowed (not ported): CAREL (Saghafian et al., TMLR 2025,
arXiv:2411.19787) attaches a contrastive episode-instruction alignment loss to
an instruction-following RL agent; here the "instruction" is the public event
report and the "trajectory" is the affected intersections' lane-feature window
while the event is active.

Data path:
  * ``EpisodeAlignAccumulator`` runs during rollout inside the agent's
    ``remember`` path.  Per control step it stores the raw lane-observation
    rows (no phase/action features -- see the leakage audit in
    ``docs/att_entity_003_stage3_alignment_audit.md``) and, for every active
    report, the step index plus its grounded-node mask column.
  * At episode end ``finish`` builds one :class:`EventAlignPair` per event:
    the positive window is the mean lane feature over the event's grounded
    nodes during its active span (capped at ``window_cap`` steps); explicit
    negatives are a wrong-node window (same span, ungrounded node) and a
    no-event window (same grounded nodes, contiguous event-free stretch).
  * ``event_traffic_align_loss`` re-encodes event payloads through the live
    ``SceneContextEncoder`` (text branch + meta branch + fusion, gradients
    enabled) and encodes windows through ``TrafficWindowEncoder``; the loss is
    symmetric InfoNCE with cosine similarity.

Public report payloads deliberately expose no event begin/end; the event
window is therefore inferred from the set of steps where the report is active
in the scene, which is the same view the policy itself sees.
"""
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class EventAlignPair:
    """One event's positive traffic window plus its explicit negatives."""
    report: Any                       # grounded report (text + meta fields)
    pos: np.ndarray                   # [T, F] mean lane feats at grounded nodes
    negs: Dict[str, np.ndarray] = field(default_factory=dict)
    # GRIF-style (experiment B): an event-free baseline window measured at
    # the SAME nodes immediately before each window's span, enabling the
    # transition target f(H_before, H_after - H_before).  ``None`` when no
    # contiguous event-free stretch exists before the span.
    pos_before: Optional[np.ndarray] = None
    negs_before: Dict[str, np.ndarray] = field(default_factory=dict)


class TrafficWindowEncoder(nn.Module):
    """[T, F] lane-feature window -> d-dim embedding via per-feature stats."""

    def __init__(self, feat_dim: int, hidden_dim: int = 128, out_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(4 * feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_batch(self, windows: Sequence[np.ndarray]) -> torch.Tensor:
        device = next(self.parameters()).device
        stats = []
        for w in windows:
            t = torch.as_tensor(np.asarray(w, dtype=np.float32), device=device)
            if t.ndim != 2 or t.shape[-1] != self.feat_dim:
                raise ValueError(
                    f'window must be [T, {self.feat_dim}], got {tuple(t.shape)}')
            stats.append(torch.cat([
                t.mean(0), t.max(0).values, t.std(0, unbiased=False),
                t[-1] - t[0],
            ], dim=0))
        z = self.net(torch.stack(stats, dim=0))
        return F.normalize(z, dim=-1)


class TrafficTransitionEncoder(nn.Module):
    """GRIF-style transition embedding: f(H_before, H_after - H_before).

    ``H_*`` are lane-feature means over an event-free baseline window and a
    candidate window measured at the SAME nodes.  Only traffic features ever
    enter here — phases/actions are excluded by construction of the
    accumulator's stored rows.
    """

    def __init__(self, feat_dim: int, hidden_dim: int = 128,
                 out_dim: int = 128):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Linear(2 * feat_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def encode_batch(self, befores: Sequence[np.ndarray],
                     afters: Sequence[np.ndarray]) -> torch.Tensor:
        device = next(self.parameters()).device
        feats = []
        for before, after in zip(befores, afters):
            b = torch.as_tensor(np.asarray(before, dtype=np.float32),
                                device=device)
            a = torch.as_tensor(np.asarray(after, dtype=np.float32),
                                device=device)
            if b.shape[-1] != self.feat_dim or a.shape[-1] != self.feat_dim:
                raise ValueError('transition features disagree with feat_dim')
            feats.append(torch.cat([b.mean(0), a.mean(0) - b.mean(0)], dim=0))
        z = self.net(torch.stack(feats, dim=0))
        return F.normalize(z, dim=-1)


def symmetric_infonce(z_e: torch.Tensor, z_w: torch.Tensor,
                      pos_index: Sequence[int], tau: float = 0.07):
    """Symmetric InfoNCE: event i aligns to window pos_index[i].

    ``z_e`` is [E, D] event embeddings; ``z_w`` is [W, D] window embeddings
    where window ``pos_index[i]`` is event i's positive.  All other windows
    (other events' positives plus explicit negatives) act as in-batch
    negatives in both directions.
    """
    if z_e.shape[0] == 0:
        raise ValueError('InfoNCE requires at least one event')
    pos_index = torch.as_tensor(list(pos_index), dtype=torch.long,
                                device=z_w.device)
    s_e2w = (z_e @ z_w.T) / tau                      # [E, W]
    loss_e = F.cross_entropy(s_e2w, pos_index)
    z_pos = z_w[pos_index]                            # [E, D]
    s_w2e = (z_pos @ z_e.T) / tau                     # [E, E]
    loss_w = F.cross_entropy(
        s_w2e, torch.arange(z_e.shape[0], device=z_e.device))
    return 0.5 * (loss_e + loss_w)


class EpisodeAlignAccumulator:
    """Accumulate one episode's (event, traffic-window) evidence."""

    def __init__(self, window_cap_steps: int = 60, min_active_steps: int = 3,
                 noevent_trials: int = 8):
        self.window_cap = int(window_cap_steps)
        self.min_active_steps = int(min_active_steps)
        self.noevent_trials = int(noevent_trials)
        self.reset()

    def reset(self):
        self.obs: List[np.ndarray] = []
        self.events: Dict[str, Dict[str, Any]] = {}

    def step(self, obs_rows: np.ndarray, scene) -> None:
        """Record one control step's lane features and active events.

        ``obs_rows`` is [N, F] in grounding/node order (the same order as
        ``ob_generator`` rows and ``Gnode.node_report_mask`` rows).
        ``scene`` is a ``utils.scene_context.SceneSnapshot`` or None.
        """
        t = len(self.obs)
        self.obs.append(np.asarray(obs_rows, dtype=np.float32))
        if scene is None:
            return
        events = getattr(scene.context, 'events', ()) or ()
        mask = getattr(scene.grounding, 'node_report_mask', None)
        if mask is None or not events:
            return
        mask = np.asarray(mask, dtype=bool)
        for j, report in enumerate(events):
            eid = getattr(report, 'report_id', None) or getattr(
                report, 'event_id', None)
            if eid is None:
                continue
            entry = self.events.get(eid)
            if entry is None:
                entry = {'report': report, 'mask': mask[:, j].copy(),
                         'steps': []}
                self.events[eid] = entry
            entry['steps'].append(t)

    def finish(self, rng: np.random.Generator) -> List[EventAlignPair]:
        """Build this episode's positive/negative window pairs."""
        if not self.events or not self.obs:
            return []
        obs = np.stack(self.obs, axis=0)               # [T_total, N, F]
        steps_with_events = np.zeros(obs.shape[0], dtype=bool)
        for entry in self.events.values():
            steps_with_events[np.asarray(entry['steps'], dtype=int)] = True
        free_steps = np.where(~steps_with_events)[0]
        pairs = []
        for entry in self.events.values():
            steps = np.asarray(entry['steps'][:self.window_cap], dtype=int)
            if steps.size < self.min_active_steps:
                continue
            grounded = np.where(entry['mask'])[0]
            if not grounded.size:
                continue
            pos = obs[steps][:, grounded, :].mean(axis=1)      # [T, F]
            negs: Dict[str, np.ndarray] = {}
            negs_before: Dict[str, np.ndarray] = {}
            # GRIF baseline: latest contiguous event-free run ending before
            # this event's first active step, at the same grounded nodes.
            pos_before_steps = self._before_run(free_steps, int(steps[0]),
                                                steps.size)
            pos_before = (obs[pos_before_steps][:, grounded, :].mean(axis=1)
                          if pos_before_steps is not None else None)
            unaff = np.where(~entry['mask'])[0]
            if unaff.size:
                node = int(unaff[rng.integers(unaff.size)])
                negs['wrong_node'] = obs[steps][:, node, :]
                if pos_before_steps is not None:
                    negs_before['wrong_node'] = \
                        obs[pos_before_steps][:, node, :]
            noevent_sel = self._noevent_run(
                free_steps, steps.size, rng)
            if noevent_sel is not None:
                negs['no_event'] = obs[noevent_sel][:, grounded, :].mean(axis=1)
                nb = self._before_run(free_steps, int(noevent_sel[0]),
                                      steps.size)
                if nb is not None:
                    negs_before['no_event'] = obs[nb][:, grounded, :].mean(
                        axis=1)
            pairs.append(EventAlignPair(report=entry['report'],
                                        pos=pos, negs=negs,
                                        pos_before=pos_before,
                                        negs_before=negs_before))
        return pairs

    def _before_run(self, free_steps, end_t, length):
        """Latest contiguous free run ending before ``end_t``.

        Returns up to ``length`` trailing steps of that run (at least
        ``min_active_steps``), or None when no qualifying run exists.
        """
        cand = free_steps[free_steps < end_t]
        if not cand.size:
            return None
        runs = np.split(cand, np.where(np.diff(cand) != 1)[0] + 1)
        run = runs[-1]
        # The last run may end after end_t minus a gap; require contiguity
        # only within itself, and take its tail.
        if run.size < self.min_active_steps:
            return None
        return run[-length:]

    def _noevent_run(self, free_steps, length, rng):
        """Random contiguous event-free stretch of ``length`` (step ids)."""
        if length < 2:
            return None
        runs = np.split(free_steps,
                        np.where(np.diff(free_steps) != 1)[0] + 1)
        runs = [r for r in runs if r.size >= length]
        if not runs:
            return None
        run = runs[int(rng.integers(len(runs)))]
        start = int(rng.integers(run.size - length + 1))
        return run[start:start + length]


class AlignmentBuffer:
    """Recent-episode pair buffer; the InfoNCE batch draws cross-episode
    negatives from this pool."""

    def __init__(self, max_episodes: int = 32):
        self.episodes: deque = deque(maxlen=int(max_episodes))
        self.total_pairs = 0

    def append(self, pairs: Sequence[EventAlignPair]) -> None:
        if pairs:
            self.episodes.append(list(pairs))
            self.total_pairs += len(pairs)

    def all_pairs(self) -> List[EventAlignPair]:
        return [p for ep in self.episodes for p in ep]

    def __len__(self):
        return sum(len(ep) for ep in self.episodes)
