"""Query gate for the expert seat (HITL channel C).

Implements the spec's ``QueryGate`` semantics for the live broadcast:

* a *query* is atomic — once the gate fires it spends one unit of the
  episode budget, regardless of whether the expert's answer differs from
  the student's proposal;
* queries respect a decision-tick cooldown;
* only *eligible* decisions count (a junction is eligible when the rank
  is under controller control — manually pinned ranks never trigger).

Modes:

* ``off``       — never fires (default when HITL is not configured)
* ``periodic``  — fires every ``period``-th eligible decision
* ``on_demand`` — fires when the junction's incoming-lane waiting queue
  exceeds ``queue_threshold`` (absolute) or grows by ``queue_growth``
  versus the previous eligible decision (relative); either suffices
* ``random``    — fires with probability ``random_prob`` per eligible
  decision (same budget; lower bound for "when to ask")
* ``margin``    — fires when the student's top-1/top-2 Q-margin falls in
  the lowest ``margin_frac`` of its recent history (uncertainty trigger;
  self-calibrating to the same budget on any scene)
* ``window``    — fires only while ``now`` (sim seconds) lies inside one
  of ``windows`` (event-window oracle: upper bound for query timing)

The gate is deliberately pure: it receives the junction's queue and the
global decision index and returns ``(fired, reason)``.  World access,
expert invocation and arbitration stay in the engine/recorder.
"""

from __future__ import annotations

import random
from collections import deque

import numpy as np

MODES = ('off', 'periodic', 'on_demand', 'random', 'margin', 'window')


class HitlGate:
    """Per-rank trigger with episode budget and cooldown."""

    def __init__(self, mode='off', period=10, queue_threshold=None,
                 queue_growth=None, budget=36, cooldown=3,
                 random_prob=0.06, margin_frac=0.06, margin_warmup=20,
                 windows=None, rng_seed=0):
        if mode not in MODES:
            raise ValueError(f'unknown gate mode: {mode}')
        self.mode = mode
        self.period = max(1, int(period))
        self.queue_threshold = (
            None if queue_threshold is None else float(queue_threshold))
        self.queue_growth = (
            None if queue_growth is None else float(queue_growth))
        self.budget = max(0, int(budget))
        self.cooldown = max(0, int(cooldown))
        self.random_prob = float(random_prob)
        self.margin_frac = float(margin_frac)
        self.margin_warmup = max(2, int(margin_warmup))
        self.windows = [
            (float(lo), float(hi)) for lo, hi in (windows or [])]
        self._rng = random.Random(rng_seed)
        self.reset()

    # ------------------------------------------------------------- episode

    def reset(self):
        """Episode boundary: spent budget, cooldowns and queue memory all
        reset so each broadcast episode gets a fresh query allowance."""
        self._spent = {}       # rank -> queries consumed this episode
        self._last_fired = {}  # rank -> decision index of last query
        self._seen = {}        # rank -> eligible decisions seen
        self._last_queue = {}  # rank -> queue at previous eligible decision
        self._margins = {}     # rank -> deque of recent student Q-margins

    # -------------------------------------------------------------- status

    def spent(self):
        return dict(self._spent)

    def total_spent(self):
        return sum(self._spent.values())

    def config(self):
        return {
            'mode': self.mode,
            'period': self.period,
            'queue_threshold': self.queue_threshold,
            'queue_growth': self.queue_growth,
            'budget': self.budget,
            'cooldown': self.cooldown,
            'random_prob': (
                self.random_prob if self.mode == 'random' else None),
            'margin_frac': (
                self.margin_frac if self.mode == 'margin' else None),
            'windows': self.windows if self.mode == 'window' else None,
        }

    # --------------------------------------------------------------- check

    def check(self, rank, queue, decision_index, margin=None, now=None):
        """Return ``(fired, reason)`` for one rank at a decision boundary.

        ``queue`` is the junction's incoming-lane waiting-vehicle count
        for this decision; ``decision_index`` is the engine's monotonic
        decision counter.  ``margin`` (student Q top-1 minus top-2) feeds
        ``margin`` mode; ``now`` (sim seconds) feeds ``window`` mode.
        Both are optional so legacy callers keep working.
        """
        rank = int(rank)
        self._seen[rank] = self._seen.get(rank, 0) + 1
        queue = float(queue)

        fire_reason = None
        if self.mode == 'periodic':
            if self._seen[rank] % self.period == 0:
                fire_reason = 'periodic'
        elif self.mode == 'random':
            if self._rng.random() < self.random_prob:
                fire_reason = 'random'
        elif self.mode == 'margin':
            hist = self._margins.setdefault(rank, deque(maxlen=240))
            if margin is not None and np.isfinite(margin):
                if (len(hist) >= self.margin_warmup
                        and float(margin) <= float(np.quantile(
                            np.asarray(hist), self.margin_frac))):
                    fire_reason = 'margin'
                hist.append(float(margin))
        elif self.mode == 'window':
            if now is not None and any(
                    lo <= float(now) <= hi for lo, hi in self.windows):
                fire_reason = 'window'
        elif self.mode == 'on_demand':
            if (self.queue_threshold is not None
                    and queue >= self.queue_threshold):
                fire_reason = 'queue_threshold'
            prev = self._last_queue.get(rank)
            if (fire_reason is None and self.queue_growth is not None
                    and prev is not None
                    and queue - prev >= self.queue_growth):
                fire_reason = 'queue_growth'

        self._last_queue[rank] = queue

        if fire_reason is None:
            return False, None
        if self._spent.get(rank, 0) >= self.budget:
            return False, 'budget_exhausted'
        last = self._last_fired.get(rank)
        if last is not None and decision_index - last < self.cooldown:
            return False, 'cooldown'

        self._spent[rank] = self._spent.get(rank, 0) + 1
        self._last_fired[rank] = decision_index
        return True, fire_reason
