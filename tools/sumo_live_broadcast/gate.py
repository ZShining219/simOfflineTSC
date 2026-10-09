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

The gate is deliberately pure: it receives the junction's queue and the
global decision index and returns ``(fired, reason)``.  World access,
expert invocation and arbitration stay in the engine/recorder.
"""

from __future__ import annotations

MODES = ('off', 'periodic', 'on_demand')


class HitlGate:
    """Per-rank trigger with episode budget and cooldown."""

    def __init__(self, mode='off', period=10, queue_threshold=None,
                 queue_growth=None, budget=36, cooldown=3):
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
        self.reset()

    # ------------------------------------------------------------- episode

    def reset(self):
        """Episode boundary: spent budget, cooldowns and queue memory all
        reset so each broadcast episode gets a fresh query allowance."""
        self._spent = {}       # rank -> queries consumed this episode
        self._last_fired = {}  # rank -> decision index of last query
        self._seen = {}        # rank -> eligible decisions seen
        self._last_queue = {}  # rank -> queue at previous eligible decision

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
        }

    # --------------------------------------------------------------- check

    def check(self, rank, queue, decision_index):
        """Return ``(fired, reason)`` for one rank at a decision boundary.

        ``queue`` is the junction's incoming-lane waiting-vehicle count
        for this decision; ``decision_index`` is the engine's monotonic
        decision counter.
        """
        rank = int(rank)
        self._seen[rank] = self._seen.get(rank, 0) + 1
        queue = float(queue)

        fire_reason = None
        if self.mode == 'periodic':
            if self._seen[rank] % self.period == 0:
                fire_reason = 'periodic'
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
