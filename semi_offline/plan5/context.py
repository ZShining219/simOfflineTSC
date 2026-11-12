"""Online causal demand context from observed network-entry events."""
from collections import deque
import math

DIRECTIONS = ("North", "South", "East", "West")


def direction_from_geometry(start, end):
    """Return the cardinal direction of a boundary edge from two coordinates.

    The dominant axis is used for diagonal/near-diagonal geometry.  Ties are
    rejected because an ambiguous mapping would make the causal feature
    irreproducible.
    """
    dx = float(end[0]) - float(start[0])
    dy = float(end[1]) - float(start[1])
    ax, ay = abs(dx), abs(dy)
    if ax == 0 and ay == 0:
        raise ValueError("Boundary edge has zero length")
    if math.isclose(ax, ay, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Boundary edge direction is geometrically ambiguous")
    if ay > ax:
        return "North" if dy > 0 else "South"
    return "East" if dx > 0 else "West"


def context_from_events(events, time_s, window_s=60):
    """Compute [N,S,E,W] rates using only events in ``(t-window, t]``."""
    if window_s <= 0:
        raise ValueError("window_s must be positive")
    counts = {direction: 0 for direction in DIRECTIONS}
    lower = float(time_s) - float(window_s)
    for event_time, direction in events:
        if lower < float(event_time) <= float(time_s):
            if direction not in counts:
                raise ValueError(f"Unknown boundary direction: {direction!r}")
            counts[direction] += 1
    return tuple(counts[d] / float(window_s) for d in DIRECTIONS)


class ContextHistory:
    """Bounded event history with zero-padded episode cold starts."""
    def __init__(self, window_s=60, max_events=100000):
        if window_s <= 0 or max_events <= 0:
            raise ValueError("window_s and max_events must be positive")
        self.window_s = float(window_s)
        self.events = deque(maxlen=int(max_events))
        self.last_time = None

    def reset(self):
        self.events.clear()
        self.last_time = None

    def observe_entry(self, time_s, direction):
        time_s = float(time_s)
        if self.last_time is not None and time_s < self.last_time:
            raise ValueError("Observed event time moved backwards")
        if direction not in DIRECTIONS:
            raise ValueError(f"Unknown boundary direction: {direction!r}")
        self.events.append((time_s, direction))
        self.last_time = time_s

    def observe_entries(self, events):
        for time_s, direction in events:
            self.observe_entry(time_s, direction)

    def value(self, time_s):
        # Drop events that can no longer affect a future query.  The strict
        # lower bound matches the protocol's (t-60,t] interval.
        lower = float(time_s) - self.window_s
        while self.events and self.events[0][0] <= lower:
            self.events.popleft()
        return context_from_events(self.events, time_s, self.window_s)

    def state_dict(self):
        return {"window_s": self.window_s, "events": list(self.events), "last_time": self.last_time}

    @classmethod
    def from_state_dict(cls, state):
        history = cls(state["window_s"])
        history.events.extend((float(t), d) for t, d in state.get("events", []))
        history.last_time = state.get("last_time")
        return history
