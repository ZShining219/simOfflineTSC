"""ID-stable local control helpers independent of simulator bindings."""

from __future__ import annotations

import re
import numpy as np


def stable_intersection_order(intersection_ids):
    def natural(value):
        return tuple(int(part) if part.isdigit() else part
                     for part in re.split(r'(\d+)', str(value)))
    ordered = tuple(sorted((str(x) for x in intersection_ids), key=natural))
    if len(set(ordered)) != len(ordered):
        raise ValueError('Intersection IDs must be unique')
    return ordered


def build_action_mask(action_dims):
    dims = tuple(int(x) for x in action_dims)
    if not dims or any(x <= 0 for x in dims):
        raise ValueError('Action dimensions must be positive')
    return np.arange(max(dims))[None, :] < np.asarray(dims)[:, None]


def actions_by_intersection(intersection_ids, actions):
    actions = np.asarray(actions).reshape(-1)
    if len(intersection_ids) != len(actions):
        raise ValueError('Action count does not match intersection count')
    return {str(key): int(value) for key, value in zip(intersection_ids, actions)}
