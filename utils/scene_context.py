"""Time-indexed event scene snapshots and replay payloads.

The event runtime keeps the latest report for every event that has occurred,
including ``cleared`` reports.  A controller must not use that report history
as its current scene.  This module provides the small, dependency-light
boundary used by both the scene encoder and replay code:

``scene_context_at(t, reports)``
    builds the unique scene for one synchronized control step.  Only active
    reports become ``SceneContext.events``; cleared reports are retained only
    as audit information.

``SceneReplayTransition``
    is the opt-in ten-field replay contract.  Legacy six-field transitions
    remain readable so existing checkpoints and non-scene agents are not
    silently reinterpreted.
"""

from dataclasses import dataclass
import math
from typing import Any, Iterable, NamedTuple, Optional, Tuple


class SceneContextError(ValueError):
    """A report snapshot cannot define one unambiguous control scene."""


def _report_id(report: Any) -> str:
    value = getattr(report, 'report_id', None)
    if value is None:
        value = getattr(report, 'event_id', None)
    if not isinstance(value, str) or not value:
        raise SceneContextError('reports must expose a non-empty report_id/event_id')
    return value


def _report_status(report: Any) -> str:
    value = getattr(report, 'report_status', None)
    if value is None:
        value = getattr(report, 'status', None)
    if value not in {'active', 'cleared'}:
        raise SceneContextError(
            f'report {_report_id(report)!r} has unsupported current status {value!r}')
    return value


def _report_time(report: Any) -> float:
    value = getattr(report, 'updated_at', None)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneContextError(f'report {_report_id(report)!r} has invalid updated_at')
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise SceneContextError(f'report {_report_id(report)!r} has invalid updated_at')
    return value


@dataclass(frozen=True)
class SceneContext:
    """The only event scene visible at one control timestamp.

    ``events`` is deliberately restricted to reports with ``active`` status.
    ``cleared_reports`` is a diagnostic copy of the current public snapshot;
    it is never part of the scene consumed by a policy or encoder.
    """

    observed_at: float
    events: Tuple[Any, ...] = ()
    cleared_reports: Tuple[Any, ...] = ()

    schema_version = 'scene-context-v1'

    def __post_init__(self):
        if isinstance(self.observed_at, bool) or not isinstance(self.observed_at, (int, float)):
            raise SceneContextError('observed_at must be a finite nonnegative number')
        observed_at = float(self.observed_at)
        if not math.isfinite(observed_at) or observed_at < 0:
            raise SceneContextError('observed_at must be a finite nonnegative number')
        object.__setattr__(self, 'observed_at', observed_at)

        events = tuple(self.events)
        cleared = tuple(self.cleared_reports)
        event_ids = [_report_id(item) for item in events]
        cleared_ids = [_report_id(item) for item in cleared]
        if len(set(event_ids)) != len(event_ids):
            raise SceneContextError('active scene contains duplicate report IDs')
        if len(set(cleared_ids)) != len(cleared_ids):
            raise SceneContextError('cleared report log contains duplicate report IDs')
        if set(event_ids) & set(cleared_ids):
            raise SceneContextError('one report cannot be active and cleared in one snapshot')
        for item in events:
            if _report_status(item) != 'active':
                raise SceneContextError('SceneContext.events may contain active reports only')
            if _report_time(item) > observed_at + 1e-8:
                raise SceneContextError('active report is from the future of observed_at')
        for item in cleared:
            if _report_status(item) != 'cleared':
                raise SceneContextError('cleared_reports may contain cleared reports only')
            # A public report snapshot may retain the latest cleared message
            # while a caller is constructing an earlier audit view.  Cleared
            # messages never enter ``events`` and therefore cannot leak future
            # control information; active reports remain subject to the strict
            # timestamp check above.
        object.__setattr__(self, 'events', events)
        object.__setattr__(self, 'cleared_reports', cleared)

    @property
    def state(self) -> str:
        """Return exactly ``normal`` or ``event`` for this control step."""
        return 'event' if self.events else 'normal'

    @property
    def is_normal(self) -> bool:
        return not self.events

    @property
    def event_ids(self) -> Tuple[str, ...]:
        return tuple(_report_id(item) for item in self.events)

    @property
    def scene_ref(self) -> Tuple[str, ...]:
        """Stable active-event reference suitable for replay diagnostics."""
        return self.event_ids

    def to_dict(self, include_cleared=True):
        """Return a JSON-friendly scene/audit record without trainable tensors."""
        def report_dict(report):
            if hasattr(report, 'to_dict'):
                return report.to_dict()
            if hasattr(report, '__dict__'):
                return dict(report.__dict__)
            return {'report_id': _report_id(report), 'status': _report_status(report)}

        result = {
            'schema_version': self.schema_version,
            'observed_at': self.observed_at,
            'state': self.state,
            'scene_ref': list(self.scene_ref),
            'events': [report_dict(item) for item in self.events],
        }
        if include_cleared:
            result['cleared_reports'] = [report_dict(item) for item in self.cleared_reports]
        return result


@dataclass(frozen=True)
class SceneSnapshot:
    """Replay-safe synchronized context plus its route-aligned Gscene."""

    context: SceneContext
    grounding: Any

    @property
    def observed_at(self):
        return self.context.observed_at

    @property
    def events(self):
        return self.context.events

    @property
    def state(self):
        return self.context.state

    @property
    def event_ids(self):
        return self.context.event_ids

    def to_dict(self):
        return {
            'context': self.context.to_dict(),
            'event_ids': list(self.event_ids),
            'has_grounding': self.grounding is not None,
        }


def scene_context_at(observed_at: float, reports: Iterable[Any]) -> SceneContext:
    """Build one deterministic ``Normal → Event → Normal`` scene snapshot.

    ``reports`` must be the latest synchronized public snapshot for
    ``observed_at`` (normally the result of ``runtime.reports()`` after the
    runtime has synchronized).  The function intentionally does not accept a
    schedule or infer an event's end time: the runtime's current ``active`` /
    ``cleared`` status is the public causal boundary.
    """
    if isinstance(observed_at, bool) or not isinstance(observed_at, (int, float)):
        raise SceneContextError('observed_at must be a finite nonnegative number')
    observed_at = float(observed_at)
    if not math.isfinite(observed_at) or observed_at < 0:
        raise SceneContextError('observed_at must be a finite nonnegative number')

    active, cleared = [], []
    seen = set()
    for report in tuple(reports):
        report_id = _report_id(report)
        if report_id in seen:
            raise SceneContextError(f'duplicate report ID in snapshot: {report_id}')
        seen.add(report_id)
        status = _report_status(report)
        _report_time(report)
        mapping_status = getattr(report, 'mapping_status', None)
        if mapping_status is not None and mapping_status != 'mapped':
            raise SceneContextError(
                f'report {report_id!r} is not mapped: {mapping_status!r}')
        (active if status == 'active' else cleared).append(report)

    # Runtime reports are normally sorted already, but scene identity must not
    # depend on a caller's container order.
    active.sort(key=_report_id)
    cleared.sort(key=_report_id)
    return SceneContext(observed_at, tuple(active), tuple(cleared))


@dataclass(frozen=True)
class SceneReplayState:
    """Replay-safe scene reference plus optional frozen text cache.

    ``cached_z_text_raw`` may be a CPU tensor/array produced by the frozen
    encoder.  ``grounding`` may carry the immutable Gnode sidecar needed for a
    later policy read.  No trainable ``z_scene`` is stored here; structured
    encoding and fusion are intentionally rerun with current parameters.
    """

    context: SceneContext
    cached_z_text_raw: Optional[Any] = None
    grounding: Optional[Any] = None

    def __post_init__(self):
        if not isinstance(self.context, SceneContext):
            raise TypeError('SceneReplayState.context must be a SceneContext')
        if (self.cached_z_text_raw is not None
                and bool(getattr(self.cached_z_text_raw, 'requires_grad', False))):
            raise ValueError('cached_z_text_raw must be detached frozen features')

    @property
    def scene_ref(self):
        return self.context.scene_ref


class SceneReplayTransition(NamedTuple):
    """Opt-in ten-field replay contract requested by the scene pipeline."""

    obs_t: Any
    phase_t: Any
    scene_t: Optional[SceneReplayState]
    action: Any
    reward: Any
    obs_t1: Any
    phase_t1: Any
    scene_t1: Optional[SceneReplayState]
    done: bool
    truncated: bool


def pack_scene_transition(obs_t, phase_t, scene_t, action, reward, obs_t1,
                          phase_t1, scene_t1, done, truncated=False):
    return SceneReplayTransition(
        obs_t, phase_t, scene_t, action, reward, obs_t1, phase_t1,
        scene_t1, bool(done), bool(truncated))


def unpack_scene_transition(payload) -> SceneReplayTransition:
    """Normalize a new ten-field or legacy six-field replay payload."""
    if isinstance(payload, SceneReplayTransition):
        return payload
    try:
        size = len(payload)
    except TypeError as error:
        raise ValueError('Replay payload must be a six- or ten-field sequence') from error
    if size == 10:
        return SceneReplayTransition(*payload)
    if size == 6:
        # Old agents did not retain termination semantics or scene inputs.
        return SceneReplayTransition(
            payload[0], payload[1], None, payload[2], payload[3],
            payload[4], payload[5], None, False, False)
    raise ValueError(f'Unsupported replay payload width: {size}')
