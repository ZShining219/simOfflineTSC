"""Control-step scene lifecycle and replay contract tests."""
from types import SimpleNamespace

import pytest

from utils.scene_context import (
    SceneContext,
    SceneContextError,
    SceneReplayState,
    pack_scene_transition,
    scene_context_at,
    unpack_scene_transition,
)


def report(event_id, status, updated_at):
    return SimpleNamespace(
        event_id=event_id,
        status=status,
        updated_at=float(updated_at),
        text=f'Event {event_id}. {status}',
    )


def test_normal_event_normal_and_cleared_is_audit_only():
    active = report('event_b', 'active', 10)
    cleared = report('event_a', 'cleared', 20)

    before = scene_context_at(9, ())
    during = scene_context_at(10, [cleared, active])
    after = scene_context_at(20, [cleared])

    assert isinstance(before, SceneContext)
    assert before.state == 'normal' and before.events == ()
    assert during.state == 'event'
    assert during.event_ids == ('event_b',)
    assert during.cleared_reports == (cleared,)
    assert after.state == 'normal' and after.events == ()
    assert after.cleared_reports == (cleared,)


def test_multiple_active_events_are_sorted_and_unique():
    value = scene_context_at(30, [report('z', 'active', 20), report('a', 'active', 10)])
    assert value.event_ids == ('a', 'z')
    assert value.scene_ref == ('a', 'z')

    with pytest.raises(SceneContextError, match='duplicate report ID'):
        scene_context_at(30, [report('a', 'active', 10), report('a', 'active', 11)])


@pytest.mark.parametrize('reports', [
    [report('future', 'active', 31)],
    [report('bad', 'pending', 1)],
])
def test_invalid_snapshot_cannot_define_a_scene(reports):
    with pytest.raises(SceneContextError):
        scene_context_at(30, reports)


def test_replay_payload_keeps_context_and_normalizes_legacy_shape():
    context = scene_context_at(10, [report('event', 'active', 10)])
    scene = SceneReplayState(context=context, cached_z_text_raw='raw')
    transition = pack_scene_transition(
        'obs', 'phase', scene, 'action', 'reward', 'next_obs',
        'next_phase', scene, True, True)
    assert len(transition) == 10
    assert unpack_scene_transition(transition).scene_t.context.event_ids == ('event',)
    legacy = unpack_scene_transition(('obs', 'phase', 'action', 'reward', 'next_obs', 'next_phase'))
    assert legacy.scene_t is None and legacy.done is False and legacy.truncated is False

