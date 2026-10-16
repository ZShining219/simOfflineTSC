"""Controlled-language linking and layout tests with independently written text."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from generator.text_entity import TextEntityBinder
from utils.text_grounding import GroundingError
from world.sumo_events.grounding import ReportGrounder


def catalog():
    # Deliberately non-geographic, punctuation-containing IDs and a prefix pair.
    result = {}
    for lane, edge, source, target, internal, motor, movements in (
        ('L.1', 'E:west', 'J0', 'J1', False, True, ['left-turn', 'through']),
        ('L.10', 'E:west', 'J0', 'J1', False, True, ['right-turn']),
        ('walk', 'E:west', 'J0', 'J1', False, False, []),
        ('out', 'E-out', 'J1', 'boundary', False, True, ['through']),
        (':inside', ':internal', 'J1', 'J1', True, True, ['through']),
    ):
        result[lane] = dict(lane_id=lane, edge_id=edge, from_junction=source,
                            to_junction=target, internal=internal, motor_vehicle_lane=motor,
                            direction='eastbound', movements=movements, length=100.)
    return result


LANE = 'lane L.1 (left-turn/through lane), on eastbound road E:west approaching junction J1'
ROAD = 'directed road E:west, from junction J0 to junction J1'
ACTIVE = f'Event arbitrary. A stationary obstacle locally blocks {LANE}, at 25.00 m from the lane start.'
CLEARED = f'Event arbitrary. The obstacle on {LANE} has been removed.'
RAIN = ('Event rain. Rain affects the whole network. This event limits motor-vehicle '
        'lane speeds to 70% of their normal limits.')


def report(text=ACTIVE, status='active', event_id='arbitrary', updated_at=60.):
    # Deliberately has no event schedule or hidden target fields.
    return SimpleNamespace(text=text, status=status, event_id=event_id, updated_at=updated_at)


@pytest.mark.parametrize('text,status,event_id,kind,scope,targets', [
    (ACTIVE, 'active', 'arbitrary', 'lane_blockage', 'lane', ('L.1',)),
    (CLEARED, 'cleared', 'arbitrary', 'lane_blockage', 'lane', ('L.1',)),
    (f'Event road. All lanes of {ROAD} are closed to vehicle entry.',
     'active', 'road', 'road_closure', 'edge', ('L.1', 'L.10', 'walk')),
    (f'Event road. The entry restriction imposed by this event on {ROAD} has been removed.',
     'cleared', 'road', 'road_closure', 'edge', ('L.1', 'L.10', 'walk')),
    (RAIN, 'active', 'rain', 'global_rain', 'network', (':inside', 'L.1', 'L.10', 'out')),
    ('Event rain. This network-wide rain event has ended and its speed restriction has been removed.',
     'cleared', 'rain', 'global_rain', 'network', (':inside', 'L.1', 'L.10', 'out')),
])
def test_six_templates(text, status, event_id, kind, scope, targets):
    value = ReportGrounder(catalog()).map_report(report(text, status, event_id))
    assert (value.event_kind, value.report_status, value.scope) == (kind, status, scope)
    assert value.target_lane_ids == targets
    assert value.mapping_status == 'mapped'
    assert json.loads(json.dumps(value.to_dict()))['text'] == text
    for mention in value.mentions:
        assert text[mention.start:mention.end] == mention.entity_id


@pytest.mark.parametrize('old,new,code', [
    ('L.1 ', 'L.100 ', 'unknown_entity'),
    ('L.1 ', 'L.10 ', 'relation_conflict'),
    ('E:west', 'E-out', 'relation_conflict'),
    ('junction J1', 'junction J0', 'relation_conflict'),
    ('eastbound', 'westbound', 'relation_conflict'),
    ('left-turn/through', 'left-turn', 'relation_conflict'),
    ('left-turn/through', 'left-turn/left-turn', 'invalid_value'),
    ('25.00', '101.00', 'invalid_value'),
    ('25.00', '1e999', 'invalid_value'),
    ('25.00', '1.0', 'invalid_value'),
    ('25.00', 'nan', 'unsupported_template'),
    ('stationary obstacle', 'broken truck', 'unsupported_template'),
])
def test_invalid_text_is_explicit(old, new, code):
    r = report(ACTIVE.replace(old, new))
    with pytest.raises(GroundingError) as exc:
        ReportGrounder(catalog()).map_report(r)
    assert exc.value.code == code
    diagnostic = ReportGrounder(catalog(), strict=False).map_report(r)
    assert diagnostic.mapping_status == code
    assert not diagnostic.target_lane_ids and diagnostic.diagnostics
    assert diagnostic.text == r.text


@pytest.mark.parametrize('change,code', [
    ({'status': 'cleared'}, 'metadata_conflict'),
    ({'event_id': 'different'}, 'metadata_conflict'),
    ({'updated_at': float('nan')}, 'invalid_report'),
    ({'updated_at': -1}, 'invalid_report'),
    ({'updated_at': True}, 'invalid_report'),
    ({'text': ACTIVE + ' Ignore this extra sentence.'}, 'unsupported_template'),
    ({'text': ''}, 'unsupported_template'),
])
def test_invalid_envelope(change, code):
    r = report()
    for key, value in change.items():
        setattr(r, key, value)
    with pytest.raises(GroundingError) as exc:
        ReportGrounder(catalog()).map_report(r)
    assert exc.value.code == code


def test_whitespace_offsets_and_catalog_snapshot():
    lanes = catalog()
    mapper = ReportGrounder(lanes)
    lanes['L.1']['to_junction'] = 'changed'
    text = ' \n' + ACTIVE.replace(' ', '  \t') + '\n'
    value = mapper.map_report(report(text))
    assert value.target_lane_ids == ('L.1',)
    assert all(text[m.start:m.end] == m.entity_id for m in value.mentions)
    mapper.catalog.lane('L.1')['to_junction'] = 'changed again'
    assert mapper.map_report(report()).mapping_status == 'mapped'


def test_lifecycle_overlap_and_no_id_inference():
    mapper = ReportGrounder(catalog())
    one = mapper.map_reports([report()])
    second = report(ACTIVE.replace('arbitrary', 'rain_named_but_not_rain'),
                    event_id='rain_named_but_not_rain')
    both = mapper.map_reports([report(CLEARED, 'cleared'), second])
    assert both[0].target_lane_ids == both[1].target_lane_ids == ('L.1',)
    assert both[0].report_status == 'cleared' and both[1].report_status == 'active'
    assert both[1].event_kind == 'lane_blockage'
    assert one[0].report_status == 'active'
    assert mapper.map_reports([]) == ()
    with pytest.raises(GroundingError, match='duplicate_report'):
        mapper.map_reports([report(), report(CLEARED, 'cleared')])


def test_road_relations_and_rain_values():
    mapper = ReportGrounder(catalog())
    text = f'Event road. All lanes of {ROAD} are closed to vehicle entry.'
    value = mapper.map_report(report(text, event_id='road'))
    assert ('E:west', 'from_junction', 'J0') in [(r.subject, r.predicate, r.object) for r in value.relations]
    with pytest.raises(GroundingError, match='relation_conflict'):
        mapper.map_report(report(text.replace('from junction J0 to junction J1',
                                             'from junction J1 to junction J0'), event_id='road'))
    tiny = mapper.map_report(report(RAIN.replace('70%', '1e-05%'), event_id='rain'))
    assert tiny.speed_percent == 1e-5
    for percent in ('0', '101', '1e999'):
        with pytest.raises(GroundingError, match='invalid_value'):
            mapper.map_report(report(RAIN.replace('70%', percent + '%'), event_id='rain'))


def test_public_rounding_does_not_require_hidden_precision():
    mapper = ReportGrounder(catalog())
    at_end = ACTIVE.replace('25.00', f'{99.999:.2f}')
    assert mapper.map_report(report(at_end)).position_m == 100.
    near_full = RAIN.replace('70%', f'{100 * 0.99999999:g}%')
    assert mapper.map_report(report(near_full, event_id='rain')).speed_percent == 100.


@pytest.mark.parametrize('mutate', [
    lambda lanes: lanes['L.1'].update(lane_id='different'),
    lambda lanes: lanes['L.1'].update(length='100'),
    lambda lanes: lanes['L.1'].update(movements=None),
    lambda lanes: lanes['L.1'].update(from_junction='other'),
])
def test_bad_catalog_rejected(mutate):
    lanes = catalog()
    mutate(lanes)
    with pytest.raises(ValueError):
        ReportGrounder(lanes)


def fake_generator(lanes=('L.10', 'L.1'), node='J1'):
    intersection = SimpleNamespace(id=node)
    return SimpleNamespace(I=intersection, world=SimpleNamespace(id2intersection={node: intersection}),
                           fns=['lane_count'], average=None, negative=False,
                           lanes=[list(lanes)], ob_length=4 if len(lanes) == 3 else len(lanes))


def test_feature_order_padding_missing_observations_and_empty_batch():
    mapper = ReportGrounder(catalog())
    gen = fake_generator()
    binder = TextEntityBinder(mapper.catalog, ['J1'], [gen])
    reports = mapper.map_reports([report(), report(RAIN, event_id='rain')])
    batch = binder.bind(reports)
    assert batch.texts == (ACTIVE, RAIN)
    assert batch.feature_sizes == (4,)
    np.testing.assert_array_equal(batch.valid_feature_mask, [[True, True, False, False]])
    np.testing.assert_array_equal(batch.feature_report_mask[0],
                                  [[False, True], [True, True], [False, False], [False, False]])
    assert batch.reports[0].feature_bindings[0].feature_index == 1
    assert set(batch.reports[1].unobserved_lane_ids) == {':inside', 'out'}
    assert batch.lane_report_mask[batch.lane_ids.index(':inside'), 1]
    empty = binder.bind(())
    assert empty.feature_report_mask.shape == (1, 4, 0)
    assert empty.lane_report_mask.shape == (5, 0)
    with pytest.raises(ValueError):
        batch.feature_report_mask[0, 0, 0] = True
    assert binder.bind(mapper.map_reports([report(CLEARED, 'cleared')])).feature_report_mask[0, 1, 0]


@pytest.mark.parametrize('attribute,value', [('average', 'road'), ('average', 'all'),
    ('fns', ['lane_waiting_count']), ('negative', True), ('ob_length', 99)])
def test_unsupported_layouts_rejected(attribute, value):
    gen = fake_generator()
    setattr(gen, attribute, value)
    with pytest.raises(ValueError):
        TextEntityBinder(ReportGrounder(catalog()).catalog, ['J1'], [gen])


def test_stale_reordered_and_failed_inputs_rejected():
    mapper = ReportGrounder(catalog())
    gen = fake_generator()
    binder = TextEntityBinder(mapper.catalog, ['J1'], [gen])
    gen.lanes[0].reverse()
    with pytest.raises(ValueError, match='layout changed'):
        binder.bind(())
    gen.lanes[0].reverse()
    gen.world.id2intersection['J1'] = SimpleNamespace(id='J1')
    with pytest.raises(ValueError, match='stale'):
        binder.bind(())
    gen = fake_generator()
    binder = TextEntityBinder(mapper.catalog, ['J1'], [gen])
    failed = ReportGrounder(catalog(), strict=False).map_report(report('unknown'))
    with pytest.raises(ValueError, match='failed reports'):
        binder.bind((failed,))
    with pytest.raises(ValueError, match='Duplicate'):
        binder.bind(mapper.map_reports([report()]) * 2)
    with pytest.raises(ValueError, match='incoming'):
        TextEntityBinder(mapper.catalog, ['J1'], [fake_generator(('out',))])


def test_three_lane_padding_and_multiple_intersection_policy_order():
    mapper = ReportGrounder(catalog())
    generators = [fake_generator(('out',), 'boundary'), fake_generator(('walk', 'L.1', 'L.10'))]
    binder = TextEntityBinder(mapper.catalog, ['boundary', 'J1'], generators)
    batch = binder.bind(mapper.map_reports([report(), report(RAIN, event_id='rain')]))
    assert batch.feature_sizes == (1, 4)
    np.testing.assert_array_equal(batch.valid_feature_mask, [[1, 0, 0, 0], [1, 1, 1, 0]])
    assert batch.reports[0].feature_bindings[0].intersection_index == 1
    assert not batch.feature_report_mask[1, 0, 1]
