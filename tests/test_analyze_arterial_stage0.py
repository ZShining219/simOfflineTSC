from tools.analyze_arterial_stage0 import round_up_25, stable_endpoint


def records(values):
    return [dict(travel_time=value, queue=value, delay=value, throughput=value)
            for value in values]


def test_round_up_25():
    assert round_up_25(200) == 200
    assert round_up_25(201) == 225


def test_stability_rule_requires_all_metrics():
    stable, changes = stable_endpoint(records([100] * 40), 40, 20, 0.10)
    assert stable
    assert set(changes) == {'travel_time', 'queue', 'delay', 'throughput'}
    unstable_values = [100] * 20 + [120] * 20
    stable, _ = stable_endpoint(records(unstable_values), 40, 20, 0.10)
    assert not stable
