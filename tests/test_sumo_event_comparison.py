"""Metric definitions used in the event evidence figures."""
import pytest

pytest.importorskip('libsumo')
from world.sumo_events.compare import summarize_window


def test_speed_uses_vehicle_time_weighting_and_queue_counts():
    rows = [dict(vehicles=1, sum_speed_mps=10, queue_vehicles=0, target_lane_queue=0,
                 closure_area_queue=0, closure_road_entries=2),
            dict(vehicles=9, sum_speed_mps=0, queue_vehicles=9, target_lane_queue=2,
                 closure_area_queue=7, closure_road_entries=0)]
    result = summarize_window(rows)
    assert result['mean_speed_mps'] == 1  # Not (10 + 0) / 2 = 5.
    assert result['mean_queue_vehicles'] == 4.5
    assert result['queued_vehicle_seconds'] == 9
    assert result['closure_road_entries'] == 2


def test_empty_traffic_has_no_observed_speed():
    result = summarize_window([dict(vehicles=0, sum_speed_mps=0, queue_vehicles=0,
                                    target_lane_queue=0, closure_area_queue=0, closure_road_entries=0)])
    assert result['mean_speed_mps'] is None
    assert result['mean_queue_vehicles'] == 0
