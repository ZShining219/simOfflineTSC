import unittest

import numpy as np

from common.metrics import Metrics


class DummyWorld:
    def __init__(self):
        self.intersections = [object(), object()]

    def get_cur_throughput(self):
        return 7

    def get_average_travel_time(self):
        return 12.5

    def get_real_delay(self):
        return 2.5


class DummyAgent:
    def __init__(self, queue, delay):
        self.queue = queue
        self.delay = delay

    def get_queue(self):
        return self.queue

    def get_delay(self):
        return self.delay


class MetricCharacterizationTest(unittest.TestCase):
    def test_existing_approximate_aggregations_are_characterized(self):
        world = DummyWorld()
        agents = [DummyAgent(3, 1), DummyAgent(5, 3)]
        metrics = Metrics(['rewards', 'queue', 'delay'], [], world, agents)
        metrics.update(np.array([-2, -4], dtype=np.float32))
        metrics.update(np.array([-2, -4], dtype=np.float32))
        self.assertEqual(-6, metrics.rewards())
        self.assertEqual(4, metrics.queue())
        self.assertEqual(2, metrics.delay())
        self.assertEqual(7, metrics.throughput())
        self.assertEqual(12.5, metrics.real_average_travel_time())

    def test_real_delay_branch_delegates_to_world(self):
        metrics = Metrics(['rewards', 'queue'], ['delay'], DummyWorld(), [])
        self.assertEqual(2.5, metrics.delay())


if __name__ == '__main__':
    unittest.main()
