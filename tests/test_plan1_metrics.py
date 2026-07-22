import unittest

from world.world_sumo import World


class DummyLaneAPI:
    @staticmethod
    def getMaxSpeed(lane):
        return 10.0

    @staticmethod
    def getLength(lane):
        return 100.0


class DummyVehicleAPI:
    def __init__(self):
        self.vehicles = ['active']

    def getIDList(self):
        return list(self.vehicles)

    @staticmethod
    def getLanePosition(vehicle):
        return 50.0

    @staticmethod
    def getAccumulatedWaitingTime(vehicle):
        return 12.0


class DummyEngine:
    def __init__(self):
        self.lane = DummyLaneAPI()
        self.vehicle = DummyVehicleAPI()


class Plan1MetricTest(unittest.TestCase):
    def setUp(self):
        self.world = World.__new__(World)
        self.world.eng = DummyEngine()
        self.world.vehicle_trajectory = {
            'active': [['lane_a', 0, 15]],
            'finished': [['lane_a', 0, 20]],
        }
        self.world.vehicle_maxspeed = {
            ('active', 'lane_a'): 10.0,
            ('finished', 'lane_a'): 10.0,
        }
        self.world.real_delay = {'sentinel': 99.0}

    def test_real_delay_is_repeatable_and_side_effect_free(self):
        first = self.world.get_real_delay()
        second = self.world.get_real_delay()
        self.assertEqual(10.0, first)
        self.assertEqual(first, second)
        self.assertEqual({'sentinel': 99.0}, self.world.real_delay)

    def test_waiting_and_unfinished_use_active_vehicles(self):
        self.assertEqual(12.0, self.world.get_average_waiting_time())
        self.assertEqual(1, self.world.get_unfinished_vehicle_count())
        self.world.eng.vehicle.vehicles = []
        self.assertEqual(0.0, self.world.get_average_waiting_time())
        self.assertEqual(0, self.world.get_unfinished_vehicle_count())


if __name__ == '__main__':
    unittest.main()
