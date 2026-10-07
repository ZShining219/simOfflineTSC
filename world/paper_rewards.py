"""SUMO-only measurement adapter for versioned paper reward profiles.

Counts actual current stopped vehicles over complete external lanes, excluding
the artificial obstacle prefix. Does not read event schedules or text. Does not
reuse World's historical waiting-count or its 200 m observation cutoff.
"""
from pathlib import Path
import hashlib

import sumolib

from world.sumo_events.trex_blockage import PREFIX


class SumoRewardSource:
    def __init__(self, world):
        self.world = world
        net = sumolib.net.readNet(world.net)
        self.net_sha256 = hashlib.sha256(Path(world.net).read_bytes()).hexdigest()
        self.incoming, self.outgoing, self.movements = {}, {}, {}
        self.unsupported_frap_lanes = {}
        for inter in world.intersections:
            ins = tuple(l for road in inter.in_roads for l in inter.road_lane_mapping[road])
            outs = tuple(l for road in inter.out_roads for l in inter.road_lane_mapping[road])
            self.incoming[inter.id] = ins
            self.outgoing[inter.id] = outs
            selected = []
            mixed = []
            for lane_id in ins:
                directions = {c.getDirection() for c in net.getLane(lane_id).getOutgoing()}
                if directions & {'s', 'l'}:
                    if not directions <= {'s', 'l'}:
                        mixed.append(lane_id)
                    selected.append(lane_id)
            self.movements[inter.id] = tuple(selected)
            self.unsupported_frap_lanes[inter.id] = tuple(mixed)
        self._key, self._queues = None, {}

    def queues(self, threshold):
        # A reset changes Intersection instances, including when time returns to
        # the same value as the previous episode. Never reuse that old snapshot.
        key = (tuple(self.world.intersections), float(self.world.get_current_time()), threshold)
        if key != self._key:
            lanes = set(l for group in (self.incoming, self.outgoing) for row in group.values() for l in row)
            eng = self.world.eng
            self._queues = {l: sum(eng.vehicle.getSpeed(v) < threshold
                                   for v in eng.lane.getLastStepVehicleIDs(l)
                                   if not v.startswith(PREFIX)) for l in lanes}
            self._key = key
        return self._queues

    def reward(self, profile, node):
        q = self.queues(profile.stopped_speed_mps)
        lanes = self._selected_incoming(profile, node)
        out = self.outgoing[node] if profile.statistic == 'absolute_queue_pressure' else ()
        result = profile.calculate(tuple(q[l] for l in lanes), tuple(q[l] for l in out))
        result.update(incoming_lanes=lanes, outgoing_lanes=out)
        return result

    def _selected_incoming(self, profile, node):
        if profile.lane_scope != 'phase_movements':
            return self.incoming[node]
        if self.unsupported_frap_lanes[node] or not self.movements[node]:
            raise ValueError('FRAP reward adapter requires dedicated through/left-turn lanes')
        return self.movements[node]

    def lane_manifest(self, profile):
        return {node: {'incoming': list(self._selected_incoming(profile, node)),
                       'outgoing': list(self.outgoing[node]) if profile.statistic == 'absolute_queue_pressure' else []}
                for node in self.incoming}
