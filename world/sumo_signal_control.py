"""Opt-in, versioned green/yellow execution for new SUMO experiments.

The legacy World is a frozen dependency of historical experiments. Install
this adapter before reset for a new experiment identity; do not reinterpret
old checkpoints as having used corrected signal control.
"""
import math
from numbers import Integral
from types import MethodType


VERSION = 'sumo-green-yellow-v1'


def install_signal_control(world, config):
    if not isinstance(config, dict) or set(config) != {'version', 'yellow_seconds'}:
        raise ValueError('signal_control requires exactly version and yellow_seconds')
    seconds = config['yellow_seconds']
    if (config['version'] != VERSION or isinstance(seconds, bool)
            or not isinstance(seconds, (int, float)) or not math.isfinite(seconds)
            or seconds <= 0 or seconds != int(seconds)):
        raise ValueError('Expected sumo-green-yellow-v1 and positive integer yellow_seconds')
    if world.step_ratio != 1 or world.step_length != 1:
        raise ValueError('Signal control v1 requires one-second World steps')
    if getattr(world, '_connection_open', False):
        raise ValueError('Install signal control before starting a new episode')
    if hasattr(world, 'signal_control_config'):
        if world.signal_control_config != config:
            raise ValueError('Cannot replace signal control identity on an existing World')
        return
    world.signal_control_config = dict(config)
    original_reset = world.reset

    def reset(self):
        original_reset()
        for intersection in self.intersections:
            _configure(intersection, int(seconds))
        self._update_infos()

    world.reset = MethodType(reset, world)


def _configure(intersection, seconds):
    eng = intersection.eng
    phases = list(intersection.green_phases)
    transitions = {}
    for i, source in enumerate(intersection.green_phases):
        for j, target in enumerate(intersection.green_phases):
            if i == j:
                continue
            # Only extinguishing green links turn yellow. Shared green links
            # and permissive right-turn states retain their source state;
            # newly enabled target links cannot receive green before clearance.
            state = ''.join('y' if a in 'Gg' and b not in 'Gg' else a
                            for a, b in zip(source.state, target.state))
            if 'y' in state:
                transitions[i, j] = len(phases)
                phases.append(eng.trafficlight.Phase(seconds, state))
    logic = eng.trafficlight.Logic(intersection.id + '_signal_v1', 0, 0, phases)
    eng.trafficlight.setProgramLogic(intersection.id, logic)
    eng.trafficlight.setProgram(intersection.id, logic.programID)
    intersection.current_phase = intersection.virtual_phase = intersection.next_phase = 0
    intersection.current_phase_time = 0
    intersection.yellow_phase_time = seconds
    intersection.signal_is_yellow = False
    intersection.signal_yellow_elapsed = 0
    intersection.signal_actual_phase = 0
    intersection.signal_transitions = transitions
    intersection.signal_phases = phases
    _hold(intersection, 0)
    intersection.pseudo_step = MethodType(_step, intersection)


def _hold(intersection, phase):
    intersection.eng.trafficlight.setPhase(intersection.id, int(phase))
    # Disable autonomous progression between one-second World callbacks. The
    # adapter owns transitions, including long green holds and yellow expiry.
    intersection.eng.trafficlight.setPhaseDuration(intersection.id, 1e6)
    intersection.signal_actual_phase = phase


def _step(intersection, action):
    if (isinstance(action, bool) or not isinstance(action, Integral)
            or not 0 <= action < len(intersection.green_phases)):
        raise ValueError('SUMO action must be a valid integer green-phase index')
    action = int(action)
    if intersection.signal_is_yellow:
        if intersection.signal_yellow_elapsed < intersection.yellow_phase_time:
            _hold(intersection, intersection.signal_actual_phase)
            intersection.signal_yellow_elapsed += 1
            return
        # Requests during clearance cannot shorten it or replace its target.
        # Give that target at least this step of green before another change.
        intersection.signal_is_yellow = False
        _hold(intersection, intersection.current_phase)
        intersection.current_phase_time = 1
        return
    if action != intersection.current_phase:
        transition = intersection.signal_transitions.get((intersection.current_phase, action))
        intersection.current_phase = intersection.virtual_phase = intersection.next_phase = action
        intersection.current_phase_time = 0
        if transition is not None:
            intersection.signal_is_yellow = True
            intersection.signal_yellow_elapsed = 1
            _hold(intersection, transition)
            return
    _hold(intersection, intersection.current_phase)
    intersection.current_phase_time += 1
