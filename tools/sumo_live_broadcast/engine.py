"""Real-time SUMO broadcast engine.

One engine instance owns one ``world_sumo.World`` driven by libsumo on a
dedicated thread.  The simulation is paced against wall-clock time (1 sim
second per ``1/speed`` wall seconds) so the browser sees real-traffic-like
motion.  Traffic never waits for a controller decision: when a controller
raises or a human does not answer, the last held action simply keeps
running.  An optional "wait at decision point" mode can freeze the
simulation at each decision boundary until the operator confirms.
"""

from __future__ import annotations

import json
import queue
import tempfile
import threading
import time
from pathlib import Path

from tools.sumo_gui_comparison import (
    _configure_registry, _make_world_config,
)
from tools.sumo_html_comparison import (
    _intersection_catalog, _intersection_details, _traffic_light_states,
    _vehicle_states, load_network_geometry,
)

DEFAULT_MODEL_PARAMS = {
    't_fixed': 30,
    't_min': 10,
    'min_green_vehicle': 3,
    'max_red_vehicle': 6,
}


def _controlled_link_triples(engine, tl_id):
    """Per-link-index mapping of incoming/via/outgoing lane ids.

    ``getControlledLinks`` returns one entry per signalized link; the first
    tuple is ``(in_lane, out_lane, via_lane)`` where ``via_lane`` is the
    internal (``:``-prefixed) lane drawn inside the junction.
    """
    links = []
    try:
        raw = engine.trafficlight.getControlledLinks(tl_id)
    except Exception:
        return links
    for entry in raw or []:
        try:
            first = entry[0] if entry else None
        except (TypeError, IndexError):
            first = None
        record = {'in': None, 'via': None, 'out': None}
        if isinstance(first, (tuple, list)) and len(first) >= 3:
            record['in'] = str(first[0])
            record['out'] = str(first[1])
            record['via'] = str(first[2])
        links.append(record)
    return links


def _phase_panel_info(inter):
    """Green-phase metadata for the sidebar phase cards."""
    phases = []
    for index, phase in enumerate(inter.green_phases):
        phases.append({
            'index': index,
            'state': str(getattr(phase, 'state', '')),
        })
    return {
        'id': inter.id,
        'phases': phases,
        'phase_count': len(phases),
        'yellow_phase_time': int(inter.yellow_phase_time),
        'yellow_dict': {str(key): int(value) for key, value in inter.yellow_dict.items()},
    }


class _BuildCancelled(Exception):
    pass


class BroadcastEngine:
    """Session manager: owns the daemon loop; SUMO world lives only while a
    session is open (idle -> building -> live -> stopping -> idle)."""

    def __init__(self, scene, source_config, controllers,
                 default_controller_id=None, action_interval=10, speed=1.0,
                 sumo_seed=None, runtime_dir=None, rank=0,
                 model_params=None, scene_configs=None):
        self.scene = scene
        self.scene_key = str(scene)
        self.source_config = Path(source_config).expanduser().resolve()
        # {scene_key: config_path} for the on-page simulation-package switch.
        self.scene_configs = {
            str(key): Path(value).expanduser().resolve()
            for key, value in (scene_configs or {}).items()
        }
        self.controllers = dict(controllers)
        if not self.controllers:
            raise ValueError('At least one controller is required')
        self.controller_id = (
            default_controller_id
            if default_controller_id in self.controllers
            else next(iter(self.controllers))
        )
        self.action_interval = max(1, int(action_interval))
        self.speed = float(speed)
        self.sumo_seed = sumo_seed
        self.rank = int(rank)
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.runtime_dir = Path(
            runtime_dir or (Path(tempfile.gettempdir()) / 'sumo_live_broadcast')
        ).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.world = None
        self.held_action = 0
        self.paused = False
        self.wait_enabled = False
        self.awaiting = None          # None | 'confirm' | 'manual'
        self._proposed_action = None
        self._next_decision_t = 0.0
        self._decision_index = 0
        self._history = []
        self._commands = queue.Queue()
        self._cond = threading.Condition()
        self._latest_frame = None
        self._frame_version = 0
        self._stop = threading.Event()
        self._thread = None
        self._bound = set()
        self._next_wall = None
        self.rebuilding = False
        self.network = None
        self._init_payload = None
        self._status_message = ''
        # Session lifecycle: idle -> building -> live -> stopping -> idle.
        # 'error' retains the failure message until the next start attempt.
        self.session_state = 'idle'
        self.session_progress = []    # [{'t': wall_ts, 'msg': str}]
        self.session_error = None
        self._build_cancel = False
        self._last_heartbeat = 0.0

    # ------------------------------------------------------------------ setup

    def start(self):
        """Launch the permanent manager thread; the SUMO world is only built
        when a session is opened via the start_session command."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run, name='sumo-live-broadcast', daemon=True,
        )
        self._thread.start()

    def _controller_entries(self):
        return [
            {
                'id': controller.id,
                'label': controller.label,
                'kind': controller.kind,
                'detail': getattr(controller, 'detail', {}),
                'path': getattr(controller, 'weights_path', None),
            }
            for controller in self.controllers.values()
        ]

    def _session_info(self):
        return {
            'state': self.session_state,
            'scene_key': self.scene_key,
            'progress': list(self.session_progress),
            'error': self.session_error,
        }

    def _prepare_static_payload(self):
        catalog = _intersection_catalog(self.world.eng)
        for tl_id, item in catalog.items():
            item['links'] = _controlled_link_triples(self.world.eng, tl_id)
        inter = self.world.id2intersection[
            self.world.intersection_ids[self.rank]
        ]
        self._init_payload = {
            'scene': self.scene,
            'network': self.network,
            'intersections': catalog,
            'intersection': _phase_panel_info(inter),
        }

    def init_payload(self):
        payload = {
            'scene_key': self.scene_key,
            'scenes': [
                {'id': key, 'label': key,
                 'network': Path(cfg).stem}
                for key, cfg in self.scene_configs.items()
            ],
            'controllers': self._controller_entries(),
            'default_controller': self.controller_id,
            'action_interval': self.action_interval,
            'speed': self.speed,
            'session': self._session_info(),
        }
        if self._init_payload is not None:
            payload.update(self._init_payload)
        return payload

    # ---------------------------------------------------------------- commands

    def command(self, op, **kwargs):
        """Queue a control command for the simulation thread."""
        self._commands.put((str(op), kwargs))

    def pause(self, flag):
        self.command('pause', value=bool(flag))

    def set_speed(self, value):
        self.command('speed', value=float(value))

    def set_wait(self, flag):
        self.command('wait', value=bool(flag))

    def switch_controller(self, controller_id):
        self.command('switch_controller', controller_id=str(controller_id))

    def manual_phase(self, phase_index):
        self.command('manual_phase', phase=int(phase_index))

    def set_scene(self, scene_key):
        self.command('set_scene', scene=str(scene_key))

    def reset(self):
        self.command('reset')

    def start_session(self, scene_key=None):
        self.command(
            'start_session',
            scene=str(scene_key) if scene_key else None,
        )

    def stop_session(self):
        self.command('stop_session')

    def cancel_build(self):
        self._build_cancel = True

    def shutdown(self):
        self._stop.set()
        self._build_cancel = True
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.world is not None:
            try:
                self.world.close()
            except Exception:
                pass

    # -------------------------------------------------------------- main loop

    # ------------------------------------------------------------ session core

    def _set_state(self, state, error=None):
        self.session_state = state
        if state == 'building':
            self.session_progress = []
            self.session_error = None
        if error is not None:
            self.session_error = error
        if state == 'idle':
            self.session_progress = []
        self._publish_status()

    def _progress(self, msg):
        self.session_progress.append({'t': time.time(), 'msg': msg})
        self._publish_status()

    def _check_build_cancel(self):
        if self._build_cancel:
            raise _BuildCancelled()

    def _start_session(self, scene_key=None):
        """Build the SUMO world in stages; runs on the manager thread so
        progress/cancel commands keep flowing between stages."""
        if self.session_state in ('building', 'live'):
            return
        scene_key = str(scene_key) if scene_key else self.scene_key
        if scene_key not in self.scene_configs:
            self._set_state('error', f'未知仿真包: {scene_key}')
            return
        self._build_cancel = False
        self._set_state('building')
        world = None
        try:
            self._progress('解析仿真配置')
            new_config = self.scene_configs[scene_key]
            world_config = _make_world_config(
                new_config, self.runtime_dir, 'live', gui=False,
            )
            world_class = _configure_registry(
                world_config, self.sumo_seed, self.runtime_dir,
            )
            from common.registry import Registry

            Registry.mapping['model_mapping']['setting'].param = dict(
                self.model_params
            )
            self._check_build_cancel()
            self._progress('启动 libsumo 并加载路网（首次约 30-60 秒）')
            world = world_class(str(world_config), interface='libsumo')
            self._check_build_cancel()
            self._progress('订阅指标并初始化仿真')
            world.subscribe(['lane_count', 'lane_waiting_count'])
            world.reset()
            self._check_build_cancel()
            self._progress('解析路口几何与相位')
            self.world = world
            self.scene = Path(new_config).stem
            self.scene_key = scene_key
            self.source_config = new_config
            self.network = load_network_geometry(new_config)
            self._prepare_static_payload()
            self._reset_session_state()
            self._check_build_cancel()
            self._progress(f'绑定控制器 {self.controller_id}')
            try:
                self._bind_controller(self.controller_id)
            except Exception as exc:
                self._record_event(
                    f'{self.controller_id} 不支持 {scene_key}: {exc}'
                )
                self.controller_id = 'fixedtime'
                self._bind_controller(self.controller_id)
            self._next_wall = time.monotonic()
            self._set_state('live')
            self._record_event(f'session start {scene_key}')
        except _BuildCancelled:
            self._teardown_world(world)
            self._set_state('idle')
            self._status_message = '已取消构建'
            self._publish_status()
        except Exception as exc:
            self._teardown_world(world)
            self._set_state('error', f'{type(exc).__name__}: {exc}')

    def _teardown_world(self, world):
        try:
            if world is not None:
                world.close()
        except Exception:
            pass
        if self.world is world:
            self.world = None
        self.network = None
        self._init_payload = None
        self._bound.clear()
        self._latest_frame = None

    def _reset_session_state(self):
        self.held_action = 0
        self._decision_index = 0
        self._next_decision_t = float(self.world.get_current_time())
        self.awaiting = None
        self._proposed_action = None
        self._bound.clear()
        self._history.clear()
        self.paused = False
        self._next_wall = time.monotonic()

    def _stop_session(self):
        if self.session_state == 'building':
            self._build_cancel = True
            return
        if self.session_state != 'live':
            self._set_state('idle')
            return
        self._set_state('stopping')
        self._teardown_world(self.world)
        self._set_state('idle')

    # -------------------------------------------------------------- main loop

    def _run(self):
        self._next_wall = time.monotonic()
        while not self._stop.is_set():
            self._drain_commands()
            if self._stop.is_set():
                break
            if self.session_state != 'live' or self.world is None:
                self._heartbeat()
                time.sleep(0.05)
                continue
            if self.paused or self.awaiting is not None:
                time.sleep(0.05)
                continue
            now = float(self.world.get_current_time())
            if now + 1e-6 >= self._next_decision_t:
                self._handle_decision(now)
                if self.awaiting is not None:
                    continue
            # Every controlled junction keeps its current target; only the
            # broadcast rank receives the held action (S1-S4 are 1x1 anyway).
            actions = [
                int(inter.virtual_phase) for inter in self.world.intersections
            ]
            actions[self.rank] = self.held_action
            self.world.step(actions)
            self._publish(self._sample_frame())
            self._pace()
        self._publish_status()

    def _heartbeat(self):
        """Keep SSE subscribers fed while idle/building (status type)."""
        now = time.monotonic()
        if now - self._last_heartbeat >= 1.0:
            self._last_heartbeat = now
            self._publish_status()

    def _pace(self):
        """Sleep until the next wall-clock slot; stays interruptible."""
        self._next_wall += 1.0 / max(self.speed, 1e-6)
        lag = self._next_wall - time.monotonic()
        if lag <= 0:
            # Behind schedule (slow step or burst after pause): re-baseline
            # instead of bursting.
            self._next_wall = time.monotonic()
            return
        deadline = self._next_wall
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self._stop.is_set():
                return
            time.sleep(min(remaining, 0.05))
            # Drain urgent commands (pause / manual) so they feel instant.
            if not self._commands.empty():
                self._drain_commands()
                if self.paused or self.awaiting is not None:
                    self._next_wall = time.monotonic()
                    return

    def _handle_decision(self, now):
        self._decision_index += 1
        controller = self.controllers[self.controller_id]
        proposal = None
        error = None
        if not (self.wait_enabled and controller.kind == 'manual'):
            try:
                proposal = int(controller.decide())
            except Exception as exc:  # never block traffic on a bad decision
                error = f'{type(exc).__name__}: {exc}'
                proposal = int(self.held_action)
        self._proposed_action = proposal
        self._next_decision_t += self.action_interval
        if self.wait_enabled:
            self.awaiting = 'manual' if controller.kind == 'manual' else 'confirm'
            self._publish(self._sample_frame())
            return
        self.held_action = proposal
        self._record_decision(proposal, controller.id, error=error)

    def _drain_commands(self):
        while True:
            try:
                op, kwargs = self._commands.get_nowait()
            except queue.Empty:
                return
            try:
                self._apply_command(op, kwargs)
            except Exception as exc:
                self._status_message = f'{op} failed: {exc}'
            if op in ('pause', 'speed', 'wait', 'switch_controller',
                      'manual_phase', 'reset', 'resolve_decision'):
                self._publish_status()

    def _apply_command(self, op, kwargs):
        if op == 'start_session':
            self._start_session(kwargs.get('scene'))
        elif op == 'stop_session':
            self._stop_session()
        elif op == 'cancel_build':
            self._build_cancel = True
        elif self.session_state != 'live':
            self._status_message = f'仿真未运行，忽略命令 {op}'
            self._publish_status()
        elif op == 'pause':
            self.paused = bool(kwargs.get('value'))
            self._next_wall = time.monotonic()
        elif op == 'speed':
            self.speed = min(max(float(kwargs.get('value')), 0.05), 32.0)
            self._next_wall = time.monotonic()
        elif op == 'wait':
            self.wait_enabled = bool(kwargs.get('value'))
            if not self.wait_enabled and self.awaiting is not None:
                self._resolve_awaiting(self._proposed_action)
        elif op == 'switch_controller':
            self._switch_controller(kwargs['controller_id'])
        elif op == 'manual_phase':
            self._apply_manual_phase(int(kwargs['phase']))
        elif op == 'set_scene':
            self._switch_scene(kwargs['scene'])
        elif op == 'resolve_decision':
            if self.awaiting is not None:
                action = kwargs.get('phase')
                if action is None:
                    action = (
                        self._proposed_action
                        if self._proposed_action is not None
                        else self.held_action
                    )
                self._resolve_awaiting(int(action))
        elif op == 'reset':
            self._reset_world()

    def _resolve_awaiting(self, action):
        controller = self.controllers[self.controller_id]
        if (
            controller.kind != 'manual'
            and self._proposed_action is not None
            and int(action) != int(self._proposed_action)
        ):
            # Picking a different phase than the model proposed is an
            # operator override: hand control to the manual controller.
            self._switch_controller('manual')
            controller = self.controllers[self.controller_id]
        if controller.kind == 'manual':
            controller.request_phase(action)
        self.held_action = int(action)
        self._record_decision(int(action), controller.id)
        self.awaiting = None
        self._proposed_action = None

    def _apply_manual_phase(self, phase):
        controller = self.controllers[self.controller_id]
        if controller.kind != 'manual':
            # Clicking a phase while a model is active takes over control.
            self._switch_controller('manual')
            controller = self.controllers[self.controller_id]
        controller.request_phase(phase)
        if self.awaiting is not None:
            self._resolve_awaiting(phase)
            return
        self.held_action = int(phase)
        self._record_decision(
            int(phase), 'manual', note='operator override',
        )

    def _switch_controller(self, controller_id):
        if controller_id not in self.controllers:
            raise ValueError(f'Unknown controller: {controller_id}')
        self._bind_controller(controller_id)
        self.controller_id = controller_id
        self._record_event(
            f"controller -> {self.controllers[controller_id].label}"
        )

    def _bind_controller(self, controller_id):
        if controller_id in self._bound:
            return
        self.controllers[controller_id].bind(self.world, rank=self.rank)
        self._bound.add(controller_id)

    def _switch_scene(self, scene_key):
        """Live scene switch = close current session and rebuild for the new
        package; progress is reported through the normal building stages."""
        if scene_key not in self.scene_configs:
            raise ValueError(f'Unknown scene: {scene_key}')
        if scene_key == self.scene_key:
            return
        self._record_event(f'scene switch -> {scene_key}')
        self._set_state('stopping')
        self._teardown_world(self.world)
        self._set_state('idle')
        self._start_session(scene_key)

    def _reset_world(self):
        self.world.reset()
        self.held_action = 0
        self._decision_index = 0
        self._next_decision_t = 0.0
        self.awaiting = None
        self._proposed_action = None
        self._bound.clear()
        self._bind_controller(self.controller_id)
        self._history.clear()
        self._record_event('simulation reset')
        self._next_wall = time.monotonic()

    # ----------------------------------------------------------------- frames

    def _sim_time(self):
        if self.world is None:
            return 0.0
        return float(self.world.get_current_time())

    def _record_decision(self, action, source, note=None, error=None):
        entry = {
            't': self._sim_time(),
            'decision': self._decision_index,
            'action': int(action),
            'source': source,
        }
        if note:
            entry['note'] = note
        if error:
            entry['error'] = error
        self._history.append(entry)
        del self._history[:-80]

    def _record_event(self, message):
        self._history.append({
            't': self._sim_time(),
            'event': message,
        })
        del self._history[:-80]

    def _sample_frame(self):
        world = self.world
        vehicles = _vehicle_states(world.eng)
        lights = _traffic_light_states(world.eng)
        catalog = self._init_payload['intersections']
        intersections = _intersection_details(world, catalog, vehicles, lights)
        lane_queue = world.get_lane_waiting_vehicle_count()
        inter = world.id2intersection[world.intersection_ids[self.rank]]
        now = float(world.get_current_time())
        controller = self.controllers[self.controller_id]
        return {
            'type': 'frame',
            't': now,
            'scene': self.scene,
            'scene_key': self.scene_key,
            'rebuilding': bool(self.rebuilding),
            'vehicles': [
                [v['id'], v['x'], v['y'], v['angle'], v['speed']]
                for v in vehicles
            ],
            'lights': {
                tl: [state['phase'], state['state']]
                for tl, state in lights.items()
            },
            'intersections': intersections,
            'queue': float(sum(lane_queue.values())),
            'throughput': int(world.get_cur_throughput()),
            'vehicles_total': len(vehicles),
            'halting': sum(1 for v in vehicles if v['speed'] < 0.1),
            'controller': {'id': controller.id, 'label': controller.label,
                           'kind': controller.kind},
            'held_action': int(self.held_action),
            'virtual_phase': int(inter.virtual_phase),
            'current_phase_raw': int(inter.get_current_phase()),
            'decision_index': self._decision_index,
            'next_decision_in': max(0.0, self._next_decision_t - now),
            'proposed_action': self._proposed_action,
            'awaiting': self.awaiting,
            'paused': bool(self.paused),
            'wait_enabled': bool(self.wait_enabled),
            'speed': float(self.speed),
            'status': self._status_message,
            'history': self._history[-25:],
        }

    def _publish(self, frame):
        with self._cond:
            self._latest_frame = frame
            self._frame_version += 1
            self._cond.notify_all()

    def _publish_status(self):
        if self.session_state == 'live' and self.world is not None:
            self._publish(self._sample_frame())
            return
        self._publish({
            'type': 'status',
            'session': self._session_info(),
            'status': self._status_message,
        })

    def latest_frame(self, since_version=0, timeout=30.0):
        """Return (frame, version) blocking until a newer frame exists."""
        with self._cond:
            if self._latest_frame is None or self._frame_version <= since_version:
                self._cond.wait_for(
                    lambda: self._stop.is_set()
                    or (self._latest_frame is not None
                        and self._frame_version > since_version),
                    timeout=timeout,
                )
            return self._latest_frame, self._frame_version
