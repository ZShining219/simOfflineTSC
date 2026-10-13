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

import numpy as np

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
                 sumo_seed=None, runtime_dir=None, ranks=None,
                 model_params=None, scene_configs=None, injects=None,
                 recorder=None, expert_id=None, gate=None,
                 human_timeout=30.0):
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
        # Intersection ranks the active controller drives; None -> every
        # signalized junction once the world is known.
        self._ranks_param = None if ranks is None else [int(r) for r in ranks]
        self.controlled_ranks = [0]
        self._rank_tl = {}
        self._tl_rank = {}
        self.model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        # raw spec strings; re-parsed per session so flags reset on restart
        self.inject_specs = list(injects or [])
        self.injects = []
        self._last_dispatch_child = None
        # Optional per-decision transition recorder (HITL data plane);
        # default None keeps the session completely unrecorded.
        self.recorder = recorder
        # HITL expert seat: ``expert_id`` names a controller that answers
        # only when ``gate`` fires (spec: no expert lookup without a
        # trigger, no queries when the seat is empty).
        self.expert_id = expert_id
        self.gate = gate
        self.human_timeout = float(human_timeout)
        self._expert_ctx = None     # expert answers for the open boundary
        self._expert_deadline = None  # wall-clock deadline for a human answer
        self._last_query = None     # {'t','junction','reason','intervened'}
        self._budget_logged = False
        self.runtime_dir = Path(
            runtime_dir or (Path(tempfile.gettempdir()) / 'sumo_live_broadcast')
        ).resolve()
        self.runtime_dir.mkdir(parents=True, exist_ok=True)

        self.world = None
        # Per-rank held green-phase targets; junctions without an entry fall
        # back to their own virtual_phase (frozen hold).
        self.held_actions = {}
        # Operator-pinned green phases, per rank; pinned junctions skip the
        # controller at every decision tick until released.
        self.manual_overrides = {}
        self.paused = False
        self.wait_enabled = False
        self.awaiting = None          # None | 'confirm'
        self._proposed_actions = None
        self._pending_choices = {}
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
        # recorder helpers keep instrumentation failures from ever
        # breaking traffic control
        self._recorder_errors = 0
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
        panels = {
            tl_id: _phase_panel_info(inter)
            for tl_id, inter in self.world.id2intersection.items()
        }
        self._init_payload = {
            'scene': self.scene,
            'network': self.network,
            'intersections': catalog,
            'panels': panels,
            'controlled': [
                self._rank_tl[rank] for rank in self.controlled_ranks
            ],
            'focus': self._rank_tl[self.controlled_ranks[0]],
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

    def manual_phase(self, phase_index, target=None):
        self.command(
            'manual_phase', phase=int(phase_index), target=target,
        )

    def release_manual(self, target=None):
        self.command('release_manual', target=target)

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
        if self.recorder is not None:
            self._recorder_end('shutdown')
            try:
                self.recorder.close()
            except Exception:
                pass
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

    # ------------------------------------------------------ recorder hooks
    # All hooks are failure-isolated: instrumentation must never break
    # traffic control.

    def _recorder_call(self, method, *args, **kwargs):
        if self.recorder is None:
            return
        try:
            getattr(self.recorder, method)(*args, **kwargs)
        except Exception as exc:
            self._recorder_errors += 1
            if self._recorder_errors <= 3:
                self._record_event(
                    f'recorder {method} 失败: {type(exc).__name__}: {exc}')

    def _recorder_begin(self):
        if self.recorder is None:
            self._expert_begin()
            return
        try:
            self.recorder.bind(self.world, self.controlled_ranks)
            self.recorder.begin_episode({'scene_key': self.scene_key})
        except Exception as exc:
            self._recorder_errors += 1
            self._record_event(
                f'recorder 绑定失败: {type(exc).__name__}: {exc}')
        self._expert_begin()

    def _recorder_end(self, reason):
        self._recorder_call('end_episode', reason)

    def _expert_begin(self):
        """Episode-start for the expert seat: bind the expert controller
        and reset the query gate's episode budget."""
        self._expert_ctx = None
        self._last_query = None
        self._budget_logged = False
        if self.gate is not None:
            self.gate.reset()
        if not self.expert_id:
            return
        if self.expert_id not in self.controllers:
            self._record_event(f'无效专家席: {self.expert_id}')
            return
        if self.expert_id == 'manual':
            self._record_event('专家席不接受 manual（人工接管走相位卡）')
            return
        try:
            self._bind_controller(self.expert_id)
        except Exception as exc:
            self._record_event(
                f'专家 {self.expert_id} 绑定失败: {type(exc).__name__}: {exc}')

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
            world.subscribe(
                ['lane_count', 'lane_waiting_count',
                 'lane_waiting_time_count'])
            world.reset()
            self._check_build_cancel()
            self._progress('解析路口几何与相位')
            self.world = world
            self.scene = Path(new_config).stem
            self.scene_key = scene_key
            self.source_config = new_config
            self._rank_tl = {
                rank: tl_id
                for rank, tl_id in enumerate(world.intersection_ids)
            }
            self._tl_rank = {
                tl_id: rank for rank, tl_id in self._rank_tl.items()
            }
            requested = (
                self._ranks_param
                if self._ranks_param is not None
                else list(self._rank_tl)
            )
            self.controlled_ranks = [
                rank for rank in requested if rank in self._rank_tl
            ] or [0]
            self.network = load_network_geometry(new_config)
            self._prepare_static_payload()
            self._reset_session_state()
            from tools.sumo_live_broadcast.injects import parse_injects
            self.injects = parse_injects(self.inject_specs)
            self._last_dispatch_child = None
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
            self._recorder_begin()
            self._next_wall = time.monotonic()
            self._set_state('live')
            self._record_event(f'session start {scene_key}')
        except _BuildCancelled:
            self._teardown_world(world, reason='cancel')
            self._set_state('idle')
            self._status_message = '已取消构建'
            self._publish_status()
        except Exception as exc:
            self._teardown_world(world, reason='error')
            self._set_state('error', f'{type(exc).__name__}: {exc}')

    def _teardown_world(self, world, reason='teardown'):
        self._recorder_end(reason)
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
        self.held_actions = {
            rank: int(inter.virtual_phase)
            for rank, inter in enumerate(self.world.intersections)
        }
        self.manual_overrides = {}
        self._decision_index = 0
        self._next_decision_t = float(self.world.get_current_time())
        self.awaiting = None
        self._proposed_actions = None
        self._pending_choices = {}
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
        self._teardown_world(self.world, reason='stop')
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
                if (self.awaiting == 'expert_request' and not self.paused
                        and self._expert_deadline is not None
                        and time.monotonic() >= self._expert_deadline):
                    self._resolve_awaiting({'timeout': True})
                    continue
                # Republish ~1/s while the banner is open so the human
                # sees the countdown tick down toward timeout.
                if (self.awaiting == 'expert_request'
                        and time.monotonic() - self._last_heartbeat >= 1.0):
                    self._last_heartbeat = time.monotonic()
                    self._publish(self._sample_frame())
                time.sleep(0.05)
                continue
            now = float(self.world.get_current_time())
            if now + 1e-6 >= self._next_decision_t:
                self._handle_decision(now)
                if self.awaiting is not None:
                    continue
            # Every junction steps toward its held target; junctions the
            # controller does not drive keep their own virtual_phase (frozen).
            if self.injects:
                from tools.sumo_live_broadcast.injects import apply_injects
                apply_injects(self.world, self.injects, now)
            actions = [
                self.held_actions.get(rank, int(inter.virtual_phase))
                for rank, inter in enumerate(self.world.intersections)
            ]
            self.world.step(actions)
            self._recorder_call('on_step', now)
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

    def _junction_queues(self):
        """Incoming-lane waiting count per controlled rank — the gate's
        trigger signal."""
        queues = {}
        try:
            lane_waiting = self.world.get_lane_waiting_vehicle_count()
        except Exception:
            return queues
        for rank in self.controlled_ranks:
            inter = self.world.id2intersection[self._rank_tl[rank]]
            lanes = [
                lane for road in getattr(inter, 'in_roads', [])
                for lane in inter.road_lane_mapping.get(road, [])
            ]
            queues[rank] = float(
                sum(lane_waiting.get(lane, 0.0) for lane in lanes))
        return queues

    def _student_margin(self, rank):
        """Student Q top-1/top-2 margin for one rank — the ``margin``
        gate's uncertainty signal.  ``None`` when the student exposes no
        Q-values (non-snapshot controllers)."""
        controller = self.controllers.get(self.controller_id)
        agent = getattr(controller, '_agents', {}).get(int(rank))
        if agent is None:
            return None
        try:
            from tools.sumo_live_broadcast.controllers import (
                inference_q_values)
            q = np.asarray(inference_q_values(agent), dtype=float)
            if q.size < 2:
                return None
            top2 = np.partition(q, -2)[-2:]
            return float(top2.max() - top2.min())
        except Exception:
            return None

    def _expert_consult(self, now, proposals):
        """Ask the expert seat where the gate fired; sets _expert_ctx."""
        self._expert_ctx = None
        self._expert_deadline = None
        if self.gate is None or not self.expert_id:
            return
        if self.expert_id == self.controller_id:
            return
        queues = self._junction_queues()
        if self.expert_id == 'manual':
            # Human banner seat: every gate-fired rank becomes an
            # ``expert_request`` shown to the operator; the query is billed
            # at gate.check time even if the human stays silent (timeout
            # resolves to the student's proposal — silence ≠ free).
            queries = {}
            for rank in self.controlled_ranks:
                if rank in self.manual_overrides or rank not in proposals:
                    continue
                fired, reason = self.gate.check(
                    rank, queues.get(rank, 0.0), self._decision_index,
                    margin=self._student_margin(rank), now=now)
                if not fired:
                    if (reason == 'budget_exhausted'
                            and not self._budget_logged):
                        self._record_event('人工干预预算耗尽')
                        self._budget_logged = True
                    continue
                queries[rank] = reason
            if not queries:
                return
            self._expert_ctx = {
                'id': 'human', 'actions': {}, 'queries': queries,
            }
            self._expert_deadline = time.monotonic() + self.human_timeout
            self.awaiting = 'expert_request'
            for rank, reason in queries.items():
                self._last_query = {
                    't': now, 'junction': self._rank_tl[rank],
                    'reason': reason, 'expert': 'human',
                    'intervened': False,
                }
                self._record_event(
                    f'人工请求 {self._rank_tl[rank]}'
                    f'（{reason}）@t={now:.0f}s')
            self._publish(self._sample_frame())
            return
        expert = self.controllers.get(self.expert_id)
        if expert is None or self.expert_id not in self._bound:
            return
        actions, queries = {}, {}
        for rank in self.controlled_ranks:
            if rank in self.manual_overrides:
                continue                      # already human-controlled
            if rank not in proposals:
                continue
            fired, reason = self.gate.check(
                rank, queues.get(rank, 0.0), self._decision_index,
                margin=self._student_margin(rank), now=now)
            if not fired:
                if reason == 'budget_exhausted' and not self._budget_logged:
                    self._record_event('专家干预预算耗尽')
                    self._budget_logged = True
                continue
            queries[rank] = reason
            try:
                actions[rank] = int(expert.decide(rank))
            except Exception as exc:
                actions[rank] = None
                self._record_event(
                    f'专家 {expert.label} 答题失败: {exc}')
        if not queries:
            return
        self._expert_ctx = {
            'id': expert.id, 'actions': actions, 'queries': queries,
        }
        for rank, reason in queries.items():
            changed = (
                actions.get(rank) is not None
                and actions[rank] != proposals.get(rank)
            )
            self._last_query = {
                't': now, 'junction': self._rank_tl[rank],
                'reason': reason, 'expert': expert.id,
                'intervened': bool(changed),
            }
            mark = '接管' if changed else '同判'
            self._record_event(
                f'专家 {expert.label} {mark} {self._rank_tl[rank]}'
                f'（{reason}）@t={now:.0f}s')

    def _accepted_actions(self, proposals):
        """Student proposals overlaid with expert answers (when queried)."""
        merged = dict(proposals)
        for rank, action in (self._expert_ctx or {}).get(
                'actions', {}).items():
            if action is not None:
                merged[rank] = action
        return merged

    def _handle_decision(self, now):
        self._decision_index += 1
        controller = self.controllers[self.controller_id]
        proposals = {}
        errors = {}
        for rank in self.controlled_ranks:
            if rank in self.manual_overrides:
                proposals[rank] = int(self.manual_overrides[rank])
                continue
            try:
                proposals[rank] = int(controller.decide(rank))
            except Exception as exc:  # never block traffic on a bad decision
                errors[self._rank_tl[rank]] = f'{type(exc).__name__}: {exc}'
                proposals[rank] = int(self.held_actions.get(rank, 0))
        self._proposed_actions = proposals
        self._pending_choices = {}
        self._next_decision_t += self.action_interval
        self._expert_consult(now, proposals)
        child = getattr(controller, 'child_label', None)
        if child is not None and child != self._last_dispatch_child:
            self._record_event(
                f'{controller.label} 调度 → {child} @t={now:.0f}s')
            self._last_dispatch_child = child
        if self.awaiting == 'expert_request':
            return                          # banner published in consult
        if self.wait_enabled:
            self.awaiting = 'confirm'
            self._publish(self._sample_frame())
            return
        self._commit_actions(
            self._accepted_actions(proposals), controller.id,
            errors=errors)

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
                      'manual_phase', 'release_manual', 'reset',
                      'resolve_decision'):
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
                self._resolve_awaiting({})
        elif op == 'switch_controller':
            self._switch_controller(kwargs['controller_id'])
        elif op == 'manual_phase':
            self._apply_manual_phase(int(kwargs['phase']), kwargs.get('target'))
        elif op == 'release_manual':
            self._release_manual(kwargs.get('target'))
        elif op == 'set_scene':
            self._switch_scene(kwargs['scene'])
        elif op == 'resolve_decision':
            if self.awaiting is not None:
                self._resolve_awaiting(kwargs)
        elif op == 'reset':
            self._reset_world()

    def _commit_actions(self, actions, source, note=None, errors=None):
        for rank, phase in actions.items():
            self.held_actions[rank] = int(phase)
        self._recorder_call(
            'on_decision', self._sim_time(), actions, source=source,
            proposals=self._proposed_actions,
            pending=self._pending_choices,
            manual=self.manual_overrides,
            expert=self._expert_ctx, note=note, errors=errors,
        )
        self._record_decision(
            actions, source, note=note, errors=errors,
        )
        self.awaiting = None
        self._proposed_actions = None
        self._pending_choices = {}
        self._expert_ctx = None

    def _resolve_awaiting(self, kwargs):
        if self.awaiting == 'expert_request':
            self._resolve_expert_request(kwargs)
            return
        controller = self.controllers[self.controller_id]
        # Order matters: proposals < expert answers < operator choices.
        merged = self._accepted_actions(self._proposed_actions or {})
        for rank, phase in self._pending_choices.items():
            merged[rank] = phase
        # Optional single-junction override carried on the resolve call.
        if kwargs.get('phase') is not None:
            merged[self._resolve_rank(kwargs.get('junction'))] = int(
                kwargs['phase']
            )
        note = None
        adjusted = [
            self._rank_tl[rank]
            for rank, phase in self._pending_choices.items()
            if self._proposed_actions is None
            or self._proposed_actions.get(rank) != phase
        ]
        for rank in self._pending_choices:
            if self._rank_tl[rank] in adjusted:
                self.manual_overrides[rank] = self._pending_choices[rank]
        if adjusted:
            note = '人工调整 ' + ','.join(adjusted)
        self._commit_actions(merged, controller.id, note=note)
        # Operator adjustments confirmed at a decision boundary are
        # interventions but did not alter a mid-flight interval.
        for tl in adjusted:
            rank = self._tl_rank.get(tl)
            self._recorder_call(
                'on_intervention', 'manual_pin', self._sim_time(),
                rank=rank, phase=merged.get(rank),
                detail={'via': 'resolve'}, mark_interval=False,
            )
        self._publish_status()

    def _resolve_expert_request(self, kwargs):
        """Close an ``expert_request`` banner: human answers (``choices``
        {rank-or-tl: phase}, or a single ``junction``/``phase`` pair) overlay
        the proposals as expert answers.  Timeout resolves to silence — the
        query stays billed (gate already spent it) and the student's
        proposal stands."""
        controller = self.controllers[self.controller_id]
        ctx = self._expert_ctx or {'id': 'human', 'actions': {}, 'queries': {}}
        timed_out = bool(kwargs.get('timeout'))
        answered = {}
        if not timed_out:
            for key, phase in (kwargs.get('choices') or {}).items():
                rank = self._resolve_rank(key)
                if rank in ctx['queries']:
                    answered[rank] = int(phase)
            if kwargs.get('phase') is not None:
                rank = self._resolve_rank(kwargs.get('junction'))
                if rank in ctx['queries']:
                    answered[rank] = int(kwargs['phase'])
        ctx['actions'] = dict(answered)
        self._expert_ctx = ctx
        proposals = self._proposed_actions or {}
        for rank, phase in answered.items():
            intervened = phase != proposals.get(rank)
            self._last_query = {
                't': self._sim_time(), 'junction': self._rank_tl[rank],
                'reason': ctx['queries'].get(rank), 'expert': 'human',
                'intervened': bool(intervened),
            }
            mark = '接管' if intervened else '同判'
            self._record_event(
                f'人工{mark} {self._rank_tl[rank]}')
        silent = [rank for rank in ctx['queries'] if rank not in answered]
        if timed_out:
            self._record_event('人工请求超时沉默（查询已计费）')
        for rank in silent:
            self._record_event(f'人工沉默 {self._rank_tl[rank]}（沿用学生方案）')
        merged = self._accepted_actions(proposals)
        self._commit_actions(merged, controller.id)
        self._publish_status()

    def _resolve_rank(self, target):
        """Map a junction selector value (tl id, or int rank) to a rank."""
        if target is None:
            return self.controlled_ranks[0]
        tl_id = str(target)
        if tl_id in self._tl_rank:
            return self._tl_rank[tl_id]
        try:
            rank = int(target)
        except (TypeError, ValueError):
            rank = -1
        if 0 <= rank < len(self._rank_tl):
            return rank
        raise ValueError(f'未知路口: {target}')

    def _apply_manual_phase(self, phase, target=None):
        rank = self._resolve_rank(target)
        if self.awaiting is not None:
            # While paused at a decision point a click only marks the choice
            # for that junction; '应用并继续' applies the merged vector.
            self._pending_choices[rank] = int(phase)
            if len(self.controlled_ranks) == 1:
                self._resolve_awaiting({})
            self._publish_status()
            return
        self.manual_overrides[rank] = int(phase)
        self.held_actions[rank] = int(phase)
        self._recorder_call(
            'on_intervention', 'manual_pin', self._sim_time(),
            rank=rank, phase=int(phase), detail={'via': 'override'},
        )
        self._record_decision(
            int(phase), 'manual',
            junction=self._rank_tl[rank], note='operator override',
        )
        self._publish_status()

    def _release_manual(self, target=None):
        controller = self.controllers[self.controller_id]
        ranks = (
            list(self.controlled_ranks)
            if target is None
            else [self._resolve_rank(target)]
        )
        for rank in ranks:
            if rank not in self.manual_overrides:
                continue
            self.manual_overrides.pop(rank, None)
            self._pending_choices.pop(rank, None)
            try:
                self.held_actions[rank] = int(controller.decide(rank))
            except Exception:
                pass
            self._recorder_call(
                'on_intervention', 'manual_release', self._sim_time(),
                rank=rank, detail={'controller': controller.id},
            )
            self._record_event(
                f'{self._rank_tl[rank]} 解除人工接管 → {controller.id}'
            )
        self._publish_status()

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
        self.controllers[controller_id].bind(
            self.world, ranks=self.controlled_ranks,
        )
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
        self._teardown_world(self.world, reason='scene_switch')
        self._set_state('idle')
        self._start_session(scene_key)

    def _reset_world(self):
        self._recorder_end('reset')
        self.world.reset()
        self._reset_session_state()
        self._bind_controller(self.controller_id)
        self._recorder_begin()
        self._record_event('simulation reset')

    # ----------------------------------------------------------------- frames

    def _sim_time(self):
        if self.world is None:
            return 0.0
        return float(self.world.get_current_time())

    def _record_decision(self, action, source, note=None, errors=None,
                         junction=None):
        entry = {
            't': self._sim_time(),
            'decision': self._decision_index,
            'source': source,
        }
        if isinstance(action, dict):
            entry['actions'] = {
                self._rank_tl[rank]: int(phase)
                for rank, phase in action.items()
                if rank in self._rank_tl
            }
        else:
            entry['action'] = int(action)
        if junction is not None:
            entry['junction'] = junction
        if note:
            entry['note'] = note
        if errors:
            entry['errors'] = dict(errors)
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
                           'kind': controller.kind,
                           'dispatch': getattr(
                               controller, 'child_label', None)},
            'held_actions': {
                tl: int(self.held_actions.get(rank, 0))
                for rank, tl in self._rank_tl.items()
            },
            'manual': {
                self._rank_tl[rank]: int(phase)
                for rank, phase in self.manual_overrides.items()
            },
            'decision_index': self._decision_index,
            'next_decision_in': max(0.0, self._next_decision_t - now),
            'proposed_actions': (
                {
                    self._rank_tl[rank]: int(phase)
                    for rank, phase in self._proposed_actions.items()
                }
                if self._proposed_actions is not None else None
            ),
            'pending_choices': {
                self._rank_tl[rank]: int(phase)
                for rank, phase in self._pending_choices.items()
            },
            'awaiting': self.awaiting,
            'expert_request': self._expert_request_payload(),
            'paused': bool(self.paused),
            'wait_enabled': bool(self.wait_enabled),
            'hitl': {
                'expert': self.expert_id if self.gate else None,
                'gate': self.gate.config() if self.gate else None,
                'queries_spent': (
                    self.gate.total_spent() if self.gate else 0),
                'last_query': self._last_query,
            },
            'speed': float(self.speed),
            'status': self._status_message,
            'history': self._history[-25:],
        }

    def _expert_request_payload(self):
        """Banner payload for the human seat; only built while an
        ``expert_request`` is open (per-rank queues are expensive)."""
        if self.awaiting != 'expert_request' or not self._expert_ctx:
            return None
        queries = self._expert_ctx.get('queries') or {}
        if not queries:
            return None
        queues = self._junction_queues()
        proposals = self._proposed_actions or {}
        answers = self._expert_ctx.get('actions') or {}
        return {
            'deadline_in': max(
                0.0, (self._expert_deadline or 0.0) - time.monotonic()),
            'ranks': [
                {
                    'rank': rank,
                    'junction': self._rank_tl[rank],
                    'reason': queries[rank],
                    'queue': queues.get(rank, 0.0),
                    'student': proposals.get(rank),
                    'answered': rank in answers,
                }
                for rank in sorted(queries)
            ],
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
