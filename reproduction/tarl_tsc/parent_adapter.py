"""One graph per decision/replay item, using the parent SUMO trainer and DQN helpers.

All five methods share observation, reward, exploration, replay and TD updates.
Frozen embeddings are collected with observations, so training never queries
live simulator text for an old replay transition.
"""
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils import clip_grad_norm_
from common.registry import Registry
from generator import LaneVehicleGenerator, IntersectionPhaseGenerator
from .models.tarl import TARLPolicy
from .models.paper_tarl import PaperTARLPolicy
from .text.encoder import FrozenTextEncoder
from .text.conditions import apply_condition, CONDITIONS
from .events.runtime import install_event_runtime, load_schedule
from .models.paper_tarl import ZTASK_FIELDS

HERE = Path(__file__).resolve().parent
TEXT_METHODS = ('concat', 'attention', 'gating')
IMPLS = ('legacy', 'paper')


class _QWrapper(nn.Module):
    def __init__(self, traffic_dim, actions, method, adjacency, lane_dim,
                 impl='legacy', text_width=768, text_input='bert',
                 semantic_aux=False, fusion_gate=False, ztask_vocabs=None):
        super().__init__()
        if impl == 'paper':
            self.policy = PaperTARLPolicy(
                traffic_dim, actions, method=method, text_input=text_input,
                semantic_aux=semantic_aux, fusion_gate=fusion_gate,
                ztask_vocabs=ztask_vocabs)
            self.text_methods = PaperTARLPolicy.TEXT_METHODS
        else:
            self.policy = TARLPolicy(traffic_dim, actions, method=method)
            self.text_methods = TEXT_METHODS
        self.method, self.lane_dim = method, lane_dim
        self.text_width = text_width
        self.register_buffer('adjacency', adjacency)

    def forward(self, x, train=False):
        if self.method in self.text_methods:
            w = self.lane_dim + self.text_width
            traffic = torch.cat([x[..., :self.lane_dim], x[..., w:]], dim=-1)
            text_features = x[..., self.lane_dim:w]
        else:
            traffic, text_features = x, None
        if isinstance(self.policy, PaperTARLPolicy):
            return self.policy(traffic, text_features, self.adjacency, train=train)
        return self.policy(traffic, text_features, self.adjacency)


def make_tarl_agent(method):
    # Lazy dependency avoids a cycle through agent.__init__ -> agent.tarl when
    # this independent adapter is imported directly by validation tools.
    from agent.dqn import DQNAgent

    class TARLAgent(DQNAgent):
        def __init__(self, world, rank):
            if rank != 0:
                raise ValueError('TARL uses one shared agent for the complete graph')
            self.method = method
            self.node_ids = tuple(world.intersection_ids)
            graph = json.loads((HERE / 'data/hangzhou_4x4_gudang_18041610_1h/adjacency.json').read_text())
            if set(graph) == set(self.node_ids):
                self.adjacency = torch.tensor([[a == b or b in graph[a] for b in self.node_ids] for a in self.node_ids])
            elif len(self.node_ids) == 1:
                # Single-junction net (e.g. sumohz1x1 quick validation): the
                # GAT degenerates to a self-loop; cross-node attention/fusion
                # paths still execute for conditional-behavior checks.
                self.adjacency = torch.ones((1, 1), dtype=torch.bool)
            else:
                raise ValueError('TARL adjacency does not match the SUMO intersection IDs')
            self.impl = Registry.mapping['model_mapping']['setting'].param.get(
                'tarl_impl', 'legacy')
            if self.impl not in IMPLS:
                raise ValueError(f'Invalid tarl_impl {self.impl!r}')
            if self.impl == 'paper' and method not in PaperTARLPolicy.METHODS:
                raise ValueError(f'paper impl has no method {method!r}')
            # ATT-ENTITY-004 revised line: 'structured' replaces the frozen BERT
            # text slot with per-node grounded event fields (present/type/
            # scope/direction/movement/severity) mapped through the existing
            # ReportGrounder + TextEntityBinder pipeline.  Must be set before
            # super().__init__ because DQNAgent.__init__ calls _build_model().
            model_param = Registry.mapping['model_mapping']['setting'].param
            self.event_input = model_param.get('event_input', 'bert')
            self.semantic_aux = dict(model_param.get('semantic_aux') or {})
            # ATT-ENTITY-004 repair knobs (defaults preserve prior behaviour):
            self.td_loss_fn = model_param.get('td_loss_fn', 'mse')
            self.huber_beta = float(model_param.get('huber_beta', 10.0))
            self.text_dropout = float(model_param.get('text_dropout', 0.0))
            self.fusion_gate = bool(model_param.get('fusion_gate', False))
            # demand_surge arms widen the z_task field vocabularies
            # (lifecycle 0/1/2 + type code 4): model.ztask_vocabs supplies the
            # full tuple, e.g. [3,5,4,5,6]; model.ztask_type_vocab remains a
            # shorthand that only widens the type column.  Default None keeps
            # all pre-surge checkpoints loadable.
            vocabs = model_param.get('ztask_vocabs')
            if vocabs is None and model_param.get('ztask_type_vocab') is not None:
                from reproduction.tarl_tsc.models.paper_tarl import ZTASK_VOCABS
                vocabs = list(ZTASK_VOCABS)
                vocabs[1] = int(model_param['ztask_type_vocab'])
            self.ztask_vocabs = vocabs
            self.starvation_guard = dict(model_param.get('starvation_guard') or {})
            self._guard_on = bool(self.starvation_guard.get('enabled'))
            self._guard_stats = {'fired_steps': 0, 'fired_nodes': 0}
            # closure_bind arm: unbindable edge-scope road_closure reports are
            # bound at node level to the nearest controlled upstream junction
            # (TextEntityBinder closure_binding='nearest_upstream').  Default
            # None preserves the legacy coverage gap for all other arms.
            self.closure_binding = model_param.get('closure_binding')
            self._closure_bind_hops = int(
                model_param.get('closure_bind_max_hops', 4))
            # Event-semantic auxiliary reward (pairaux line): a localised
            # shaping term added to the REPLAYED reward only — metrics keep the
            # native reward.  aux_i = -lambda * 12 * mean(lane_waiting_count)
            # over the lanes feeding the event-affected corridor at actionable
            # nodes (closure feeder + optional 2-hop upstream; blockage's
            # bound lane).  It amplifies the event-caused share of the true
            # cost so credit assignment can tie text->action->local outcome.
            self.reward_aux = dict(model_param.get('reward_aux') or {})
            self._aux_on = bool(self.reward_aux.get('enabled'))
            self._aux_lambda = float(self.reward_aux.get('lambda', 0.2))
            self._aux_upstream = bool(self.reward_aux.get('include_upstream', True))
            self._aux_window = self.reward_aux.get('window', 'active')
            self._aux_recovery_s = float(self.reward_aux.get('recovery_s', 300.0))
            self._aux_lane_cache = {}
            self._aux_net = None
            self._aux_stats = {'applied_steps': 0, 'total': 0.0}
            if self.event_input not in ('bert', 'structured'):
                raise ValueError(f'Invalid event_input {self.event_input!r}')
            self.text_width = (768 if self.event_input == 'bert'
                               else ZTASK_FIELDS)
            self._ztask = None   # lazily built grounder/binder pair
            super().__init__(world, rank)
            self.sub_agents = len(self.node_ids)
            if any(len(i.phases) != self.action_space.n for i in world.intersections):
                raise ValueError('The hz4x4 reference expects equal local action dimensions')
            self.condition = os.environ.get('TARL_EVENT_CONDITION', 'normal')
            self.text_condition = os.environ.get('TARL_TEXT_CONDITION', 'canonical')
            if self.text_condition not in CONDITIONS:
                raise ValueError('Invalid TARL text condition')
            self.runtime = install_event_runtime(world, load_schedule(self.condition))
            text_methods = (PaperTARLPolicy.TEXT_METHODS if self.impl == 'paper'
                            else TEXT_METHODS)
            self.text_encoder = (FrozenTextEncoder()
                                 if method in text_methods
                                 and self.event_input == 'bert' else None)
            self._generator_objects = None
            self.reset()
            self.protocol_version = 'tarl_graph_v2' if self.impl == 'paper' else 'tarl_graph_v1'
            if self.event_input == 'structured':
                self.protocol_version += '_struct'
            self.output_dir = Path(Registry.mapping['logger_mapping']['path'].path)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            qualification = self.output_dir / 'reproduction_qualification.json'
            if qualification.exists():
                previous = json.loads(qualification.read_text())
                if previous.get('protocol_version') != self.protocol_version:
                    raise ValueError('Cannot resume a legacy run into the corrected graph protocol')
            qualification.write_text(json.dumps({
                'protocol_version': self.protocol_version, 'eligible_for_paper_results': False,
                'protocol_id': os.environ.get('TARL_PROTOCOL_ID'),
                'frozen_manifest': os.environ.get('TARL_FROZEN_MANIFEST'),
                'impl': self.impl,
                'status': 'validation_pending', 'method': method, 'event_condition': self.condition,
                'text_condition': self.text_condition if self.text_encoder is not None else 'none',
                'event_input': self.event_input,
                'semantic_aux': bool(self.semantic_aux.get('enabled')),
                'encoder': None if self.text_encoder is None else self.text_encoder.model_name,
                'reason': 'Corrected implementation must pass behavioral pilots before formal admission',
            }, indent=2) + '\n')
            self._init_sga_gb()

        # ------------------------------------------------------------------
        # ATT-ENTITY-004: staged-freeze / traffic-lr-scale / SGA-GB plumbing.
        # All knobs come from model config; defaults keep legacy behaviour.
        TRAFFIC_MODULES = ('traffic_mlp', 'state_projection', 'traffic')

        def _init_sga_gb(self):
            param = Registry.mapping['model_mapping']['setting'].param
            self.freeze_params = frozenset(param.get('freeze_params') or ())
            self.traffic_lr_scale = param.get('traffic_lr_scale')
            self.sga_gb = param.get('sga_gb') or {}
            self._sga_gb_on = bool(self.sga_gb.get('enabled'))
            self.grad_ledger = bool(param.get('grad_ledger', False)) or self._sga_gb_on
            # Probe cadence: diagnostic ledger runs sample r_t every
            # grad_probe_interval updates (0 disables probing entirely, e.g.
            # formal performance runs); SGA-GB refreshes r_t/r_ema every
            # sga_gb.update_interval updates and reuses the cached value in
            # between.  Both default to 1 = legacy every-step behaviour.
            self._probe_interval = int(param.get('grad_probe_interval', 1))
            self._sga_interval = int(self.sga_gb.get('update_interval', 1))
            self._r_ema = None
            self._train_steps = 0
            self._empty_emb_cache = None
            if self.freeze_params:
                frozen = []
                for name, p in self.model.named_parameters():
                    if name.split('.')[1] in self.freeze_params:
                        p.requires_grad_(False)
                        frozen.append(name)
                if not frozen:
                    raise ValueError(f'freeze_params matched nothing: {sorted(self.freeze_params)}')
            if self.traffic_lr_scale is not None:
                lr = self.learning_rate
                traffic, rest = [], []
                for name, p in self.model.named_parameters():
                    (traffic if name.split('.')[1] in self.TRAFFIC_MODULES
                     else rest).append(p)
                if not traffic:
                    raise ValueError('traffic_lr_scale set but no traffic params found')
                self.optimizer = torch.optim.RMSprop(
                    [{'params': traffic, 'lr': lr * float(self.traffic_lr_scale)},
                     {'params': rest, 'lr': lr}],
                    lr=lr, alpha=0.9, centered=False, eps=1e-7)
            if self.grad_ledger and hasattr(self.model.policy, 'grad_probe'):
                interval = (self._sga_interval if self._sga_gb_on
                            else self._probe_interval)
                self.model.policy.grad_probe = interval != 0
            self._ledger_path = self.output_dir / 'grad_ledger.jsonl'

        def _traffic_params(self):
            return [p for n, p in self.model.named_parameters()
                    if n.split('.')[1] in self.TRAFFIC_MODULES and p.requires_grad]

        def _empty_emb(self):
            if self._empty_emb_cache is None:
                self._empty_emb_cache = self.text_encoder([''])[0].numpy()
            return self._empty_emb_cache

        def _event_active(self, obs):
            """last_obs[..., lane:lane+text_width] carries a live event."""
            if self.event_input == 'structured':
                a = np.asarray(obs, dtype=np.float32)
                lane = self.ob_generator.ob_length
                if a.shape[-1] < lane + ZTASK_FIELDS:
                    return 0.0
                return float(a.reshape(-1, a.shape[-1])[:, lane].max() > 0.5)
            if self.text_encoder is None:
                return 0.0
            a = np.asarray(obs, dtype=np.float32)
            lane = self.ob_generator.ob_length
            if a.shape[-1] < lane + 768:
                return 0.0
            emb = a.reshape(-1, a.shape[-1])[0, lane:lane + 768]
            return float(np.linalg.norm(emb - self._empty_emb()) > 1e-3)

        def _build_model(self):
            return _QWrapper(self.ob_length, self.action_space.n, method,
                             self.adjacency, self.ob_generator.ob_length,
                             impl=self.impl, text_width=self.text_width,
                             text_input=self.event_input,
                             semantic_aux=bool(self.semantic_aux.get('enabled')),
                             fusion_gate=self.fusion_gate,
                             ztask_vocabs=self.ztask_vocabs)

        def reset(self):
            self._generator_objects = tuple(self.world.intersections)
            self._guard_stats = {'fired_steps': 0, 'fired_nodes': 0}
            # event ids repeat across episode schedules; lane caches are
            # per-episode (the world/runtime objects are rebuilt on reset).
            self._aux_lane_cache = {}
            self._aux_stats = {'applied_steps': 0, 'total': 0.0}
            self._aux_last = 0.0
            self.obs_generators = [LaneVehicleGenerator(self.world, i, ['lane_count'], in_only=True) for i in self.world.intersections]
            self.phase_generators = [IntersectionPhaseGenerator(self.world, i, ['phase'], targets=['cur_phase'], negative=False) for i in self.world.intersections]
            self.reward_generators = [LaneVehicleGenerator(self.world, i, ['lane_waiting_count'], in_only=True, average='all', negative=True) for i in self.world.intersections]
            self.queue_generators = [LaneVehicleGenerator(self.world, i, ['lane_waiting_count'], in_only=True) for i in self.world.intersections]
            self.delay_generators = [LaneVehicleGenerator(self.world, i, ['lane_delay'], in_only=True, average='all') for i in self.world.intersections]

        def _refresh_generators(self):
            if self._generator_objects != tuple(self.world.intersections):
                self.reset()
                # world.reset() replaces Intersection objects, so the
                # TextEntityBinder's captured generators go stale.
                self._ztask = None

        def policy_texts(self):
            # Prefer the project's own SUMO event runtime when the shared
            # plan-book machinery is installed (world._sumo_event_runtime).
            # Its texts() broadcasts the joined public reports to every node,
            # matching the paper's global e_t semantics. The TARL-internal
            # runtime stays at its configured condition (normal = no events)
            # and only tracks resets when the project runtime is present.
            runtime = getattr(self.world, '_sumo_event_runtime', None) or self.runtime
            raw = runtime.texts(float(self.world.get_current_time()), self.node_ids)
            if self.event_input == 'structured':
                # Text is audit-logging only in structured mode; utility
                # conditions are applied on the grounded feature matrix.
                return raw
            c = getattr(self.world, '_episode_text_condition', None) \
                or self.text_condition
            return apply_condition(raw, c)

        # ---- structured event input (event_input='structured') ------------
        # Per-node raw slot: [present, type, scope, direction, movement, sev].
        # produced by ReportGrounder over runtime.reports() then localized by
        # TextEntityBinder's node_report_mask; Normal rows stay all-zero.
        _ZTASK_TYPE = {'lane_blockage': 1, 'road_closure': 2, 'global_rain': 3,
                       'demand_surge': 4}
        _ZTASK_SCOPE = {'lane': 1, 'edge': 2, 'network': 3}
        _ZTASK_DIR = {'eastbound': 1, 'westbound': 2, 'northbound': 3,
                      'southbound': 4}
        _ZTASK_MOV = {'left-turn': 1, 'through': 2, 'right-turn': 3,
                      'U-turn': 4, 'multiple': 5}   # 0 = none/no single movement

        def _ztask_pipeline(self):
            if self._ztask is None:
                from world.sumo_events.grounding import ReportGrounder
                from generator.text_entity import TextEntityBinder
                runtime = getattr(self.world, '_sumo_event_runtime', None)
                if runtime is None:
                    raise RuntimeError('structured input requires '
                                       'world._sumo_event_runtime')
                grounder = ReportGrounder(runtime.network_catalog(),
                                          strict=False)
                binder = TextEntityBinder(
                    grounder.catalog, self.node_ids, self.obs_generators,
                    closure_binding=self.closure_binding,
                    closure_bind_max_hops=self._closure_bind_hops)
                self._ztask = (runtime, grounder, binder)
            return self._ztask

        def _ztask_features(self):
            runtime, grounder, binder = self._ztask_pipeline()
            feat = np.zeros((len(self.node_ids), ZTASK_FIELDS),
                            dtype=np.float32)
            grounded = [g for g in grounder.map_reports(runtime.reports())
                        if g.mapping_status == 'mapped']
            # Live reports: announced (surge advance notice) + active.  Field 0
            # is the lifecycle code 0/1/2 — a present-gate distinguishes
            # bound-ness, the code value itself is semantic.
            _LIFECYCLE = {'announced': 1.0, 'active': 2.0}
            live = sorted(
                (g for g in grounded if g.report_status in _LIFECYCLE),
                key=lambda g: g.report_id)
            if live:
                batch = binder.bind(grounded)
                g = batch.to_gnode(active_only=True)
                mask = g.node_report_mask          # [N, M_live]
                fallback_mask = getattr(g, 'node_fallback_mask', None)
                if fallback_mask is not None and fallback_mask.any():
                    self._audit_closure_bind(live, fallback_mask)
                node_rank = {n: i for i, n in enumerate(self.node_ids)}
                for j, rep in enumerate(live):
                    movements = rep.movements or ()
                    mov = (movements[0] if len(movements) == 1
                           else ('multiple' if movements else 'none'))
                    if rep.rate_vph is not None:
                        sev = rep.rate_vph / 3600.0
                    else:
                        sev = (1.0 - rep.speed_percent / 100.0
                               if rep.speed_percent is not None else 1.0)
                    row = np.array(
                        [_LIFECYCLE[rep.report_status],
                         self._ZTASK_TYPE.get(rep.event_kind, 0),
                         self._ZTASK_SCOPE.get(rep.scope, 0),
                         self._ZTASK_DIR.get(rep.direction, 0),
                         self._ZTASK_MOV.get(mov, 0), sev],
                        dtype=np.float32)
                    nodes = list(np.where(mask[:, j])[0])
                    # demand_surge reports name a public destination junction
                    # ('bound for junction J'): the surge row lands on that
                    # controlled node too, so the corridor endpoint is visible
                    # without leaking the privileged route.
                    if rep.event_kind == 'demand_surge':
                        for m in rep.mentions:
                            if m.role == 'bound' and m.entity_id in node_rank:
                                nodes.append(node_rank[m.entity_id])
                    for i in nodes:
                        # First live report wins on multi-event nodes (rare;
                        # deterministic by sorted report_id).
                        if feat[i, 0] == 0:
                            feat[i] = row
            return self._apply_ztask_condition(feat)

        def _audit_closure_bind(self, active, fallback_mask):
            """Append one row per upstream-fallback-bound closure report."""
            try:
                path = self.output_dir / 'closure_bind_log.jsonl'
                now = float(self.world.get_current_time())
                with path.open('a', encoding='utf-8') as handle:
                    for j, rep in enumerate(active):
                        nodes = np.where(fallback_mask[:, j])[0]
                        if nodes.size == 0:
                            continue
                        edge = None
                        if rep.target_lane_ids:
                            _, grounder, _ = self._ztask
                            edge = grounder.catalog.lane(
                                rep.target_lane_ids[0])['edge_id']
                        handle.write(json.dumps({
                            't': now, 'report_id': rep.report_id,
                            'edge_id': edge,
                            'bound_node': self.node_ids[int(nodes[0])]},
                            sort_keys=True) + '\n')
            except Exception:
                pass

        def _effective_text_condition(self):
            """Text condition that actually gates this episode's inputs:
            paired-curriculum plan rows override the process-level
            TARL_TEXT_CONDITION; eval-role resets never set the override."""
            if self.event_input != 'structured' and self.text_encoder is None:
                return 'none'
            return getattr(self.world, '_episode_text_condition', None) \
                or self.text_condition

        def _apply_ztask_condition(self, feat):
            c = self._effective_text_condition()
            if c == 'canonical':
                return feat
            if c == 'empty':
                return np.zeros_like(feat)
            if c == 'wrong_location':
                # Rotate event rows across nodes: grounded content stays
                # coherent but the bound junction is wrong.
                return np.roll(feat, 4, axis=0)
            if c == 'wrong_type':
                out = feat.copy()
                hit = out[:, 0] > 0.5
                out[hit, 1] = (out[hit, 1] % 3) + 1   # cycle 1->2->3->1
                return out
            if c == 'both_wrong':
                return np.roll(self._ztask_wrong_type(feat), 4, axis=0)
            raise ValueError(f'Unsupported structured text condition {c!r}')

        def _ztask_wrong_type(self, feat):
            out = feat.copy()
            hit = out[:, 0] > 0.5
            out[hit, 1] = (out[hit, 1] % 3) + 1
            return out

        def get_ob(self):
            self._refresh_generators()
            traffic = np.asarray([g.generate() for g in self.obs_generators], dtype=np.float32)
            if self.event_input == 'structured':
                return np.concatenate([traffic, self._ztask_features()],
                                      axis=-1)
            if self.text_encoder is None:
                return traffic
            texts = self.policy_texts()
            embeddings = self.text_encoder(texts).numpy()
            return np.concatenate([traffic, embeddings], axis=-1)

        def get_phase(self):
            self._refresh_generators()
            return np.concatenate([g.generate() for g in self.phase_generators]).astype(np.int64)

        def get_reward(self):
            return np.asarray([g.generate().item() for g in self.reward_generators], dtype=np.float32) * 12

        def get_queue(self):
            return np.asarray([np.sum(g.generate()) for g in self.queue_generators])

        def get_delay(self):
            return np.asarray([np.mean(g.generate()) for g in self.delay_generators])

        def _features(self, observations, phases):
            if not self.phase:
                return observations
            phase = np.eye(self.action_space.n, dtype=np.float32)[np.asarray(phases, dtype=int)] if self.one_hot else np.asarray(phases)[..., None]
            return np.concatenate([observations, phase], axis=-1)

        def _starvation_guard(self, actions):
            """ATT-ENTITY-004 repair: per-node starvation pocket guard.
            If a node's inflow queue exceeds theta AND its current phase has
            held longer than t_max seconds, override that node's action with
            the MaxPressure rule (argmax over phases of inflow-outflow
            pressure).  Returns (actions, n_overridden)."""
            theta = float(self.starvation_guard.get('theta', 30.0))
            t_max = float(self.starvation_guard.get('t_max_s', 180.0))
            lvc = self.world.get_info('lane_count')
            actions = np.asarray(actions).copy()
            fired = 0
            for i, inter in enumerate(self.world.intersections):
                q_i = float(np.sum(self.queue_generators[i].generate()))
                if q_i <= theta or inter.current_phase_time <= t_max:
                    continue
                best, best_p = int(actions[i]), None
                for p, lanelinks in enumerate(inter.phase_available_lanelinks):
                    pressure = sum(lvc.get(s, 0) - lvc.get(e, 0)
                                   for s, e in lanelinks)
                    if best_p is None or pressure > best_p:
                        best, best_p = p, pressure
                if best != int(actions[i]):
                    actions[i] = best
                    fired += 1
            return actions, fired

        def get_action(self, ob, phase, test=False):
            if not test and np.random.rand() <= self.epsilon:
                return self.sample()
            feats = self._features(ob, phase)
            dropped = False
            if not test and self.text_dropout > 0 and np.random.rand() < self.text_dropout:
                # CFG-style conditioning dropout: train the empty-text mode
                # online by blanking the structured event slot for this
                # decision.
                feats = feats.copy()
                lane = self.ob_generator.ob_length
                feats[..., lane:lane + self.text_width] = 0.0
                dropped = True
            with torch.no_grad():
                q = self.model(torch.as_tensor(feats, dtype=torch.float32))
            actions = q.argmax(dim=-1).numpy()
            fired = 0
            if self._guard_on:
                actions, fired = self._starvation_guard(actions)
                self._guard_stats['fired_steps'] += int(fired > 0)
                self._guard_stats['fired_nodes'] += fired
            weights = self.model.policy.last_fusion_weights
            # Mean over both Attention tokens is always 0.5; report the traffic
            # token separately so the diagnostic can detect modality preference.
            traffic_weights = weights[..., 0] if weights is not None and method == 'attention' else weights
            record = {'simulation_time': float(self.world.get_current_time()), 'reset_index': self.runtime.resets,
                      'test': bool(test), 'event_condition': self.condition,
                      'text_source': ('project_runtime' if getattr(
                          self.world, '_sumo_event_runtime', None) is not None
                          else self.condition),
                      'texts': self.policy_texts() if self.text_encoder is not None else [],
                      'text_condition': self._effective_text_condition(),
                      'fusion_weight_mean': None if weights is None else weights.mean().item(),
                      'fusion_weight_min': None if weights is None else weights.min().item(),
                      'fusion_weight_max': None if weights is None else weights.max().item()}
            record['traffic_weight_mean'] = None if traffic_weights is None else traffic_weights.mean().item()
            record['text_weight_mean'] = None if traffic_weights is None else 1. - traffic_weights.mean().item()
            record['text_dropped'] = dropped
            record['guard_fired_nodes'] = fired
            alpha = getattr(self.model.policy, 'fusion_alpha', None)
            record['fusion_alpha'] = None if alpha is None else float(alpha.detach())
            record['reward_aux_last'] = getattr(self, '_aux_last', 0.0)
            with (self.output_dir / 'tarl_inputs.jsonl').open('a') as f:
                f.write(json.dumps(record) + '\n')
            return actions

        def _in_lanes_to(self, node_id, to_lanes):
            node = self.world.id2intersection.get(node_id)
            if node is None or not to_lanes:
                return ()
            return tuple(sorted({fr for links in node.phase_available_lanelinks
                                 for fr, to in links if to in to_lanes}))

        def _aux_lanes_for(self, ev):
            """Event -> {node_id: [lane_ids]} of corridor lanes whose waiting
            the aux reward amplifies.  Cached per event id; lane sets are
            static for the whole event."""
            if ev.event_id in self._aux_lane_cache:
                return self._aux_lane_cache[ev.event_id]
            runtime = self.world._sumo_event_runtime
            lanes_by_node = {}
            if ev.kind == 'road_closure' and ev.edge_id:
                closed = set(runtime._edges.get(ev.edge_id, ()))
                feeder = runtime._lanes.get(next(iter(closed), ''),
                                            {}).get('from_junction')
                hit = self._in_lanes_to(feeder, closed)
                if hit:
                    lanes_by_node.setdefault(feeder, set()).update(hit)
                if self._aux_upstream:
                    for lane in hit:
                        info = runtime._lanes.get(lane) or {}
                        j0 = info.get('from_junction')
                        e0 = info.get('edge_id')
                        if j0 and e0:
                            lanes_by_node.setdefault(j0, set()).update(
                                self._in_lanes_to(
                                    j0, set(runtime._edges.get(e0, ()))))
            elif ev.kind == 'lane_blockage' and ev.lane_id:
                info = runtime._lanes.get(ev.lane_id) or {}
                j = info.get('to_junction')
                if j:
                    lanes_by_node.setdefault(j, set()).add(ev.lane_id)
            lanes_by_node = {k: sorted(v) for k, v in lanes_by_node.items()
                             if k in self.world.id2intersection}
            self._aux_lane_cache[ev.event_id] = lanes_by_node
            return lanes_by_node

        def _event_aux_vector(self):
            """Per-node aux reward vector for currently ACTIVE events, or None."""
            if not self._aux_on:
                return None
            runtime = getattr(self.world, '_sumo_event_runtime', None)
            if runtime is None or runtime.schedule is None:
                return None
            now = float(self.world.get_current_time())
            vec = np.zeros(len(self.node_ids), dtype=np.float32)
            fired = False
            idx_of = {nid: i for i, nid in enumerate(self.node_ids)}
            for ev in runtime.schedule.events:
                if self._aux_window == 'active':
                    in_win = ev.begin <= now < ev.end
                elif self._aux_window == 'recovery':
                    in_win = ev.end <= now < ev.end + self._aux_recovery_s
                else:  # 'both'
                    in_win = ev.begin <= now < ev.end + self._aux_recovery_s
                if not in_win:
                    continue
                for nid, lanes in self._aux_lanes_for(ev).items():
                    if nid not in idx_of:
                        continue
                    inter = self.world.id2intersection[nid]
                    obs = inter.full_observation or {}
                    qs = [obs[l]['lane_waiting_count'] for l in lanes
                          if l in obs]
                    if not qs:
                        continue
                    # scale by the corridor lanes' share of the node's observed
                    # incoming lanes so lambda reads as "extra weight on the
                    # corridor's native reward contribution"
                    n_in = max(1, len(inter.full_observation or {}))
                    vec[idx_of[nid]] -= (self._aux_lambda * 12.0
                                         * float(np.mean(qs))
                                         * len(qs) / n_in)
                    fired = True
            return vec if fired else None

        def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
            aux = self._event_aux_vector()
            if aux is not None:
                rewards = np.asarray(rewards, dtype=np.float32) + aux
                self._aux_stats['applied_steps'] += 1
                self._aux_stats['total'] += float(aux.sum())
                self._aux_last = float(aux.sum())
            else:
                self._aux_last = 0.0
            # Reuse the parent's replay accounting and retention rules.
            super().remember(last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key)
            _, values = self.replay_buffer[-1]
            horizon = Registry.mapping['trainer_mapping']['setting'].param['steps']
            terminal = bool(done) or float(self.world.get_current_time()) >= horizon
            # ATT-ENTITY-004: store event_active explicitly in the transition
            # (index 7). Analysis/training-gating only; never enters the
            # policy state. Pre-flag items are re-derived from obs at batch time.
            event_active = self._event_active(last_obs)
            self.replay_buffer[-1] = (key, values + (terminal, event_active))

        def _batchwise(self, samples):
            values = [item[1] for item in samples]
            before = np.stack([self._features(v[0], v[1]) for v in values])
            after = np.stack([self._features(v[4], v[5]) for v in values])
            events = np.asarray([v[7] if len(v) > 7 else self._event_active(v[0])
                                 for v in values], dtype=np.float32)
            return (torch.as_tensor(before, dtype=torch.float32), torch.as_tensor(after, dtype=torch.float32),
                    torch.as_tensor(np.stack([v[3] for v in values]), dtype=torch.float32),
                    torch.as_tensor(np.stack([v[2] for v in values]), dtype=torch.long),
                    torch.as_tensor([v[6] for v in values], dtype=torch.float32).unsqueeze(-1),
                    torch.as_tensor(events))

        def is_training_ready(self):
            return len(self.replay_buffer) >= self.batch_size

        def train(self):
            if not self.is_training_ready():
                return np.array(0., dtype=np.float32)
            samples = random.sample(self.replay_buffer, self.batch_size)
            self._record_replay_samples(samples)
            before, after, rewards, actions, terminal, events = self._batchwise(samples)
            if self.text_dropout > 0 and self.event_input == 'structured':
                # CFG-style conditioning dropout in the replay path: blank the
                # z_task slot of a random subset of samples (same mask on
                # before/after keeps conditioning consistent per transition).
                lane = self.ob_generator.ob_length
                drop = torch.rand(before.shape[0]) < self.text_dropout
                before = before.clone()
                after = after.clone()
                before[drop, :, lane:lane + ZTASK_FIELDS] = 0.0
                after[drop, :, lane:lane + ZTASK_FIELDS] = 0.0
            with torch.no_grad():
                target = rewards + self.gamma * (1-terminal) * self.target_model(after).max(dim=-1).values
            prediction = self.model(before, train=True).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            if self.td_loss_fn == 'huber':
                per_sample = torch.nn.functional.smooth_l1_loss(
                    prediction, target, reduction='none', beta=self.huber_beta)
            else:
                per_sample = torch.nn.functional.mse_loss(prediction, target, reduction='none')
            td_loss = per_sample.mean()
            loss = td_loss
            aux_stats = {}
            # Semantic auxiliary loss (ATT-ENTITY-004 revised line): the aux
            # head reads h_it via policy._aux; labels come straight out of the
            # structured obs slot, so no extra replay metadata is needed.
            # present is supervised on every node (it IS the location target);
            # type/movement CEs are masked to present nodes only.
            if (self.semantic_aux.get('enabled')
                    and self.event_input == 'structured'):
                aux = getattr(self.model.policy, '_aux', None)
                if aux is not None:
                    lane = self.ob_generator.ob_length
                    zslot = before[..., lane:lane + ZTASK_FIELDS]
                    # field 0 is a lifecycle code (0/1/2); the aux present head
                    # is binary bound-ness, so binarize the target.
                    present_t = (zslot[..., 0] > 0).float()
                    mask = present_t > 0.5
                    pred = aux['present'] > 0
                    l_present = F.binary_cross_entropy_with_logits(
                        aux['present'], present_t)
                    l_type = l_movement = torch.zeros((), device=loss.device)
                    if bool(mask.any()):
                        l_type = F.cross_entropy(
                            aux['type'][mask], zslot[..., 1].long()[mask])
                        l_movement = F.cross_entropy(
                            aux['movement'][mask], zslot[..., 4].long()[mask])
                        aux_stats['aux_type_acc'] = float(
                            (aux['type'][mask].argmax(-1)
                             == zslot[..., 1].long()[mask]).float().mean())
                        aux_stats['aux_movement_acc'] = float(
                            (aux['movement'][mask].argmax(-1)
                             == zslot[..., 4].long()[mask]).float().mean())
                    aux_stats.update({
                        'aux_td_loss': float(td_loss.detach()),
                        'aux_l_present': float(l_present.detach()),
                        'aux_l_type': float(l_type.detach()),
                        'aux_l_movement': float(l_movement.detach()),
                        'aux_present_tp': float((pred & mask).sum()),
                        'aux_present_fp': float((pred & ~mask).sum()),
                        'aux_present_fn': float((~pred & mask).sum()),
                    })
                    l_sem = l_present + l_type + l_movement
                    lam = float(self.semantic_aux.get('lambda', 1.0))
                    aux_stats['aux_loss'] = float(l_sem.detach())
                    loss = td_loss + lam * l_sem
            self.optimizer.zero_grad()
            # Input-level R_t measurement (fusion inputs sp/e_p) via
            # autograd.grad so the probe never pollutes .grad.  Sampled at the
            # configured cadence; SGA-GB reuses the cached r_ema between
            # refreshes while non-SGA ledgers only emit rows on probe steps.
            interval = (self._sga_interval if self._sga_gb_on
                        else self._probe_interval)
            probe_step = (self.grad_ledger and interval > 0
                          and self._train_steps % interval == 0)
            r_t = None
            if probe_step:
                probe = getattr(self.model.policy, '_probe', {})
                sp_t, ep_t = probe.get('s_p'), probe.get('e_p')
                if sp_t is not None and ep_t is not None:
                    gs, ge = torch.autograd.grad(
                        td_loss, [sp_t, ep_t], retain_graph=True,
                        allow_unused=True)
                    if gs is not None and ge is not None:
                        r_t = float(gs.pow(2).mean().sqrt()
                                    / ge.pow(2).mean().sqrt().clamp_min(1e-12))
            alpha_s = None
            if self._sga_gb_on and self._r_ema is not None and \
                    self._train_steps >= int(self.sga_gb.get('warmup_steps', 500)):
                alpha_s = float(np.clip(
                    float(self.sga_gb.get('r_star', 3.0)) / self._r_ema,
                    float(self.sga_gb.get('alpha_min', 0.1)), 1.0))
            gated = False
            # Per-sample event gating (SGA-GB): the traffic parameter group keeps
            # the normal-sample gradient untouched; only the event samples'
            # contribution is damped by alpha_s.  g_t = g_full - (1-a)*g_event.
            # Selective rescaling of a gradient *component* survives RMSprop's
            # g/sqrt(v) normalisation, unlike a uniform group multiplier.
            if alpha_s is not None and alpha_s < 1.0 and float(events.sum()) > 0:
                tparams = self._traffic_params()
                if tparams:
                    mask = events.unsqueeze(-1).expand_as(per_sample)
                    l_event = (per_sample * mask).sum() / per_sample.numel()
                    g_event = torch.autograd.grad(l_event, tparams, retain_graph=True)
                    loss.backward()
                    for p, ge in zip(tparams, g_event):
                        p.grad = p.grad - (1.0 - alpha_s) * ge
                    gated = True
            if not gated:
                loss.backward()
            if r_t is not None:
                beta = float(self.sga_gb.get('ema_beta', 0.98))
                self._r_ema = r_t if self._r_ema is None else \
                    beta * self._r_ema + (1.0 - beta) * r_t
            # Aux metrics (A3) are cheap and valuable per-step; r_t probes stay
            # on their interval.  `probe_step or sga_gb_on` preserves the old
            # emission rule for runs without semantic aux.
            ledger_row = self._sga_gb_on or probe_step or bool(aux_stats)
            if self.grad_ledger and ledger_row:
                with self._ledger_path.open('a') as f:
                    f.write(json.dumps({
                        'train_step': self._train_steps,
                        'r_t': r_t, 'r_ema': self._r_ema,
                        'alpha_s': alpha_s, 'gated': bool(gated),
                        'm_b': float(events.mean().item()),
                        'eff_mult': (None if alpha_s is None else
                                     1.0 - float(events.mean().item()) * (1.0 - alpha_s)),
                        'loss': float(loss.detach()),
                        'fusion_alpha': (
                            None if getattr(self.model.policy, 'fusion_alpha', None) is None
                            else float(self.model.policy.fusion_alpha.detach())),
                        'guard_fired_steps': self._guard_stats['fired_steps'],
                        'guard_fired_nodes': self._guard_stats['fired_nodes'],
                        **aux_stats,
                    }) + '\n')
            clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            self._train_steps += 1
            return loss.detach().numpy()

    TARLAgent.__name__ = f'TARL_{method}_Agent'
    return TARLAgent
